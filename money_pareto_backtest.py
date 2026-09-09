#!/usr/bin/env python3
"""money_pareto_backtest.py: 帕累托多维选股回测（非支配排序 + 拥挤度收敛）。

维度（5 个，均横截面 rank 后，越高越好 = 越值得买）：
  1. 低波 low_vol（-年化波动率）
  2. 低ATR atr_amp（-ATR/收盘）
  3. 小市值 liq_factor（-log(20日均成交额)）
  4. 均值回归 bias_health（接近 MA20）
  5. 反转 mom60_rev（-60日动量，回避强势）

口径（无前视）：T日算因子 → 非支配排序找第一前沿 → 拥挤度收敛 top-N
  → T+1 开盘买入，持有 HOLD 交易日收盘卖出，复利滚动。含手续费。

对照模式 --mode weighted：同样的 5 维 rank 后按 ICIR 权重加权求和（一维标量化）。

用法: python3 money_pareto_backtest.py -n 5 --hold 20 [--mode pareto|weighted]
"""
import sys, os, argparse, sqlite3, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factors import compute_technical_factors

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_cache.db")
MAINBOARD = ("600", "601", "603", "605", "000", "001", "002", "003")
CAPITAL = 1_000_000
START_DATE = "2016-09-08"
COMMISSION = 0.0003
STAMP_DUTY = 0.0005

# 5 个维度 + 各自 ICIR 权重（对应 factors.FACTOR_WEIGHTS，用于一维加权对照）
DIMS = {
    "low_vol":   0.154,   # 低波
    "atr_amp":   0.147,   # 低ATR
    "liq_factor": 0.144,  # 小市值
    "bias_health": 0.101, # 均值回归
    "mom60_rev":  0.085,  # 反转（= |mom_60 权重|）
}


def non_dominated_set(points, chunk=512):
    """返回第一前沿（非支配集）的布尔 mask。points: (N,K)，值越高越好。
    分块广播实现，避免 O(N^2) 的 Python 循环。"""
    N = len(points)
    dominated = np.zeros(N, dtype=bool)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        block = points[start:end]            # (c, K)
        diff = block[:, None, :] - points[None, :, :]   # (c, N, K)
        ge = (diff >= 0).all(axis=2)         # (c, N): 块内 j 所有维 >= i
        gt = (diff > 0).any(axis=2)          # (c, N): 块内 j 至少一维 > i
        dominated |= (ge & gt).any(axis=0)   # (N,)
    return ~dominated


def crowding_distance(points):
    """拥挤度距离：每维邻居差之和，越大越分散。边界=inf。"""
    M, K = points.shape
    if M <= 2:
        return np.full(M, np.inf)
    dist = np.zeros(M)
    for k in range(K):
        order = np.argsort(points[:, k])
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        for j in range(1, M - 1):
            dist[order[j]] += points[order[j + 1], k] - points[order[j - 1], k]
    return dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--hold", type=int, default=20)
    ap.add_argument("--min-price", type=float, default=5.0)
    ap.add_argument("--mode", choices=["pareto", "weighted"], default="pareto")
    args = ap.parse_args()
    ap_lim = int(os.environ.get("LIMIT", "0"))

    t0 = time.time()
    db = sqlite3.connect(DB)
    k = pd.read_sql(
        "SELECT code,date,open,high,low,close,volume FROM klines "
        "WHERE date >= '2015-09-01' ORDER BY code,date", db)
    db.close()

    codes = [c for c in k["code"].unique() if c.startswith(MAINBOARD)]
    if ap_lim:
        codes = codes[:ap_lim]

    all_dates = sorted(k["date"].unique().tolist())
    s0 = all_dates.index(START_DATE) if START_DATE in all_dates else 0
    rebalance_dates = all_dates[s0::args.hold]
    print(f"股票 {len(codes)} 只 | 调仓 {len(rebalance_dates)-1} 期 | "
          f"模式 {args.mode} | 维度 {list(DIMS.keys())}")

    # 因子缓存（只存 5 个维度列）
    k_by_code = {c: g for c, g in k.groupby("code")}
    factor_cache = {}
    for code in codes:
        sub = k_by_code.get(code)
        if sub is None or len(sub) < 300:
            continue
        df = sub.set_index("date")[["open", "high", "low", "close", "volume"]].astype(float)
        f = compute_technical_factors(df)
        f["mom60_rev"] = -f["mom_60"]
        factor_cache[code] = f[list(DIMS.keys())]
    print(f"因子计算完成: {len(factor_cache)} 只 | {time.time()-t0:.0f}s")

    k2 = k.set_index(["code", "date"])
    open_map = k2["open"].to_dict()
    close_map = k2["close"].to_dict()

    dim_names = list(DIMS.keys())
    total_pnl = 0.0
    pool = CAPITAL * args.n
    n_trades = 0
    n_front_avg = []
    picked = set()

    for di, T in enumerate(rebalance_dates[:-1]):
        rows = []
        row_codes = []
        for code, f in factor_cache.items():
            if T in f.index:
                v = f.loc[T]
                if close_map.get((code, T), 0) < args.min_price:
                    continue
                rows.append([v[d] for d in dim_names])
                row_codes.append(code)
        if len(rows) < args.n:
            continue
        raw = pd.DataFrame(rows, columns=dim_names, index=row_codes)
        # 横截面 rank（NaN 填中性 50）
        ranked = raw.rank(pct=True) * 100.0
        ranked = ranked.fillna(50.0)
        pts = ranked.values  # (N, K)

        if args.mode == "pareto":
            front_mask = non_dominated_set(pts)
            front_idx = np.where(front_mask)[0]
            if len(front_idx) > args.n:
                cd = crowding_distance(pts[front_idx])
                order = front_idx[np.argsort(-cd)][:args.n]
                top_codes = [row_codes[i] for i in order]
            else:
                top_codes = [row_codes[i] for i in front_idx]
            n_front_avg.append(len(front_idx))
        else:
            w = np.array([DIMS[d] for d in dim_names])
            wsum = np.abs(w).sum()
            score = (pts * w).sum(axis=1) / wsum
            order = np.argsort(-score)[:args.n]
            top_codes = [row_codes[i] for i in order]

        if len(top_codes) < args.n:
            continue

        ti = all_dates.index(T)
        buy_d = all_dates[ti + 1] if ti + 1 < len(all_dates) else T
        sell_d = all_dates[ti + 1 + args.hold] if ti + 1 + args.hold < len(all_dates) else all_dates[-1]
        period_rets = []
        for code in top_codes:
            bp = open_map.get((code, buy_d))
            sp = close_map.get((code, sell_d))
            if bp and sp and bp > 0:
                net_ret = sp * (1 - COMMISSION - STAMP_DUTY) / (bp * (1 + COMMISSION)) - 1
                period_rets.append(net_ret)
                n_trades += 1
        if period_rets:
            period_ret = sum(period_rets) / args.n
            total_pnl += sum(r * CAPITAL for r in period_rets)
            pool *= (1 + period_ret)
        picked.update(top_codes)

        if (di + 1) % 20 == 0:
            print(f"  调仓 {di+1}/{len(rebalance_dates)-1} ({T}) | "
                  f"累计盈亏 {total_pnl/1e4:+,.0f}万 | 复利池 {pool/1e4:.0f}万 | "
                  f"{time.time()-t0:.0f}s", flush=True)

    print(f"\n=== {'帕累托多维' if args.mode=='pareto' else '一维加权'} 选股回测结果 ===")
    print(f"回测 {rebalance_dates[0]} ~ {rebalance_dates[-1]} | 调仓 {len(rebalance_dates)-1} 期")
    print(f"每期选 {args.n} 只 | 共 {n_trades} 笔 | 去重 {len(picked)} 只不同股票")
    if n_front_avg:
        print(f"第一前沿平均规模: {np.mean(n_front_avg):.0f} 只")
    print(f"累计盈亏(固定100万/只): {total_pnl/1e4:+,.0f} 万元 ({total_pnl/1e8:+.4f} 亿)")
    print(f"复利资金池: {CAPITAL*args.n/1e4:.0f}万 → {pool/1e4:.0f}万 ({pool/(CAPITAL*args.n)-1:+.2%})")
    print(f"耗时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
