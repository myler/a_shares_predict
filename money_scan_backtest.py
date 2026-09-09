#!/usr/bin/env python3
"""money_scan_backtest.py: top-N 海选策略资金回测（横截面低波+反转+小市值）。

口径（无前视偏差）:
  调仓日 T（每 HOLD 个交易日一次）：用 T 日及之前数据算 12 技术因子
    → 横截面 percentile rank + ICIR 加权合成总分
    → 过滤（主板 / ≥300日历史 / 收盘价≥min_price）
    → 取 top-N，T+1 开盘买入，持有 HOLD 个交易日收盘卖出
  每只独立 100 万本金，累计总盈亏。

用法: python3 money_scan_backtest.py [-n 5] [--hold 20] [--min-price 5]
"""
import sys, os, argparse, sqlite3, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factors import compute_technical_factors, composite_scores, FACTOR_WEIGHTS

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_cache.db")
MAINBOARD = ("600", "601", "603", "605", "000", "001", "002", "003")
CAPITAL = 1_000_000
START_DATE = "2016-09-08"
COMMISSION = 0.0003   # 佣金万3（双边）
STAMP_DUTY = 0.0005   # 印花税万5（卖出）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--hold", type=int, default=20, help="持有交易日数（调仓频率）")
    ap.add_argument("--min-price", type=float, default=5.0)
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

    # 交易日历
    all_dates = sorted(k["date"].unique().tolist())
    try:
        s0 = all_dates.index(START_DATE)
    except ValueError:
        s0 = 0
    rebalance_dates = all_dates[s0::args.hold]
    print(f"股票 {len(codes)} 只 | 调仓日 {len(rebalance_dates)} 个 "
          f"(每{args.hold}交易日) | 回测 {rebalance_dates[0]} ~ {rebalance_dates[-1]}")

    # 每只股票算因子序列（缓存）—— 预分组避免循环里全表扫描
    k_by_code = {c: g for c, g in k.groupby("code")}
    factor_cache = {}
    for code in codes:
        sub = k_by_code.get(code)
        if sub is None or len(sub) < 300:
            continue
        df = sub.set_index("date")[["open", "high", "low", "close", "volume"]].astype(float)
        factor_cache[code] = compute_technical_factors(df)
    print(f"因子计算完成: {len(factor_cache)} 只 | {time.time()-t0:.0f}s")

    # 价格映射（日期 → (open, close)）
    k2 = k.set_index(["code", "date"])
    open_map = k2["open"].to_dict()
    close_map = k2["close"].to_dict()

    # 每个调仓日选股 + 算收益
    fac_list = list(FACTOR_WEIGHTS.keys())
    total_pnl = 0.0       # 累计盈亏（每期每只固定100万）
    pool = CAPITAL * args.n   # 复利资金池（500万滚动）
    n_picks = 0
    n_trades = 0
    picked_codes = set()  # 去重：整个回测涉及多少只不同股票
    for di, T in enumerate(rebalance_dates[:-1]):
        # 当日因子快照
        rows = []
        row_codes = []
        for code, f in factor_cache.items():
            if T in f.index:
                v = f.loc[T]
                # 价格过滤：T 日收盘价 >= min_price
                if close_map.get((code, T), 0) < args.min_price:
                    continue
                rows.append([v[fc] for fc in fac_list])
                row_codes.append(code)
        if len(rows) < args.n:
            continue
        frame = pd.DataFrame(rows, columns=fac_list, index=row_codes)
        score = composite_scores(frame)
        top = score.nlargest(args.n)
        if len(top) < args.n:
            continue

        # 买入 T+1 开盘，持有 HOLD 个交易日卖出
        ti = all_dates.index(T)
        buy_d = all_dates[ti + 1] if ti + 1 < len(all_dates) else T
        sell_d = all_dates[ti + 1 + args.hold] if ti + 1 + args.hold < len(all_dates) else all_dates[-1]
        period_rets = []
        for code in top.index:
            bp = open_map.get((code, buy_d))
            sp = close_map.get((code, sell_d))
            if bp and sp and bp > 0:
                # 净收益率（扣佣金双边 + 印花税卖出）
                net_ret = sp * (1 - COMMISSION - STAMP_DUTY) / (bp * (1 + COMMISSION)) - 1
                period_rets.append(net_ret)
                n_trades += 1
        if period_rets:
            period_ret = sum(period_rets) / args.n   # 缺失股票记 0 收益
            total_pnl += sum(r * CAPITAL for r in period_rets)
            pool *= (1 + period_ret)
        n_picks += len(top)
        picked_codes.update(top.index)

        if (di + 1) % 20 == 0:
            print(f"  调仓 {di+1}/{len(rebalance_dates)-1} ({T}) | "
                  f"累计盈亏 {total_pnl/1e4:+,.0f}万 | 复利池 {pool/1e4:.0f}万 | "
                  f"{time.time()-t0:.0f}s", flush=True)

    print(f"\n=== top-N 海选资金回测结果 ===")
    print(f"回测 {rebalance_dates[0]} ~ {rebalance_dates[-1]} | 调仓 {len(rebalance_dates)-1} 期")
    print(f"每期选 {args.n} 只 × 100万 | 共 {n_trades} 笔交易 | 去重 {len(picked_codes)} 只不同股票")
    print(f"累计盈亏(固定100万/只): {total_pnl/1e4:+,.0f} 万元 ({total_pnl/1e8:+.4f} 亿)")
    print(f"复利资金池: {CAPITAL*args.n/1e4:.0f}万 → {pool/1e4:.0f}万 "
          f"({pool/(CAPITAL*args.n)-1:+.2%})")
    print(f"耗时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
