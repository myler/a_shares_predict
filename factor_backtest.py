#!/usr/bin/env python3
"""横截面多因子回测：算每个因子的 IC/ICIR 定权重，再回测合成总分 vs 基线。

用法: python3 factor_backtest.py [--limit N] [--reb-interval 10] [--hold 20] [--top 50]
"""
import sys, os, argparse, sqlite3
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factors import compute_technical_factors, TECH_FACTORS, rank_cross_section

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_cache.db")
MAINBOARD = ("600", "601", "603", "605", "000", "001", "002", "003")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reb-interval", type=int, default=10)
    ap.add_argument("--hold", type=int, default=20)
    ap.add_argument("--top", type=int, default=50)
    args = ap.parse_args()

    db = sqlite3.connect(DB)
    cur = db.cursor()
    codes = [r[0] for r in cur.execute(
        "SELECT DISTINCT code FROM klines").fetchall()]
    codes = [c for c in codes if c.startswith(MAINBOARD)]
    if args.limit:
        codes = codes[:args.limit]
    print(f"回测 {len(codes)} 只主板")

    # 交易日历（用 000001 平安银行，主板常青树）
    cal = [r[0] for r in cur.execute(
        "SELECT date FROM klines WHERE code='000001' ORDER BY date").fetchall()]
    cal = cal[250:]  # 留 mom_250 预热
    reb_dates = cal[:-args.hold][::args.reb_interval]
    print(f"日历 {len(cal)} 日 | 调仓日期 {len(reb_dates)} 个 | 持仓 {args.hold} 日")

    # 每只股票算因子，抽取调仓日的因子值 + 前向收益
    cross = {f: [] for f in TECH_FACTORS}
    fwd_ret = []
    date_tags = []

    for code in codes:
        rows = cur.execute(
            "SELECT date,open,high,low,close,volume FROM klines "
            "WHERE code=? ORDER BY date", (code,)).fetchall()
        if len(rows) < 300:
            continue
        df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df = df.set_index("date")
        f = compute_technical_factors(df)
        f = f.reindex(reb_dates)  # 只保留调仓日
        c = df["close"].reindex(reb_dates)
        # 前向收益：用原始 df 的位置（close[t+hold]/close[t]-1）
        close_all = df["close"]
        pos = close_all.index.get_indexer(reb_dates)
        fwd = np.full(len(reb_dates), np.nan)
        valid = (pos >= 0) & (pos + args.hold < len(close_all))
        pv = pos[valid]
        fwd[valid] = (close_all.iloc[pv + args.hold].values / close_all.iloc[pv].values) - 1.0
        for fac in TECH_FACTORS:
            cross[fac].append(f[fac].values)
        fwd_ret.append(fwd)
        date_tags.append(reb_dates)

    # 合并为长表：每行 (date, factor..., fwd_ret)
    big = {}
    for fac in TECH_FACTORS:
        big[fac] = np.concatenate(cross[fac])
    fwd_all = np.concatenate(fwd_ret)
    dates_all = np.concatenate(date_tags)
    print(f"样本点 {len(fwd_all)} 个")

    # IC 计算：每个因子与 fwd_ret 的横截面 Spearman（按日期分组）
    # 无 scipy，用 pandas rank + pearson = Spearman
    ic_rows = []
    for fac in TECH_FACTORS:
        ic_list = []
        for d in np.unique(dates_all):
            m = dates_all == d
            x = big[fac][m]; y = fwd_all[m]
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 30:
                continue
            rx = pd.Series(x[mask]).rank()
            ry = pd.Series(y[mask]).rank()
            ic_list.append(rx.corr(ry))
        ic = np.array([v for v in ic_list if not np.isnan(v)])
        mean_ic = ic.mean(); std_ic = ic.std()
        icir = mean_ic / std_ic if std_ic > 0 else 0.0
        ic_rows.append((fac, mean_ic, std_ic, icir, len(ic)))
        print(f"  {fac:18s} IC={mean_ic:+.4f} std={std_ic:.4f} ICIR={icir:+.3f} n={len(ic)}")

    # 权重 = ICIR（保留正 ICIR 方向；负 ICIR 因子剔除或反号）
    weights = {}
    for fac, mean_ic, std_ic, icir, n in ic_rows:
        if icir > 0.05 and mean_ic > 0:
            weights[fac] = icir
        elif icir < -0.05 and mean_ic < 0:
            weights[fac] = icir  # 负因子反号后仍有信息
    # 归一化（按 |ICIR|）
    tot = sum(abs(w) for w in weights.values())
    for k in weights:
        weights[k] = weights[k] / tot
    print(f"\n采用因子权重(按ICIR归一化):")
    for k, w in sorted(weights.items(), key=lambda x: -abs(x[1])):
        print(f"  {k:18s} {w:+.3f}")

    # 合成总分回测：每个调仓日，composite = Σ w*rank(因子)，取 top-N
    ranked = {f: None for f in TECH_FACTORS}
    # 对每个调仓日做横截面 rank
    comp_ret = []  # top-N 组合的每次调仓日平均前向收益
    for d in np.unique(dates_all):
        m = dates_all == d
        n_stk = m.sum()
        if n_stk < args.top * 2:
            continue
        comp = np.zeros(n_stk)
        valid_any = np.zeros(n_stk, bool)
        for f, w in weights.items():
            x = big[f][m]
            r = pd.Series(x).rank(pct=True).values * 100.0
            comp += w * r
            valid_any |= ~np.isnan(x)
        y = fwd_all[m]
        ok = valid_any & ~np.isnan(y)
        if ok.sum() < args.top:
            continue
        order = np.argsort(-comp)  # 降序
        # 只取有效且排前 top 的
        top_idx = [i for i in order if ok[i]][:args.top]
        if len(top_idx) < args.top:
            continue
        comp_ret.append(np.nanmean(y[top_idx]))

    comp_ret = np.array(comp_ret)
    win_rate = (comp_ret > 0).mean() * 100
    mean_ret = comp_ret.mean() * 100
    print(f"\n=== 合成模型回测 (top{args.top}, 持有{args.hold}日) ===")
    print(f"调仓次数 {len(comp_ret)} | 组合平均收益 {mean_ret:+.2f}% | 胜率 {win_rate:.1f}%")
    print(f"参考基线：现 S* 全量回测胜率 45.4%（mainboard_baseline 口径）")


if __name__ == "__main__":
    main()
