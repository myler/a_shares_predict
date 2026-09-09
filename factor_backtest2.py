#!/usr/bin/env python3
"""多因子回测 v2：加市场 regime（牛熊过滤）+ 流动性因子，看胜率能否突破。

在 v1 基础上新增：
1. 等权市场指数 + 牛熊 regime（市场价 > MA200 为牛）
2. liq_factor（低成交额=小盘/低流动性效应）
3. 分牛熊报告组合胜率（核心：牛市中选股胜率应显著更高）
"""
import sys, os, argparse, sqlite3
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factors import compute_technical_factors, TECH_FACTORS

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_cache.db")
MAINBOARD = ("600", "601", "603", "605", "000", "001", "002", "003")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reb-interval", type=int, default=20)
    ap.add_argument("--hold", type=int, default=60)
    ap.add_argument("--top", type=int, default=50)
    args = ap.parse_args()

    db = sqlite3.connect(DB)
    cur = db.cursor()
    codes = [r[0] for r in cur.execute(
        "SELECT DISTINCT code FROM klines").fetchall()]
    codes = [c for c in codes if c.startswith(MAINBOARD)]
    if args.limit:
        codes = codes[:args.limit]

    cal = [r[0] for r in cur.execute(
        "SELECT date FROM klines WHERE code='000001' ORDER BY date").fetchall()]
    cal = cal[250:]
    reb_dates = cal[:-args.hold][::args.reb_interval]
    print(f"{len(codes)} 只 | {len(reb_dates)} 调仓日 | 持有{args.hold}日")

    # 第一遍：加载所有 close，算等权市场指数
    print("算市场指数...")
    closes_all = {}
    for code in codes:
        rec = cur.execute(
            "SELECT date, close FROM klines WHERE code=? ORDER BY date", (code,)).fetchall()
        if len(rec) < 300:
            continue
        s = pd.Series([float(r[1]) for r in rec], index=[r[0] for r in rec])
        closes_all[code] = s
    closes_df = pd.DataFrame(closes_all).sort_index()
    mkt_ret = closes_df.pct_change().mean(axis=1)  # 等权日收益
    mkt_idx = (1 + mkt_ret).cumprod()
    mkt_ma200 = mkt_idx.rolling(200).mean()
    regime = (mkt_idx > mkt_ma200).reindex(reb_dates)
    print(f"牛(市场>MA200)调仓日 {regime.sum()} / {len(regime)}")

    # 第二遍：算因子 + 组装横截面
    cross = {f: [] for f in TECH_FACTORS}
    fwd_ret = []; date_tags = []
    for code in codes:
        rec = cur.execute(
            "SELECT date,open,high,low,close,volume FROM klines "
            "WHERE code=? ORDER BY date", (code,)).fetchall()
        if len(rec) < 300:
            continue
        df = pd.DataFrame(rec, columns=["date","open","high","low","close","volume"])
        for col in ["open","high","low","close","volume"]:
            df[col] = df[col].astype(float)
        df = df.set_index("date")
        f = compute_technical_factors(df).reindex(reb_dates)
        close_all = df["close"]
        pos = close_all.index.get_indexer(reb_dates)
        fwd = np.full(len(reb_dates), np.nan)
        valid = (pos >= 0) & (pos + args.hold < len(close_all))
        pv = pos[valid]
        fwd[valid] = (close_all.iloc[pv + args.hold].values / close_all.iloc[pv].values) - 1.0
        for fac in TECH_FACTORS:
            cross[fac].append(f[fac].values)
        fwd_ret.append(fwd); date_tags.append(reb_dates)

    big = {f: np.concatenate(cross[f]) for f in TECH_FACTORS}
    fwd_all = np.concatenate(fwd_ret)
    dates_all = np.concatenate(date_tags)

    # IC
    ic_rows = []
    for fac in TECH_FACTORS:
        ic_list = []
        for d in np.unique(dates_all):
            m = dates_all == d
            x = big[fac][m]; y = fwd_all[m]
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 30:
                continue
            rx = pd.Series(x[mask]).rank(); ry = pd.Series(y[mask]).rank()
            ic_list.append(rx.corr(ry))
        ic = np.array([v for v in ic_list if not np.isnan(v)])
        mean_ic = ic.mean(); std_ic = ic.std()
        icir = mean_ic / std_ic if std_ic > 0 else 0.0
        ic_rows.append((fac, mean_ic, icir))
        print(f"  {fac:16s} IC={mean_ic:+.4f} ICIR={icir:+.3f}")

    # 权重 = ICIR（保留自然符号：正IC→正权重买入，负IC→负权重=反转）
    weights = {}
    for fac, mean_ic, icir in ic_rows:
        if abs(icir) > 0.05:
            weights[fac] = icir
    tot = sum(abs(w) for w in weights.values())
    weights = {k: w / tot for k, w in weights.items()}
    print(f"\n权重: { {k: round(v,3) for k,v in weights.items()} }")

    # 合成 + 分牛熊报告胜率
    def composite_winrate(mask_dates, label):
        comp_ret = []
        for d in mask_dates:
            m = dates_all == d
            n_stk = m.sum()
            if n_stk < args.top * 2:
                continue
            comp = np.zeros(n_stk); ok = np.zeros(n_stk, bool)
            for f, w in weights.items():
                x = big[f][m]
                r = pd.Series(x).rank(pct=True).values * 100.0
                comp += w * r
                ok |= ~np.isnan(x)
            y = fwd_all[m]
            ok &= ~np.isnan(y)
            if ok.sum() < args.top:
                continue
            order = np.argsort(-comp)
            top_idx = [i for i in order if ok[i]][:args.top]
            comp_ret.append(np.nanmean(y[top_idx]))
        comp_ret = np.array(comp_ret)
        if len(comp_ret) == 0:
            print(f"{label}: 无样本")
            return
        wr = (comp_ret > 0).mean() * 100
        print(f"{label}: {len(comp_ret)}次调仓 | 平均收益 {comp_ret.mean()*100:+.2f}% | 胜率 {wr:.1f}%")

    all_dates = np.unique(dates_all)
    bull_dates = np.array([d for d in all_dates if regime.get(d, False)])
    bear_dates = np.array([d for d in all_dates if not regime.get(d, False)])
    print()
    composite_winrate(all_dates, "全部时期")
    composite_winrate(bull_dates, "牛市(市场>MA200)")
    composite_winrate(bear_dates, "熊市")


if __name__ == "__main__":
    main()
