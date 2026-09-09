#!/usr/bin/env python3
"""多因子回测 v3：技术因子 + 基本面因子（价值/成长/质量）联合回测。

基本面因子从 financials 表读（point-in-time：取 rebalance 日前最近一期报告）。
"""
import sys, os, argparse, sqlite3
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factors import compute_technical_factors, TECH_FACTORS

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_cache.db")
MAINBOARD = ("600", "601", "603", "605", "000", "001", "002", "003")

FUND_FACTORS = ["gross_margin", "net_margin", "np_yoy", "rev_yoy"]


def load_financials(cur):
    """返回 {code: DataFrame(report_date, revenue, net_profit, gross_margin, net_margin)}"""
    rows = cur.execute(
        "SELECT code, report_date, revenue, net_profit, gross_margin, net_margin "
        "FROM financials").fetchall()
    out = {}
    for code, rd, rev, np_, gm, nm in rows:
        out.setdefault(code, []).append(
            (rd, rev, np_, gm, nm))
    res = {}
    for code, lst in out.items():
        lst.sort(key=lambda x: x[0])
        df = pd.DataFrame(lst, columns=["report_date", "revenue", "net_profit",
                                        "gross_margin", "net_margin"])
        df = df.drop_duplicates(subset="report_date", keep="last")
        res[code] = df
    return res


def build_fund_series(fin_df):
    """预计算一只股票的 (report_dates, {fac: array})，含 YoY。"""
    fin_df = fin_df.sort_values("report_date")
    rd = fin_df["report_date"].values
    gm = fin_df["gross_margin"].astype(float).values
    nm = fin_df["net_margin"].astype(float).values
    np_ = fin_df["net_profit"].astype(float).values
    rev = fin_df["revenue"].astype(float).values
    np_yoy = np.full(len(rd), np.nan)
    rev_yoy = np.full(len(rd), np.nan)
    idx = {d: i for i, d in enumerate(rd)}
    for i, d in enumerate(rd):
        py = str(int(d[:4]) - 1) + d[4:]
        if py in idx:
            j = idx[py]
            if np_[j] and np_[j] != 0:
                np_yoy[i] = (np_[i] - np_[j]) / abs(np_[j]) * 100
            if rev[j] and rev[j] != 0:
                rev_yoy[i] = (rev[i] - rev[j]) / abs(rev[j]) * 100
    return rd, {"gross_margin": gm, "net_margin": nm,
                "np_yoy": np_yoy, "rev_yoy": rev_yoy}


def fund_factor_at(fin_df, reb_date):
    """返回该股票在 reb_date 的基本面因子值 dict（用最近一期报告，报告日<=reb_date）。"""
    sub = fin_df[fin_df["report_date"] <= reb_date]
    if sub.empty:
        return None
    last = sub.iloc[-1]
    rd = last["report_date"]
    gm = last["gross_margin"]; nm = last["net_margin"]
    np_ = last["net_profit"]; rev = last["revenue"]
    # YoY：找去年同期报告
    yoy_target = rd[:4] + rd[4:]  # 占位
    np_yoy = rev_yoy = None
    prev_rd = str(int(rd[:4]) - 1) + rd[4:]
    prev = fin_df[fin_df["report_date"] == prev_rd]
    if not prev.empty:
        p_np = prev.iloc[-1]["net_profit"]; p_rev = prev.iloc[-1]["revenue"]
        if p_np and p_np != 0:
            np_yoy = (np_ - p_np) / abs(p_np) * 100
        if p_rev and p_rev != 0:
            rev_yoy = (rev - p_rev) / abs(p_rev) * 100
    return {"gross_margin": gm, "net_margin": nm,
            "np_yoy": np_yoy, "rev_yoy": rev_yoy}


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

    print("加载财务...")
    fins = load_financials(cur)
    print(f"有财务数据 {len(fins)} 只")

    # 技术因子
    cross_tech = {f: [] for f in TECH_FACTORS}
    fwd_ret = []; date_tags = []; valid_codes = []
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
            cross_tech[fac].append(f[fac].values)
        fwd_ret.append(fwd); date_tags.append(reb_dates)
        valid_codes.append(code)

    big = {f: np.concatenate(cross_tech[f]) for f in TECH_FACTORS}
    fwd_all = np.concatenate(fwd_ret)
    dates_all = np.concatenate(date_tags)
    codes_all = np.repeat(valid_codes, len(reb_dates))

    # 基本面因子（point-in-time，向量化）
    print("算基本面因子...")
    fund_series = {}
    for code, fin_df in fins.items():
        fund_series[code] = build_fund_series(fin_df)

    for fac in FUND_FACTORS:
        vals = np.full(len(dates_all), np.nan)
        for code in np.unique(codes_all):
            if code not in fund_series:
                continue
            rd, facs = fund_series[code]
            m = codes_all == code
            pos = np.searchsorted(rd, dates_all[m], side="right") - 1
            vals[m] = np.where(pos >= 0, facs[fac][np.clip(pos, 0, len(rd) - 1)],
                               np.nan)
        big[fac] = vals

    # IC
    print("\n因子 IC/ICIR:")
    ic_rows = []
    all_factors = TECH_FACTORS + FUND_FACTORS
    for fac in all_factors:
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

    # 权重 + 合成
    weights = {}
    for fac, mean_ic, icir in ic_rows:
        if abs(icir) > 0.05:
            weights[fac] = icir
    tot = sum(abs(w) for w in weights.values())
    weights = {k: w / tot for k, w in weights.items()}
    print(f"\n权重: { {k: round(v,3) for k,v in weights.items()} }")

    comp_ret = []
    for d in np.unique(dates_all):
        m = dates_all == d
        if m.sum() < args.top * 2:
            continue
        comp = np.zeros(m.sum()); ok = np.zeros(m.sum(), bool)
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
    print(f"\n=== 技术+基本面 合成模型 (top{args.top}, 持有{args.hold}日) ===")
    print(f"{len(comp_ret)}次调仓 | 平均收益 {comp_ret.mean()*100:+.2f}% | 胜率 {(comp_ret>0).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
