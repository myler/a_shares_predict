#!/usr/bin/env python3
"""多因子回测 v4：在 v3 技术+基本面因子基础上，加 市场择时 + 退出机制。

对比 4 个配置：
  A. 无择时 + 固定持有（= v3 基线，应复现 ~56.4%）
  B. 无择时 + 退出机制（止损/止盈/时间）
  C. 择时   + 固定持有
  D. 择时   + 退出机制

择时信号：全市场宽度中位数 mom20>0 且 mom60>0（中期上升趋势确认，牛市才买）。
退出机制：止损 -8% / 止盈 +15% / 持有满 hold 日时间止损。日内触价成交（high/low）。
"""
import sys, os, argparse, sqlite3, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factors import compute_technical_factors, TECH_FACTORS

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_cache.db")
MAINBOARD = ("600", "601", "603", "605", "000", "001", "002", "003")

FUND_FACTORS = ["gross_margin", "net_margin", "np_yoy", "rev_yoy"]


def load_financials(cur):
    rows = cur.execute(
        "SELECT code, report_date, revenue, net_profit, gross_margin, net_margin "
        "FROM financials").fetchall()
    out = {}
    for code, rd, rev, np_, gm, nm in rows:
        out.setdefault(code, []).append((rd, rev, np_, gm, nm))
    res = {}
    for code, lst in out.items():
        lst.sort(key=lambda x: x[0])
        df = pd.DataFrame(lst, columns=["report_date", "revenue", "net_profit",
                                        "gross_margin", "net_margin"])
        df = df.drop_duplicates(subset="report_date", keep="last")
        res[code] = df
    return res


def build_fund_series(fin_df):
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


def simulate_trade(dates, opens, highs, lows, closes, reb_date, hold, stop, take,
                   do_exit):
    """模拟一笔从 reb_date 收盘买入的交易，返回收益（NaN=无法交易）。"""
    pos = np.searchsorted(dates, reb_date)
    if pos >= len(dates) or dates[pos] != reb_date:
        return np.nan                      # 该日停牌/无数据
    if pos + hold >= len(closes):
        return np.nan                      # 持有期数据不足
    buy = closes[pos]
    if not do_exit:
        return closes[pos + hold] / buy - 1.0
    for t in range(pos + 1, pos + hold + 1):
        lo, hi = lows[t], highs[t]
        if lo <= buy * (1 - stop):         # 先触止损（保守）
            return -stop
        if hi >= buy * (1 + take):         # 触止盈
            return +take
    return closes[pos + hold] / buy - 1.0  # 到期时间止损


def run_config(dates_all, codes_all, big, mom20_all, close_maps, reb_dates,
               weights, top, hold, stop, take, do_timing, do_exit):
    trades = []
    skipped = 0
    for d in reb_dates:
        m = dates_all == d
        if m.sum() < top * 2:
            continue
        if do_timing:
            mom20_med = np.nanmedian(mom20_all[m])
            mom60_med = np.nanmedian(big["mom_60"][m])
            if not (mom20_med > 0 and mom60_med > 0):
                skipped += 1
                continue
        comp = np.zeros(m.sum()); ok = np.zeros(m.sum(), bool)
        for f, w in weights.items():
            x = big[f][m]
            r = pd.Series(x).rank(pct=True).values * 100.0
            comp += w * r
            ok |= ~np.isnan(x)
        if ok.sum() < top:
            continue
        order = np.argsort(-comp)
        top_idx = [i for i in order if ok[i]][:top]
        codes_here = codes_all[m][top_idx]
        rets = []
        for c in codes_here:
            dm = close_maps.get(c)
            if dm is None:
                continue
            r = simulate_trade(dm[0], dm[1], dm[2], dm[3], dm[4], d, hold,
                               stop, take, do_exit)
            if not np.isnan(r):
                rets.append(r)
        if rets:
            trades.append(np.mean(rets))
    trades = np.array(trades)
    return trades, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reb-interval", type=int, default=20)
    ap.add_argument("--hold", type=int, default=60)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--stop", type=float, default=0.08)
    ap.add_argument("--take", type=float, default=0.15)
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
    print(f"{len(codes)} 只 | {len(reb_dates)} 调仓日 | 持有{args.hold}日 | "
          f"止损-{args.stop:.0%} 止盈+{args.take:.0%}")

    print("加载财务...")
    fins = load_financials(cur)
    print(f"有财务数据 {len(fins)} 只")

    # E/P 因子：总股本（假设不变）+ 年报净利润（point-in-time）
    shares = {}
    try:
        shares = json.load(open(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "shares.json")))
    except Exception:
        pass
    ann = {}
    for code, rd, np_ in cur.execute(
            "SELECT code, report_date, net_profit FROM financials "
            "WHERE report_date LIKE '%1231' AND net_profit IS NOT NULL"):
        ann.setdefault(code, []).append((rd, float(np_)))
    for c in ann:
        ann[c].sort(key=lambda x: x[0])
        ann[c] = (np.array([f"{r[0][:4]}-{r[0][4:6]}-{r[0][6:]}"
                            for r in ann[c]]),
                  np.array([r[1] for r in ann[c]]))

    cross_tech = {f: [] for f in TECH_FACTORS}
    ep_list = []
    fwd_ret = []
    market_mom20 = []
    close_maps = {}
    valid_codes = []
    for code in codes:
        rec = cur.execute(
            "SELECT date,open,high,low,close,volume FROM klines "
            "WHERE code=? ORDER BY date", (code,)).fetchall()
        if len(rec) < 300:
            continue
        df = pd.DataFrame(rec, columns=["date", "open", "high", "low",
                                        "close", "volume"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df = df.set_index("date")
        f = compute_technical_factors(df).reindex(reb_dates)
        for fac in TECH_FACTORS:
            cross_tech[fac].append(f[fac].values)
        close_all = df["close"]
        pos = close_all.index.get_indexer(reb_dates)
        fwd = np.full(len(reb_dates), np.nan)
        valid = (pos >= 0) & (pos + args.hold < len(close_all))
        pv = pos[valid]
        fwd[valid] = (close_all.iloc[pv + args.hold].values /
                      close_all.iloc[pv].values) - 1.0
        fwd_ret.append(fwd)
        mom20 = df["close"].pct_change(20)
        market_mom20.append(mom20.reindex(reb_dates).values)
        close_maps[code] = (df.index.values, df["open"].values,
                            df["high"].values, df["low"].values,
                            df["close"].values)
        ep = np.full(len(reb_dates), np.nan)
        if code in shares and code in ann:
            rds, nps = ann[code]
            pn = np.searchsorted(rds, np.asarray(df.index), side="right") - 1
            pn = np.clip(pn, 0, len(rds) - 1)
            ep_series = nps[pn] / (shares[code] * 1e8 * df["close"].values)
            ep = pd.Series(ep_series, index=df.index).reindex(reb_dates).values
        ep_list.append(ep)
        valid_codes.append(code)

    big = {f: np.concatenate(cross_tech[f]) for f in TECH_FACTORS}
    big["ep"] = np.concatenate(ep_list)
    fwd_all = np.concatenate(fwd_ret)
    mom20_all = np.concatenate(market_mom20)
    dates_all = np.concatenate([reb_dates] * len(valid_codes))
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

    # IC / ICIR → 权重（与 v3 一致，用固定60日前向收益）
    all_factors = TECH_FACTORS + FUND_FACTORS + ["ep"]
    ic_rows = []
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

    weights = {}
    for fac, mean_ic, icir in ic_rows:
        if abs(icir) > 0.05:
            weights[fac] = icir
    tot = sum(abs(w) for w in weights.values())
    weights = {k: w / tot for k, w in weights.items()}
    print(f"权重: { {k: round(v, 3) for k, v in weights.items()} }")

    configs = [
        ("A 无择时+固定持有", False, False),
        ("B 无择时+退出机制", False, True),
        ("C 择时+固定持有", True, False),
        ("D 择时+退出机制", True, True),
    ]
    print(f"\n=== 配置对比 (top{args.top}, 持有{args.hold}日) ===")
    print(f"{'配置':22s} {'调仓':>5s} {'空仓':>5s} {'平均收益':>9s} "
          f"{'组合胜率':>8s}")
    for name, do_timing, do_exit in configs:
        trades, skipped = run_config(dates_all, codes_all, big, mom20_all,
                                     close_maps, reb_dates, weights, args.top,
                                     args.hold, args.stop, args.take,
                                     do_timing, do_exit)
        if len(trades) == 0:
            print(f"{name:22s} 无有效交易")
            continue
        print(f"{name:22s} {len(trades):5d} {skipped:5d} "
              f"{trades.mean()*100:+8.2f}% {(trades>0).mean()*100:7.1f}%")


if __name__ == "__main__":
    main()
