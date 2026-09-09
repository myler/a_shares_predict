#!/usr/bin/env python3
"""factor_backtest7.py: 启动突破策略回测（抓涨停前夕 v2）。

区别于 v6 的纯动量追涨（会追到见顶股），这里聚焦「低位放量突破平台」：
突破20日新高 + 放量 + 温和启动，即首板/主升浪起点的量价特征。
持有 1/3/5 天，看平均收益、胜率、涨停命中率。
"""
import sys, os, argparse, sqlite3
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_cache.db")
MAINBOARD = ("600", "601", "603", "605", "000", "001", "002", "003")

# 启动突破因子（值越高 = 越像启动）
BO_FACTORS = ["breakout", "vol_ratio", "mom_3", "pos250_low"]


def compute_bo_factors(df):
    c = df["close"]; v = df["volume"]
    out = pd.DataFrame(index=df.index)
    out["breakout"] = c / c.rolling(20).max()          # 突破20日新高(≥1=创新高)
    out["vol_ratio"] = v / v.rolling(5).mean()          # 量比(放量)
    out["mom_3"] = c.pct_change(3)                      # 3日动量(温和启动)
    # 250日位置取负 = 低位启动优先（避免高位接盘）
    rng250 = (c.rolling(250).max() - c.rolling(250).min())
    pos250 = (c - c.rolling(250).min()) / rng250
    out["pos250_low"] = -pos250
    return out


def run(top=50, days=250, limit=0):
    db = sqlite3.connect(DB)
    cur = db.cursor()
    codes = [r[0] for r in cur.execute(
        "SELECT DISTINCT code FROM klines").fetchall()]
    codes = [c for c in codes if c.startswith(MAINBOARD)]
    if limit:
        codes = codes[:limit]

    cal = [r[0] for r in cur.execute(
        "SELECT date FROM klines WHERE code='000001' ORDER BY date").fetchall()]
    cal = cal[250:]
    cal = cal[-days:]

    factor_seq = {}
    close_map = {}
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
        f = compute_bo_factors(df).reindex(cal)
        factor_seq[code] = {fac: f[fac].values for fac in BO_FACTORS}
        close_map[code] = df["close"].reindex(cal).values
        valid_codes.append(code)

    n_stocks = len(valid_codes)
    n_days = len(cal)
    print(f"有效 {n_stocks} 只 | {n_days} 交易日")

    for hold in [1, 3, 5]:
        rets, up5, limit = [], [], []
        for di in range(n_days - hold):
            scores = np.zeros(n_stocks)
            valid_mask = np.ones(n_stocks, bool)
            for fac in BO_FACTORS:
                col = np.array([factor_seq[c][fac][di] for c in valid_codes])
                valid_mask &= ~np.isnan(col)
            idx_valid = np.where(valid_mask)[0]
            if len(idx_valid) < top * 2:
                continue
            for fac in BO_FACTORS:
                col = np.array([factor_seq[c][fac][di] for c in valid_codes])
                r = pd.Series(col[idx_valid]).rank(pct=True).values * 100.0
                scores[idx_valid] += r
            order = np.argsort(-scores)[:top]
            top_idx = [i for i in order if valid_mask[i]][:top]
            for i in top_idx:
                c = close_map[valid_codes[i]]
                base = c[di]
                if base <= 0 or np.isnan(base):
                    continue
                fwd = c[di + hold] / base - 1.0
                if np.isnan(fwd):
                    continue
                rets.append(fwd)
                window = c[di + 1: di + hold + 1]
                window = window[~np.isnan(window)]
                if len(window):
                    max_ret = window.max() / base - 1.0
                    up5.append(max_ret >= 0.05)
                    limit.append(max_ret >= 0.095)
        rets = np.array(rets)
        if len(rets) == 0:
            continue
        print(f"\n=== 启动突破策略 持有{hold}天 (top{top}) ===")
        print(f"  样本 {len(rets)} 笔")
        print(f"  平均收益 {rets.mean()*100:+.2f}%  胜率 {(rets>0).mean()*100:.1f}%")
        print(f"  大涨命中(≥5%) {np.mean(up5)*100:.1f}%  涨停命中(≥9.5%) {np.mean(limit)*100:.1f}%")
        print(f"  最好10% {np.quantile(rets,0.9)*100:+.2f}%  最差10% {np.quantile(rets,0.1)*100:+.2f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--days", type=int, default=250)
    ap.add_argument("--top", type=int, default=50)
    args = ap.parse_args()
    run(top=args.top, days=args.days, limit=args.limit)
