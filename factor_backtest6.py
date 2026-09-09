#!/usr/bin/env python3
"""factor_backtest6.py: 短线强势股策略回测（抓涨停前夕）。

与 v1-v5 的低波反转完全相反：因子全部指向「强势启动」——
短期动量、多头排列、量能放大、量价齐升、接近新高。
持有 1/3/5 天，看平均收益(期望)、胜率、大涨/涨停命中率。

对比基准：旧低波反转策略 60日持有 = 56.4% 胜率 / +1.41% 平均收益。
"""
import sys, os, argparse, sqlite3
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_cache.db")
MAINBOARD = ("600", "601", "603", "605", "000", "001", "002", "003")

# 追强势因子（值越高 = 越强势）
STRONG_FACTORS = ["mom_5", "mom_10", "mom_20", "mom_60",
                  "ma_align", "vol_ratio", "pv_corr", "near_high"]


def compute_strong_factors(df):
    """df 含 [open,high,low,close,volume]，返回强势因子 DataFrame。"""
    c = df["close"]; v = df["volume"]
    out = pd.DataFrame(index=df.index)
    out["mom_5"] = c.pct_change(5)
    out["mom_10"] = c.pct_change(10)
    out["mom_20"] = c.pct_change(20)
    out["mom_60"] = c.pct_change(60)
    ma5 = c.rolling(5).mean(); ma10 = c.rolling(10).mean()
    ma20 = c.rolling(20).mean()
    out["ma_align"] = (ma5 / ma10 - 1) + (ma10 / ma20 - 1)
    out["vol_ratio"] = v / v.rolling(5).mean()          # 量比，追放量
    ret20 = c.pct_change(20)
    voltrend = v.rolling(5).mean() / v.rolling(20).mean()
    out["pv_corr"] = ret20 * (voltrend - 1)             # 量价齐升
    out["near_high"] = c / c.rolling(20).max()          # 接近20日新高(≤1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)     # 股票数限制(调试)
    ap.add_argument("--days", type=int, default=250)    # 回测最近N个交易日
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
    cal = cal[-args.days:]
    print(f"{len(codes)} 只 | 最近 {len(cal)} 个交易日 | top{args.top}")

    # 预计算每只股票的强势因子序列
    factor_seq = {}   # code -> {fac: array}
    fwd_seq = {}      # code -> dict: hold -> array(未来hold天收益)
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
        f = compute_strong_factors(df).reindex(cal)
        factor_seq[code] = {fac: f[fac].values for fac in STRONG_FACTORS}
        close_map[code] = df["close"].reindex(cal).values
        valid_codes.append(code)

    n_stocks = len(valid_codes)
    n_days = len(cal)
    print(f"有效 {n_stocks} 只 | 算横截面...")

    # 逐日横截面选 top N，持有 hold 天
    for hold in [1, 3, 5]:
        rets = []
        up5 = []   # 大涨命中(hold天内最大涨幅>=5%)
        limit = [] # 涨停命中(hold天内最大涨幅>=9.5%)
        for di in range(n_days - hold):
            # 当日横截面 rank 合成（等权）
            scores = np.zeros(n_stocks)
            valid_mask = np.ones(n_stocks, bool)
            for fac in STRONG_FACTORS:
                col = np.array([factor_seq[c][fac][di] for c in valid_codes])
                valid_mask &= ~np.isnan(col)
            idx_valid = np.where(valid_mask)[0]
            if len(idx_valid) < args.top * 2:
                continue
            for fac in STRONG_FACTORS:
                col = np.array([factor_seq[c][fac][di] for c in valid_codes])
                r = pd.Series(col[idx_valid]).rank(pct=True).values * 100.0
                scores[idx_valid] += r
            order = np.argsort(-scores)[:args.top]
            top_idx = [i for i in order if valid_mask[i]][:args.top]
            # 未来 hold 天收益：用 close[di+hold]/close[di]-1，及期内最大涨幅
            for i in top_idx:
                c = close_map[valid_codes[i]]
                base = c[di]
                if base <= 0 or np.isnan(base):
                    continue
                fwd = c[di + hold] / base - 1.0
                if np.isnan(fwd):
                    continue
                rets.append(fwd)
                # 期内最大涨幅（近似用收盘，忽略日内高低）
                window = c[di + 1: di + hold + 1]
                window = window[~np.isnan(window)]
                if len(window):
                    max_ret = window.max() / base - 1.0
                    up5.append(max_ret >= 0.05)
                    limit.append(max_ret >= 0.095)
        rets = np.array(rets)
        if len(rets) == 0:
            print(f"持有{hold}天: 无有效样本")
            continue
        print(f"\n=== 追强势策略 持有{hold}天 (top{args.top}) ===")
        print(f"  样本 {len(rets)} 笔")
        print(f"  平均收益 {rets.mean()*100:+.2f}%  胜率 {(rets>0).mean()*100:.1f}%")
        print(f"  大涨命中(≥5%) {(np.mean(up5))*100:.1f}%  涨停命中(≥9.5%) {(np.mean(limit))*100:.1f}%")
        print(f"  最好10%平均 {np.quantile(rets,0.9)*100:+.2f}%  最差10%平均 {np.quantile(rets,0.1)*100:+.2f}%")


if __name__ == "__main__":
    main()
