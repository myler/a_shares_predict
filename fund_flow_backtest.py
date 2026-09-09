#!/usr/bin/env python3
"""fund_flow_backtest.py: 主力净流入「涨停前夕」全市场回测（干净数据）。

复杂度注意：全部向量化（merge / groupby），禁止逐行 dict.get 关联。
"""
import sqlite3
import pandas as pd
import numpy as np

DB = "stock_cache.db"


def load():
    db = sqlite3.connect(DB)
    ff = pd.read_sql(
        "SELECT code,date,main_net FROM fund_flow WHERE main_net IS NOT NULL "
        "AND date >= '2025-01-01'", db)
    k = pd.read_sql("SELECT code,date,close FROM klines", db)
    db.close()
    k = k.sort_values(["code", "date"])
    k["pct"] = k.groupby("code")["close"].pct_change() * 100         # 当日涨幅
    k["ret_next"] = k.groupby("code")["close"].pct_change(-1) * 100  # 次日涨幅
    ff = ff.merge(k[["code", "date", "pct", "ret_next"]],
                  on=["code", "date"], how="left")                   # 向量化
    ff = ff.rename(columns={"pct": "pct_today"})
    ff["rank"] = ff.groupby("date")["main_net"].rank(pct=True)
    ff = ff.dropna(subset=["ret_next"]).reset_index(drop=True)
    return ff


def show(label, mask, ff):
    sub = ff.loc[mask]
    r = sub["ret_next"]
    print(f"{label:30s} n={len(sub):6d} | 次日均 {r.mean():+.2f}% | "
          f"胜率 {(r>0).mean()*100:.1f}% | 涨停率 {(r>=9.5).mean()*100:.2f}%")


def main():
    ff = load()
    print(f"全市场资金流样本: {len(ff)} 条 | 基准涨停率 "
          f"{(ff['ret_next']>=9.5).mean()*100:.2f}% | 基准次日均 "
          f"{ff['ret_next'].mean():+.2f}%\n")

    print("=== 1. 主力净流入分位 × 次日表现（全市场） ===")
    for lo, hi, label in [(0, 0.05, "净流出top5%"),
                          (0.5, 0.8, "偏流入(50-80%)"),
                          (0.8, 0.95, "净流入强(80-95%)"),
                          (0.95, 1.0, "净流入top5%")]:
        show(label, (ff["rank"] >= lo) & (ff["rank"] < hi), ff)

    print("\n=== 2. 主力净流入top5% 按当日涨幅分组（找未启动） ===")
    top5 = ff["rank"] >= 0.95
    show("top5 + 当日大涨(>7%)", top5 & (ff["pct_today"] > 7), ff)
    show("top5 + 当日涨3-7%", top5 & (ff["pct_today"] > 3) & (ff["pct_today"] <= 7), ff)
    show("top5 + 当日涨0-3%", top5 & (ff["pct_today"] > 0) & (ff["pct_today"] <= 3), ff)
    show("top5 + 当日下跌(<0%)", top5 & (ff["pct_today"] < 0), ff)

    print("\n=== 3. 净流入top5% + 当日涨3-7%（最优提前信号） 分位细分 ===")
    m = top5 & (ff["pct_today"] > 3) & (ff["pct_today"] <= 7)
    sub = ff.loc[m].copy()
    sub["r2"] = sub.groupby("date")["main_net"].rank(pct=True)
    show("全部(净流入top5%+涨3-7%)", m, ff)
    ff["r2"] = np.nan
    ff.loc[m, "r2"] = sub["r2"].to_numpy()
    m2 = m & (ff["r2"] >= 0.5)
    show("  其中净流入再前50%", m2, ff)

    print("\n=== 4. 连续净流入天数 × 次日表现 ===")
    ff2 = ff.sort_values(["code", "date"]).reset_index(drop=True)
    net_pos = (ff2["main_net"] > 0).astype(int).to_numpy()
    code = ff2["code"].to_numpy()
    streak = np.zeros(len(ff2), dtype=int)
    s, last = 0, None
    for i in range(len(ff2)):
        if code[i] != last:
            s, last = 0, code[i]
        s = s + 1 if net_pos[i] == 1 else 0
        streak[i] = s
    ff2["streak"] = streak
    show("连续3天+净流入", ff2["streak"] >= 3, ff2)
    show("连续5天+净流入", ff2["streak"] >= 5, ff2)


if __name__ == "__main__":
    main()
