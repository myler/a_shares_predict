#!/usr/bin/env python3
"""fund_flow_analysis.py: 验证主力净流入能否提前锁定涨停（涨停前夕）。

核心问题：涨停前的 1-2 天，主力资金是否已经净流入？
若成立，则可用「主力净流入」在涨停前锁定标的，而不是等涨停后的龙虎榜。

方法（在龙虎榜活跃池内）：
  对每个交易日 d，看每只股票当日主力净流入(main_net)，
  标签 = 次日(d+1) 是否涨停(涨幅>=9.5%)。
  验证：主力净流入强的股票，次日涨停率是否显著高于池内基准。
"""
import sqlite3
import pandas as pd
import numpy as np

DB = "stock_cache.db"


def load():
    db = sqlite3.connect(DB)
    ff = pd.read_sql(
        "SELECT code,date,main_net,main_pct,xl_net FROM fund_flow "
        "WHERE main_net IS NOT NULL", db)
    # 涨停标签：从 klines 算每日涨幅
    k = pd.read_sql("SELECT code,date,close FROM klines", db)
    db.close()
    k = k.sort_values(["code", "date"])
    k["pct"] = k.groupby("code")["close"].pct_change() * 100
    k["is_zt_next"] = k.groupby("code")["pct"].shift(-1) >= 9.5  # 次日涨停
    zt_map = k.set_index(["code", "date"])["is_zt_next"]
    ff["is_zt_next"] = [
        zt_map.get((c, d), False) for c, d in zip(ff["code"], ff["date"])]
    return ff


def main():
    ff = load()
    print(f"资金流样本: {len(ff)} 条（日×股）")
    print(f"次日涨停基准率: {ff['is_zt_next'].mean()*100:.2f}%\n")

    # 主力净流入强度分位（按日横截面）
    ff["main_rank"] = ff.groupby("date")["main_net"].rank(pct=True)

    print("=== 主力净流入分位 × 次日涨停率 ===")
    for lo, hi, label in [(0, 0.2, "净流出最大(0-20%)"),
                          (0.2, 0.5, "偏流出(20-50%)"),
                          (0.5, 0.8, "偏流入(50-80%)"),
                          (0.8, 0.95, "净流入强(80-95%)"),
                          (0.95, 1.0, "净流入最强(95-100%)")]:
        sub = ff[(ff["main_rank"] >= lo) & (ff["main_rank"] < hi)]
        zt = sub["is_zt_next"].mean() * 100
        print(f"  {label:<16} n={len(sub):6d} | 次日涨停率 {zt:.2f}%")

    print("\n=== 主力净流入(绝对额)符号 × 次日涨停率 ===")
    for name, m in [("净流入(>0)", ff["main_net"] > 0),
                    ("净流出(<0)", ff["main_net"] < 0)]:
        sub = ff[m]
        print(f"  {name:<12} n={len(sub):6d} | 次日涨停率 {sub['is_zt_next'].mean()*100:.2f}%")

    print("\n=== 连续净流入天数 × 次日涨停率 ===")
    ff = ff.sort_values(["code", "date"])
    ff["net_pos"] = (ff["main_net"] > 0).astype(int)
    def streak(g):
        s = 0
        out = []
        for x in g["net_pos"]:
            s = s + 1 if x == 1 else 0
            out.append(s)
        return pd.Series(out, index=g.index)
    ff["streak"] = ff.groupby("code", group_keys=False).apply(streak)
    for name, m in [("首日净流入(1天)", ff["streak"] == 1),
                    ("连续2天净流入", ff["streak"] == 2),
                    ("连续3天+净流入", ff["streak"] >= 3)]:
        sub = ff[m]
        if len(sub) < 100:
            continue
        print(f"  {name:<14} n={len(sub):6d} | 次日涨停率 {sub['is_zt_next'].mean()*100:.2f}%")


if __name__ == "__main__":
    main()
