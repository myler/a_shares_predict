#!/usr/bin/env python3
"""lhb_lianban_height.py: 用 K线精确计算龙虎榜上榜股的连板高度，回测最优高度。

之前用 reason 字段"连续三个交易日"判断连板是模糊的（实为连续3日涨幅偏离20%）。
这里用 klines 真实涨幅精确算连续涨停天数，区分 首板/2板/3板/4板+。
"""
import sqlite3
import pandas as pd
import numpy as np

DB = "stock_cache.db"


def main():
    db = sqlite3.connect(DB)
    lhb = pd.read_sql(
        "SELECT code,name,list_date,reason,interpretation,net_buy_ratio,"
        "turnover_rate,pct_change,market_turnover,after_1d,after_2d,after_5d,after_10d "
        "FROM lhb", db)
    lhb = lhb.drop_duplicates(subset=["code", "list_date"], keep="first")
    lhb = lhb[~lhb["name"].str.contains("退", na=False)]
    lhb = lhb[lhb["market_turnover"] > 3e7]
    lhb["nbr"] = lhb["net_buy_ratio"].clip(-100, 100)

    # 读涉及股票的 K线
    codes = lhb["code"].unique().tolist()
    k = pd.read_sql(
        f"SELECT code,date,close FROM klines WHERE code IN "
        f"({','.join('?' * len(codes))}) ORDER BY code,date", db, params=codes)
    db.close()

    k["pct"] = k.groupby("code")["close"].pct_change() * 100

    # 算每只股票每天的连板高度（连续涨幅>=9.5%的天数）
    def lianban_height(g):
        zt = (g["pct"] >= 9.5).astype(int)
        # 连续涨停天数（从当天往前数）
        h = np.zeros(len(g), dtype=int)
        run = 0
        for i in range(len(g)):
            if zt.iloc[i] == 1:
                run += 1
            else:
                run = 0
            h[i] = run
        return pd.Series(h, index=g.index)

    k["height"] = k.groupby("code", group_keys=False).apply(lianban_height)

    # 关联 lhb：上榜日的连板高度
    k_map = k.set_index(["code", "date"])["height"]
    lhb["height"] = [
        k_map.get((c, d), 0) for c, d in zip(lhb["code"], lhb["list_date"])]

    print(f"样本 {len(lhb)} 条（上榜日连板高度已算）\n")

    def show(label, mask):
        sub = lhb[mask]
        if len(sub) < 30:
            print(f"{label:26s} n={len(sub):4d} 太少"); return
        s1 = sub["after_1d"].dropna(); s2 = sub["after_2d"].dropna()
        s5 = sub["after_5d"].dropna()
        print(f"{label:26s} n={len(sub):4d} | 1日 {s1.mean():+.2f}%(胜{(s1>0).mean()*100:.0f}%)"
              f" | 2日 {s2.mean():+.2f}%(胜{(s2>0).mean()*100:.0f}%)"
              f" | 5日 {s5.mean():+.2f}%(胜{(s5>0).mean()*100:.0f}%)")

    print("=== 连板高度分布 ===")
    for h in [0, 1, 2, 3, 4, 5]:
        print(f"  {h}板(连续{h}涨停): {(lhb['height']==h).sum()} 条")

    print("\n=== 连板高度 × 净买占比>15%（不额外过滤换手） ===")
    base = lhb["nbr"] > 15
    for name, m in [("首板(h=0)", base & (lhb["height"] == 0)),
                    ("1板(h=1)", base & (lhb["height"] == 1)),
                    ("2板(h=2)", base & (lhb["height"] == 2)),
                    ("3板(h=3)", base & (lhb["height"] == 3)),
                    ("4板+(h>=4)", base & (lhb["height"] >= 4))]:
        show(name, m)

    print("\n=== 完整最优：净买>15% + 低换手 + 各连板高度 ===")
    low_turn = lhb["turnover_rate"] <= lhb["turnover_rate"].median()
    for name, m in [("首板+低换手", base & low_turn & (lhb["height"] == 0)),
                    ("2板+低换手", base & low_turn & (lhb["height"] == 2)),
                    ("3板+低换手", base & low_turn & (lhb["height"] == 3)),
                    ("4板++低换手", base & low_turn & (lhb["height"] >= 4)),
                    ("2板及以上+低换手", base & low_turn & (lhb["height"] >= 2))]:
        show(name, m)


if __name__ == "__main__":
    main()
