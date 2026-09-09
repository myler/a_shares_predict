#!/usr/bin/env python3
"""lhb_factor_analysis.py: 龙虎榜因子系统挖掘 —— 找最优选股策略。

从 lhb 表提取所有可用特征，扫描单因子 IC + 分组收益，再组合寻优。
标签：after_1d/after_2d/after_5d/after_10d（上榜后收益，百分比数值）。
"""
import sqlite3, re, sys
import pandas as pd
import numpy as np

DB = "stock_cache.db"


def load():
    db = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM lhb", db)
    db.close()
    df = df.drop_duplicates(subset=["code", "list_date"], keep="first")
    # 清洗：退市股 + 僵尸股(成交额<3000万) + 净买占比 clip
    df = df[~df["name"].str.contains("退", na=False)]
    df = df[df["market_turnover"] > 3e7]
    df["nbr"] = df["net_buy_ratio"].clip(-100, 100)
    return df


def featurize(df):
    """特征工程。"""
    interp = df["interpretation"].fillna("")
    reason = df["reason"].fillna("")
    # 机构数量
    df["n_inst"] = interp.str.extract(r"(\d+)家机构").astype(float).fillna(0)
    df["is_inst"] = df["n_inst"] > 0
    # 游资/普通席位（无机构）
    df["is_retail"] = ~df["is_inst"]
    # 买一主买（大单游资）
    df["is_buy1"] = interp.str.contains("买一主买", na=False)
    # 连板（reason 含"连续三个交易日"或"连续三个交易日内"）
    df["is_lianban"] = reason.str.contains("连续三个交易日", na=False)
    # 涨停上榜（涨幅 >= 9.5%）
    df["is_zt"] = df["pct_change"] >= 9.5
    # 大涨上榜（5%~9.5%）
    df["is_da"] = (df["pct_change"] >= 5) & (df["pct_change"] < 9.5)
    # 净买入额（符号 + 量级）
    df["net_buy_sign"] = df["lhb_net_buy"] > 0
    df["net_buy_abs"] = np.log1p(df["lhb_net_buy"].abs())
    # 流通市值对数
    df["log_fcap"] = np.log1p(df["float_market_cap"])
    # 龙虎榜成交额占比（龙虎榜成交额/市场总成交额）
    df["lhb_ratio"] = (df["lhb_turnover"] / df["market_turnover"]).clip(0, 1)
    return df


def ic_scan(df, factors, labels):
    """每个因子对每个标签的 IC（spearman 秩相关）。"""
    print("=" * 70)
    print("单因子 IC（spearman，正=因子高→收益高）")
    print("=" * 70)
    hdr = f"{'因子':<16}" + "".join(f"{l:>10}" for l in labels)
    print(hdr)
    for name, series in factors.items():
        row = f"{name:<16}"
        for l in labels:
            sub = pd.DataFrame({"f": series, "y": df[l]})
            sub = sub.dropna()
            if len(sub) < 100:
                row += f"{'NA':>10}"
            else:
                ic = sub["f"].rank().corr(sub["y"].rank())   # spearman = pearson of ranks
                row += f"{ic:>+10.3f}"
        print(row)


def group_scan(df, factor, label, bins=5):
    """分组收益：因子分 5 组，看每组标签均值。"""
    sub = df[[factor, label]].dropna()
    if len(sub) < 200:
        return None
    sub["g"] = pd.qcut(sub[factor].rank(method="first"), bins, labels=False)
    out = {}
    for g in range(bins):
        s = sub[sub["g"] == g][label]
        out[g] = (len(s), s.mean(), (s > 0).mean())
    return out


def main():
    df = load()
    df = featurize(df)
    print(f"清洗后样本: {len(df)} 条\n")

    labels = ["after_1d", "after_2d", "after_5d", "after_10d"]
    factors = {
        "nbr(净买占比)": df["nbr"],
        "lhb_net_buy(净买额)": df["lhb_net_buy"],
        "net_buy_abs(净买额量)": df["net_buy_abs"],
        "pct_change(涨跌幅)": df["pct_change"],
        "turnover_rate(换手)": df["turnover_rate"],
        "log_fcap(流通市值)": df["log_fcap"],
        "lhb_ratio(龙虎成交占比)": df["lhb_ratio"],
    }
    ic_scan(df, factors, labels)

    # 分类因子的分组收益（涨停/连板/机构/游资）
    print("\n" + "=" * 70)
    print("分类因子分组收益（after_1d / after_5d）")
    print("=" * 70)
    cats = {
        "涨停上榜(>=9.5%)": df["is_zt"],
        "大涨上榜(5-9.5%)": df["is_da"],
        "连板(连续3日)": df["is_lianban"],
        "机构参与": df["is_inst"],
        "游资/普通席位": df["is_retail"],
        "买一主买": df["is_buy1"],
        "净买入(额>0)": df["net_buy_sign"],
    }
    for name, mask in cats.items():
        sub = df[mask]
        s1 = sub["after_1d"].dropna()
        s5 = sub["after_5d"].dropna()
        print(f"{name:<16} n={len(sub):5d} | 1日 {s1.mean():+.2f}%(胜{(s1>0).mean()*100:.0f}%) | 5日 {s5.mean():+.2f}%(胜{(s5>0).mean()*100:.0f}%)")


if __name__ == "__main__":
    main()
