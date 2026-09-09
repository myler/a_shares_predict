#!/usr/bin/env python3
"""横截面多因子模型 — 因子计算模块。

每个因子输出"值越高越好"的连续序列。因子在横截面上做 percentile rank 标准化后合成总分。
只含可回测的技术因子（从 OHLCV 推导）；基本面/资金因子见 collect_fundamentals.py，属前向叠加层。
"""
import numpy as np
import pandas as pd

TECH_FACTORS = [
    "mom_60", "mom_250", "ma_align",
    "rsi_health", "boll_health", "kdj_health", "bias_health",
    "vol_ratio_health", "obv_flow", "pv_corr",
    "low_vol", "atr_amp", "liq_factor",
]

# 回测训练出的因子权重（60日持仓 ICIR 归一化，2026-09 全量3330只主板回测）
# 正权重=买入方向，负权重=回避方向（反号后仍含信息，体现 A股短中期反转+低波+小市值效应）
FACTOR_WEIGHTS = {
    "low_vol": +0.154,          # 低波动率（经典低波异象，最强正因子）
    "atr_amp": +0.147,          # 低ATR相对幅度
    "liq_factor": +0.144,       # 低成交额（小市值/低流动性效应）
    "bias_health": +0.101,      # 接近MA20（均值回归）
    "boll_health": +0.019,      # 接近布林中轨
    "ma_align": -0.090,         # 回避强多头排列（反转）
    "mom_60": -0.085,           # 回避60日强势股（反转）
    "pv_corr": -0.084,          # 回避量价齐升（反转）
    "mom_250": -0.065,          # 回避250日强势股
    "vol_ratio_health": -0.055, # 回避高量比
    "obv_flow": -0.037,         # 回避OBV流入
    "kdj_health": -0.021,       # 回避KDJ过热
}


def compute_technical_factors(df):
    """df 需含列 [open, high, low, close, volume]（按日期升序）。
    返回 DataFrame（同 index，一列一个因子），值越高越好。"""
    c = df["close"]; h = df["high"]; l = df["low"]; v = df["volume"]
    out = pd.DataFrame(index=df.index)

    # 均线
    ma5 = c.rolling(5).mean(); ma10 = c.rolling(10).mean()
    ma20 = c.rolling(20).mean(); ma60 = c.rolling(60).mean()

    # 1-3 动量 + 均线多头度
    out["mom_60"] = c.pct_change(60)
    out["mom_250"] = c.pct_change(250)
    out["ma_align"] = (ma5 / ma10 - 1) + (ma10 / ma20 - 1) + (ma20 / ma60 - 1)

    # RSI(14)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss)

    # KDJ(9,3,3)
    low_n = l.rolling(9).min(); high_n = h.rolling(9).max()
    rsv = (c - low_n) / (high_n - low_n) * 100
    k = rsv.ewm(com=2).mean(); d = k.ewm(com=2).mean(); j = 3 * k - 2 * d

    # 布林(20,2)
    bm = c.rolling(20).mean(); bs = c.rolling(20).std()
    bb_pos = (c - (bm - 2 * bs)) / (4 * bs)

    # 4-7 超买超卖健康度（钟形，极端值低分）
    out["rsi_health"] = 100 - 2 * (rsi - 50).abs()
    out["boll_health"] = 100 - 200 * (bb_pos - 0.5).abs()
    out["kdj_health"] = 100 - 2 * (j - 50).abs()
    out["bias_health"] = 100 - 200 * (c / ma20 - 1).abs()

    # 8 量比健康度（量比 2.0 最优）
    vr = v / v.rolling(5).mean()
    out["vol_ratio_health"] = 100 - 40 * (vr - 2.0).abs()

    # 9 OBV 5日净量能流
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    out["obv_flow"] = obv.diff(5) / v.rolling(5).sum()

    # 10 量价配合度（20日涨幅 × 量能趋势）
    ret20 = c.pct_change(20)
    voltrend = v.rolling(5).mean() / v.rolling(20).mean()
    out["pv_corr"] = ret20 * (voltrend - 1)

    # 11 低波（年化波动率取负）
    vol = c.pct_change().rolling(20).std() * np.sqrt(252)
    out["low_vol"] = -vol

    # 12 ATR 相对幅度（取负）
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    out["atr_amp"] = -atr / c

    # 13 流动性/市值因子（低成交额 = 小盘/低关注 = 高预期收益，A股小市值+低换手异象）
    amount = c * v
    out["liq_factor"] = -np.log(amount.rolling(20).mean())

    return out


def rank_cross_section(factor_frame):
    """横截面 percentile rank → 0~100（NaN 保留）。输入：每列一个因子、每行一只股票。"""
    return factor_frame.rank(axis=0, pct=True) * 100.0


def composite_scores(factor_frame):
    """因子原始值 → 横截面 rank → 加权合成总分。
    输入 factor_frame：index=股票代码，columns=因子名（原始连续值）。
    返回 Series：每只股票的合成总分（0~100 量级）。"""
    ranked = rank_cross_section(factor_frame)
    score = pd.Series(0.0, index=factor_frame.index)
    w_sum = 0.0
    for fac, w in FACTOR_WEIGHTS.items():
        if fac not in ranked.columns:
            continue
        score += w * ranked[fac].fillna(50.0)  # 缺省中性50
        w_sum += abs(w)
    return score / w_sum if w_sum else score
