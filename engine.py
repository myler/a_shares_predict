#!/usr/bin/env python3
"""核心引擎：MACD计算、背离检测、回测、预测、画图"""

import os, sys, io, base64
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── 字体 ──
FONT_PATH = os.path.join(ROOT, 'wqy-zenhei.ttf')
if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    plt.rcParams['font.family'] = fm.FontProperties(fname=FONT_PATH).get_name()
    plt.rcParams['axes.unicode_minus'] = False
else:
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    plt.rcParams['axes.unicode_minus'] = False


def make_output_dir(code):
    today = datetime.now().strftime('%Y%m%d')
    base = os.path.join(ROOT, 'output')
    n = 1
    while os.path.exists(os.path.join(base, f'{today}_{code}_{n:02d}')):
        n += 1
    d = os.path.join(base, f'{today}_{code}_{n:02d}')
    os.makedirs(d, exist_ok=True)
    return d


# ═══════════════════════════
# MACD 计算
# ═══════════════════════════
def ema(data, period):
    r = np.full(len(data), np.nan)
    if len(data) < period: return r
    r[period-1] = np.mean(data[:period])
    m = 2/(period+1)
    for i in range(period, len(data)): r[i] = data[i]*m + r[i-1]*(1-m)
    return r

def calc_macd(closes, fast=12, slow=26, signal=9):
    n = len(closes)
    ef, es = ema(closes, fast), ema(closes, slow)
    dif = ef - es
    dea = np.full(n, np.nan); dea[slow-1:] = ema(dif[slow-1:], signal)
    bar = (dif - dea) * 2
    return dif, dea, bar


# ═══════════════════════════
# 经典形态检测
# ═══════════════════════════
def detect_classic_patterns(closes, highs, lows, lookback=250):
    """检测经典技术形态，返回 (patterns, score_adjustment)"""
    n = len(closes)
    if n < 60:
        return [], 0
    
    patterns = []
    total_adj = 0
    
    # ── 局部极值查找 ──
    def find_peaks(arr, min_dist=10):
        peaks = []
        for i in range(min_dist, len(arr) - min_dist):
            if arr[i] == max(arr[i-min_dist:i+min_dist+1]):
                peaks.append(i)
        # deduplicate nearby peaks, keep highest
        merged = []
        for p in peaks:
            if not merged or p - merged[-1] > min_dist * 2:
                merged.append(p)
            elif arr[p] > arr[merged[-1]]:
                merged[-1] = p
        return merged
    
    def find_valleys(arr, min_dist=10):
        valleys = []
        for i in range(min_dist, len(arr) - min_dist):
            if arr[i] == min(arr[i-min_dist:i+min_dist+1]):
                valleys.append(i)
        merged = []
        for v in valleys:
            if not merged or v - merged[-1] > min_dist * 2:
                merged.append(v)
            elif arr[v] < arr[merged[-1]]:
                merged[-1] = v
        return merged
    
    peaks = find_peaks(closes)
    valleys = find_valleys(closes)
    
    # ── M顶检测 ──
    if len(peaks) >= 2:
        p1, p2 = peaks[-2], peaks[-1]
        peak_diff_pct = abs(closes[p1] - closes[p2]) / max(closes[p1], closes[p2]) * 100
        if peak_diff_pct < 5 and p2 - p1 >= 20:
            between = closes[p1:p2+1]
            neckline = np.min(between)
            drawdown = (max(closes[p1], closes[p2]) - neckline) / max(closes[p1], closes[p2]) * 100
            if drawdown > 3:
                neckline_broken = closes[-1] < neckline
                score = -25 if neckline_broken else -15
                patterns.append({
                    'type': '双头(M顶)', 'peaks': [p1, p2],
                    'peak_price': round(max(closes[p1], closes[p2]), 2),
                    'neckline': round(neckline, 2),
                    'neckline_broken': neckline_broken, 'score': score
                })
                total_adj += score
    
    # ── 头肩顶检测 ──
    if len(peaks) >= 3:
        p_left, p_head, p_right = peaks[-3], peaks[-2], peaks[-1]
        head_higher = closes[p_head] > closes[p_left] and closes[p_head] > closes[p_right]
        shoulders_similar = abs(closes[p_left] - closes[p_right]) / max(closes[p_left], closes[p_right]) * 100 < 8
        if head_higher and shoulders_similar:
            left_valley = min(closes[p_left:p_head+1]) if p_left < p_head else closes[p_left]
            right_valley = min(closes[p_head:p_right+1]) if p_head < p_right else closes[p_head]
            neckline = min(left_valley, right_valley)
            neckline_broken = closes[-1] < neckline
            score = -30 if neckline_broken else -20
            patterns.append({
                'type': '头肩顶', 'left_shoulder': p_left, 'head': p_head, 'right_shoulder': p_right,
                'head_price': round(closes[p_head], 2),
                'shoulder_price': round((closes[p_left] + closes[p_right]) / 2, 2),
                'neckline': round(neckline, 2),
                'neckline_broken': neckline_broken, 'score': score
            })
            total_adj += score
    
    # ── K线形态: 晨星 / 看涨吞没 / 三白兵 / 看跌吞没 ──
    if n >= 3:
        o = np.array([float('nan')] * n)
        c = closes
        h = highs
        l = lows
        for i in range(1, n):
            o[i] = (closes[i] + closes[i-1]) / 2  # approximate open
        
        # 三白兵 (last 3 days)
        if n >= 4:
            up3 = all(closes[-i] > closes[-i-1] for i in [1, 2, 3])
            if up3:
                patterns.append({'signals': ['三白兵'], 'score': 15})
                total_adj += 15
        
        # 晨星
        if n >= 3:
            day1_bear = closes[-3] < closes[-4] if n >= 4 else True
            day2_small = abs(closes[-2] - closes[-3]) / closes[-3] < 0.02
            day3_bull = closes[-1] > closes[-2] and closes[-1] > closes[-3]
            if day1_bear and day2_small and day3_bull:
                patterns.append({'signals': ['晨星'], 'score': 15})
                total_adj += 15
        
        # 看跌吞没
        if n >= 3:
            prev_up = closes[-3] < closes[-2] if n >= 3 else False
            curr_bear = closes[-1] < closes[-2] * 0.98
            if prev_up and curr_bear:
                patterns.append({'signals': ['看跌吞没'], 'score': -15})
                total_adj -= 15
    
    return patterns, total_adj


def calc_obv(closes, volumes):
    """OBV 能量潮"""
    n = len(closes)
    obv = np.zeros(n)
    obv[0] = volumes[0]
    for i in range(1, n):
        if closes[i] > closes[i-1]:
            obv[i] = obv[i-1] + volumes[i]
        elif closes[i] < closes[i-1]:
            obv[i] = obv[i-1] - volumes[i]
        else:
            obv[i] = obv[i-1]
    return obv


def calc_atr(highs, lows, closes, period=14):
    """ATR 平均真实波幅"""
    n = len(closes)
    tr = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period-1] = np.nanmean(tr[1:period])
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


# ═══════════════════════════
# 三合资本综合策略 — 三面量化
# ═══════════════════════════
def score_technical(closes, highs, lows, volumes):
    """技术面评分 0-100 (权重 30%)"""
    n = len(closes)
    if n < 30:
        return 50, {}, []
    
    dif, dea, bar = calc_macd(closes)
    rsi = calc_rsi(closes)
    k_line, d_line, j_line = calc_kdj(highs, lows, closes)
    bb_upper, bb_mid, bb_lower = calc_bollinger(closes)
    wr = calc_wr(highs, lows, closes)
    atr = calc_atr(highs, lows, closes)
    obv = calc_obv(closes, volumes)
    
    i = n - 1
    while i >= 0 and np.isnan(dif[i]): i -= 1
    if i < 30: return 50, {}, []
    
    patterns, pattern_adj = detect_classic_patterns(closes, highs, lows)
    
    dimensions = {}
    
    # ── 趋势 (25%) ──
    ma60 = np.mean(closes[-60:]) if n >= 60 else np.mean(closes)
    ma20 = np.mean(closes[-20:]) if n >= 20 else np.mean(closes)
    ma10 = np.mean(closes[-10:]) if n >= 10 else np.mean(closes)
    trend_score = 50
    if closes[i] > ma60: trend_score += 15
    if ma10 > ma20 > ma60: trend_score += 20
    elif ma10 > ma20: trend_score += 10
    
    # MACD
    macd_score = 50
    if dif[i] > dea[i]: macd_score += 15
    if dif[i] > 0: macd_score += 15
    if i >= 5 and dif[i] > dif[i-5]: macd_score += 10
    if bar[i] > 0 and i >= 3 and bar[i] > bar[i-3]: macd_score += 10
    dimensions['趋势'] = round(trend_score * 0.5 + macd_score * 0.5, 1)
    
    # ── 动量 (20%) ──
    # RSI
    rsi_val = rsi[i] if not np.isnan(rsi[i]) else 50
    rsi_score = 50
    if 30 <= rsi_val <= 70: rsi_score += 20  # healthy range
    elif rsi_val < 30: rsi_score += 15  # oversold, potential reversal
    elif rsi_val > 70: rsi_score -= 15  # overbought
    
    # KDJ
    kdj_score = 50
    kd_golden = k_line[i] > d_line[i]
    if kd_golden: kdj_score += 20
    elif k_line[i] < d_line[i]: kdj_score -= 15
    if not np.isnan(j_line[i]):
        if j_line[i] < 0: kdj_score += 15
        elif j_line[i] > 100: kdj_score -= 15
    
    # WR
    wr_val = wr[i] if not np.isnan(wr[i]) else 50
    wr_score = 50
    if wr_val > 80: wr_score += 15
    elif wr_val < 20: wr_score -= 15
    
    dimensions['动量'] = round(rsi_score * 0.4 + kdj_score * 0.35 + wr_score * 0.25, 1)
    
    # ── 量能 (20%) ──
    # 量比
    vol_ma20 = np.mean(volumes[-21:-1]) if n >= 21 else np.mean(volumes)
    vol_ratio = volumes[i] / vol_ma20 if vol_ma20 > 0 else 1
    vol_score = 50
    if 1.0 <= vol_ratio <= 2.0: vol_score += 15
    elif vol_ratio > 2.0: vol_score += 10
    elif vol_ratio < 0.5: vol_score -= 10
    
    # OBV
    obv_ma20 = np.mean(obv[-21:-1]) if n >= 21 else np.mean(obv)
    obv_trend = obv[i] > obv_ma20
    obv_score = 50
    if obv_trend: obv_score += 25
    else: obv_score -= 15
    
    # 换手率代理：成交量相对变化
    turnover_score = 50
    if vol_ratio > 1.5: turnover_score += 10
    elif vol_ratio < 0.7: turnover_score -= 10
    
    dimensions['量能'] = round(vol_score * 0.4 + obv_score * 0.35 + turnover_score * 0.25, 1)
    
    # ── 通道/波动 (15%) ──
    bb_pos = (closes[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) * 100 if not np.isnan(bb_upper[i]) and bb_upper[i] != bb_lower[i] else 50
    bb_score = 50
    if 20 <= bb_pos <= 80: bb_score += 15
    elif bb_pos < 5: bb_score += 20  # bottom reversal
    elif bb_pos > 95: bb_score -= 15  # top
    
    atr_val = atr[i] if not np.isnan(atr[i]) else 0
    atr_ratio = atr_val / closes[i] * 100
    atr_score = 50
    if 1.5 <= atr_ratio <= 4: atr_score += 15
    
    dimensions['通道/波动'] = round(bb_score * 0.6 + atr_score * 0.4, 1)
    
    # ── 形态/结构 (10%) ──
    pattern_score = 50 + pattern_adj
    pattern_score = max(10, min(100, pattern_score))
    
    # 斐波那契
    if n >= 60:
        high_60 = np.max(highs[-60:])
        low_60 = np.min(lows[-60:])
        fib_382 = low_60 + (high_60 - low_60) * 0.382
        fib_618 = low_60 + (high_60 - low_60) * 0.618
        fib_score = 50
        if closes[i] > fib_618: fib_score += 10
        elif closes[i] < fib_382: fib_score -= 10
    else:
        fib_score = 50
    
    dimensions['形态/结构'] = round(pattern_score * 0.6 + fib_score * 0.4, 1)
    
    # ── 筹码 (10%) — 用MA位置代理 ──
    chip_score = 50
    if closes[i] > ma20: chip_score += 15
    if closes[i] > ma60: chip_score += 10
    if closes[i] < ma60: chip_score -= 15
    dimensions['筹码'] = round(chip_score, 1)
    
    # 加权总分
    weights = {'趋势': 0.25, '动量': 0.20, '量能': 0.20, '通道/波动': 0.15, '形态/结构': 0.10, '筹码': 0.10}
    total = sum(dimensions[k] * weights[k] for k in weights)
    
    return total, dimensions, patterns


def score_game_theory(closes, volumes, highs, lows):
    """博弈面评分 0-100 (权重 45%) — 量价关系 + 资金流向代理"""
    n = len(closes)
    if n < 30:
        return 50, {}
    
    obv = calc_obv(closes, volumes)
    dimensions = {}
    
    # ── OBV背离检测 ──
    i = n - 1
    obv_ma5 = np.mean(obv[-5:])
    obv_ma20 = np.mean(obv[-20:])
    obv_div_score = 50
    if obv_ma5 > obv_ma20 * 1.05: obv_div_score += 20  # OBV上行
    elif obv_ma5 < obv_ma20 * 0.95: obv_div_score -= 20  # OBV下行
    
    # 量价背离: 价格涨OBV不跟
    price_5d = closes[i] - closes[max(0, i-5)]
    obv_5d_chg = obv[i] - obv[max(0, i-5)]
    if price_5d > 0 and obv_5d_chg < 0: obv_div_score -= 25  # 量价背离
    elif price_5d < 0 and obv_5d_chg > 0: obv_div_score += 20  # 底部吸筹
    
    dimensions['OBV背离'] = round(obv_div_score, 1)
    
    # ── 量价关系 ──
    vol_price_score = 50
    # 近5日量价趋势
    for j in range(max(0, i-4), i+1):
        if closes[j] > closes[j-1] and volumes[j] > volumes[j-1] * 1.2:
            vol_price_score += 5  # 放量上涨
        elif closes[j] < closes[j-1] and volumes[j] > volumes[j-1] * 1.2:
            vol_price_score -= 8  # 放量下跌
    vol_price_score = max(10, min(100, vol_price_score))
    dimensions['量价关系'] = round(vol_price_score, 1)
    
    # ── 主力资金趋势代理 ──
    # 用大成交量日的净方向判断
    vol_threshold = np.percentile(volumes[-60:], 70) if n >= 60 else np.mean(volumes)
    big_vol_days = []
    for j in range(max(0, i-20), i+1):
        if volumes[j] > vol_threshold:
            big_vol_days.append(closes[j] > closes[j-1])  # True = 净买入
    
    fund_score = 50
    if big_vol_days:
        buy_ratio = sum(big_vol_days) / len(big_vol_days)
        if buy_ratio > 0.6: fund_score += 20
        elif buy_ratio > 0.5: fund_score += 10
        elif buy_ratio < 0.4: fund_score -= 15
    
    # 近5日净方向
    net_dir = sum(1 for j in range(max(0, i-4), i+1) if closes[j] > closes[j-1])
    if net_dir >= 4: fund_score += 10
    elif net_dir <= 1: fund_score -= 10
    
    dimensions['资金趋势'] = round(fund_score, 1)
    
    # ── 板块/催化剂代理 ──
    # 用近期波动率判断是否有事件驱动
    recent_volatility = np.std([(closes[j] - closes[j-1]) / closes[j-1] for j in range(max(1, i-10), i+1)]) * 100
    catalyst_score = 50
    if 2 <= recent_volatility <= 5: catalyst_score += 15
    elif recent_volatility > 5: catalyst_score += 25  # 高波动=有催化剂
    dimensions['事件驱动'] = round(catalyst_score, 1)
    
    # 加权
    weights = {'OBV背离': 0.30, '量价关系': 0.25, '资金趋势': 0.30, '事件驱动': 0.15}
    total = sum(dimensions[k] * weights[k] for k in weights)
    
    return total, dimensions


def score_fundamental(closes, volumes, code=None):
    """基本面评分 0-100 (权重 25%) — 基于可计算指标 + PE代理"""
    n = len(closes)
    if n < 60:
        return 50, {}
    
    dimensions = {}
    i = n - 1
    
    # ── PE代理：通过股价相对历史位置判断估值 ──
    high_250 = np.max(closes[-250:]) if n >= 250 else np.max(closes)
    low_250 = np.min(closes[-250:]) if n >= 250 else np.min(closes)
    price_position = (closes[i] - low_250) / (high_250 - low_250) * 100 if high_250 != low_250 else 50
    
    pe_score = 50
    if price_position < 20: pe_score += 25  # 估值低位
    elif price_position < 40: pe_score += 15
    elif price_position > 80: pe_score -= 20  # 估值高位
    elif price_position > 60: pe_score -= 10
    dimensions['估值位置'] = round(pe_score, 1)
    
    # ── 营收增长代理：近一年价格趋势 ──
    if n >= 250:
        yoy_change = (closes[i] - closes[i-250]) / closes[i-250] * 100
        growth_score = 50
        if yoy_change > 30: growth_score += 25
        elif yoy_change > 15: growth_score += 15
        elif yoy_change > 0: growth_score += 5
        elif yoy_change > -15: growth_score -= 10
        else: growth_score -= 20
    else:
        growth_score = 50
    dimensions['增长趋势'] = round(growth_score, 1)
    
    # ── ROE代理：近期盈利能力(涨跌幅) ──
    if n >= 60:
        q_change = (closes[i] - closes[i-60]) / closes[i-60] * 100
        roe_score = 50
        if q_change > 20: roe_score += 25
        elif q_change > 10: roe_score += 15
        elif q_change > 0: roe_score += 5
        elif q_change > -10: roe_score -= 5
        else: roe_score -= 15
    
        # 稳定性加分
        returns = [(closes[j] - closes[j-1]) / closes[j-1] for j in range(i-59, i+1)]
        positive_days = sum(1 for r in returns if r > 0)
        if positive_days > 35: roe_score += 10
    else:
        roe_score = 50
    dimensions['盈利质量'] = round(roe_score, 1)
    
    weights = {'估值位置': 0.40, '增长趋势': 0.30, '盈利质量': 0.30}
    total = sum(dimensions[k] * weights[k] for k in weights)
    
    return total, dimensions


def _quick_pattern_sell(closes, highs, idx, lookback=250):
    """快速形态检测（仅检测已破颈线的看跌形态，用于回测卖出）"""
    n = len(closes)
    if idx < 60:
        return None
    
    start = max(0, idx - lookback)
    window = closes[start:idx+1]
    wlen = len(window)
    
    # ── 局部峰值查找 ──
    peaks = []
    for i in range(20, wlen - 10):
        segment = window[max(0,i-15):min(wlen,i+16)]
        if window[i] == max(segment) and window[i] > np.mean(window[max(0,i-30):i]):
            if not peaks or i - peaks[-1] > 15:
                peaks.append(i)
            elif window[i] > window[peaks[-1]]:
                peaks[-1] = i
    
    # M顶：最近两个峰
    if len(peaks) >= 2:
        p1, p2 = peaks[-2], peaks[-1]
        peak_diff = abs(window[p1] - window[p2]) / max(window[p1], window[p2]) * 100
        if peak_diff < 5 and p2 - p1 >= 15:
            between = window[p1:p2+1]
            neckline = np.min(between)
            if closes[idx] < neckline:
                return f'M顶破颈线({window[max(p1,p2)]:.1f})'
    
    # 头肩顶：最近三个峰
    if len(peaks) >= 3:
        pl, ph, pr = peaks[-3], peaks[-2], peaks[-1]
        if window[ph] > window[pl] and window[ph] > window[pr]:
            shoulder_diff = abs(window[pl] - window[pr]) / max(window[pl], window[pr]) * 100
            if shoulder_diff < 10:
                left_valley = np.min(window[pl:ph+1])
                right_valley = np.min(window[ph:pr+1])
                neckline = min(left_valley, right_valley)
                if closes[idx] < neckline:
                    return f'头肩顶破颈线({window[ph]:.1f})'
    
    # 看跌吞没
    if idx >= 3:
        prev_up = closes[idx-3] < closes[idx-2]
        curr_bear = closes[idx] < closes[idx-1] * 0.97 and closes[idx] < closes[idx-2]
        if prev_up and curr_bear:
            return '看跌吞没'
    
    return None


def backtest_comprehensive(dates, closes, highs, lows, volumes):
    """融合策略回测：MACD核心40% + 多因子30% + 基本面15% + 量能15% + 形态卖出"""
    n = len(closes)
    if n < 60:
        return []

    # ── 预计算指标 ──
    dif, dea, bar = calc_macd(closes)
    rsi_arr = calc_rsi(closes)
    k_arr, d_arr, j_arr = calc_kdj(highs, lows, closes)
    bb_u, bb_m, bb_l = calc_bollinger(closes)
    wr_arr = calc_wr(highs, lows, closes)
    obv_full = calc_obv(closes, volumes)

    trades = []
    pos = None

    for i in range(60, n):
        if np.isnan(dif[i]):
            continue

        # ══════ MACD核心 40% ══════
        macd_score = 50
        if dif[i] > dea[i]: macd_score += 15
        else: macd_score -= 15
        if dif[i] > 0: macd_score += 12
        else: macd_score -= 8
        if i >= 5 and dif[i] > dif[i-5]: macd_score += 8
        else: macd_score -= 5
        if i >= 3 and bar[i] > bar[i-3]: macd_score += 5
        if i >= 30 and closes[i] > np.min(closes[max(0,i-30):i]) * 1.03 and dif[i] > dif[max(0,i-30)]:
            macd_score += 10
        if i >= 30 and closes[i] >= np.max(closes[max(0,i-30):i]) * 0.98 and dif[i] < dif[max(0,i-30)] * 0.9:
            macd_score -= 15

        # ══════ 多因子 30% ══════
        mf_score = 50
        rv = rsi_arr[i] if not np.isnan(rsi_arr[i]) else 50
        kv = k_arr[i] if not np.isnan(k_arr[i]) else 50
        dv = d_arr[i] if not np.isnan(d_arr[i]) else 50
        jv = j_arr[i] if not np.isnan(j_arr[i]) else 50
        wv = wr_arr[i] if not np.isnan(wr_arr[i]) else 50

        if 30 <= rv <= 65: mf_score += 10
        elif rv < 30: mf_score += 15
        elif rv > 80: mf_score -= 15
        elif rv > 70: mf_score -= 8
        if kv > dv: mf_score += 10
        elif kv < dv: mf_score -= 8
        if jv < 0: mf_score += 8
        elif jv > 100: mf_score -= 8
        bb_pos = (closes[i] - bb_l[i]) / (bb_u[i] - bb_l[i]) * 100 if not np.isnan(bb_u[i]) and bb_u[i] != bb_l[i] else 50
        if bb_pos < 10: mf_score += 12
        elif bb_pos > 90: mf_score -= 8
        if wv > 80: mf_score += 8
        elif wv < 20: mf_score -= 8

        # ══════ 基本面 15% ══════
        fund_score = 50
        if i >= 249:
            h250 = np.max(closes[i-249:i+1])
            l250 = np.min(closes[i-249:i+1])
            pos250 = (closes[i] - l250) / (h250 - l250) * 100 if h250 != l250 else 50
            if pos250 < 25: fund_score += 20
            elif pos250 < 40: fund_score += 10
            elif pos250 > 80: fund_score -= 15
        if i >= 250:
            yoy = (closes[i] - closes[i-250]) / closes[i-250] * 100
            if yoy > 20: fund_score += 10
            elif yoy < -20: fund_score -= 10

        # ══════ 量能/博弈 15% ══════
        game_score = 50
        obv_ma20 = np.mean(obv_full[max(0,i-20):i]) if i >= 20 else np.mean(obv_full[:i])
        obv_ma5 = np.mean(obv_full[max(0,i-4):i+1])
        obv_ratio = obv_full[i] / obv_ma20 if obv_ma20 > 0 else 1
        if obv_ma5 > obv_ma20 * 1.08: game_score += 12
        elif obv_ma5 < obv_ma20 * 0.92: game_score -= 12
        chg5 = closes[i] - closes[max(0,i-5)]
        obv_chg5 = obv_full[i] - obv_full[max(0,i-5)]
        if chg5 > 0 and obv_chg5 < 0: game_score -= 18
        elif chg5 < 0 and obv_chg5 > 0: game_score += 12

        # ── 机构参与度代理（回测用：下跌+低波动=机构吸筹）──
        if i >= 60:
            ret_5d = (closes[i] - closes[max(0,i-5)]) / closes[max(0,i-5)] * 100
            ret_20d = (closes[i] - closes[max(0,i-20)]) / closes[max(0,i-20)] * 100
            vol_60d = np.std([(closes[j]-closes[j-1])/closes[j-1] for j in range(max(1,i-60), i+1)]) * 100
            if ret_5d < -3 and vol_60d < 25: game_score += 10  # 下跌低波→机构吸筹
            elif ret_20d < -10 and vol_60d < 25: game_score += 15  # 深跌低波→强吸筹
            elif ret_5d < -3 and vol_60d > 40: game_score -= 8  # 下跌高波→散户恐慌

        # ══════ 综合 ══════
        composite = macd_score * 0.40 + mf_score * 0.30 + fund_score * 0.15 + game_score * 0.15

        # ══════ 门禁 ══════
        gates_pass = obv_ratio >= 0.90 and rv <= 92 and not (macd_score < 35 and mf_score < 40)

        # ══════ 买卖 ══════
        if pos is None:
            if composite >= 65 and gates_pass:
                pos = {
                    'bd': dates[i], 'bp': closes[i], 'bi': i,
                    'macd': round(macd_score, 1), 'mf': round(mf_score, 1),
                    'fund': round(fund_score, 1), 'game': round(game_score, 1),
                    'comp': round(composite, 1)
                }
        else:
            pnl = (closes[i] - pos['bp']) / pos['bp'] * 100
            sell = False
            reason = ''

            if pnl < -8:
                sell = True; reason = f'止损{pnl:.1f}%'
            elif pnl > 25:
                sell = True; reason = f'止盈+{pnl:.1f}%'
            elif composite < 35:
                sell = True; reason = f'综合分{composite:.0f}<35'
            elif pnl > 12 and composite < 50:
                sell = True; reason = f'获利回吐+{pnl:.1f}%'

            # 顶背离强制卖出
            if not sell and i >= 30:
                rh = np.max(closes[max(0,i-30):i])
                if closes[i] >= rh * 0.98 and dif[i] < dif[max(0,i-30)] * 0.85:
                    sell = True; reason = '顶背离'

            # 形态卖出
            if not sell and i - pos['bi'] > 10 and (i - pos['bi']) % 5 == 0:
                pat = _quick_pattern_sell(closes, highs, i)
                if pat:
                    sell = True; reason = pat

            if sell:
                days = i - pos['bi']
                trades.append({
                    'buy_date': pos['bd'], 'sell_date': dates[i],
                    'buy_price': pos['bp'], 'sell_price': closes[i],
                    'profit_pct': round(pnl, 2), 'hold_days': days,
                    'buy_reason': f'融合{pos["comp"]:.0f}(M{pos["macd"]:.0f}/F{pos["mf"]:.0f}/基{pos["fund"]:.0f}/量{pos["game"]:.0f})',
                    'sell_reason': reason
                })
                pos = None

    return trades


def predict_comprehensive(dates, closes, highs, lows, volumes, holding=False):
    """融合策略当前状态预测：MACD核心40% + 多因子30% + 基本面15% + 量能15%"""
    n = len(closes)
    if n < 60:
        return ["数据不足，需要至少60根K线"]

    dif, dea, bar = calc_macd(closes)
    rsi_arr = calc_rsi(closes)
    k_arr, d_arr, j_arr = calc_kdj(highs, lows, closes)
    bb_u, bb_m, bb_l = calc_bollinger(closes)
    wr_arr = calc_wr(highs, lows, closes)
    obv_full = calc_obv(closes, volumes)

    i = n - 1

    # MACD核心
    macd_score = 50
    if dif[i] > dea[i]: macd_score += 15
    else: macd_score -= 15
    if dif[i] > 0: macd_score += 12
    else: macd_score -= 8
    if i >= 5 and dif[i] > dif[i-5]: macd_score += 8
    else: macd_score -= 5
    if i >= 3 and bar[i] > bar[i-3]: macd_score += 5
    if i >= 30 and closes[i] > np.min(closes[max(0,i-30):i]) * 1.03 and dif[i] > dif[max(0,i-30)]:
        macd_score += 10

    # 多因子
    mf_score = 50
    rv = rsi_arr[i] if not np.isnan(rsi_arr[i]) else 50
    kv = k_arr[i] if not np.isnan(k_arr[i]) else 50
    dv = d_arr[i] if not np.isnan(d_arr[i]) else 50
    jv = j_arr[i] if not np.isnan(j_arr[i]) else 50
    wv = wr_arr[i] if not np.isnan(wr_arr[i]) else 50
    if 30 <= rv <= 65: mf_score += 10
    elif rv < 30: mf_score += 15
    elif rv > 80: mf_score -= 15
    if kv > dv: mf_score += 10
    elif kv < dv: mf_score -= 8
    if jv < 0: mf_score += 8
    elif jv > 100: mf_score -= 8
    bb_pos = (closes[i] - bb_l[i]) / (bb_u[i] - bb_l[i]) * 100 if not np.isnan(bb_u[i]) and bb_u[i] != bb_l[i] else 50
    if bb_pos < 10: mf_score += 12
    elif bb_pos > 90: mf_score -= 8
    if wv > 80: mf_score += 8

    # 基本面
    fund_score = 50
    if i >= 249:
        h250 = np.max(closes[i-249:i+1])
        l250 = np.min(closes[i-249:i+1])
        pos250 = (closes[i] - l250) / (h250 - l250) * 100 if h250 != l250 else 50
        if pos250 < 25: fund_score += 20
        elif pos250 < 40: fund_score += 10
        elif pos250 > 80: fund_score -= 15

    # 量能
    game_score = 50
    obv_ma20 = np.mean(obv_full[max(0,i-20):i]) if i >= 20 else np.mean(obv_full[:i])
    obv_ma5 = np.mean(obv_full[max(0,i-4):i+1])
    obv_ratio = obv_full[i] / obv_ma20 if obv_ma20 > 0 else 1
    if obv_ma5 > obv_ma20 * 1.08: game_score += 12
    elif obv_ma5 < obv_ma20 * 0.92: game_score -= 12
    chg5 = closes[i] - closes[max(0,i-5)]
    obv_chg5 = obv_full[i] - obv_full[max(0,i-5)]
    if chg5 > 0 and obv_chg5 < 0: game_score -= 18
    elif chg5 < 0 and obv_chg5 > 0: game_score += 12

    composite = macd_score * 0.40 + mf_score * 0.30 + fund_score * 0.15 + game_score * 0.15

    # 门禁
    gates_pass = obv_ratio >= 0.90 and rv <= 92 and not (macd_score < 35 and mf_score < 40)

    tech_score, tech_dim, patterns = score_technical(closes, highs, lows, volumes)

    lines = []
    lines.append(f"{'='*50}")
    lines.append(f"  三合资本 · 融合策略 — {dates[-1]}")
    lines.append(f"{'='*50}")

    lines.append(f"\n── 📊 四维评分 ──")
    lines.append(f"  MACD核心 (40%): {macd_score:.1f} 分 — DIF{dif[i]:.2f}/DEA{dea[i]:.2f} BAR{bar[i]:.2f}")
    lines.append(f"  多因子 (30%): {mf_score:.1f} 分 — RSI{rv:.0f} K{kv:.0f}/D{dv:.0f}/J{jv:.0f} WR{wv:.0f}")
    lines.append(f"  基本面 (15%): {fund_score:.1f} 分")
    lines.append(f"  量能 (15%): {game_score:.1f} 分 — OBV比值{obv_ratio:.2f}")

    # 机构参与度代理
    if i >= 60:
        ret_5d_val = (closes[i] - closes[max(0,i-5)]) / closes[max(0,i-5)] * 100
        ret_20d_val = (closes[i] - closes[max(0,i-20)]) / closes[max(0,i-20)] * 100
        vol_60d_val = np.std([(closes[j]-closes[j-1])/closes[j-1] for j in range(max(1,i-60), i+1)]) * 100
        if vol_60d_val < 20: inst_label = '🏛️ 机构主导 (低波)'
        elif vol_60d_val < 30: inst_label = '🤝 均衡型'
        else: inst_label = '👤 散户活跃 (高波)'
        inst_sig = ''
        if ret_5d_val < -3 and vol_60d_val < 25: inst_sig = ' ⚡下跌低波→机构吸筹信号'
        elif ret_20d_val < -10 and vol_60d_val < 25: inst_sig = ' ⚡⚡深跌低波→强吸筹信号'
        lines.append(f"  机构代理: {inst_label} 波动率{vol_60d_val:.1f}%{inst_sig}")

    lines.append(f"\n── 🎯 综合判断 ──")
    lines.append(f"  加权总分: {composite:.1f} (M{macd_score:.0f}×0.40 + F{mf_score:.0f}×0.30 + 基{fund_score:.0f}×0.15 + 量{game_score:.0f}×0.15)")

    # 门禁
    if not gates_pass:
        lines.append(f"\n── ⚠️ 质量门禁 ──")
        if obv_ratio < 0.90: lines.append(f"  🔴 OBV严重背离(比值{obv_ratio:.2f}<0.90) → 否决")
        if rv > 92: lines.append(f"  🔴 RSI极度超买({rv:.0f}>92) → 否决")
        if macd_score < 35 and mf_score < 40: lines.append(f"  🔴 双弱(MACD{macd_score:.0f}<35 且 多因子{mf_score:.0f}<40) → 否决")

    # 顶背离
    if i >= 30 and closes[i] >= np.max(closes[max(0,i-30):i]) * 0.98 and dif[i] < dif[max(0,i-30)] * 0.9:
        lines.append(f"  🔴 顶背离警告: 价格近30日高但DIF未确认")

    # 经典形态
    if patterns:
        lines.append(f"\n── 📐 经典形态 ──")
        for p in patterns[:5]:
            if 'type' in p:
                status = '⛔颈线已破' if p.get('neckline_broken') else '未破颈线'
                lines.append(f"  {p['type']}: 评分{p['score']:+d}  {status}")
            elif 'signals' in p:
                lines.append(f"  {', '.join(p['signals'])}: 评分{p['score']:+d}")

    # 信号
    if gates_pass:
        if composite >= 70: signal = '🟢 强烈看多'; action = '增持' if holding else '买入'
        elif composite >= 65: signal = '🟢 偏多'; action = '拿住' if holding else '买入'
        elif composite >= 50: signal = '🟡 中性'; action = '减持' if holding else '观望'
        elif composite >= 40: signal = '🟠 偏空'; action = '减持' if holding else '不买'
        else: signal = '🔴 看空'; action = '卖出' if holding else '不买'
    else:
        signal = '🔴 门禁否决'; action = '观望'

    lines.append(f"\n  信号: {signal}  |  建议: {action}")

    lines.append(f"\n── 📍 关键位 ──")
    ma20 = np.mean(closes[-20:])
    ma60 = np.mean(closes[-60:])
    lines.append(f"  MA20={ma20:.2f}  MA60={ma60:.2f}  布林: 上{bb_u[-1]:.2f} 下{bb_l[-1]:.2f}")

    lines.append(f"\n{'='*50}")
    n_icon = max(1, min(10, int(composite / 10)))
    if holding:
        if '卖出' in action: lines.append(f"  {'❌' * n_icon}  {action}")
        elif '增持' in action or '拿住' in action: lines.append(f"  {'✅' * n_icon}  {action}")
        else: lines.append(f"  {'⚠️' * n_icon}  {action}")
    else:
        if '买入' in action: lines.append(f"  {'✅' * n_icon}  {action}")
        else: lines.append(f"  {'❌' * n_icon}  {action}")
    lines.append(f"{'='*50}")

    return lines


# ═══════════════════════════
# 多因子共振策略 — 激进指标
# ═══════════════════════════
def calc_rsi(closes, period=14):
    """RSI 相对强弱指标 0-100"""
    n = len(closes)
    if n < period + 1:
        return np.full(n, np.nan)
    gains = np.maximum(np.diff(closes, prepend=closes[0]), 0)
    losses = np.abs(np.minimum(np.diff(closes, prepend=closes[0]), 0))
    rsi = np.full(n, np.nan)
    avg_gain = np.mean(gains[1:period+1])
    avg_loss = np.mean(losses[1:period+1])
    if avg_loss == 0:
        rsi[period] = 100
    else:
        rsi[period] = 100 - 100 / (1 + avg_gain / avg_loss)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i] = 100
        else:
            rsi[i] = 100 - 100 / (1 + avg_gain / avg_loss)
    return rsi


def calc_kdj(highs, lows, closes, n=9, m1=3, m2=3):
    """KDJ 随机指标 返回 (k, d, j)"""
    length = len(closes)
    k, d, j = np.full(length, np.nan), np.full(length, np.nan), np.full(length, np.nan)
    if length < n:
        return k, d, j
    for i in range(n - 1, length):
        hh = np.max(highs[i-n+1:i+1])
        ll = np.min(lows[i-n+1:i+1])
        rsv = (closes[i] - ll) / (hh - ll) * 100 if hh != ll else 50
        if i == n - 1:
            k[i] = 50
            d[i] = 50
        else:
            k[i] = rsv / m1 + k[i-1] * (m1 - 1) / m1
            d[i] = k[i] / m2 + d[i-1] * (m2 - 1) / m2
        j[i] = 3 * k[i] - 2 * d[i]
    return k, d, j


def calc_bollinger(closes, period=20, nbdev=2):
    """布林带 返回 (upper, middle, lower)"""
    n = len(closes)
    upper = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    if n < period:
        return upper, middle, lower
    for i in range(period - 1, n):
        window = closes[i-period+1:i+1]
        middle[i] = np.mean(window)
        std = np.std(window, ddof=0)
        upper[i] = middle[i] + nbdev * std
        lower[i] = middle[i] - nbdev * std
    return upper, middle, lower


def calc_wr(highs, lows, closes, period=10):
    """威廉指标 WR 0-100，超卖>80，超买<20"""
    n = len(closes)
    wr = np.full(n, np.nan)
    if n < period:
        return wr
    for i in range(period - 1, n):
        hh = np.max(highs[i-period+1:i+1])
        ll = np.min(lows[i-period+1:i+1])
        wr[i] = (hh - closes[i]) / (hh - ll) * 100 if hh != ll else 50
    return wr


def backtest_multifactor(dates, closes, highs, lows, volumes):
    """多因子共振回测：RSI + KDJ + Bollinger + WR + 量能确认
    无冷却期，信号更频繁，偏向激进短线。
    买入: score>=2 且无强空信号
    卖出: score<=-2 或止损-6%
    """
    n = len(closes)
    if n < 30:
        return []

    rsi = calc_rsi(closes)
    k_line, d_line, j_line = calc_kdj(highs, lows, closes)
    bb_upper, bb_mid, bb_lower = calc_bollinger(closes)
    wr = calc_wr(highs, lows, closes)
    ma20 = np.full(n, np.nan)
    vol_ma20 = np.full(n, np.nan)
    for i in range(19, n):
        ma20[i] = np.mean(closes[i-19:i+1])
        vol_ma20[i] = np.mean(volumes[i-19:i+1])

    sigs = []
    pos = None
    for i in range(30, n):
        if np.isnan(rsi[i]) or np.isnan(k_line[i]) or np.isnan(d_line[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(wr[i]):
            continue

        score = 0
        # RSI: 超卖加分，超买减分
        if rsi[i] < 25: score += 3
        elif rsi[i] < 35: score += 2
        elif rsi[i] < 45: score += 1
        elif rsi[i] > 80: score -= 3
        elif rsi[i] > 70: score -= 2
        elif rsi[i] > 60: score -= 1
        # KDJ
        if k_line[i] > d_line[i] and k_line[i-1] <= d_line[i-1]:
            score += 3 if k_line[i] < 50 else 2  # 金叉
        elif k_line[i] < d_line[i] and k_line[i-1] >= d_line[i-1]:
            score -= 3 if k_line[i] > 50 else 2  # 死叉
        elif k_line[i] > d_line[i]: score += 1
        else: score -= 1
        # J值极端信号
        if not np.isnan(j_line[i]):
            if j_line[i] < 0: score += 2   # J<0 超卖
            elif j_line[i] > 100: score -= 2  # J>100 超买
        # Bollinger
        if closes[i] < bb_lower[i]: score += 2  # 跌破下轨
        elif closes[i] > bb_upper[i]: score -= 1  # 突破上轨
        # WR
        if wr[i] > 85: score += 2
        elif wr[i] < 15: score -= 2
        # 趋势 MA20
        if not np.isnan(ma20[i]):
            if closes[i] > ma20[i]: score += 1
            else: score -= 1
        # 量能加分
        vol_strong = not np.isnan(vol_ma20[i]) and volumes[i] > vol_ma20[i] * 1.3

        if pos is None:
            # 买入：score>=2 且不在超买区（RSI<75）
            if score >= 2 and rsi[i] < 75:
                sigs.append({'date': dates[i], 'idx': i, 'action': 'buy',
                             'price': closes[i],
                             'reason': f'多因子共振(score{score:+d}/RSI{int(rsi[i])}/KDJ{int(k_line[i])}/{int(d_line[i])})'})
                pos = i
        else:
            pnl = (closes[i] - closes[pos]) / closes[pos] * 100
            # 卖出：空头信号强，或获利回吐，或止损
            sell = False
            reason = ''
            if score <= -3:
                sell = True; reason = f'强空信号(score{score:+d})'
            elif pnl > 8 and score <= -1:
                sell = True; reason = f'获利回吐+{pnl:.1f}%'
            elif pnl < -6:
                sell = True; reason = f'止损{pnl:.1f}%'
            elif pnl > 15:
                sell = True; reason = f'止盈+{pnl:.1f}%'

            if sell:
                sigs.append({'date': dates[i], 'idx': i, 'action': 'sell',
                             'price': closes[i], 'reason': reason})
                pos = None

    sigs.sort(key=lambda x: x['idx'])
    trades, pos_state = [], None
    for s in sigs:
        if s['action'] == 'buy' and pos_state is None:
            pos_state = {'bd': s['date'], 'bp': s['price'], 'bi': s['idx'], 'br': s['reason']}
        elif s['action'] == 'sell' and pos_state is not None:
            pct = (s['price'] - pos_state['bp']) / pos_state['bp'] * 100
            trades.append({
                'buy_date': pos_state['bd'], 'sell_date': s['date'],
                'buy_price': pos_state['bp'], 'sell_price': s['price'],
                'profit_pct': pct, 'buy_reason': pos_state['br'],
                'sell_reason': s['reason'], 'hold_days': s['idx'] - pos_state['bi']
            })
            pos_state = None
    return trades


def predict_multifactor(dates, closes, highs, lows, volumes, holding=False):
    """多因子共振当前状态预测"""
    n = len(closes)
    if n < 35:
        return ["数据不足，无法预测"]

    rsi = calc_rsi(closes)
    k_line, d_line, j_line = calc_kdj(highs, lows, closes)
    bb_upper, bb_mid, bb_lower = calc_bollinger(closes)
    wr = calc_wr(highs, lows, closes)
    ma20 = np.full(n, np.nan)
    vol_ma20 = np.full(n, np.nan)
    for i in range(19, n):
        ma20[i] = np.mean(closes[i-19:i+1])
        vol_ma20[i] = np.mean(volumes[i-19:i+1])

    i = n - 1
    while i >= 0 and np.isnan(rsi[i]): i -= 1
    if i < 35: return ["数据不足，无法预测"]

    cur = {
        'date': dates[i], 'close': closes[i],
        'rsi': rsi[i], 'k': k_line[i], 'd': d_line[i], 'j': j_line[i],
        'bb_upper': bb_upper[i], 'bb_mid': bb_mid[i], 'bb_lower': bb_lower[i],
        'wr': wr[i], 'ma20': ma20[i],
    }

    lines = []
    lines.append(f"{'='*50}")
    lines.append(f"  多因子共振预测 — {dates[i]}")
    lines.append(f"{'='*50}")

    lines.append(f"\n── 当前指标 ──")
    lines.append(f"  收盘: {cur['close']:.2f}  |  MA20: {cur['ma20']:.1f}" if not np.isnan(cur['ma20']) else f"  收盘: {cur['close']:.2f}")
    lines.append(f"  RSI: {cur['rsi']:.1f}  |  KDJ: K={cur['k']:.1f} D={cur['d']:.1f} J={cur['j']:.1f}")
    lines.append(f"  布林: 上{cur['bb_upper']:.2f} 中{cur['bb_mid']:.2f} 下{cur['bb_lower']:.2f}")
    lines.append(f"  WR: {cur['wr']:.1f}")

    score = 0
    reasons = []
    if cur['rsi'] < 35: score += 2; reasons.append('RSI超卖')
    elif cur['rsi'] > 70: score -= 2; reasons.append('RSI超买')
    elif cur['rsi'] < 50: score += 1; reasons.append('RSI偏弱')
    else: score -= 1; reasons.append('RSI偏强')

    if cur['k'] > cur['d']: score += 2; reasons.append('KDJ金叉')
    else: score -= 2; reasons.append('KDJ死叉')

    if not np.isnan(cur['j']):
        if cur['j'] < 0: score += 2; reasons.append('J<0超卖')
        elif cur['j'] > 100: score -= 2; reasons.append('J>100超买')

    if cur['close'] < cur['bb_lower']: score += 2; reasons.append('跌破布林下轨')
    elif cur['close'] > cur['bb_upper']: score -= 1; reasons.append('突破布林上轨')

    if cur['wr'] > 80: score += 1; reasons.append('WR超卖')
    elif cur['wr'] < 20: score -= 1; reasons.append('WR超买')

    if not np.isnan(cur['ma20']) and cur['close'] > cur['ma20']:
        score += 1; reasons.append('MA20之上')
    else:
        score -= 1; reasons.append('MA20之下')

    # 量能
    if not np.isnan(vol_ma20[i]) and volumes[i] > vol_ma20[i] * 1.3:
        reasons.append('放量')

    lines.append(f"\n── 综合判断 ──")
    if score >= 4: signal = '🟢 强烈看多'; action = '增持' if holding else '买入'
    elif score >= 2: signal = '🟢 偏多'; action = '拿住' if holding else '买入'
    elif score >= 0: signal = '🟡 中性'; action = '减持' if holding else '观望'
    elif score >= -2: signal = '🟠 偏空'; action = '减持' if holding else '不买'
    else: signal = '🔴 看空'; action = '卖出' if holding else '不买'

    lines.append(f"  评分: {score:+d}  |  信号: {signal}")
    lines.append(f"  建议: {action}")
    lines.append(f"  依据: {', '.join(reasons) if reasons else '无明确信号'}")

    lines.append(f"\n{'='*50}")
    n_icon = max(1, min(10, abs(score)))
    if holding:
        if action == '卖出':       lines.append(f"  {'❌' * n_icon}  {action}")
        elif action in ('增持','拿住'): lines.append(f"  {'✅' * n_icon}  {action}")
        else:                     lines.append(f"  {'⚠️' * n_icon}  {action}")
    else:
        if action == '买入':       lines.append(f"  {'✅' * n_icon}  {action}")
        else:                     lines.append(f"  {'❌' * n_icon}  {action}")
    lines.append(f"{'='*50}")

    return lines


# ═══════════════════════════
# 背离 + 牛熊
# ═══════════════════════════
def find_divergences(dates, closes, dif, lookback=60, cluster_gap=20):
    tops, bottoms = [], []
    n = len(closes)
    for i in range(34 + lookback, n):
        pw = closes[i-lookback:i+1]; dw = dif[i-lookback:i+1]
        if np.isnan(dw).any(): continue
        mid = lookback // 2
        if np.argmax(pw) == lookback and closes[i] > np.mean(pw[:mid]):
            if dif[i] < np.max(dw[:mid]) * 0.95:
                tops.append({'date':datetime.strptime(dates[i],'%Y-%m-%d'),'idx':i,'price':closes[i],'dif':dif[i]})
        if np.argmin(pw) == lookback and closes[i] < np.mean(pw[:mid]):
            if dif[i] > np.min(dw[:mid]) * 1.05:
                bottoms.append({'date':datetime.strptime(dates[i],'%Y-%m-%d'),'idx':i,'price':closes[i],'dif':dif[i]})
    def cluster(sigs, rev=False):
        if not sigs: return []
        clustered, cur = [], [sigs[0]]
        for s in sigs[1:]:
            if s['idx']-cur[-1]['idx']<=cluster_gap: cur.append(s)
            else: clustered.append(min(cur,key=lambda x:x['dif']) if rev else max(cur,key=lambda x:x['dif'])); cur=[s]
        clustered.append(min(cur,key=lambda x:x['dif']) if rev else max(cur,key=lambda x:x['dif']))
        return clustered
    return cluster(tops), cluster(bottoms, rev=True)

def detect_regime(closes, ma_period=60, slope_days=10, confirm=5):
    n = len(closes)
    ma = np.full(n, np.nan)
    for i in range(ma_period-1, n): ma[i] = np.mean(closes[i-ma_period+1:i+1])
    raw = []
    for i in range(n):
        if i < ma_period + max(slope_days, confirm) or np.isnan(ma[i]): raw.append('bear')
        else: raw.append('bull' if (closes[i]>ma[i] and ma[i]>ma[i-slope_days]) else 'bear')
    reg = [raw[0]]*confirm
    for i in range(confirm, n): reg.append(reg[-1])
    for i in range(confirm, n):
        if raw[i]!=reg[i-1] and all(r==raw[i] for r in raw[i-confirm+1:i+1]): reg[i]=raw[i]
        else: reg[i]=reg[i-1]
    return np.array(reg), ma

def zero_line_cycles(dates, dif):
    cycles, current, start = [], None, 0
    for i in range(34, len(dif)):
        if np.isnan(dif[i]) or np.isnan(dif[i-1]): continue
        if dif[i-1]<=0 and dif[i]>0:
            if current=='below': cycles.append({'zone':'below','start':start,'end':i-1,'start_date':dates[start],'end_date':dates[i-1],'duration':i-start})
            start,current=i,'above'
        elif dif[i-1]>=0 and dif[i]<0:
            if current=='above': cycles.append({'zone':'above','start':start,'end':i-1,'start_date':dates[start],'end_date':dates[i-1],'duration':i-start})
            start,current=i,'below'
    if current: cycles.append({'zone':current,'start':start,'end':len(dif)-1,'start_date':dates[start],'end_date':dates[-1],'duration':len(dif)-start,'ongoing':True})
    return cycles

# ═══════════════════════════
# 预测
# ═══════════════════════════
def predict(dates, closes, dif, dea, bar, regimes, tops, bottoms, holding=False):
    """基于当前MACD状态，预测近期买卖信号触发条件"""
    i = len(dif)-1
    while i >= 0 and np.isnan(dif[i]): i -= 1
    if i < 35: return ["数据不足，无法预测"]

    cur = {
        'date': dates[i], 'close': closes[i],
        'dif': dif[i], 'dea': dea[i], 'bar': bar[i],
        'regime': regimes[i],
        'dif_slope_5d': dif[i] - dif[max(0,i-5)] if i>=5 else None,
        'dif_slope_10d': dif[i] - dif[max(0,i-10)] if i>=10 else None,
    }

    lines = []
    lines.append(f"{'='*50}")
    lines.append(f"  MACD 预测 — {dates[i]}")
    lines.append(f"{'='*50}")

    lines.append(f"\n── 当前状态 ──")
    lines.append(f"  收盘: {cur['close']:.2f}  |  {'🟢牛市' if cur['regime']=='bull' else '🔴熊市'}")
    lines.append(f"  DIF: {cur['dif']:.2f}  |  DEA: {cur['dea']:.2f}  |  BAR: {cur['bar']:.2f}")

    if cur['dif'] > cur['dea']:
        lines.append(f"  信号: 🟢 金叉 (DIF > DEA)")
        gap = cur['dif'] - cur['dea']
        if cur['dif_slope_5d'] is not None and cur['dif_slope_5d'] < 0:
            days = int(gap / abs(cur['dif_slope_5d'] / 5)) if cur['dif_slope_5d'] < 0 else '?'
            lines.append(f"  死叉预警: DIF每5日下降{abs(cur['dif_slope_5d']):.2f}，约{days}天后死叉")
        else:
            lines.append(f"  金叉安全: DIF斜率向上，距死叉{gap:.2f}")
    else:
        lines.append(f"  信号: 🔴 死叉 (DIF < DEA)")
        gap = cur['dea'] - cur['dif']
        if cur['dif_slope_5d'] is not None and cur['dif_slope_5d'] > 0:
            days = int(gap / (cur['dif_slope_5d'] / 5))
            lines.append(f"  金叉预期: DIF每5日上升{cur['dif_slope_5d']:.2f}，约{days}天后金叉")

    lines.append(f"\n── 零轴分析 ──")
    if cur['dif'] > 0:
        lines.append(f"  DIF在零轴上方 (做多区) → {cur['dif']:.2f}")
        if cur['dif_slope_10d'] is not None:
            if cur['dif_slope_10d'] < 0:
                days = int(cur['dif'] / abs(cur['dif_slope_10d'] / 10))
                lines.append(f"  ⚠️ DIF下滑中，约{days}天后下穿零轴")
            else:
                lines.append(f"  ✅ DIF向上，零轴支撑稳固")
    else:
        lines.append(f"  DIF在零轴下方 (做空区) → {cur['dif']:.2f}")
        if cur['dif_slope_10d'] is not None and cur['dif_slope_10d'] > 0:
            days = int(abs(cur['dif']) / (cur['dif_slope_10d'] / 10))
            lines.append(f"  DIF上升中，约{days}天后上穿零轴")

    lines.append(f"\n── 背离检查 ──")
    recent_tops = [t for t in tops if (datetime.strptime(dates[i],'%Y-%m-%d') - t['date']).days < 120]
    recent_bots = [b for b in bottoms if (datetime.strptime(dates[i],'%Y-%m-%d') - b['date']).days < 120]

    if recent_tops:
        last_top = recent_tops[-1]
        days_ago = (datetime.strptime(dates[i],'%Y-%m-%d') - last_top['date']).days
        lines.append(f"  最近顶背离: {days_ago}天前 ({last_top['date'].strftime('%Y-%m-%d')} 价{last_top['price']:.2f})")
        if closes[i] > last_top['price'] and dif[i] < last_top['dif']:
            lines.append(f"  🔴 当前正在形成新的顶背离！(价创新高 {closes[i]:.2f}>{last_top['price']:.2f} 但DIF {dif[i]:.2f}<{last_top['dif']:.2f})")
    else:
        lines.append(f"  近4个月无顶背离")

    if recent_bots:
        last_bot = recent_bots[-1]
        days_ago = (datetime.strptime(dates[i],'%Y-%m-%d') - last_bot['date']).days
        lines.append(f"  最近底背离: {days_ago}天前 ({last_bot['date'].strftime('%Y-%m-%d')} 价{last_bot['price']:.2f})")
    else:
        lines.append(f"  近4个月无底背离")

    lines.append(f"\n── 综合预测 ──")
    score = 0
    reasons = []
    if cur['dif'] > cur['dea']: score += 2; reasons.append('金叉')
    else: score -= 2; reasons.append('死叉')
    if cur['dif'] > 0: score += 1; reasons.append('零轴上')
    else: score -= 1; reasons.append('零轴下')
    if cur['regime'] == 'bull': score += 2; reasons.append('牛市')
    else: score -= 2; reasons.append('熊市')
    if cur['dif_slope_5d'] is not None:
        if cur['dif_slope_5d'] > 0: score += 1; reasons.append('DIF↑')
        else: score -= 1; reasons.append('DIF↓')
    if recent_tops and closes[i] > recent_tops[-1]['price'] and dif[i] < recent_tops[-1]['dif']:
        score -= 3; reasons.append('顶背离进行中')

    if score >= 4: signal = '🟢 强烈看多'; action = '增持' if holding else '买入'
    elif score >= 2: signal = '🟢 偏多'; action = '拿住' if holding else '买入'
    elif score >= 0: signal = '🟡 中性'; action = '减持' if holding else '观望'
    elif score >= -2: signal = '🟠 偏空'; action = '减持' if holding else '不买'
    else: signal = '🔴 看空'; action = '卖出' if holding else '不买'

    lines.append(f"  评分: {score:+d}  |  信号: {signal}")
    lines.append(f"  建议: {action}")
    lines.append(f"  依据: {', '.join(reasons) if reasons else '无明确信号'}")

    lines.append(f"\n── 关注点位 ──")
    if cur['dif'] > cur['dea']:
        gap = cur['dif'] - cur['dea']
        lines.append(f"  卖出触发: DIF下穿DEA (当前差{gap:.2f})")
        lines.append(f"  卖出触发: 出现顶背离")
    if cur['regime'] == 'bull':
        lines.append(f"  牛市护盾: 死叉不卖，只等顶背离")
    else:
        lines.append(f"  熊市警告: 死叉立即卖出")
    if cur['bar'] < 0 and i > 0 and bar[i] > bar[i-1]:
        lines.append(f"  BAR柱缩窄中 → 空方力量减弱")
    elif cur['bar'] > 0 and i > 0 and bar[i] < bar[i-1]:
        lines.append(f"  BAR柱缩窄中 → 多方力量减弱")

    lines.append(f"\n{'='*50}")
    n = max(1, min(10, abs(score)))
    if holding:
        if action == '卖出':       lines.append(f"  {'❌' * n}  {action}")
        elif action in ('增持','拿住'): lines.append(f"  {'✅' * n}  {action}")
        else:                     lines.append(f"  {'⚠️' * n}  {action}")
    else:
        if action == '买入':       lines.append(f"  {'✅' * n}  {action}")
        else:                     lines.append(f"  {'❌' * n}  {action}")
    lines.append(f"{'='*50}")

    return lines


def backtest(dates, closes, dif, dea, tops, bottoms, regimes, cooldown=25):
    sigs = []
    for t in tops: sigs.append({'date':t['date'].strftime('%Y-%m-%d'),'idx':t['idx'],'action':'sell','price':t['price'],'reason':'顶背离'})
    for b in bottoms: sigs.append({'date':b['date'].strftime('%Y-%m-%d'),'idx':b['idx'],'action':'buy','price':b['price'],'reason':'底背离'})
    ls=-999
    for i in range(35,len(dif)-1):
        if np.isnan(dif[i]) or np.isnan(dea[i]): continue
        if (dif[i-1]<=dea[i-1] and dif[i]>dea[i] and dif[i]>0 and dif[i]>dif[i-3] and i-ls>cooldown and regimes[i]=='bull'):
            if i+2<len(dif) and dif[i+1]>dea[i+1] and dif[i+2]>dea[i+2]:
                if not any(abs(s['idx']-i)<15 for s in sigs if s['action']=='buy'):
                    sigs.append({'date':dates[i],'idx':i,'action':'buy','price':closes[i],'reason':'零轴上金叉'})
        if dif[i-1]>=dea[i-1] and dif[i]<dea[i]:
            if regimes[i]=='bear' and not any(abs(s['idx']-i)<10 for s in sigs if s['action']=='sell'):
                sigs.append({'date':dates[i],'idx':i,'action':'sell','price':closes[i],'reason':'死叉(熊市)'}); ls=i
    sigs.sort(key=lambda x:x['idx'])
    trades,pos=[],None
    for s in sigs:
        if s['action']=='buy' and pos is None: pos={'bd':s['date'],'bp':s['price'],'bi':s['idx'],'br':s['reason']}
        elif s['action']=='sell' and pos is not None:
            pct=(s['price']-pos['bp'])/pos['bp']*100
            trades.append({'buy_date':pos['bd'],'sell_date':s['date'],'buy_price':pos['bp'],'sell_price':s['price'],'profit_pct':pct,'buy_reason':pos['br'],'sell_reason':s['reason'],'hold_days':s['idx']-pos['bi']})
            pos=None
    return trades


def plot_multifactor(code, name, dates, closes, highs, lows, trades):
    """多因子共振图表：价格+布林带/RSI/KDJ+买卖点"""
    fig = plt.figure(figsize=(18, 14))
    import matplotlib.lines as mlines

    # ── 子图1: 价格 + 布林带 ──
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(dates, closes, color='#1565C0', linewidth=1, alpha=0.8, label='收盘价')
    bb_upper, bb_mid, bb_lower = calc_bollinger(closes)
    ax1.plot(dates, bb_upper, color='#FF6F00', linewidth=0.8, alpha=0.5, linestyle='--', label='布林上轨')
    ax1.plot(dates, bb_mid, color='#FF6F00', linewidth=1, alpha=0.6, label='布林中轨')
    ax1.plot(dates, bb_lower, color='#FF6F00', linewidth=0.8, alpha=0.5, linestyle='--', label='布林下轨')
    dt_idx = {d.strftime('%Y-%m-%d'): i for i, d in enumerate(dates)}
    for t in trades:
        bi = dt_idx.get(t['buy_date']); si = dt_idx.get(t['sell_date'])
        if bi is not None: ax1.scatter(dates[bi], closes[bi], color='lime', s=80, marker='o', zorder=6, edgecolors='black')
        if si is not None: ax1.scatter(dates[si], closes[si], color='orange', s=80, marker='s', zorder=6, edgecolors='black')
    h1, _ = ax1.get_legend_handles_labels()
    h1 += [mlines.Line2D([],[],color='lime',marker='o',linestyle='',markersize=8,markeredgecolor='black',label='买入'),
           mlines.Line2D([],[],color='orange',marker='s',linestyle='',markersize=8,markeredgecolor='black',label='卖出')]
    ax1.legend(handles=h1, loc='upper left', fontsize=7, ncol=2)
    ax1.set_title(f'{name}({code}) — 多因子共振', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3); ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

    # ── 子图2: RSI ──
    ax2 = plt.subplot(3, 1, 2)
    rsi = calc_rsi(closes)
    ax2.plot(dates, rsi, color='#7B1FA2', linewidth=1, label='RSI(14)')
    ax2.axhline(y=70, color='#ef5350', linewidth=0.8, linestyle='--', alpha=0.5)
    ax2.axhline(y=30, color='#26a69a', linewidth=0.8, linestyle='--', alpha=0.5)
    ax2.fill_between(dates, 70, 100, alpha=0.08, color='#ef5350')
    ax2.fill_between(dates, 0, 30, alpha=0.08, color='#26a69a')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.set_title('RSI(14) — 红区超买 / 绿区超卖', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3); ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

    # ── 子图3: KDJ ──
    ax3 = plt.subplot(3, 1, 3)
    k_line, d_line, j_line = calc_kdj(highs, lows, closes)
    ax3.plot(dates, k_line, color='#1565C0', linewidth=1, label='K')
    ax3.plot(dates, d_line, color='#FF6F00', linewidth=1, label='D')
    ax3.plot(dates, j_line, color='#E91E63', linewidth=0.8, alpha=0.6, label='J')
    ax3.axhline(y=80, color='#ef5350', linewidth=0.8, linestyle='--', alpha=0.5)
    ax3.axhline(y=20, color='#26a69a', linewidth=0.8, linestyle='--', alpha=0.5)
    ax3.set_ylim(-20, 120)
    ax3.legend(loc='upper left', fontsize=8, ncol=3)
    ax3.set_title('KDJ(9,3,3) — K蓝 D橙 J粉', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3); ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()


# ═══════════════════════════
# 画图 (CLI用)
# ═══════════════════════════
def plot_all(code, name, dates, closes, dif, dea, bar, cycles, tops, bottoms, trades, regimes, ma60, outdir):
    fig = plt.figure(figsize=(20,14))
    ax1 = plt.subplot(3,1,1)
    ax1.plot(dates, closes, color='#1565C0', linewidth=1, alpha=0.7); ax1.plot(dates, ma60, color='#FF6F00', linewidth=1.5, alpha=0.6, label='MA60')
    in_bull,bull_start=False,0
    for i in range(len(regimes)):
        if regimes[i]=='bull' and not in_bull: bull_start=i; in_bull=True
        elif regimes[i]=='bear' and in_bull: ax1.axvspan(dates[bull_start],dates[i-1],alpha=0.12,color='#e8f5e9'); in_bull=False
    if in_bull: ax1.axvspan(dates[bull_start],dates[-1],alpha=0.12,color='#e8f5e9')
    for t in tops: ax1.scatter(t['date'],t['price'],color='red',s=80,marker='v',zorder=5); ax1.annotate('顶背离',(t['date'],t['price']),textcoords="offset points",xytext=(0,-15),fontsize=8,color='red',ha='center')
    for b in bottoms: ax1.scatter(b['date'],b['price'],color='green',s=80,marker='^',zorder=5); ax1.annotate('底背离',(b['date'],b['price']),textcoords="offset points",xytext=(0,15),fontsize=8,color='green',ha='center')
    dt_idx={d.strftime('%Y-%m-%d'):i for i,d in enumerate(dates)}
    for tr in trades:
        bi=dt_idx.get(tr['buy_date']); si=dt_idx.get(tr['sell_date'])
        if bi is not None: ax1.scatter(dates[bi],closes[bi],color='lime',s=150,marker='o',zorder=6,edgecolors='black',linewidth=2)
        if si is not None: ax1.scatter(dates[si],closes[si],color='orange',s=150,marker='s',zorder=6,edgecolors='black',linewidth=2)
    ax1.set_title(f'{name}({code}) — 绿底=牛市  MA60橙色',fontsize=13,fontweight='bold'); ax1.legend(loc='upper left',fontsize=8); ax1.grid(True,alpha=0.3); ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')); plt.setp(ax1.xaxis.get_majorticklabels(),rotation=45,ha='right',fontsize=8)

    ax2=plt.subplot(3,1,(2,3))
    colors=['#ef5350' if v>=0 else '#26a69a' for v in bar]
    ax2.bar(dates,bar,color=colors,width=0.8,alpha=0.75); ax2.plot(dates,dif,color='#FFB300',linewidth=1.5,label='DIF'); ax2.plot(dates,dea,color='#212121',linewidth=1.5,label='DEA'); ax2.axhline(y=0,color='#9e9e9e',linewidth=1)
    for c in cycles: ax2.axvspan(datetime.strptime(c['start_date'],'%Y-%m-%d'),datetime.strptime(c['end_date'],'%Y-%m-%d'),alpha=0.15,color='#e8f5e9' if c['zone']=='above' else '#fce4ec')
    for t in tops: ax2.scatter(t['date'],t['dif'],color='red',s=60,marker='v',zorder=5)
    for b in bottoms: ax2.scatter(b['date'],b['dif'],color='green',s=60,marker='^',zorder=5)
    for tr in trades:
        bi=dt_idx.get(tr['buy_date']); si=dt_idx.get(tr['sell_date'])
        if bi is not None: ax2.scatter(dates[bi],dif[bi],color='lime',s=120,marker='o',zorder=6,edgecolors='black')
        if si is not None: ax2.scatter(dates[si],dif[si],color='orange',s=120,marker='s',zorder=6,edgecolors='black')
    ax2.set_title('MACD (12,26,9) — 绿底=做多区 红底=做空区',fontsize=13,fontweight='bold'); ax2.legend(loc='upper left',fontsize=8); ax2.grid(True,alpha=0.3); ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')); plt.setp(ax2.xaxis.get_majorticklabels(),rotation=45,ha='right',fontsize=8)

    plt.tight_layout()
    path=os.path.join(outdir,f'{code}_macd_cycle.png')
    fig.savefig(path,dpi=150,bbox_inches='tight'); plt.close()
    return path
