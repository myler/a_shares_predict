#!/usr/bin/env python3
"""核心引擎：MACD计算、背离检测、回测、预测 — 纯计算层，不依赖 matplotlib"""

import os
from datetime import datetime
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

COMPREHENSIVE_BUY_SCORE = 69
COMPREHENSIVE_STOP_LOSS_PCT = -12
COMPREHENSIVE_TAKE_PROFIT_PCT = 15
COMPREHENSIVE_PROFIT_PROTECT_PCT = 12
COMPREHENSIVE_PROFIT_PROTECT_SCORE = 50
COMPREHENSIVE_AUXILIARY_CONTRARIAN_BETA = 0.08


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
    if len(data) < period:
        return r
    r[period - 1] = np.mean(data[:period])
    m = 2 / (period + 1)
    for i in range(period, len(data)):
        r[i] = data[i] * m + r[i - 1] * (1 - m)
    return r


def calc_macd(closes, fast=12, slow=26, signal=9):
    n = len(closes)
    ef, es = ema(closes, fast), ema(closes, slow)
    dif = ef - es
    dea = np.full(n, np.nan)
    dea[slow - 1:] = ema(dif[slow - 1:], signal)
    bar = (dif - dea) * 2
    return dif, dea, bar


# ═══════════════════════════
# 技术指标
# ═══════════════════════════
def calc_rsi(closes, period=14):
    """RSI 相对强弱指标 0-100"""
    n = len(closes)
    if n < period + 1:
        return np.full(n, np.nan)
    gains = np.maximum(np.diff(closes, prepend=closes[0]), 0)
    losses = np.abs(np.minimum(np.diff(closes, prepend=closes[0]), 0))
    rsi = np.full(n, np.nan)
    avg_gain = np.mean(gains[1:period + 1])
    avg_loss = np.mean(losses[1:period + 1])
    if avg_gain == 0 and avg_loss == 0:
        rsi[period] = 50
    elif avg_loss == 0:
        rsi[period] = 100
    else:
        rsi[period] = 100 - 100 / (1 + avg_gain / avg_loss)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_gain == 0 and avg_loss == 0:
            rsi[i] = 50
        elif avg_loss == 0:
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
        hh = np.max(highs[i - n + 1:i + 1])
        ll = np.min(lows[i - n + 1:i + 1])
        rsv = (closes[i] - ll) / (hh - ll) * 100 if hh != ll else 50
        if i == n - 1:
            k[i] = 50
            d[i] = 50
        else:
            k[i] = rsv / m1 + k[i - 1] * (m1 - 1) / m1
            d[i] = k[i] / m2 + d[i - 1] * (m2 - 1) / m2
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
        window = closes[i - period + 1:i + 1]
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
        hh = np.max(highs[i - period + 1:i + 1])
        ll = np.min(lows[i - period + 1:i + 1])
        wr[i] = (hh - closes[i]) / (hh - ll) * 100 if hh != ll else 50
    return wr


def calc_obv(closes, volumes):
    """OBV 能量潮"""
    n = len(closes)
    obv = np.zeros(n)
    for i in range(1, n):
        if closes[i] > closes[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]
    return obv


def calc_obv_flow(obv, volumes, i, period=5):
    """返回指定窗口内 OBV 的净量能流，范围通常在 -1 到 1。"""
    start = max(0, i - period)
    if i <= start:
        return 0.0
    volume_sum = np.sum(volumes[start + 1:i + 1])
    if volume_sum <= 0:
        return 0.0
    return float((obv[i] - obv[start]) / volume_sum)


def calc_net_trade_return(buy_price, sell_price, commission_rate=0.0003,
                          stamp_duty_rate=0.0005):
    """计算已含双边佣金和卖出印花税的单笔收益率。"""
    if buy_price <= 0 or sell_price <= 0:
        return 0.0
    buy_cost = 1 + commission_rate
    sell_proceeds = 1 - commission_rate - stamp_duty_rate
    return (sell_price * sell_proceeds / (buy_price * buy_cost) - 1) * 100


def summarize_trades(trades):
    """汇总顺序全仓交易的复利收益；分红字段存在时一并纳入。"""
    if not trades:
        return {
            'trades': 0, 'wins': 0, 'win_rate_pct': 0.0,
            'price_return_pct': 0.0, 'total_return_pct': 0.0,
        }

    price_returns = [trade['profit_pct'] / 100 for trade in trades]
    total_returns = [trade.get('total_return_pct', trade['profit_pct']) / 100
                     for trade in trades]
    wins = sum(trade['profit_pct'] > 0 for trade in trades)
    return {
        'trades': len(trades),
        'wins': wins,
        'win_rate_pct': wins / len(trades) * 100,
        'price_return_pct': (np.prod([1 + value for value in price_returns]) - 1) * 100,
        'total_return_pct': (np.prod([1 + value for value in total_returns]) - 1) * 100,
    }


def calc_atr(highs, lows, closes, period=14):
    """ATR 平均真实波幅"""
    n = len(closes)
    tr = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = np.nanmean(tr[1:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


# ═══════════════════════════
# 辅助指标 — 投票面板用（不作为主策略因子）
# ═══════════════════════════

def calc_dmi(highs, lows, closes, period=14):
    """DMI 趋向指标 — 返回 (pdi, mdi, adx, adxr)
    PDI/MDI：多空方向力；ADX：趋势强度；ADXR：ADX平滑
    """
    n = len(closes)
    pdi = np.full(n, np.nan)
    mdi = np.full(n, np.nan)
    adx = np.full(n, np.nan)
    adxr = np.full(n, np.nan)
    if n < period + 1:
        return pdi, mdi, adx, adxr

    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]))
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down

    tr_sm = np.full(n, np.nan)
    pdm_sm = np.full(n, np.nan)
    mdm_sm = np.full(n, np.nan)
    tr_sm[period] = np.sum(tr[1:period + 1])
    pdm_sm[period] = np.sum(plus_dm[1:period + 1])
    mdm_sm[period] = np.sum(minus_dm[1:period + 1])

    for i in range(period + 1, n):
        tr_sm[i] = tr_sm[i - 1] - tr_sm[i - 1] / period + tr[i]
        pdm_sm[i] = pdm_sm[i - 1] - pdm_sm[i - 1] / period + plus_dm[i]
        mdm_sm[i] = mdm_sm[i - 1] - mdm_sm[i - 1] / period + minus_dm[i]

    dx_arr = np.full(n, np.nan)
    for i in range(period, n):
        if tr_sm[i] > 0:
            pdi[i] = pdm_sm[i] / tr_sm[i] * 100
            mdi[i] = mdm_sm[i] / tr_sm[i] * 100
            denom = pdi[i] + mdi[i]
            dx_arr[i] = abs(pdi[i] - mdi[i]) / denom * 100 if denom > 0 else 0

    adx[period * 2 - 1] = np.nanmean(dx_arr[period:period * 2])
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx_arr[i]) / period

    for i in range(period * 2 - 1 + period, n):
        adxr[i] = (adx[i] + adx[i - period]) / 2

    return pdi, mdi, adx, adxr


def calc_cci(highs, lows, closes, period=14):
    """CCI 商品通道指数 — 超买>100，超卖<-100"""
    n = len(closes)
    cci = np.full(n, np.nan)
    if n < period:
        return cci
    tp = (highs + lows + closes) / 3
    for i in range(period - 1, n):
        tp_slice = tp[i - period + 1:i + 1]
        ma_tp = np.mean(tp_slice)
        mad = np.mean(np.abs(tp_slice - ma_tp))
        cci[i] = (tp[i] - ma_tp) / (0.015 * mad) if mad > 0 else 0
    return cci


def calc_bias(closes, period=6):
    """BIAS 乖离率 — 价格偏离MA的百分比"""
    n = len(closes)
    bias = np.full(n, np.nan)
    if n < period:
        return bias
    for i in range(period - 1, n):
        ma = np.mean(closes[i - period + 1:i + 1])
        bias[i] = (closes[i] - ma) / ma * 100 if ma > 0 else 0
    return bias


def calc_expma(closes, period=12):
    """EXPMA 指数移动平均"""
    return ema(closes, period)


def calc_bbi(closes):
    """BBI 多空指数 — 3/6/12/24日MA的均值"""
    n = len(closes)
    bbi = np.full(n, np.nan)
    periods = [3, 6, 12, 24]
    if n < max(periods):
        return bbi
    for i in range(max(periods) - 1, n):
        avg = 0
        for p in periods:
            avg += np.mean(closes[i - p + 1:i + 1])
        bbi[i] = avg / len(periods)
    return bbi


def calc_trix(closes, period=12):
    """TRIX 三重指数平滑 — 趋势指标"""
    n = len(closes)
    trix = np.full(n, np.nan)
    if n < period * 3:
        return trix
    
    def _ema_skip_nan(data, period, start):
        """从 start 位置开始计算 EMA（跳过前导 NaN）"""
        r = np.full(n, np.nan)
        r[start + period - 1] = np.mean(data[start:start + period])
        m = 2 / (period + 1)
        for j in range(start + period, n):
            r[j] = data[j] * m + r[j - 1] * (1 - m)
        return r

    e1 = _ema_skip_nan(closes, period, 0)
    e2_start = period - 1  # e1 从这开始有效
    e2 = _ema_skip_nan(e1, period, e2_start)
    e3_start = e2_start + period - 1
    e3 = _ema_skip_nan(e2, period, e3_start)
    
    for i in range(e3_start + period - 1, n):
        if not np.isnan(e3[i]) and not np.isnan(e3[i - 1]) and e3[i - 1] != 0:
            trix[i] = (e3[i] - e3[i - 1]) / e3[i - 1] * 100
    return trix


def calc_vr(closes, volumes, period=26):
    """VR 成交量比率 — >450超买，<70超卖"""
    n = len(closes)
    vr = np.full(n, np.nan)
    if n < period + 1:
        return vr
    for i in range(period, n):
        up_vol = 0.0
        down_vol = 0.0
        flat_vol = 0.0
        for j in range(i - period + 1, i + 1):
            if closes[j] > closes[j - 1]:
                up_vol += volumes[j]
            elif closes[j] < closes[j - 1]:
                down_vol += volumes[j]
            else:
                flat_vol += volumes[j]
        denom = down_vol + flat_vol / 2
        vr[i] = (up_vol + flat_vol / 2) / denom * 100 if denom > 0 else 100
    return vr


def calc_brar(opens, highs, lows, closes, period=26):
    """BRAR 情绪指标 — BR>300超买<40超卖；AR>200超买<60超卖
    返回 (br, ar)
    """
    n = len(opens)
    br = np.full(n, np.nan)
    ar = np.full(n, np.nan)
    if n < period + 1:
        return br, ar
    for i in range(period, n):
        br_sum_h, br_sum_l = 0.0, 0.0
        ar_sum_h, ar_sum_l = 0.0, 0.0
        for j in range(i - period + 1, i + 1):
            br_sum_h += max(0, highs[j] - closes[j - 1])
            br_sum_l += max(0, closes[j - 1] - lows[j])
            ar_sum_h += highs[j] - opens[j]
            ar_sum_l += opens[j] - lows[j]
        br[i] = br_sum_h / br_sum_l * 100 if br_sum_l > 0 else 100
        ar[i] = ar_sum_h / ar_sum_l * 100 if ar_sum_l > 0 else 100
    return br, ar


def calc_cr(highs, lows, closes, period=26):
    """CR 能量指标 — 用中间价计算买卖力量"""
    n = len(closes)
    cr = np.full(n, np.nan)
    if n < period + 1:
        return cr
    mid = np.array([(highs[i] + lows[i]) / 2 for i in range(n)])
    for i in range(period, n):
        sum_h, sum_l = 0.0, 0.0
        for j in range(i - period + 1, i + 1):
            sum_h += max(0, highs[j] - mid[j - 1]) if j > 0 else 0
            sum_l += max(0, mid[j - 1] - lows[j]) if j > 0 else 0
        cr[i] = sum_h / sum_l * 100 if sum_l > 0 else 100
    return cr


def calc_dma(closes, short=10, long_=50):
    """DMA 均线差 — 快线MA10 - 慢线MA50"""
    n = len(closes)
    dma = np.full(n, np.nan)
    if n < long_:
        return dma
    for i in range(long_ - 1, n):
        ma_s = np.mean(closes[i - short + 1:i + 1])
        ma_l = np.mean(closes[i - long_ + 1:i + 1])
        dma[i] = ma_s - ma_l
    return dma


def calc_dpo(closes, period=20):
    """DPO 区间震荡线 — 去除长期趋势后的价格摆动"""
    n = len(closes)
    dpo = np.full(n, np.nan)
    if n < period:
        return dpo
    half = period // 2 + 1
    for i in range(period - 1, n):
        ma = np.mean(closes[i - period + 1:i + 1])
        src_idx = i - half
        if src_idx >= 0:
            dpo[i] = closes[src_idx] - ma
    return dpo


def calc_mtm(closes, period=12):
    """MTM 动量线 — C - C[n]"""
    n = len(closes)
    mtm = np.full(n, np.nan)
    for i in range(period, n):
        mtm[i] = closes[i] - closes[i - period]
    return mtm


def calc_skdj(highs, lows, closes, n=9, m=3):
    """SKDJ 慢速KDJ — KDJ的慢速版，信号更稳定
    返回 (k, d)
    """
    length = len(closes)
    rsv_k = np.full(length, np.nan)
    # 先算普通KDJ
    k_raw, d_raw, _ = calc_kdj(highs, lows, closes, n=n, m1=m, m2=m)
    # SKDJ: K=原D, D=原D再平滑
    k_s = np.full(length, np.nan)
    d_s = np.full(length, np.nan)
    for i in range(length):
        if not np.isnan(d_raw[i]):
            k_s[i] = d_raw[i]
    # D再平滑一次
    for i in range(n + m - 1, length):
        if not np.isnan(k_s[i]):
            if np.isnan(d_s[i - 1]):
                d_s[i] = k_s[i]
            else:
                d_s[i] = d_s[i - 1] * (m - 1) / m + k_s[i] / m
    return k_s, d_s


def calc_lwr(highs, lows, closes, n=9, m1=3, m2=3):
    """LWR 威廉变异离散量 — KDJ的倒置版，LWR1=100-K, LWR2=100-D
    返回 (lwr1, lwr2)
    """
    k, d, _ = calc_kdj(highs, lows, closes, n=n, m1=m1, m2=m2)
    lwr1 = np.where(~np.isnan(k), 100 - k, np.nan)
    lwr2 = np.where(~np.isnan(d), 100 - d, np.nan)
    return lwr1, lwr2


def calc_ene(closes, period=26, pct=0.06):
    """ENE 轨道线 — 中轨=MA(N)，上轨=MA*(1+pct)，下轨=MA*(1-pct)
    返回 (upper, mid, lower)
    """
    n = len(closes)
    upper = np.full(n, np.nan)
    mid = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    if n < period:
        return upper, mid, lower
    for i in range(period - 1, n):
        m = np.mean(closes[i - period + 1:i + 1])
        mid[i] = m
        upper[i] = m * (1 + pct)
        lower[i] = m * (1 - pct)
    return upper, mid, lower


def calc_lon(highs, lows, closes, period=20):
    """LON 唐奇安通道 — 上=N日最高价，下=N日最低价，中=均价
    返回 (upper, mid, lower)
    """
    n = len(closes)
    upper = np.full(n, np.nan)
    mid = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    if n < period:
        return upper, mid, lower
    for i in range(period - 1, n):
        upper[i] = np.max(highs[i - period + 1:i + 1])
        lower[i] = np.min(lows[i - period + 1:i + 1])
        mid[i] = (upper[i] + lower[i]) / 2
    return upper, mid, lower


# ═══════════════════════════
# 辅助指标投票面板 — 多数共识信号
# ═══════════════════════════

def calc_auxiliary_consensus(closes, highs, lows, volumes, opens=None):
    """辅助指标投票面板。
    18个指标各自投票（+1看多 / −1看空 / 0中性），汇总共识度。
    
    返回 dict:
      - consensus_score: −100~+100，正=多数看多
      - consensus_pct: 0~100，非中性票中看多的占比
      - bullish_count / bearish_count / neutral_count
      - votes: 每个指标的投票详情
    """
    n = len(closes)
    if n < 60:
        return {'consensus_score': 0, 'consensus_pct': 50,
                'bullish_count': 0, 'bearish_count': 0, 'neutral_count': 18,
                'votes': [{'name': n, 'value': 0, 'vote': 0, 'signal': '数据不足'}
                          for n in _INDICATOR_NAMES]}

    i = n - 1
    if opens is None:
        opens = np.full_like(closes, closes[0])
        for j in range(1, n):
            opens[j] = closes[j - 1]

    votes = []

    def vote(name, value, bull_cond, bear_cond, display_val=None):
        """统一投票函数"""
        if display_val is None:
            display_val = value
        v = display_val if not isinstance(display_val, str) else 0
        try:
            v = round(float(v), 2)
        except (ValueError, TypeError):
            v = 0
        if bull_cond:
            votes.append({'name': name, 'value': v, 'vote': 1, 'signal': '看多'})
        elif bear_cond:
            votes.append({'name': name, 'value': v, 'vote': -1, 'signal': '看空'})
        else:
            votes.append({'name': name, 'value': v, 'vote': 0, 'signal': '中性'})

    # ── 1. DMI ──
    pdi, mdi, adx_dmi, adxr = calc_dmi(highs, lows, closes)
    pdi_v = pdi[i] if not np.isnan(pdi[i]) else 0
    mdi_v = mdi[i] if not np.isnan(mdi[i]) else 0
    adx_v = adx_dmi[i] if not np.isnan(adx_dmi[i]) else 0
    adxr_v = adxr[i] if not np.isnan(adxr[i]) else 0
    vote('DMI', {'pdi': round(pdi_v, 1), 'mdi': round(mdi_v, 1), 'adx': round(adx_v, 1), 'adxr': round(adxr_v, 1)},
         pdi_v > mdi_v and adx_v > 20,
         mdi_v > pdi_v and adx_v > 20,
         f'PDI{pdi_v:.0f}/MDI{mdi_v:.0f} ADX{adx_v:.0f}')

    # ── 2. CCI ──
    cci_arr = calc_cci(highs, lows, closes)
    cci_v = cci_arr[i] if not np.isnan(cci_arr[i]) else 0
    vote('CCI', round(cci_v, 1),
         cci_v < -100,
         cci_v > 100)

    # ── 3. BIAS(6) ──
    bias6 = calc_bias(closes, 6)
    bias6_v = bias6[i] if not np.isnan(bias6[i]) else 0
    vote('BIAS(6)', round(bias6_v, 2),
         bias6_v < -3,  # 超卖看多
         bias6_v > 5)   # 超买看空

    # ── 4. BIAS(12) ──
    bias12 = calc_bias(closes, 12)
    bias12_v = bias12[i] if not np.isnan(bias12[i]) else 0
    vote('BIAS(12)', round(bias12_v, 2),
         bias12_v < -5,
         bias12_v > 8)

    # ── 5. EXPMA ──
    expma12 = calc_expma(closes, 12)
    expma50 = calc_expma(closes, 50)
    ema12_v = expma12[i] if not np.isnan(expma12[i]) else 0
    ema50_v = expma50[i] if not np.isnan(expma50[i]) else 0
    vote('EXPMA', round(ema12_v, 2),
         ema12_v > ema50_v and closes[i] > ema12_v,
         ema12_v < ema50_v and closes[i] < ema12_v,
         f'12={ema12_v:.2f}/50={ema50_v:.2f}')

    # ── 6. BBI ──
    bbi_arr = calc_bbi(closes)
    bbi_v = bbi_arr[i] if not np.isnan(bbi_arr[i]) else 0
    vote('BBI', round(bbi_v, 2),
         closes[i] > bbi_v,
         closes[i] < bbi_v)

    # ── 7. TRIX ──
    trix_arr = calc_trix(closes)
    trix_v = trix_arr[i] if not np.isnan(trix_arr[i]) else 0
    vote('TRIX', round(trix_v, 3),
         trix_v > 0.05,
         trix_v < -0.05)

    # ── 8. VR ──
    vr_arr = calc_vr(closes, volumes)
    vr_v = vr_arr[i] if not np.isnan(vr_arr[i]) else 100
    vote('VR', round(vr_v, 1),
         vr_v < 70,
         vr_v > 350)

    # ── 9. BR ──
    br_arr, ar_arr = calc_brar(opens, highs, lows, closes)
    br_v = br_arr[i] if not np.isnan(br_arr[i]) else 100
    vote('BR', round(br_v, 1),
         br_v < 60,
         br_v > 300)

    # ── 10. AR ──
    ar_v = ar_arr[i] if not np.isnan(ar_arr[i]) else 100
    vote('AR', round(ar_v, 1),
         ar_v < 60,
         ar_v > 200)

    # ── 11. CR ──
    cr_arr = calc_cr(highs, lows, closes)
    cr_v = cr_arr[i] if not np.isnan(cr_arr[i]) else 100
    vote('CR', round(cr_v, 1),
         cr_v < 50,
         cr_v > 300)

    # ── 12. DMA ──
    dma_arr = calc_dma(closes)
    dma_v = dma_arr[i] if not np.isnan(dma_arr[i]) else 0
    vote('DMA', round(dma_v, 2),
         dma_v > 0 and closes[i] > np.mean(closes[-10:]),
         dma_v < 0 and closes[i] < np.mean(closes[-20:]))

    # ── 13. DPO ──
    dpo_arr = calc_dpo(closes)
    dpo_v = dpo_arr[i] if not np.isnan(dpo_arr[i]) else 0
    vote('DPO', round(dpo_v, 2),
         dpo_v < -closes[i] * 0.05,  # 深度超卖
         dpo_v > closes[i] * 0.05)   # 深度超买

    # ── 14. MTM ──
    mtm_arr = calc_mtm(closes)
    mtm_v = mtm_arr[i] if not np.isnan(mtm_arr[i]) else 0
    vote('MTM', round(mtm_v, 2),
         mtm_v > 0,
         mtm_v < 0)

    # ── 15. SKDJ ──
    k_s, d_s = calc_skdj(highs, lows, closes)
    ks_v = k_s[i] if not np.isnan(k_s[i]) else 50
    ds_v = d_s[i] if not np.isnan(d_s[i]) else 50
    vote('SKDJ', round(ks_v, 1),
         ks_v < 20 or (ks_v > ds_v and ks_v < 50),
         ks_v > 80 or (ks_v < ds_v and ks_v > 50),
         f'K{ks_v:.0f}/D{ds_v:.0f}')

    # ── 16. LWR ──
    lwr1, lwr2 = calc_lwr(highs, lows, closes)
    lwr1_v = lwr1[i] if not np.isnan(lwr1[i]) else 50
    lwr2_v = lwr2[i] if not np.isnan(lwr2[i]) else 50
    vote('LWR', round(lwr1_v, 1),
         lwr1_v > 80 and lwr2_v > 70,  # 超卖看多（倒置）
         lwr1_v < 20 and lwr2_v < 30,  # 超买看空
         f'LWR1{lwr1_v:.0f}/LWR2{lwr2_v:.0f}')

    # ── 17. ENE ──
    ene_u, ene_m, ene_l = calc_ene(closes)
    ene_u_v = ene_u[i] if not np.isnan(ene_u[i]) else 0
    ene_l_v = ene_l[i] if not np.isnan(ene_l[i]) else 0
    ene_pos = (closes[i] - ene_l_v) / (ene_u_v - ene_l_v) * 100 if ene_u_v != ene_l_v else 50
    vote('ENE', round(ene_pos, 1),
         ene_pos < 10,  # 下轨附近
         ene_pos > 90)  # 上轨附近

    # ── 18. LON ──
    lon_u, lon_m, lon_l = calc_lon(highs, lows, closes)
    lon_pos = (closes[i] - lon_l[i]) / (lon_u[i] - lon_l[i]) * 100 \
        if not np.isnan(lon_u[i]) and lon_u[i] != lon_l[i] else 50
    vote('LON', round(lon_pos, 1),
         lon_pos < 20,
         lon_pos > 80)

    # ── 汇总 ──
    bullish = [v for v in votes if v['vote'] == 1]
    bearish = [v for v in votes if v['vote'] == -1]
    neutral = [v for v in votes if v['vote'] == 0]

    n_active = len(bullish) + len(bearish)
    if n_active > 0:
        consensus_pct = len(bullish) / n_active * 100
        consensus_score = round((len(bullish) - len(bearish)) / len(votes) * 100, 1)
    else:
        consensus_pct = 50
        consensus_score = 0

    return {
        'consensus_score': consensus_score,
        'consensus_pct': round(consensus_pct, 1),
        'bullish_count': len(bullish),
        'bearish_count': len(bearish),
        'neutral_count': len(neutral),
        'total_indicators': len(votes),
        'votes': votes,
    }


def calc_auxiliary_consensus_series(closes, highs, lows, volumes, opens=None):
    """返回每个交易日的 18 项辅助指标共识度，计算只使用该日及此前数据。"""
    closes = np.asarray(closes, dtype=float)
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    n = len(closes)
    consensus = np.zeros(n)
    if n < 60:
        return consensus

    if opens is None:
        opens = np.empty(n, dtype=float)
        opens[0] = closes[0]
        opens[1:] = closes[:-1]
    else:
        opens = np.asarray(opens, dtype=float)
        if len(opens) != n:
            raise ValueError('opens 长度必须与 closes 一致')

    def finite_or(values, default):
        return np.where(np.isfinite(values), values, default)

    def rolling_mean(period):
        result = np.full(n, np.nan)
        if n >= period:
            sums = np.concatenate(([0.0], np.cumsum(closes, dtype=float)))
            result[period - 1:] = (sums[period:] - sums[:-period]) / period
        return result

    bullish_count = np.zeros(n, dtype=int)
    bearish_count = np.zeros(n, dtype=int)

    def add_votes(bullish, bearish):
        bullish_count[:] += bullish.astype(int)
        bearish_count[:] += (~bullish & bearish).astype(int)

    pdi, mdi, adx, _ = calc_dmi(highs, lows, closes)
    pdi = finite_or(pdi, 0)
    mdi = finite_or(mdi, 0)
    adx = finite_or(adx, 0)
    add_votes((pdi > mdi) & (adx > 20), (mdi > pdi) & (adx > 20))

    cci = finite_or(calc_cci(highs, lows, closes), 0)
    add_votes(cci < -100, cci > 100)

    bias6 = finite_or(calc_bias(closes, 6), 0)
    add_votes(bias6 < -3, bias6 > 5)

    bias12 = finite_or(calc_bias(closes, 12), 0)
    add_votes(bias12 < -5, bias12 > 8)

    expma12 = finite_or(calc_expma(closes, 12), 0)
    expma50 = finite_or(calc_expma(closes, 50), 0)
    add_votes((expma12 > expma50) & (closes > expma12),
              (expma12 < expma50) & (closes < expma12))

    bbi = finite_or(calc_bbi(closes), 0)
    add_votes(closes > bbi, closes < bbi)

    trix = finite_or(calc_trix(closes), 0)
    add_votes(trix > 0.05, trix < -0.05)

    vr = finite_or(calc_vr(closes, volumes), 100)
    add_votes(vr < 70, vr > 350)

    br, ar = calc_brar(opens, highs, lows, closes)
    br = finite_or(br, 100)
    ar = finite_or(ar, 100)
    add_votes(br < 60, br > 300)
    add_votes(ar < 60, ar > 200)

    cr = finite_or(calc_cr(highs, lows, closes), 100)
    add_votes(cr < 50, cr > 300)

    dma = finite_or(calc_dma(closes), 0)
    ma10 = rolling_mean(10)
    ma20 = rolling_mean(20)
    add_votes((dma > 0) & (closes > ma10),
              (dma < 0) & (closes < ma20))

    dpo = finite_or(calc_dpo(closes), 0)
    add_votes(dpo < -closes * 0.05, dpo > closes * 0.05)

    mtm = finite_or(calc_mtm(closes), 0)
    add_votes(mtm > 0, mtm < 0)

    skdj_k, skdj_d = calc_skdj(highs, lows, closes)
    skdj_k = finite_or(skdj_k, 50)
    skdj_d = finite_or(skdj_d, 50)
    add_votes((skdj_k < 20) | ((skdj_k > skdj_d) & (skdj_k < 50)),
              (skdj_k > 80) | ((skdj_k < skdj_d) & (skdj_k > 50)))

    lwr1, lwr2 = calc_lwr(highs, lows, closes)
    lwr1 = finite_or(lwr1, 50)
    lwr2 = finite_or(lwr2, 50)
    add_votes((lwr1 > 80) & (lwr2 > 70),
              (lwr1 < 20) & (lwr2 < 30))

    ene_upper, _, ene_lower = calc_ene(closes)
    ene_valid = np.isfinite(ene_upper) & np.isfinite(ene_lower) & (ene_upper != ene_lower)
    ene_position = np.full(n, 50.0)
    ene_position[ene_valid] = (
        (closes[ene_valid] - ene_lower[ene_valid]) /
        (ene_upper[ene_valid] - ene_lower[ene_valid]) * 100)
    add_votes(ene_position < 10, ene_position > 90)

    lon_upper, _, lon_lower = calc_lon(highs, lows, closes)
    lon_valid = np.isfinite(lon_upper) & np.isfinite(lon_lower) & (lon_upper != lon_lower)
    lon_position = np.full(n, 50.0)
    lon_position[lon_valid] = (
        (closes[lon_valid] - lon_lower[lon_valid]) /
        (lon_upper[lon_valid] - lon_lower[lon_valid]) * 100)
    add_votes(lon_position < 20, lon_position > 80)

    consensus[59:] = np.round(
        (bullish_count[59:] - bearish_count[59:]) / 18 * 100, 1)
    return consensus


def apply_auxiliary_consensus_adjustment(base_score, consensus_score, beta):
    """以辅助共识作小幅反向修正，保留四维主评分的主导地位。"""
    if not 0 <= beta <= 0.2:
        raise ValueError('auxiliary_beta 必须在 0 到 0.2 之间')
    return base_score - beta * np.clip(consensus_score, -100, 100)


_INDICATOR_NAMES = [
    'DMI', 'CCI', 'BIAS(6)', 'BIAS(12)', 'EXPMA', 'BBI', 'TRIX', 'VR', 'BR',
    'AR', 'CR', 'DMA', 'DPO', 'MTM', 'SKDJ', 'LWR', 'ENE', 'LON',
]


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

    def find_peaks(arr, min_dist=10):
        peaks = []
        for i in range(min_dist, len(arr) - min_dist):
            if arr[i] == max(arr[i - min_dist:i + min_dist + 1]):
                peaks.append(i)
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
            if arr[i] == min(arr[i - min_dist:i + min_dist + 1]):
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
            between = closes[p1:p2 + 1]
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
            left_valley = min(closes[p_left:p_head + 1]) if p_left < p_head else closes[p_left]
            right_valley = min(closes[p_head:p_right + 1]) if p_head < p_right else closes[p_head]
            neckline = min(left_valley, right_valley)
            neckline_broken = closes[-1] < neckline
            score = -30 if neckline_broken else -20
            patterns.append({
                'type': '头肩顶', 'left_shoulder': p_left, 'head': p_head,
                'right_shoulder': p_right,
                'head_price': round(closes[p_head], 2),
                'shoulder_price': round((closes[p_left] + closes[p_right]) / 2, 2),
                'neckline': round(neckline, 2),
                'neckline_broken': neckline_broken, 'score': score
            })
            total_adj += score

    # ── K线形态 ──
    if n >= 3:
        # 三白兵
        if n >= 4:
            up3 = all(closes[-i] > closes[-i - 1] for i in [1, 2, 3])
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
            prev_up = closes[-3] < closes[-2]
            curr_bear = closes[-1] < closes[-2] * 0.98
            if prev_up and curr_bear:
                patterns.append({'signals': ['看跌吞没'], 'score': -15})
                total_adj -= 15

    return patterns, total_adj


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
    while i >= 0 and np.isnan(dif[i]):
        i -= 1
    if i < 30:
        return 50, {}, []

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

    macd_score = 50
    if dif[i] > dea[i]: macd_score += 15
    if dif[i] > 0: macd_score += 15
    if i >= 5 and dif[i] > dif[i - 5]: macd_score += 10
    if bar[i] > 0 and i >= 3 and bar[i] > bar[i - 3]: macd_score += 10
    dimensions['趋势'] = round(trend_score * 0.5 + macd_score * 0.5, 1)

    # ── 动量 (20%) ──
    rsi_val = rsi[i] if not np.isnan(rsi[i]) else 50
    rsi_score = 50
    if 30 <= rsi_val <= 70: rsi_score += 20
    elif rsi_val < 30: rsi_score += 15
    elif rsi_val > 70: rsi_score -= 15

    kdj_score = 50
    kd_golden = k_line[i] > d_line[i]
    if kd_golden: kdj_score += 20
    elif k_line[i] < d_line[i]: kdj_score -= 15
    if not np.isnan(j_line[i]):
        if j_line[i] < 0: kdj_score += 15
        elif j_line[i] > 100: kdj_score -= 15

    wr_val = wr[i] if not np.isnan(wr[i]) else 50
    wr_score = 50
    if wr_val > 80: wr_score += 15
    elif wr_val < 20: wr_score -= 15

    dimensions['动量'] = round(rsi_score * 0.4 + kdj_score * 0.35 + wr_score * 0.25, 1)

    # ── 量能 (20%) ──
    vol_ma20 = np.mean(volumes[-21:-1]) if n >= 21 else np.mean(volumes)
    vol_ratio = volumes[i] / vol_ma20 if vol_ma20 > 0 else 1
    vol_score = 50
    if 1.0 <= vol_ratio <= 2.0: vol_score += 15
    elif vol_ratio > 2.0: vol_score += 10
    elif vol_ratio < 0.5: vol_score -= 10

    obv_ma20 = np.mean(obv[-21:-1]) if n >= 21 else np.mean(obv)
    obv_trend = obv[i] > obv_ma20
    obv_score = 50
    if obv_trend: obv_score += 25
    else: obv_score -= 15

    turnover_score = 50
    if vol_ratio > 1.5: turnover_score += 10
    elif vol_ratio < 0.7: turnover_score -= 10

    dimensions['量能'] = round(vol_score * 0.4 + obv_score * 0.35 + turnover_score * 0.25, 1)

    # ── 通道/波动 (15%) ──
    bb_pos = (closes[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) * 100 \
        if not np.isnan(bb_upper[i]) and bb_upper[i] != bb_lower[i] else 50
    bb_score = 50
    if 20 <= bb_pos <= 80: bb_score += 15
    elif bb_pos < 5: bb_score += 20
    elif bb_pos > 95: bb_score -= 15

    atr_val = atr[i] if not np.isnan(atr[i]) else 0
    atr_ratio = atr_val / closes[i] * 100
    atr_score = 50
    if 1.5 <= atr_ratio <= 4: atr_score += 15

    dimensions['通道/波动'] = round(bb_score * 0.6 + atr_score * 0.4, 1)

    # ── 形态/结构 (10%) ──
    pattern_score = 50 + pattern_adj
    pattern_score = max(10, min(100, pattern_score))

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

    # ── 筹码 (10%) ──
    chip_score = 50
    if closes[i] > ma20: chip_score += 15
    if closes[i] > ma60: chip_score += 10
    if closes[i] < ma60: chip_score -= 15
    dimensions['筹码'] = round(chip_score, 1)

    weights = {'趋势': 0.25, '动量': 0.20, '量能': 0.20, '通道/波动': 0.15,
               '形态/结构': 0.10, '筹码': 0.10}
    total = sum(dimensions[k] * weights[k] for k in weights)

    return total, dimensions, patterns


def score_game_theory(closes, volumes, highs, lows):
    """博弈面评分 0-100 (权重 45%) — 量价关系 + 资金流向代理"""
    n = len(closes)
    if n < 30:
        return 50, {}

    obv = calc_obv(closes, volumes)
    dimensions = {}

    i = n - 1
    obv_ma5 = np.mean(obv[-5:])
    obv_ma20 = np.mean(obv[-20:])
    obv_div_score = 50
    if obv_ma5 > obv_ma20 * 1.05: obv_div_score += 20
    elif obv_ma5 < obv_ma20 * 0.95: obv_div_score -= 20

    price_5d = closes[i] - closes[max(0, i - 5)]
    obv_5d_chg = obv[i] - obv[max(0, i - 5)]
    if price_5d > 0 and obv_5d_chg < 0: obv_div_score -= 25
    elif price_5d < 0 and obv_5d_chg > 0: obv_div_score += 20

    dimensions['OBV背离'] = round(obv_div_score, 1)

    vol_price_score = 50
    for j in range(max(0, i - 4), i + 1):
        if closes[j] > closes[j - 1] and volumes[j] > volumes[j - 1] * 1.2:
            vol_price_score += 5
        elif closes[j] < closes[j - 1] and volumes[j] > volumes[j - 1] * 1.2:
            vol_price_score -= 8
    vol_price_score = max(10, min(100, vol_price_score))
    dimensions['量价关系'] = round(vol_price_score, 1)

    vol_threshold = np.percentile(volumes[-60:], 70) if n >= 60 else np.mean(volumes)
    big_vol_days = []
    for j in range(max(0, i - 20), i + 1):
        if volumes[j] > vol_threshold:
            big_vol_days.append(closes[j] > closes[j - 1])

    fund_score = 50
    if big_vol_days:
        buy_ratio = sum(big_vol_days) / len(big_vol_days)
        if buy_ratio > 0.6: fund_score += 20
        elif buy_ratio > 0.5: fund_score += 10
        elif buy_ratio < 0.4: fund_score -= 15

    net_dir = sum(1 for j in range(max(0, i - 4), i + 1) if closes[j] > closes[j - 1])
    if net_dir >= 4: fund_score += 10
    elif net_dir <= 1: fund_score -= 10

    dimensions['资金趋势'] = round(fund_score, 1)

    recent_volatility = np.std(
        [(closes[j] - closes[j - 1]) / closes[j - 1] for j in range(max(1, i - 10), i + 1)]
    ) * 100
    catalyst_score = 50
    if 2 <= recent_volatility <= 5: catalyst_score += 15
    elif recent_volatility > 5: catalyst_score += 25
    dimensions['事件驱动'] = round(catalyst_score, 1)

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

    high_250 = np.max(closes[-250:]) if n >= 250 else np.max(closes)
    low_250 = np.min(closes[-250:]) if n >= 250 else np.min(closes)
    price_position = (closes[i] - low_250) / (high_250 - low_250) * 100 \
        if high_250 != low_250 else 50

    pe_score = 50
    if price_position < 20: pe_score += 25
    elif price_position < 40: pe_score += 15
    elif price_position > 80: pe_score -= 20
    elif price_position > 60: pe_score -= 10
    dimensions['估值位置'] = round(pe_score, 1)

    if n >= 250:
        yoy_change = (closes[i] - closes[i - 250]) / closes[i - 250] * 100
        growth_score = 50
        if yoy_change > 30: growth_score += 25
        elif yoy_change > 15: growth_score += 15
        elif yoy_change > 0: growth_score += 5
        elif yoy_change > -15: growth_score -= 10
        else: growth_score -= 20
    else:
        growth_score = 50
    dimensions['增长趋势'] = round(growth_score, 1)

    if n >= 60:
        q_change = (closes[i] - closes[i - 60]) / closes[i - 60] * 100
        roe_score = 50
        if q_change > 20: roe_score += 25
        elif q_change > 10: roe_score += 15
        elif q_change > 0: roe_score += 5
        elif q_change > -10: roe_score -= 5
        else: roe_score -= 15

        returns = [(closes[j] - closes[j - 1]) / closes[j - 1]
                   for j in range(i - 59, i + 1)]
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
    window = closes[start:idx + 1]
    wlen = len(window)

    peaks = []
    for i in range(20, wlen - 10):
        segment = window[max(0, i - 15):min(wlen, i + 16)]
        if window[i] == max(segment) and window[i] > np.mean(
                window[max(0, i - 30):i]):
            if not peaks or i - peaks[-1] > 15:
                peaks.append(i)
            elif window[i] > window[peaks[-1]]:
                peaks[-1] = i

    if len(peaks) >= 2:
        p1, p2 = peaks[-2], peaks[-1]
        peak_diff = abs(window[p1] - window[p2]) / max(window[p1], window[p2]) * 100
        if peak_diff < 5 and p2 - p1 >= 15:
            between = window[p1:p2 + 1]
            neckline = np.min(between)
            if closes[idx] < neckline:
                return f'M顶破颈线({window[max(p1, p2)]:.1f})'

    if len(peaks) >= 3:
        pl, ph, pr = peaks[-3], peaks[-2], peaks[-1]
        if window[ph] > window[pl] and window[ph] > window[pr]:
            shoulder_diff = abs(window[pl] - window[pr]) / max(window[pl], window[pr]) * 100
            if shoulder_diff < 10:
                left_valley = np.min(window[pl:ph + 1])
                right_valley = np.min(window[ph:pr + 1])
                neckline = min(left_valley, right_valley)
                if closes[idx] < neckline:
                    return f'头肩顶破颈线({window[ph]:.1f})'

    if idx >= 3:
        prev_up = closes[idx - 3] < closes[idx - 2]
        curr_bear = closes[idx] < closes[idx - 1] * 0.97 and closes[idx] < closes[idx - 2]
        if prev_up and curr_bear:
            return '看跌吞没'

    return None


def _recent_price_divergence(closes, dif, i, lookback=30):
    """比较当前价格与此前极值的 DIF，检测无未来数据的背离。"""
    start = max(0, i - lookback)
    indices = np.arange(start, i)
    valid_indices = indices[np.isfinite(dif[indices])]
    if len(valid_indices) < 5:
        return False, False

    low_idx = valid_indices[np.argmin(closes[valid_indices])]
    high_idx = valid_indices[np.argmax(closes[valid_indices])]
    bottom = closes[i] > closes[low_idx] * 1.03 and dif[i] > dif[low_idx]
    top = closes[i] >= closes[high_idx] * 0.98 and dif[i] < dif[high_idx] * 0.9
    return bottom, top


# ═══════════════════════════
# 综合策略四维评分（回测 & 预测共用）
# ═══════════════════════════
def _score_comprehensive(closes, highs, lows, volumes,
                         dif, dea, bar, rsi_arr, k_arr, d_arr, j_arr,
                         bb_u, bb_l, wr_arr, obv_full, i):
    """融合策略四维评分引擎。
    返回 (macd_score, mf_score, fund_score, game_score, obv_flow, composite, gates_pass, details, breakdown)
    """
    # ══════ MACD核心 40% ══════
    macd_base = 50
    macd_items = []
    
    macd_bullish = dif[i] > dea[i]
    if macd_bullish:
        macd_base += 15
        macd_items.append({'rule': 'DIF 高于 DEA（多头状态）', 'adj': '+15'})
    else:
        macd_base -= 15
        macd_items.append({'rule': 'DIF 低于 DEA（空头状态）', 'adj': '−15'})

    if dif[i] > 0:
        macd_base += 12
        macd_items.append({'rule': 'DIF 零轴上 (做多区)', 'adj': '+12'})
    else:
        macd_base -= 8
        macd_items.append({'rule': 'DIF 零轴下 (做空区)', 'adj': '−8'})

    if i >= 5 and dif[i] > dif[i - 5]:
        macd_base += 8
        macd_items.append({'rule': f'DIF 5日斜率 ↑ ({dif[i]-dif[i-5]:+.2f})', 'adj': '+8'})
    else:
        macd_base -= 5
        macd_items.append({'rule': 'DIF 5日斜率 ↓', 'adj': '−5'})

    if i >= 3 and bar[i] > bar[i - 3]:
        macd_base += 5
        macd_items.append({'rule': f'BAR 3日趋势 ↑ ({bar[i]-bar[i-3]:+.2f})', 'adj': '+5'})
    else:
        macd_items.append({'rule': 'BAR 3日无上升趋势', 'adj': '0'})

    bottom_divergence, top_divergence = _recent_price_divergence(closes, dif, i)
    if bottom_divergence:
        macd_base += 10
        macd_items.append({'rule': '前低后动能修复（价离前低 DIF↑）', 'adj': '+10'})
    else:
        macd_items.append({'rule': '无前低动能修复', 'adj': '0'})

    if top_divergence:
        macd_base -= 15
        macd_items.append({'rule': '前高附近动能转弱（价近高 DIF↓）', 'adj': '−15'})
    else:
        macd_items.append({'rule': '无前高动能转弱', 'adj': '0'})

    macd_items.insert(0, {'rule': '基准分', 'adj': '50'})
    macd_score = macd_base

    # ══════ 多因子 30% ══════
    mf_base = 50
    mf_items = []

    rv = rsi_arr[i] if not np.isnan(rsi_arr[i]) else 50
    if 30 <= rv <= 65:
        mf_base += 10; mf_items.append({'rule': f'RSI={rv:.0f} 在健康区间30-65', 'adj': '+10'})
    elif rv < 30:
        mf_base += 15; mf_items.append({'rule': f'RSI={rv:.0f} 超卖 (<30)', 'adj': '+15'})
    elif rv > 80:
        mf_base -= 15; mf_items.append({'rule': f'RSI={rv:.0f} 超买 (>80)', 'adj': '−15'})
    elif rv > 70:
        mf_base -= 8; mf_items.append({'rule': f'RSI={rv:.0f} 偏强 (>70)', 'adj': '−8'})
    else:
        mf_items.append({'rule': f'RSI={rv:.0f} 中性', 'adj': '0'})

    kv = k_arr[i] if not np.isnan(k_arr[i]) else 50
    dv = d_arr[i] if not np.isnan(d_arr[i]) else 50
    jv = j_arr[i] if not np.isnan(j_arr[i]) else 50
    if kv > dv:
        mf_base += 10; mf_items.append({'rule': f'KDJ 多头状态 (K={kv:.0f}>D={dv:.0f})', 'adj': '+10'})
    elif kv < dv:
        mf_base -= 8; mf_items.append({'rule': f'KDJ 空头状态 (K={kv:.0f}<D={dv:.0f})', 'adj': '−8'})
    else:
        mf_items.append({'rule': 'KDJ 持平', 'adj': '0'})

    if jv < 0:
        mf_base += 8; mf_items.append({'rule': f'J={jv:.0f}<0 超卖', 'adj': '+8'})
    elif jv > 100:
        mf_base -= 8; mf_items.append({'rule': f'J={jv:.0f}>100 超买', 'adj': '−8'})
    else:
        mf_items.append({'rule': f'J={jv:.0f} 正常', 'adj': '0'})

    bb_pos = (closes[i] - bb_l[i]) / (bb_u[i] - bb_l[i]) * 100 \
        if not np.isnan(bb_u[i]) and bb_u[i] != bb_l[i] else 50
    if bb_pos < 10:
        mf_base += 12; mf_items.append({'rule': f'布林位置{bb_pos:.0f}% 下轨 (<10%)', 'adj': '+12'})
    elif bb_pos > 90:
        mf_base -= 8; mf_items.append({'rule': f'布林位置{bb_pos:.0f}% 上轨 (>90%)', 'adj': '−8'})
    else:
        mf_items.append({'rule': f'布林位置{bb_pos:.0f}% 中轨区间', 'adj': '0'})

    wv = wr_arr[i] if not np.isnan(wr_arr[i]) else 50
    if wv > 80:
        mf_base += 8; mf_items.append({'rule': f'WR={wv:.0f} 超卖 (>80)', 'adj': '+8'})
    elif wv < 20:
        mf_base -= 8; mf_items.append({'rule': f'WR={wv:.0f} 超买 (<20)', 'adj': '−8'})
    else:
        mf_items.append({'rule': f'WR={wv:.0f} 中性', 'adj': '0'})

    mf_items.insert(0, {'rule': '基准分', 'adj': '50'})
    mf_score = mf_base

    # ══════ 价格位置/趋势 15% ══════
    fund_base = 50
    fund_items = []

    if i >= 249:
        h250 = np.max(closes[i - 249:i + 1])
        l250 = np.min(closes[i - 249:i + 1])
        pos250 = (closes[i] - l250) / (h250 - l250) * 100 if h250 != l250 else 50
        if pos250 < 25:
            fund_base += 20; fund_items.append({'rule': f'250日价格位置{pos250:.0f}% 接近区间低位 (<25%)', 'adj': '+20'})
        elif pos250 < 40:
            fund_base += 10; fund_items.append({'rule': f'250日价格位置{pos250:.0f}% 偏低 (<40%)', 'adj': '+10'})
        elif pos250 > 80:
            fund_base -= 15; fund_items.append({'rule': f'250日价格位置{pos250:.0f}% 接近区间高位 (>80%)', 'adj': '−15'})
        else:
            fund_items.append({'rule': f'250日价格位置{pos250:.0f}% 居中', 'adj': '0'})
    else:
        fund_items.append({'rule': '250日价格数据不足', 'adj': '0'})

    if i >= 250:
        yoy = (closes[i] - closes[i - 250]) / closes[i - 250] * 100
        if yoy > 20:
            fund_base += 10; fund_items.append({'rule': f'年价格涨幅{yoy:+.0f}% 强势 (>20%)', 'adj': '+10'})
        elif yoy < -20:
            fund_base -= 10; fund_items.append({'rule': f'年价格涨幅{yoy:+.0f}% 弱势 (<−20%)', 'adj': '−10'})
        else:
            fund_items.append({'rule': f'年价格涨幅{yoy:+.0f}% 中性', 'adj': '0'})
    else:
        fund_items.append({'rule': '年价格数据不足', 'adj': '0'})

    fund_items.insert(0, {'rule': '基准分', 'adj': '50'})
    fund_score = fund_base

    # ══════ 量能 15% ══════
    game_base = 50
    game_items = []

    obv_flow = calc_obv_flow(obv_full, volumes, i)
    if obv_flow > 0.4:
        game_base += 12; game_items.append({'rule': f'OBV 5日净量能流 {obv_flow:+.0%}（明显流入）', 'adj': '+12'})
    elif obv_flow < -0.4:
        game_base -= 12; game_items.append({'rule': f'OBV 5日净量能流 {obv_flow:+.0%}（明显流出）', 'adj': '−12'})
    else:
        game_items.append({'rule': f'OBV 5日净量能流 {obv_flow:+.0%}（平稳）', 'adj': '0'})

    chg5 = closes[i] - closes[max(0, i - 5)]
    if chg5 > 0 and obv_flow < -0.2:
        game_base -= 18; game_items.append({'rule': f'量价背离 (价涨{chg5:+.2f} OBV跌)', 'adj': '−18'})
    elif chg5 < 0 and obv_flow > 0.2:
        game_base += 12; game_items.append({'rule': f'底部吸筹 (价跌{chg5:+.2f} OBV涨)', 'adj': '+12'})
    else:
        game_items.append({'rule': '无量价背离', 'adj': '0'})

    # 机构参与度代理
    inst_agent = None
    if i >= 60:
        ret_5d = (closes[i] - closes[max(0, i - 5)]) / closes[max(0, i - 5)] * 100
        ret_20d = (closes[i] - closes[max(0, i - 20)]) / closes[max(0, i - 20)] * 100
        vol_60d = np.std(
            [(closes[j] - closes[j - 1]) / closes[j - 1]
             for j in range(max(1, i - 60), i + 1)]
        ) * np.sqrt(252) * 100
        if ret_5d < -3 and vol_60d < 25:
            inst_agent = {'type': '机构吸筹', 'strength': 1, 'ret_5d': ret_5d,
                          'vol_60d': vol_60d}
            game_base += 10; game_items.append({'rule': f'下跌{ret_5d:+.0f}%+低波{vol_60d:.0f}%→机构吸筹', 'adj': '+10'})
        elif ret_20d < -10 and vol_60d < 25:
            inst_agent = {'type': '强吸筹', 'strength': 2, 'ret_20d': ret_20d,
                          'vol_60d': vol_60d}
            game_base += 15; game_items.append({'rule': f'深跌{ret_20d:+.0f}%+低波{vol_60d:.0f}%→强吸筹', 'adj': '+15'})
        elif ret_5d < -3 and vol_60d > 40:
            inst_agent = {'type': '散户恐慌', 'strength': -1, 'ret_5d': ret_5d,
                          'vol_60d': vol_60d}
            game_base -= 8; game_items.append({'rule': f'下跌{ret_5d:+.0f}%+高波{vol_60d:.0f}%→恐慌', 'adj': '−8'})
        else:
            game_items.append({'rule': '无特殊机构信号', 'adj': '0'})
    else:
        game_items.append({'rule': '60日数据不足，无机构代理', 'adj': '0'})

    game_items.insert(0, {'rule': '基准分', 'adj': '50'})
    game_score = game_base

    composite = macd_score * 0.40 + mf_score * 0.30 + fund_score * 0.15 + game_score * 0.15
    gates_pass = obv_flow >= -0.6 and rv <= 92 and not (macd_score < 35 and mf_score < 40)

    details = {
        'rv': rv, 'kv': kv, 'dv': dv, 'jv': jv, 'wv': wv, 'bb_pos': bb_pos,
        'obv_flow': obv_flow,
        'inst_agent': inst_agent,
    }

    breakdown = {
        'macd': macd_items,
        'multifactor': mf_items,
        'fundamental': fund_items,
        'game': game_items,
    }

    return macd_score, mf_score, fund_score, game_score, obv_flow, composite, gates_pass, details, breakdown


# ═══════════════════════════
# 融合策略回测
# ═══════════════════════════
def backtest_comprehensive(dates, closes, highs, lows, volumes, opens=None,
                           commission_rate=0.0003, stamp_duty_rate=0.0005,
                           buy_score=COMPREHENSIVE_BUY_SCORE,
                           stop_loss_pct=COMPREHENSIVE_STOP_LOSS_PCT,
                           take_profit_pct=COMPREHENSIVE_TAKE_PROFIT_PCT,
                           profit_protect_pct=COMPREHENSIVE_PROFIT_PROTECT_PCT,
                           profit_protect_score=COMPREHENSIVE_PROFIT_PROTECT_SCORE,
                           auxiliary_beta=COMPREHENSIVE_AUXILIARY_CONTRARIAN_BETA):
    """融合策略回测：收盘生成信号，下一交易日开盘成交并计入比例交易成本。"""
    n = len(closes)
    if n < 60:
        return []

    if opens is None:
        opens = closes.copy()
    opens = np.asarray(opens, dtype=float)
    if len(opens) != n:
        raise ValueError('opens 长度必须与 closes 一致')
    if not 0 <= auxiliary_beta <= 0.2:
        raise ValueError('auxiliary_beta 必须在 0 到 0.2 之间')

    dif, dea, bar = calc_macd(closes)
    rsi_arr = calc_rsi(closes)
    k_arr, d_arr, j_arr = calc_kdj(highs, lows, closes)
    bb_u, bb_m, bb_l = calc_bollinger(closes)
    wr_arr = calc_wr(highs, lows, closes)
    obv_full = calc_obv(closes, volumes)
    auxiliary_consensus = calc_auxiliary_consensus_series(
        closes, highs, lows, volumes, opens) if auxiliary_beta else None

    trades = []
    pos = None
    pending = None

    for i in range(60, n):
        # 前一交易日收盘形成的委托，在本交易日开盘执行。
        if pending and pending['side'] == 'buy':
            pos = {
                'bd': dates[i], 'bp': opens[i], 'bi': i,
                'signal_date': pending['signal_date'],
                'macd': pending['macd'], 'mf': pending['mf'],
                'fund': pending['fund'], 'game': pending['game'],
                'comp': pending['comp'], 'aux': pending['aux'],
            }
            pending = None
        elif pending and pending['side'] == 'sell' and pos is not None:
            sell_price = opens[i]
            pnl = calc_net_trade_return(
                pos['bp'], sell_price, commission_rate, stamp_duty_rate)
            trades.append({
                'buy_date': pos['bd'], 'sell_date': dates[i],
                'buy_signal_date': pos['signal_date'],
                'sell_signal_date': pending['signal_date'],
                'buy_price': pos['bp'], 'sell_price': sell_price,
                'profit_pct': round(pnl, 2), 'hold_days': i - pos['bi'],
                'buy_reason': f'融合{pos["comp"]:.0f}(M{pos["macd"]:.0f}/F{pos["mf"]:.0f}/价{pos["fund"]:.0f}/量{pos["game"]:.0f}/辅{pos["aux"]:+.1f})',
                'sell_reason': pending['reason'],
                'commission_rate': commission_rate,
                'stamp_duty_rate': stamp_duty_rate,
            })
            pos = None
            pending = None

        if np.isnan(dif[i]):
            continue

        macd_score, mf_score, fund_score, game_score, obv_flow, composite, gates_pass, details, _ = \
            _score_comprehensive(closes, highs, lows, volumes,
                                 dif, dea, bar, rsi_arr, k_arr, d_arr, j_arr,
                                 bb_u, bb_l, wr_arr, obv_full, i)
        if auxiliary_consensus is not None:
            auxiliary_adjustment = -auxiliary_beta * auxiliary_consensus[i]
            composite = apply_auxiliary_consensus_adjustment(
                composite, auxiliary_consensus[i], auxiliary_beta)
        else:
            auxiliary_adjustment = 0.0

        if pos is None:
            if (pending is None and i + 1 < n and
                    composite >= buy_score and gates_pass):
                pending = {
                    'side': 'buy', 'signal_date': dates[i],
                    'macd': round(macd_score, 1), 'mf': round(mf_score, 1),
                    'fund': round(fund_score, 1), 'game': round(game_score, 1),
                    'comp': round(composite, 1),
                    'aux': round(auxiliary_adjustment, 1),
                }
        else:
            pnl = calc_net_trade_return(
                pos['bp'], closes[i], commission_rate, stamp_duty_rate)
            sell = False
            reason = ''

            if pnl < stop_loss_pct:
                sell = True; reason = f'止损{pnl:.1f}%'
            elif pnl > take_profit_pct:
                sell = True; reason = f'止盈+{pnl:.1f}%'
            elif composite < 35:
                sell = True; reason = f'综合分{composite:.0f}<35'
            elif pnl > profit_protect_pct and composite < profit_protect_score:
                sell = True; reason = f'获利回吐+{pnl:.1f}%'

            if not sell:
                _, top_divergence = _recent_price_divergence(closes, dif, i)
                if top_divergence:
                    sell = True; reason = '顶背离'

            if not sell and i - pos['bi'] > 10 and (i - pos['bi']) % 5 == 0:
                pat = _quick_pattern_sell(closes, highs, i)
                if pat:
                    sell = True; reason = pat

            if sell and i + 1 < n:
                pending = {
                    'side': 'sell', 'signal_date': dates[i], 'reason': reason,
                }

    return trades


# ═══════════════════════════
# 融合策略预测 (结构化)
# ═══════════════════════════
def predict_comprehensive(dates, closes, highs, lows, volumes, holding=False,
                          opens=None,
                          auxiliary_beta=COMPREHENSIVE_AUXILIARY_CONTRARIAN_BETA):
    """融合策略当前状态预测 — 返回结构化 dict"""
    n = len(closes)
    if n < 60:
        return {'error': '数据不足，需要至少60根K线'}
    if not 0 <= auxiliary_beta <= 0.2:
        raise ValueError('auxiliary_beta 必须在 0 到 0.2 之间')

    dif, dea, bar = calc_macd(closes)
    rsi_arr = calc_rsi(closes)
    k_arr, d_arr, j_arr = calc_kdj(highs, lows, closes)
    bb_u, bb_m, bb_l = calc_bollinger(closes)
    wr_arr = calc_wr(highs, lows, closes)
    obv_full = calc_obv(closes, volumes)

    i = n - 1
    legacy_obv_ma20 = np.mean(obv_full[max(0, i - 20):i]) if i >= 20 else np.mean(obv_full[:i])
    legacy_obv_ratio = obv_full[i] / legacy_obv_ma20 if legacy_obv_ma20 > 0 else 1

    macd_score, mf_score, fund_score, game_score, obv_flow, composite, gates_pass, details, breakdown = \
        _score_comprehensive(closes, highs, lows, volumes,
                             dif, dea, bar, rsi_arr, k_arr, d_arr, j_arr,
                             bb_u, bb_l, wr_arr, obv_full, i)
    base_composite = composite
    aux_consensus = calc_auxiliary_consensus(
        closes, highs, lows, volumes, opens=opens)
    auxiliary_adjustment = -auxiliary_beta * aux_consensus['consensus_score']
    composite = apply_auxiliary_consensus_adjustment(
        base_composite, aux_consensus['consensus_score'], auxiliary_beta)

    # 信号映射
    if gates_pass:
        if composite >= 70:
            signal = '强烈看多'
            action = '增持' if holding else '买入'
        elif composite >= COMPREHENSIVE_BUY_SCORE:
            signal = '偏多'
            action = '拿住' if holding else '买入'
        elif composite >= 50:
            signal = '中性'
            action = '减持' if holding else '观望'
        elif composite >= 40:
            signal = '偏空'
            action = '减持' if holding else '不买'
        else:
            signal = '看空'
            action = '卖出' if holding else '不买'
    else:
        signal = '门禁否决'
        action = '观望'

    # 机构参与度代理
    inst_info = None
    if i >= 60:
        ret_5d_val = (closes[i] - closes[max(0, i - 5)]) / closes[max(0, i - 5)] * 100
        ret_20d_val = (closes[i] - closes[max(0, i - 20)]) / closes[max(0, i - 20)] * 100
        vol_60d_val = np.std(
            [(closes[j] - closes[j - 1]) / closes[j - 1]
             for j in range(max(1, i - 60), i + 1)]
        ) * np.sqrt(252) * 100
        inst_info = {
            'volatility_60d': round(vol_60d_val, 1),
            'ret_5d': round(ret_5d_val, 1),
            'ret_20d': round(ret_20d_val, 1),
        }

    # 门禁详情
    gate_details = []
    if obv_flow < -0.6:
        gate_details.append({'gate': 'OBV净量能流', 'value': round(obv_flow, 2),
                             'threshold': -0.6, 'passed': False})
    if details['rv'] > 92:
        gate_details.append({'gate': 'RSI', 'value': round(details['rv'], 0),
                             'threshold': 92, 'passed': False})
    if macd_score < 35 and mf_score < 40:
        gate_details.append({'gate': '双弱', 'passed': False})

    # 顶背离
    _, top_divergence = _recent_price_divergence(closes, dif, i)

    # 经典形态
    patterns, _ = detect_classic_patterns(closes, highs, lows)

    # 关键位
    ma20 = np.mean(closes[-20:])
    ma60 = np.mean(closes[-60:])

    return {
        'strategy': 'comprehensive',
        'date': dates[-1],
        'close': round(float(closes[i]), 2),
        'holding': holding,

        # 四维评分
        'scores': {
            'macd': {'weight': 0.40, 'score': round(macd_score, 1),
                     'dif': round(float(dif[i]), 2), 'dea': round(float(dea[i]), 2),
                     'bar': round(float(bar[i]), 2)},
            'multifactor': {'weight': 0.30, 'score': round(mf_score, 1),
                            'rsi': round(float(details['rv']), 0),
                            'k': round(float(details['kv']), 0),
                            'd': round(float(details['dv']), 0),
                            'j': round(float(details['jv']), 0),
                            'wr': round(float(details['wv']), 0)},
            'fundamental': {'weight': 0.15, 'score': round(fund_score, 1),
                            'label': '价格位置/趋势'},
            'game': {'weight': 0.15, 'score': round(game_score, 1),
                     'obv_flow': round(float(obv_flow), 2),
                     # 小程序尚未迁移；该字段只用于兼容旧接口，非当前门禁依据。
                     'obv_ratio': round(float(legacy_obv_ratio), 2)},
        },
        'base_composite': round(base_composite, 1),
        'auxiliary_adjustment': round(auxiliary_adjustment, 1),
        'auxiliary_beta': auxiliary_beta,
        'composite': round(composite, 1),

        # 门禁
        'gates': {
            'passed': gates_pass,
            'details': gate_details,
        },

        # 信号
        'signal': signal,
        'action': action,

        # 顶背离
        'top_divergence': top_divergence,

        # 形态
        'patterns': patterns[:5] if patterns else [],

        # 关键位
        'key_levels': {
            'ma20': round(float(ma20), 2),
            'ma60': round(float(ma60), 2),
            'bb_upper': round(float(bb_u[-1]), 2),
            'bb_lower': round(float(bb_l[-1]), 2),
        },

        # 机构代理
        'institution_proxy': inst_info,

        # 计算明细
        'details': {
            'bb_position': round(float(details['bb_pos']), 1),
        },
        # 评分明细 (每项评分规则)
        'breakdown': breakdown,

        # 辅助指标投票面板
        'auxiliary_consensus': aux_consensus,
    }


def format_predict_comprehensive(pred, holding=False):
    """将结构化融合策略预测转为文本行列表"""
    if 'error' in pred:
        return [pred['error']]

    s = pred['scores']
    lines = []
    lines.append(f"{'=' * 50}")
    lines.append(f"  三合资本 · 融合策略 — {pred['date']}")
    lines.append(f"{'=' * 50}")

    lines.append(f"\n── 📊 四维评分 ──")
    lines.append(f"  MACD核心 (40%): {s['macd']['score']:.1f} 分 — "
                 f"DIF{s['macd']['dif']:.2f}/DEA{s['macd']['dea']:.2f} "
                 f"BAR{s['macd']['bar']:.2f}")
    lines.append(f"  多因子 (30%): {s['multifactor']['score']:.1f} 分 — "
                 f"RSI{s['multifactor']['rsi']:.0f} "
                 f"K{s['multifactor']['k']:.0f}/D{s['multifactor']['d']:.0f}/"
                 f"J{s['multifactor']['j']:.0f} WR{s['multifactor']['wr']:.0f}")
    lines.append(f"  {s['fundamental'].get('label', '价格位置/趋势')} (15%): "
                 f"{s['fundamental']['score']:.1f} 分")
    lines.append(f"  量能 (15%): {s['game']['score']:.1f} 分 — "
                 f"OBV 5日净量能流{s['game']['obv_flow']:+.0%}")

    # 机构代理
    if pred.get('institution_proxy'):
        ip = pred['institution_proxy']
        v = ip['volatility_60d']
        if v < 20: inst_label = '🏛️ 机构主导 (低波)'
        elif v < 30: inst_label = '🤝 均衡型'
        else: inst_label = '👤 散户活跃 (高波)'
        inst_sig = ''
        if ip.get('ret_5d', 0) < -3 and v < 25:
            inst_sig = ' ⚡下跌低波→机构吸筹信号'
        elif ip.get('ret_20d', 0) < -10 and v < 25:
            inst_sig = ' ⚡⚡深跌低波→强吸筹信号'
        lines.append(f"  机构代理: {inst_label} 波动率{v:.1f}%{inst_sig}")

    lines.append(f"\n── 🎯 综合判断 ──")
    lines.append(f"  四维基础分 S: {pred['base_composite']:.1f} "
                 f"(M{s['macd']['score']:.0f}×0.40 + "
                 f"F{s['multifactor']['score']:.0f}×0.30 + "
                 f"基{s['fundamental']['score']:.0f}×0.15 + "
                 f"量{s['game']['score']:.0f}×0.15)")
    lines.append(f"  辅助共识修正: {pred['auxiliary_adjustment']:+.1f} "
                 f"(A={pred['auxiliary_consensus']['consensus_score']:+.0f}, "
                 f"β={pred['auxiliary_beta']:.2f})")
    lines.append(f"  有效评分 S*: {pred['composite']:.1f}")

    # 门禁
    if not pred['gates']['passed']:
        lines.append(f"\n── ⚠️ 质量门禁 ──")
        for g in pred['gates']['details']:
            if g['gate'] == 'OBV净量能流':
                lines.append(f"  🔴 OBV 5日净量能流过低({g['value']:+.0%}<-60%) → 否决")
            elif g['gate'] == 'RSI':
                lines.append(f"  🔴 RSI极度超买({g['value']:.0f}>92) → 否决")
            elif g['gate'] == '双弱':
                lines.append(f"  🔴 双弱(MACD<35 且 多因子<40) → 否决")

    if pred['top_divergence']:
        lines.append(f"  🔴 顶背离警告: 价格近30日高但DIF未确认")

    if pred['patterns']:
        lines.append(f"\n── 📐 经典形态 ──")
        for p in pred['patterns']:
            if 'type' in p:
                status = '⛔颈线已破' if p.get('neckline_broken') else '未破颈线'
                lines.append(f"  {p['type']}: 评分{p['score']:+d}  {status}")
            elif 'signals' in p:
                lines.append(f"  {', '.join(p['signals'])}: 评分{p['score']:+d}")

    # 信号
    signal_icon = {'强烈看多': '🟢', '偏多': '🟢', '中性': '🟡',
                   '偏空': '🟠', '看空': '🔴', '门禁否决': '🔴'}
    icon = signal_icon.get(pred['signal'], '⚪')
    lines.append(f"\n  信号: {icon} {pred['signal']}  |  建议: {pred['action']}")

    lines.append(f"\n── 📍 关键位 ──")
    kl = pred['key_levels']
    lines.append(f"  MA20={kl['ma20']:.2f}  MA60={kl['ma60']:.2f}  "
                 f"布林: 上{kl['bb_upper']:.2f} 下{kl['bb_lower']:.2f}")

    lines.append(f"\n{'=' * 50}")
    n_icon = max(1, min(10, int(pred['composite'] / 10)))
    if holding:
        if pred['action'] in ('卖出',):
            lines.append(f"  {'❌' * n_icon}  {pred['action']}")
        elif pred['action'] in ('增持', '拿住'):
            lines.append(f"  {'✅' * n_icon}  {pred['action']}")
        else:
            lines.append(f"  {'⚠️' * n_icon}  {pred['action']}")
    else:
        if pred['action'] == '买入':
            lines.append(f"  {'✅' * n_icon}  {pred['action']}")
        else:
            lines.append(f"  {'❌' * n_icon}  {pred['action']}")
    lines.append(f"{'=' * 50}")

    return lines


# ═══════════════════════════
# 多因子共振策略
# ═══════════════════════════
def backtest_multifactor(dates, closes, highs, lows, volumes):
    """多因子共振回测：RSI + KDJ + Bollinger + WR + 量能确认"""
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
        ma20[i] = np.mean(closes[i - 19:i + 1])
        vol_ma20[i] = np.mean(volumes[i - 19:i + 1])

    sigs = []
    pos = None
    for i in range(30, n):
        if np.isnan(rsi[i]) or np.isnan(k_line[i]) or np.isnan(d_line[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(wr[i]):
            continue

        score = 0
        if rsi[i] < 25: score += 3
        elif rsi[i] < 35: score += 2
        elif rsi[i] < 45: score += 1
        elif rsi[i] > 80: score -= 3
        elif rsi[i] > 70: score -= 2
        elif rsi[i] > 60: score -= 1

        if k_line[i] > d_line[i] and k_line[i - 1] <= d_line[i - 1]:
            score += 3 if k_line[i] < 50 else 2
        elif k_line[i] < d_line[i] and k_line[i - 1] >= d_line[i - 1]:
            score -= 3 if k_line[i] > 50 else 2
        elif k_line[i] > d_line[i]: score += 1
        else: score -= 1

        if not np.isnan(j_line[i]):
            if j_line[i] < 0: score += 2
            elif j_line[i] > 100: score -= 2

        if closes[i] < bb_lower[i]: score += 2
        elif closes[i] > bb_upper[i]: score -= 1

        if wr[i] > 85: score += 2
        elif wr[i] < 15: score -= 2

        if not np.isnan(ma20[i]):
            if closes[i] > ma20[i]: score += 1
            else: score -= 1

        if pos is None:
            if score >= 2 and rsi[i] < 75:
                sigs.append({'date': dates[i], 'idx': i, 'action': 'buy',
                             'price': closes[i],
                             'reason': f'多因子共振(score{score:+d}/RSI{int(rsi[i])}/KDJ{int(k_line[i])}/{int(d_line[i])})'})
                pos = i
        else:
            pnl = (closes[i] - closes[pos]) / closes[pos] * 100
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
            pos_state = {'bd': s['date'], 'bp': s['price'], 'bi': s['idx'],
                         'br': s['reason']}
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
    """多因子共振当前状态预测 — 返回结构化 dict"""
    n = len(closes)
    if n < 35:
        return {'error': '数据不足，无法预测'}

    rsi = calc_rsi(closes)
    k_line, d_line, j_line = calc_kdj(highs, lows, closes)
    bb_upper, bb_mid, bb_lower = calc_bollinger(closes)
    wr = calc_wr(highs, lows, closes)
    ma20 = np.full(n, np.nan)
    vol_ma20 = np.full(n, np.nan)
    for i in range(19, n):
        ma20[i] = np.mean(closes[i - 19:i + 1])
        vol_ma20[i] = np.mean(volumes[i - 19:i + 1])

    i = n - 1
    while i >= 0 and np.isnan(rsi[i]):
        i -= 1
    if i < 35:
        return {'error': '数据不足，无法预测'}

    score = 0
    reasons = []

    cur_rsi = rsi[i] if not np.isnan(rsi[i]) else 50
    cur_k = k_line[i] if not np.isnan(k_line[i]) else 50
    cur_d = d_line[i] if not np.isnan(d_line[i]) else 50
    cur_j = j_line[i] if not np.isnan(j_line[i]) else 50
    cur_wr = wr[i] if not np.isnan(wr[i]) else 50
    cur_bb_u = bb_upper[i] if not np.isnan(bb_upper[i]) else closes[i] * 1.1
    cur_bb_l = bb_lower[i] if not np.isnan(bb_lower[i]) else closes[i] * 0.9
    cur_bb_m = bb_mid[i] if not np.isnan(bb_mid[i]) else closes[i]
    cur_ma20 = ma20[i]
    cur_vol_ma20 = vol_ma20[i]

    if cur_rsi < 35: score += 2; reasons.append('RSI超卖')
    elif cur_rsi > 70: score -= 2; reasons.append('RSI超买')
    elif cur_rsi < 50: score += 1; reasons.append('RSI偏弱')
    else: score -= 1; reasons.append('RSI偏强')

    if cur_k > cur_d: score += 2; reasons.append('KDJ金叉')
    else: score -= 2; reasons.append('KDJ死叉')

    if not np.isnan(cur_j):
        if cur_j < 0: score += 2; reasons.append('J<0超卖')
        elif cur_j > 100: score -= 2; reasons.append('J>100超买')

    if closes[i] < cur_bb_l: score += 2; reasons.append('跌破布林下轨')
    elif closes[i] > cur_bb_u: score -= 1; reasons.append('突破布林上轨')

    if cur_wr > 80: score += 1; reasons.append('WR超卖')
    elif cur_wr < 20: score -= 1; reasons.append('WR超买')

    if not np.isnan(cur_ma20) and closes[i] > cur_ma20:
        score += 1; reasons.append('MA20之上')
    else:
        score -= 1; reasons.append('MA20之下')

    volume_boost = not np.isnan(cur_vol_ma20) and volumes[i] > cur_vol_ma20 * 1.3
    if volume_boost:
        reasons.append('放量')

    # 信号映射
    if score >= 4:
        signal = '强烈看多'
        action = '增持' if holding else '买入'
    elif score >= 2:
        signal = '偏多'
        action = '拿住' if holding else '买入'
    elif score >= 0:
        signal = '中性'
        action = '减持' if holding else '观望'
    elif score >= -2:
        signal = '偏空'
        action = '减持' if holding else '不买'
    else:
        signal = '看空'
        action = '卖出' if holding else '不买'

    return {
        'strategy': 'multifactor',
        'date': dates[i],
        'close': round(float(closes[i]), 2),
        'holding': holding,

        'indicators': {
            'rsi': round(float(cur_rsi), 1),
            'kdj': {'k': round(float(cur_k), 1), 'd': round(float(cur_d), 1),
                    'j': round(float(cur_j), 1)},
            'bollinger': {'upper': round(float(cur_bb_u), 2),
                          'middle': round(float(cur_bb_m), 2),
                          'lower': round(float(cur_bb_l), 2)},
            'wr': round(float(cur_wr), 1),
            'ma20': round(float(cur_ma20), 2) if not np.isnan(cur_ma20) else None,
        },

        'score': score,
        'signal': signal,
        'action': action,
        'reasons': reasons,
        'volume_boost': volume_boost,

        'key_levels': {
            'ma20': round(float(cur_ma20), 2) if not np.isnan(cur_ma20) else None,
        },
    }


def format_predict_multifactor(pred, holding=False):
    """将结构化多因子预测转为文本行列表"""
    if 'error' in pred:
        return [pred['error']]

    ind = pred['indicators']
    lines = []
    lines.append(f"{'=' * 50}")
    lines.append(f"  多因子共振预测 — {pred['date']}")
    lines.append(f"{'=' * 50}")

    lines.append(f"\n── 当前指标 ──")
    ma20_str = f"  |  MA20: {ind['ma20']:.1f}" if ind['ma20'] else ""
    lines.append(f"  收盘: {pred['close']:.2f}{ma20_str}")
    lines.append(f"  RSI: {ind['rsi']:.1f}  |  KDJ: K={ind['kdj']['k']:.1f} "
                 f"D={ind['kdj']['d']:.1f} J={ind['kdj']['j']:.1f}")
    lines.append(f"  布林: 上{ind['bollinger']['upper']:.2f} "
                 f"中{ind['bollinger']['middle']:.2f} "
                 f"下{ind['bollinger']['lower']:.2f}")
    lines.append(f"  WR: {ind['wr']:.1f}")

    lines.append(f"\n── 综合判断 ──")
    signal_icon = {'强烈看多': '🟢', '偏多': '🟢', '中性': '🟡',
                   '偏空': '🟠', '看空': '🔴'}
    icon = signal_icon.get(pred['signal'], '⚪')
    lines.append(f"  评分: {pred['score']:+d}  |  信号: {icon} {pred['signal']}")
    lines.append(f"  建议: {pred['action']}")
    lines.append(f"  依据: {', '.join(pred['reasons']) if pred['reasons'] else '无明确信号'}")

    lines.append(f"\n{'=' * 50}")
    n_icon = max(1, min(10, abs(pred['score'])))
    if holding:
        if pred['action'] == '卖出':
            lines.append(f"  {'❌' * n_icon}  {pred['action']}")
        elif pred['action'] in ('增持', '拿住'):
            lines.append(f"  {'✅' * n_icon}  {pred['action']}")
        else:
            lines.append(f"  {'⚠️' * n_icon}  {pred['action']}")
    else:
        if pred['action'] == '买入':
            lines.append(f"  {'✅' * n_icon}  {pred['action']}")
        else:
            lines.append(f"  {'❌' * n_icon}  {pred['action']}")
    lines.append(f"{'=' * 50}")

    return lines


# ═══════════════════════════
# 背离 + 牛熊
# ═══════════════════════════
def find_divergences(dates, closes, dif, lookback=60, cluster_gap=20):
    tops, bottoms = [], []
    n = len(closes)
    for i in range(34 + lookback, n):
        pw = closes[i - lookback:i + 1]; dw = dif[i - lookback:i + 1]
        if np.isnan(dw).any(): continue
        mid = lookback // 2
        if np.argmax(pw) == lookback and closes[i] > np.mean(pw[:mid]):
            if dif[i] < np.max(dw[:mid]) * 0.95:
                tops.append({'date': datetime.strptime(dates[i], '%Y-%m-%d'),
                             'idx': i, 'price': closes[i], 'dif': dif[i]})
        if np.argmin(pw) == lookback and closes[i] < np.mean(pw[:mid]):
            if dif[i] > np.min(dw[:mid]) * 1.05:
                bottoms.append({'date': datetime.strptime(dates[i], '%Y-%m-%d'),
                                'idx': i, 'price': closes[i], 'dif': dif[i]})

    def cluster(sigs, rev=False):
        if not sigs: return []
        clustered, cur = [], [sigs[0]]
        for s in sigs[1:]:
            if s['idx'] - cur[-1]['idx'] <= cluster_gap: cur.append(s)
            else:
                clustered.append(
                    min(cur, key=lambda x: x['dif']) if rev
                    else max(cur, key=lambda x: x['dif']))
                cur = [s]
        clustered.append(
            min(cur, key=lambda x: x['dif']) if rev
            else max(cur, key=lambda x: x['dif']))
        return clustered

    return cluster(tops), cluster(bottoms, rev=True)


def detect_regime(closes, ma_period=60, slope_days=10, confirm=5):
    n = len(closes)
    ma = np.full(n, np.nan)
    for i in range(ma_period - 1, n):
        ma[i] = np.mean(closes[i - ma_period + 1:i + 1])
    raw = []
    for i in range(n):
        if i < ma_period + max(slope_days, confirm) or np.isnan(ma[i]):
            raw.append('bear')
        else:
            raw.append('bull' if (closes[i] > ma[i] and ma[i] > ma[i - slope_days])
                       else 'bear')
    reg = [raw[0]] * confirm
    for i in range(confirm, n):
        reg.append(reg[-1])
    for i in range(confirm, n):
        if raw[i] != reg[i - 1] and all(r == raw[i] for r in raw[i - confirm + 1:i + 1]):
            reg[i] = raw[i]
        else:
            reg[i] = reg[i - 1]
    return np.array(reg), ma


def zero_line_cycles(dates, dif):
    cycles, current, start = [], None, 0
    for i in range(34, len(dif)):
        if np.isnan(dif[i]) or np.isnan(dif[i - 1]): continue
        if dif[i - 1] <= 0 and dif[i] > 0:
            if current == 'below':
                cycles.append({'zone': 'below', 'start': start, 'end': i - 1,
                               'start_date': dates[start], 'end_date': dates[i - 1],
                               'duration': i - start})
            start, current = i, 'above'
        elif dif[i - 1] >= 0 and dif[i] < 0:
            if current == 'above':
                cycles.append({'zone': 'above', 'start': start, 'end': i - 1,
                               'start_date': dates[start], 'end_date': dates[i - 1],
                               'duration': i - start})
            start, current = i, 'below'
    if current:
        cycles.append({'zone': current, 'start': start, 'end': len(dif) - 1,
                       'start_date': dates[start], 'end_date': dates[-1],
                       'duration': len(dif) - start, 'ongoing': True})
    return cycles


# ═══════════════════════════
# MACD 预测 (结构化)
# ═══════════════════════════
def predict(dates, closes, dif, dea, bar, regimes, tops, bottoms, holding=False):
    """基于当前MACD状态预测 — 返回结构化 dict"""
    i = len(dif) - 1
    while i >= 0 and np.isnan(dif[i]):
        i -= 1
    if i < 35:
        return {'error': '数据不足，无法预测'}

    dif_slope_5d = dif[i] - dif[max(0, i - 5)] if i >= 5 else None
    dif_slope_10d = dif[i] - dif[max(0, i - 10)] if i >= 10 else None

    golden_cross = dif[i] > dea[i]
    gap = abs(dif[i] - dea[i])

    # 交叉预测
    cross_prediction = None
    if golden_cross and dif_slope_5d is not None and dif_slope_5d < 0:
        days = int(gap / abs(dif_slope_5d / 5))
        cross_prediction = {'type': '死叉预警', 'days': days}
    elif not golden_cross and dif_slope_5d is not None and dif_slope_5d > 0:
        days = int(gap / (dif_slope_5d / 5))
        cross_prediction = {'type': '金叉预期', 'days': days}

    # 零轴预测
    zero_prediction = None
    if dif[i] > 0:
        if dif_slope_10d is not None and dif_slope_10d < 0:
            days = int(dif[i] / abs(dif_slope_10d / 10))
            zero_prediction = {'type': '下穿零轴', 'days': days}
    else:
        if dif_slope_10d is not None and dif_slope_10d > 0:
            days = int(abs(dif[i]) / (dif_slope_10d / 10))
            zero_prediction = {'type': '上穿零轴', 'days': days}

    # 背离检查
    recent_tops = [t for t in tops
                   if (datetime.strptime(dates[i], '%Y-%m-%d') - t['date']).days < 120]
    recent_bottoms = [b for b in bottoms
                      if (datetime.strptime(dates[i], '%Y-%m-%d') - b['date']).days < 120]

    top_divergence_now = False
    if recent_tops:
        last_top = recent_tops[-1]
        if closes[i] > last_top['price'] and dif[i] < last_top['dif']:
            top_divergence_now = True

    # 评分
    score = 0
    reasons = []
    if golden_cross: score += 2; reasons.append('金叉')
    else: score -= 2; reasons.append('死叉')
    if dif[i] > 0: score += 1; reasons.append('零轴上')
    else: score -= 1; reasons.append('零轴下')
    if regimes[i] == 'bull': score += 2; reasons.append('牛市')
    else: score -= 2; reasons.append('熊市')
    if dif_slope_5d is not None:
        if dif_slope_5d > 0: score += 1; reasons.append('DIF↑')
        else: score -= 1; reasons.append('DIF↓')
    if top_divergence_now:
        score -= 3; reasons.append('顶背离进行中')

    if score >= 4:
        signal = '强烈看多'; action = '增持' if holding else '买入'
    elif score >= 2:
        signal = '偏多'; action = '拿住' if holding else '买入'
    elif score >= 0:
        signal = '中性'; action = '减持' if holding else '观望'
    elif score >= -2:
        signal = '偏空'; action = '减持' if holding else '不买'
    else:
        signal = '看空'; action = '卖出' if holding else '不买'

    # BAR趋势
    bar_trend = None
    if bar[i] < 0 and i > 0 and bar[i] > bar[i - 1]:
        bar_trend = '空方减弱'
    elif bar[i] > 0 and i > 0 and bar[i] < bar[i - 1]:
        bar_trend = '多方减弱'

    return {
        'strategy': 'macd',
        'date': dates[i],
        'close': round(float(closes[i]), 2),
        'holding': holding,

        'dif': round(float(dif[i]), 2),
        'dea': round(float(dea[i]), 2),
        'bar': round(float(bar[i]), 2),
        'regime': 'bull' if regimes[i] == 'bull' else 'bear',
        'golden_cross': golden_cross,

        'dif_slope_5d': round(float(dif_slope_5d), 4) if dif_slope_5d is not None else None,
        'dif_slope_10d': round(float(dif_slope_10d), 4) if dif_slope_10d is not None else None,

        'cross_prediction': cross_prediction,
        'zero_prediction': zero_prediction,
        'bar_trend': bar_trend,

        'divergences': {
            'recent_tops': len(recent_tops),
            'recent_bottoms': len(recent_bottoms),
            'top_divergence_now': top_divergence_now,
            'last_top_days_ago': (datetime.strptime(dates[i], '%Y-%m-%d')
                                  - recent_tops[-1]['date']).days if recent_tops else None,
        },

        'score': score,
        'signal': signal,
        'action': action,
        'reasons': reasons,
    }


def format_predict(pred, holding=False):
    """将结构化MACD预测转为文本行列表"""
    if 'error' in pred:
        return [pred['error']]

    lines = []
    lines.append(f"{'=' * 50}")
    lines.append(f"  MACD 预测 — {pred['date']}")
    lines.append(f"{'=' * 50}")

    lines.append(f"\n── 当前状态 ──")
    regime_icon = '🟢牛市' if pred['regime'] == 'bull' else '🔴熊市'
    lines.append(f"  收盘: {pred['close']:.2f}  |  {regime_icon}")
    lines.append(f"  DIF: {pred['dif']:.2f}  |  DEA: {pred['dea']:.2f}  |  BAR: {pred['bar']:.2f}")

    if pred['golden_cross']:
        lines.append(f"  信号: 🟢 金叉 (DIF > DEA)")
        if pred.get('cross_prediction'):
            cp = pred['cross_prediction']
            lines.append(f"  {cp['type']}: DIF每5日下降{abs(pred['dif_slope_5d']):.2f}，约{cp['days']}天后死叉")
        else:
            lines.append(f"  金叉安全: DIF斜率向上，距死叉{abs(pred['dif']-pred['dea']):.2f}")
    else:
        lines.append(f"  信号: 🔴 死叉 (DIF < DEA)")
        if pred.get('cross_prediction'):
            cp = pred['cross_prediction']
            lines.append(f"  {cp['type']}: DIF每5日上升{pred['dif_slope_5d']:.2f}，约{cp['days']}天后金叉")

    lines.append(f"\n── 零轴分析 ──")
    if pred['dif'] > 0:
        lines.append(f"  DIF在零轴上方 (做多区) → {pred['dif']:.2f}")
        if pred.get('zero_prediction'):
            zp = pred['zero_prediction']
            lines.append(f"  ⚠️ DIF下滑中，约{zp['days']}天后下穿零轴")
        else:
            lines.append(f"  ✅ DIF向上，零轴支撑稳固")
    else:
        lines.append(f"  DIF在零轴下方 (做空区) → {pred['dif']:.2f}")
        if pred.get('zero_prediction'):
            zp = pred['zero_prediction']
            lines.append(f"  DIF上升中，约{zp['days']}天后上穿零轴")

    lines.append(f"\n── 背离检查 ──")
    div = pred['divergences']
    if div['recent_tops'] > 0:
        lines.append(f"  近4个月顶背离: {div['recent_tops']}次")
        if div['top_divergence_now']:
            lines.append(f"  🔴 当前正在形成新的顶背离！")
    else:
        lines.append(f"  近4个月无顶背离")
    if div['recent_bottoms'] > 0:
        lines.append(f"  近4个月底背离: {div['recent_bottoms']}次")
    else:
        lines.append(f"  近4个月无底背离")

    lines.append(f"\n── 综合预测 ──")
    signal_icon = {'强烈看多': '🟢', '偏多': '🟢', '中性': '🟡',
                   '偏空': '🟠', '看空': '🔴'}
    icon = signal_icon.get(pred['signal'], '⚪')
    lines.append(f"  评分: {pred['score']:+d}  |  信号: {icon} {pred['signal']}")
    lines.append(f"  建议: {pred['action']}")
    lines.append(f"  依据: {', '.join(pred['reasons']) if pred['reasons'] else '无明确信号'}")

    lines.append(f"\n── 关注点位 ──")
    if pred['golden_cross']:
        lines.append(f"  卖出触发: DIF下穿DEA (当前差{abs(pred['dif']-pred['dea']):.2f})")
        lines.append(f"  卖出触发: 出现顶背离")
    if pred['regime'] == 'bull':
        lines.append(f"  牛市护盾: 死叉不卖，只等顶背离")
    else:
        lines.append(f"  熊市警告: 死叉立即卖出")
    if pred.get('bar_trend'):
        lines.append(f"  BAR柱{pred['bar_trend']}")

    lines.append(f"\n{'=' * 50}")
    n = max(1, min(10, abs(pred['score'])))
    if holding:
        if pred['action'] == '卖出':
            lines.append(f"  {'❌' * n}  {pred['action']}")
        elif pred['action'] in ('增持', '拿住'):
            lines.append(f"  {'✅' * n}  {pred['action']}")
        else:
            lines.append(f"  {'⚠️' * n}  {pred['action']}")
    else:
        if pred['action'] == '买入':
            lines.append(f"  {'✅' * n}  {pred['action']}")
        else:
            lines.append(f"  {'❌' * n}  {pred['action']}")
    lines.append(f"{'=' * 50}")

    return lines


# ═══════════════════════════
# MACD 回测
# ═══════════════════════════
def backtest(dates, closes, dif, dea, tops, bottoms, regimes, cooldown=25):
    sigs = []
    for t in tops:
        sigs.append({'date': t['date'].strftime('%Y-%m-%d'), 'idx': t['idx'],
                     'action': 'sell', 'price': t['price'], 'reason': '顶背离'})
    for b in bottoms:
        sigs.append({'date': b['date'].strftime('%Y-%m-%d'), 'idx': b['idx'],
                     'action': 'buy', 'price': b['price'], 'reason': '底背离'})
    ls = -999
    for i in range(35, len(dif) - 1):
        if np.isnan(dif[i]) or np.isnan(dea[i]): continue
        if (dif[i - 1] <= dea[i - 1] and dif[i] > dea[i] and dif[i] > 0
                and dif[i] > dif[i - 3] and i - ls > cooldown
                and regimes[i] == 'bull'):
            if i + 2 < len(dif) and dif[i + 1] > dea[i + 1] and dif[i + 2] > dea[i + 2]:
                if not any(abs(s['idx'] - i) < 15 for s in sigs if s['action'] == 'buy'):
                    sigs.append({'date': dates[i], 'idx': i, 'action': 'buy',
                                 'price': closes[i], 'reason': '零轴上金叉'})
        if dif[i - 1] >= dea[i - 1] and dif[i] < dea[i]:
            if regimes[i] == 'bear' and not any(
                    abs(s['idx'] - i) < 10 for s in sigs if s['action'] == 'sell'):
                sigs.append({'date': dates[i], 'idx': i, 'action': 'sell',
                             'price': closes[i], 'reason': '死叉(熊市)'})
                ls = i
    sigs.sort(key=lambda x: x['idx'])
    trades, pos = [], None
    for s in sigs:
        if s['action'] == 'buy' and pos is None:
            pos = {'bd': s['date'], 'bp': s['price'], 'bi': s['idx'],
                   'br': s['reason']}
        elif s['action'] == 'sell' and pos is not None:
            pct = (s['price'] - pos['bp']) / pos['bp'] * 100
            trades.append({
                'buy_date': pos['bd'], 'sell_date': s['date'],
                'buy_price': pos['bp'], 'sell_price': s['price'],
                'profit_pct': pct, 'buy_reason': pos['br'],
                'sell_reason': s['reason'], 'hold_days': s['idx'] - pos['bi']
            })
            pos = None
    return trades
