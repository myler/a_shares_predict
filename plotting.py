#!/usr/bin/env python3
"""图表层 — 所有 matplotlib 画图函数，供 CLI / Web / API 共用"""

import os, io, base64
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── 字体 ──
FONT_PATHS = [
    os.path.join(ROOT, 'wqy-zenhei.ttf'),
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
]
_font_loaded = False
for _fp in FONT_PATHS:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        try:
            plt.rcParams['font.family'] = fm.FontProperties(fname=_fp).get_name()
            _font_loaded = True
        except Exception:
            pass
        break
if not _font_loaded:
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
plt.rcParams['axes.unicode_minus'] = False

# 延迟导入 engine 纯计算函数，避免循环依赖
from engine import calc_macd, calc_rsi, calc_kdj, calc_bollinger, calc_obv


# ═══════════════════════════
# CLI 用 MACD 图表
# ═══════════════════════════
def plot_all(code, name, dates, closes, dif, dea, bar, cycles, tops, bottoms,
             trades, regimes, ma60, outdir):
    fig = plt.figure(figsize=(20, 14))
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(dates, closes, color='#1565C0', linewidth=1, alpha=0.7)
    ax1.plot(dates, ma60, color='#FF6F00', linewidth=1.5, alpha=0.6, label='MA60')
    in_bull, bull_start = False, 0
    for i in range(len(regimes)):
        if regimes[i] == 'bull' and not in_bull:
            bull_start = i; in_bull = True
        elif regimes[i] == 'bear' and in_bull:
            ax1.axvspan(dates[bull_start], dates[i-1], alpha=0.12, color='#e8f5e9')
            in_bull = False
    if in_bull:
        ax1.axvspan(dates[bull_start], dates[-1], alpha=0.12, color='#e8f5e9')
    for t in tops:
        ax1.scatter(t['date'], t['price'], color='red', s=80, marker='v', zorder=5)
        ax1.annotate('顶背离', (t['date'], t['price']), textcoords="offset points",
                     xytext=(0, -15), fontsize=8, color='red', ha='center')
    for b in bottoms:
        ax1.scatter(b['date'], b['price'], color='green', s=80, marker='^', zorder=5)
        ax1.annotate('底背离', (b['date'], b['price']), textcoords="offset points",
                     xytext=(0, 15), fontsize=8, color='green', ha='center')
    dt_idx = {d.strftime('%Y-%m-%d'): i for i, d in enumerate(dates)}
    for tr in trades:
        bi = dt_idx.get(tr['buy_date']); si = dt_idx.get(tr['sell_date'])
        if bi is not None:
            ax1.scatter(dates[bi], closes[bi], color='lime', s=150, marker='o',
                       zorder=6, edgecolors='black', linewidth=2)
        if si is not None:
            ax1.scatter(dates[si], closes[si], color='orange', s=150, marker='s',
                       zorder=6, edgecolors='black', linewidth=2)
    ax1.set_title(f'{name}({code}) — 绿底=牛市  MA60橙色', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8); ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

    ax2 = plt.subplot(3, 1, (2, 3))
    colors = ['#ef5350' if v >= 0 else '#26a69a' for v in bar]
    ax2.bar(dates, bar, color=colors, width=0.8, alpha=0.75)
    ax2.plot(dates, dif, color='#FFB300', linewidth=1.5, label='DIF')
    ax2.plot(dates, dea, color='#212121', linewidth=1.5, label='DEA')
    ax2.axhline(y=0, color='#9e9e9e', linewidth=1)
    for c in cycles:
        ax2.axvspan(datetime.strptime(c['start_date'], '%Y-%m-%d'),
                    datetime.strptime(c['end_date'], '%Y-%m-%d'),
                    alpha=0.15, color='#e8f5e9' if c['zone'] == 'above' else '#fce4ec')
    for t in tops:
        ax2.scatter(t['date'], t['dif'], color='red', s=60, marker='v', zorder=5)
    for b in bottoms:
        ax2.scatter(b['date'], b['dif'], color='green', s=60, marker='^', zorder=5)
    for tr in trades:
        bi = dt_idx.get(tr['buy_date']); si = dt_idx.get(tr['sell_date'])
        if bi is not None:
            ax2.scatter(dates[bi], dif[bi], color='lime', s=120, marker='o',
                       zorder=6, edgecolors='black')
        if si is not None:
            ax2.scatter(dates[si], dif[si], color='orange', s=120, marker='s',
                       zorder=6, edgecolors='black')
    ax2.set_title('MACD (12,26,9) — 绿底=做多区 红底=做空区', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8); ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    path = os.path.join(outdir, f'{code}_macd_cycle.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


# ═══════════════════════════
# 多因子图表
# ═══════════════════════════
def plot_multifactor(code, name, dates, closes, highs, lows, trades):
    """多因子共振图表：价格+布林带 / RSI / KDJ + 买卖点，返回 base64"""
    fig = plt.figure(figsize=(18, 14))

    # ── 子图1: 价格 + 布林带 ──
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(dates, closes, color='#1565C0', linewidth=1, alpha=0.8, label='收盘价')
    bb_upper, bb_mid, bb_lower = calc_bollinger(closes)
    ax1.plot(dates, bb_upper, color='#FF6F00', linewidth=0.8, alpha=0.5,
             linestyle='--', label='布林上轨')
    ax1.plot(dates, bb_mid, color='#FF6F00', linewidth=1, alpha=0.6, label='布林中轨')
    ax1.plot(dates, bb_lower, color='#FF6F00', linewidth=0.8, alpha=0.5,
             linestyle='--', label='布林下轨')
    dt_idx = {d.strftime('%Y-%m-%d'): i for i, d in enumerate(dates)}
    for t in trades:
        bi = dt_idx.get(t['buy_date']); si = dt_idx.get(t['sell_date'])
        if bi is not None:
            ax1.scatter(dates[bi], closes[bi], color='lime', s=80, marker='o',
                       zorder=6, edgecolors='black')
        if si is not None:
            ax1.scatter(dates[si], closes[si], color='orange', s=80, marker='s',
                       zorder=6, edgecolors='black')
    h1, _ = ax1.get_legend_handles_labels()
    h1 += [mlines.Line2D([], [], color='lime', marker='o', linestyle='', markersize=8,
                         markeredgecolor='black', label='买入'),
           mlines.Line2D([], [], color='orange', marker='s', linestyle='', markersize=8,
                         markeredgecolor='black', label='卖出')]
    ax1.legend(handles=h1, loc='upper left', fontsize=7, ncol=2)
    ax1.set_title(f'{name}({code}) — 多因子共振', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
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
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
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
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()


# ═══════════════════════════
# Web 用 MACD 图表
# ═══════════════════════════
def make_chart(code, name, dates, closes, dif, dea, bar, tops, bottoms,
               trades, regimes, ma60):
    """MACD 策略图表（Web用），返回 base64"""
    fig = plt.figure(figsize=(18, 12))

    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(dates, closes, color='#1565C0', linewidth=1, alpha=0.7, label='收盘价')
    ax1.plot(dates, ma60, color='#FF6F00', linewidth=1.5, alpha=0.6, label='MA60')

    in_bull, bs = False, 0
    bull_added = False
    for i in range(len(regimes)):
        if regimes[i] == 'bull' and not in_bull:
            bs = i; in_bull = True
        elif regimes[i] == 'bear' and in_bull:
            ax1.axvspan(dates[bs], dates[i-1], alpha=0.12, color='#e8f5e9')
            in_bull = False; bull_added = True
    if in_bull:
        ax1.axvspan(dates[bs], dates[-1], alpha=0.12, color='#e8f5e9')
        bull_added = True

    for t in tops:
        ax1.scatter(t['date'], t['price'], color='red', s=80, marker='v', zorder=5)
    for b in bottoms:
        ax1.scatter(b['date'], b['price'], color='green', s=80, marker='^', zorder=5)
    dt_idx = {d.strftime('%Y-%m-%d'): i for i, d in enumerate(dates)}
    for tr in trades:
        bi = dt_idx.get(tr['buy_date']); si = dt_idx.get(tr['sell_date'])
        if bi is not None:
            ax1.scatter(dates[bi], closes[bi], color='lime', s=120, marker='o',
                       zorder=6, edgecolors='black')
        if si is not None:
            ax1.scatter(dates[si], closes[si], color='orange', s=120, marker='s',
                       zorder=6, edgecolors='black')

    h1, l1 = ax1.get_legend_handles_labels()
    h1 += [
        mlines.Line2D([], [], color='lime', marker='o', linestyle='', markersize=8,
                      markeredgecolor='black', label='买入'),
        mlines.Line2D([], [], color='orange', marker='s', linestyle='', markersize=8,
                      markeredgecolor='black', label='卖出'),
        mlines.Line2D([], [], color='red', marker='v', linestyle='', markersize=8,
                      label='顶背离'),
        mlines.Line2D([], [], color='green', marker='^', linestyle='', markersize=8,
                      label='底背离'),
    ]
    if bull_added:
        h1.append(mpatches.Patch(color='#e8f5e9', alpha=0.5, label='牛市'))
    ax1.legend(handles=h1, loc='upper left', fontsize=8, ncol=2)
    ax1.set_title(f'{name}({code})', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

    ax2 = plt.subplot(3, 1, (2, 3))
    colors = ['#ef5350' if v >= 0 else '#26a69a' for v in bar]
    ax2.bar(dates, bar, color=colors, width=0.8, alpha=0.75, label='BAR')
    ax2.plot(dates, dif, color='#FFB300', linewidth=1.5, label='DIF金线')
    ax2.plot(dates, dea, color='#212121', linewidth=1.5, label='DEA黑线')
    ax2.axhline(y=0, color='#9e9e9e', linewidth=1, label='零轴')
    for t in tops:
        ax2.scatter(t['date'], t['dif'], color='red', s=60, marker='v', zorder=5)
    for b in bottoms:
        ax2.scatter(b['date'], b['dif'], color='green', s=60, marker='^', zorder=5)
    for tr in trades:
        bi = dt_idx.get(tr['buy_date']); si = dt_idx.get(tr['sell_date'])
        if bi is not None:
            ax2.scatter(dates[bi], dif[bi], color='lime', s=100, marker='o',
                       zorder=6, edgecolors='black')
        if si is not None:
            ax2.scatter(dates[si], dif[si], color='orange', s=100, marker='s',
                       zorder=6, edgecolors='black')

    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(handles=h2 + [
        mlines.Line2D([], [], color='lime', marker='o', linestyle='', markersize=8,
                      markeredgecolor='black', label='买入'),
        mlines.Line2D([], [], color='orange', marker='s', linestyle='', markersize=8,
                      markeredgecolor='black', label='卖出'),
    ], loc='upper left', fontsize=8, ncol=2)
    ax2.set_title('MACD (12,26,9)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()


# ═══════════════════════════
# Web 用综合策略图表
# ═══════════════════════════
def make_comprehensive_chart(code, name, dates, closes, highs, lows, vols, trades):
    """综合策略图表：价格+MACD+OBV，返回 base64"""
    fig = plt.figure(figsize=(14, 8))

    # 降采样：超过1000个点就每隔N个取一个
    step = max(1, len(dates) // 800)
    d_dates = dates[::step]
    d_closes = closes[::step]

    dif, dea, bar = calc_macd(closes)
    obv = calc_obv(closes, vols)
    d_dif = dif[::step]; d_dea = dea[::step]; d_bar = bar[::step]
    d_obv = obv[::step]

    dt_idx = {}
    for i, d in enumerate(dates):
        if i % step == 0:
            dt_idx[d.strftime('%Y-%m-%d')] = i // step

    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(d_dates, d_closes, color='#1565C0', linewidth=1, alpha=0.8, label='收盘价')
    ma20 = np.array([np.mean(closes[max(0, i-19):i+1]) for i in range(0, len(closes), step)])
    ma60 = np.array([np.mean(closes[max(0, i-59):i+1]) for i in range(0, len(closes), step)])
    ax1.plot(d_dates, ma20, color='#FF6F00', linewidth=1, alpha=0.6, label='MA20')
    ax1.plot(d_dates, ma60, color='#E91E63', linewidth=1, alpha=0.5, label='MA60')
    for t in trades:
        bi = dt_idx.get(t['buy_date']); si = dt_idx.get(t['sell_date'])
        if bi is not None:
            ax1.scatter(d_dates[bi], d_closes[bi], color='lime', s=100, marker='o',
                       zorder=6, edgecolors='black')
        if si is not None:
            ax1.scatter(d_dates[si], d_closes[si], color='orange', s=100, marker='s',
                       zorder=6, edgecolors='black')
    h1, _ = ax1.get_legend_handles_labels()
    h1 += [mlines.Line2D([], [], color='lime', marker='o', linestyle='', markersize=8,
                         markeredgecolor='black', label='买入'),
           mlines.Line2D([], [], color='orange', marker='s', linestyle='', markersize=8,
                         markeredgecolor='black', label='卖出')]
    ax1.legend(handles=h1, loc='upper left', fontsize=7, ncol=2)
    ax1.set_title(f'{name}({code}) — 三合资本综合策略', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

    ax2 = plt.subplot(3, 1, 2)
    colors_bar = ['#ef5350' if v >= 0 else '#26a69a' for v in d_bar]
    ax2.bar(d_dates, d_bar, color=colors_bar, width=0.8, alpha=0.75, label='BAR')
    ax2.plot(d_dates, d_dif, color='#FFB300', linewidth=1.5, label='DIF')
    ax2.plot(d_dates, d_dea, color='#212121', linewidth=1.5, label='DEA')
    ax2.axhline(y=0, color='#9e9e9e', linewidth=1, label='零轴')
    for t in trades:
        bi = dt_idx.get(t['buy_date']); si = dt_idx.get(t['sell_date'])
        if bi is not None:
            ax2.scatter(d_dates[bi], d_dif[bi], color='lime', s=80, marker='o',
                       zorder=6, edgecolors='black')
        if si is not None:
            ax2.scatter(d_dates[si], d_dif[si], color='orange', s=80, marker='s',
                       zorder=6, edgecolors='black')
    ax2.legend(loc='upper left', fontsize=7, ncol=2)
    ax2.set_title('MACD (12,26,9)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(d_dates, d_obv, color='#7B1FA2', linewidth=1, label='OBV')
    obv_ma20_arr = np.array([np.mean(obv[max(0, i-19):i+1]) for i in range(0, len(obv), step)])
    ax3.plot(d_dates, obv_ma20_arr, color='#FF6F00', linewidth=0.8, alpha=0.6,
             linestyle='--', label='OBV MA20')
    for t in trades:
        bi = dt_idx.get(t['buy_date']); si = dt_idx.get(t['sell_date'])
        if bi is not None:
            ax3.scatter(d_dates[bi], d_obv[bi], color='lime', s=80, marker='o',
                       zorder=6, edgecolors='black')
        if si is not None:
            ax3.scatter(d_dates[si], d_obv[si], color='orange', s=80, marker='s',
                       zorder=6, edgecolors='black')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.set_title('OBV 能量潮 (量价背离检测)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=90, bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()
