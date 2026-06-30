#!/usr/bin/env python3
"""核心引擎：MACD计算、背离检测、回测、预测、画图"""

import os, sys
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
