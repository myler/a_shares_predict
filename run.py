#!/usr/bin/env python3
"""
MACD 三维分析 + 回测引擎
═══════════════════════════════════════════════
用法:
  ./run.py chart   603893    出MACD图+回测报告
  ./run.py backtest 002371   仅回测
  ./run.py help              帮助

│  输出 → output/YYYYMMDD_CODE_NN/
├── report.txt
├── XXX_macd_cycle.png
└── predict.txt ← predict 模式额外生成
═══════════════════════════════════════════════
"""
import sys, os, json, urllib.request, time
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── 本地缓存 ──
from db import load_klines, save_klines, load_dividends, save_dividends, save_stock_name

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm

# ── 字体 ──
FONT_PATH = os.path.join(ROOT, 'wqy-zenhei.ttf')
if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    plt.rcParams['font.family'] = fm.FontProperties(fname=FONT_PATH).get_name()
    plt.rcParams['axes.unicode_minus'] = False
else:
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    # No Chinese font, just avoid crash
    plt.rcParams['axes.unicode_minus'] = False

# ═══════════════════════════
# 工具函数
# ═══════════════════════════
def make_output_dir(code):
    today = datetime.now().strftime('%Y%m%d')
    base = os.path.join(ROOT, 'output')
    n = 1
    while os.path.exists(os.path.join(base, f'{today}_{code}_{n:02d}')):
        n += 1
    d = os.path.join(base, f'{today}_{code}_{n:02d}')
    os.makedirs(d, exist_ok=True)
    return d

def fetch_kline(code, days=2500, max_retries=3):
    """
    多源降级拉取K线: 优先本地DB → 新浪 → 腾讯 → 东方财富
    首次拉全量并缓存，之后从DB秒读。
    """
    # 1. 先查本地缓存
    cached = load_klines(code)
    if cached:
        return cached

    sc = f'sh{code}' if code.startswith('6') else f'sz{code}'
    backoff = [3, 10, 30]
    
    sources = [
        ('新浪', f'https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={sc}&scale=240&ma=no&datalen={days}',
         {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://finance.sina.com.cn'}),
        ('腾讯(前复权)', f'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={sc},day,,,{days},qfq',
         {'User-Agent': 'Mozilla/5.0'}),
        ('东方财富(前复权)', f'https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={1 if code.startswith("6") else 0}.{code}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt=101&fqt=1&end=20500101&lmt={days}',
         {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://quote.eastmoney.com/'}),
    ]
    
    errors = []
    for name, url, headers in sources:
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(url, headers=headers)
                raw = urllib.request.urlopen(req, timeout=15).read().decode('utf-8')
                
                if name.startswith('腾讯'):
                    data = json.loads(raw)
                    klines = data.get('data', {}).get(sc, {}).get('qfqday', []) or data.get('data', {}).get(sc, {}).get('day', [])
                    if not klines: raise ValueError('腾讯: 空数据')
                    result = []
                    for r in klines:
                        if isinstance(r, list):
                            result.append({'day': r[0], 'open': str(r[1]), 'high': str(r[2]), 'low': str(r[3]), 'close': str(r[4]), 'volume': str(r[5])})
                        elif isinstance(r, dict):
                            result.append({'day': r.get('date',''), 'open': str(r.get('open','')), 'high': str(r.get('high','')), 'low': str(r.get('low','')), 'close': str(r.get('close','')), 'volume': str(r.get('volume',''))})
                    if result:
                        save_klines(code, result)
                        return result
                    raise ValueError('腾讯: 解析失败')
                
                elif name.startswith('东方财富'):
                    data = json.loads(raw)
                    klines = data.get('data', {}).get('klines', [])
                    if not klines: raise ValueError('东方财富: 空数据')
                    result = []
                    for r in klines:
                        parts = r.split(',')
                        result.append({'day': parts[0], 'open': parts[1], 'high': parts[3], 'low': parts[4], 'close': parts[2], 'volume': parts[5]})
                    if result:
                        save_klines(code, result)
                        return result
                    raise ValueError('东方财富: 解析失败')
                
                else:  # 新浪
                    data = json.loads(raw)
                    if not data: raise ValueError('新浪: 空数据')
                    save_klines(code, data)
                    return data
                    
            except Exception as e:
                err = f'{name}(尝试{attempt+1}/{max_retries}): {e}'
                errors.append(err)
                if attempt < max_retries - 1:
                    wait = backoff[min(attempt, len(backoff)-1)]
                    print(f'  ⚠ {err}，{wait}秒后重试...', file=sys.stderr)
                    time.sleep(wait)
                else:
                    print(f'  ✗ {err} (已达上限)', file=sys.stderr)
        
        # 当前源所有重试都失败，尝试下一个源
        continue
    
    # 所有源都失败
    print('\n'.join(errors), file=sys.stderr)
    raise RuntimeError(f'所有数据源失败 ({len(errors)}次尝试)。请检查网络连接。')

def get_name(code):
    """多源获取股票名称，失败返回代码本身"""
    sc = f'sh{code}' if code.startswith('6') else f'sz{code}'
    sources = [
        ('腾讯', f'http://qt.gtimg.cn/q={sc}', 'gbk', lambda raw: raw.split('~')[1]),
        ('新浪', f'https://hq.sinajs.cn/list={sc}', 'gbk', lambda raw: raw.split('"')[1].split(',')[0] if '"' in raw else ''),
        ('东方财富', f'https://push2.eastmoney.com/api/qt/stock/get?secid={1 if code.startswith("6") else 0}.{code}&fields=f57,f58',
         'utf-8', lambda raw: json.loads(raw).get('data',{}).get('f58','') if 'data' in raw else ''),
    ]
    for name, url, enc, parser in sources:
        try:
            req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
            raw = urllib.request.urlopen(req, timeout=5).read().decode(enc)
            result = parser(raw)
            if result and result != code:
                return result
        except: continue
    return code

# ═══════════════════════════
# 分红数据
# ═══════════════════════════
def fetch_dividends(code):
    """从新浪获取A股分红送配数据（优先本地DB缓存）
    返回: [{date, bonus_share, transfer_share, dividend_10, dividend_per_share, status, ex_date, record_date}, ...]
    """
    # 1. 先查本地缓存
    cached = load_dividends(code)
    if cached:
        return cached

    import re
    url = f'https://vip.stock.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/stockid/{code}.phtml'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://finance.sina.com.cn'})
    try:
        raw = urllib.request.urlopen(req, timeout=15).read()
        html = raw.decode('gbk', errors='replace')
    except Exception as e:
        print(f'  ⚠ 分红数据获取失败: {e}', file=sys.stderr)
        return []

    tables = re.findall(r'(<table[^>]*>.*?</table>)', html, re.S)
    target = None
    for t in tables:
        if '派息' in t or ('除权除息日' in t and '送股' in t):
            target = t
            break
    if not target:
        return []

    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', target, re.S)
    results = []
    for row in rows:
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.S)
        clean = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
        clean = [re.sub(r'&nbsp;', '', c).strip() for c in clean]

        if not clean or clean[0] in ('公告日期', '') or '暂时没有数据' in str(clean):
            continue
        if not re.match(r'\d{4}-\d{2}-\d{2}', clean[0]):
            continue

        # 列: 公告日期, 送股, 转增, 派息(每10股), 进度, 除权除息日, 股权登记日, 红股上市日, 查看详细
        if len(clean) >= 9:
            try:
                dividend_10 = float(clean[3]) if clean[3] and clean[3] != '--' else 0
                bonus_share = float(clean[1]) if clean[1] and clean[1] != '--' else 0
                transfer_share = float(clean[2]) if clean[2] and clean[2] != '--' else 0

                entry = {
                    'date': clean[0],
                    'bonus_share': bonus_share,
                    'transfer_share': transfer_share,
                    'dividend_10': dividend_10,
                    'dividend_per_share': dividend_10 / 10,
                    'status': clean[4] if len(clean) > 4 else '',
                    'ex_date': clean[5] if len(clean) > 5 and clean[5] != '--' else '',
                    'record_date': clean[6] if len(clean) > 6 and clean[6] != '--' else '',
                }
                results.append(entry)
            except (ValueError, IndexError):
                continue

    implemented = [r for r in results if r['status'] == '实施']
    save_dividends(code, implemented)
    return implemented


def enrich_trades_with_dividends(trades, dividends):
    """为每笔交易计算分红收益，并累加到总收益中。
    价差收益 + 分红收益 = 总收益
    """
    for tr in trades:
        dlist = []
        for d in dividends:
            if d['ex_date'] and tr['buy_date'] <= d['ex_date'] < tr['sell_date']:
                dlist.append(d)
        tr['dividends'] = dlist
        tr['dividend_total'] = sum(d['dividend_per_share'] for d in dlist)
        # 分红收益率 = 每股累计分红 / 买入价
        tr['dividend_yield_pct'] = tr['dividend_total'] / tr['buy_price'] * 100
        # 总收益 = 价差收益 + 分红收益
        tr['total_return_pct'] = tr['profit_pct'] + tr['dividend_yield_pct']
    return trades


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
    """基于当前MACD状态，预测近期买卖信号触发条件
    holding=False: 未持仓 → 买入/不买
    holding=True:  已持仓 → 拿住/增持/减持/卖出
    """
    i = len(dif)-1
    # 找最后一个有效值
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
    
    # 1. 当前状态
    lines.append(f"\n── 当前状态 ──")
    lines.append(f"  收盘: {cur['close']:.2f}  |  {'🟢牛市' if cur['regime']=='bull' else '🔴熊市'}")
    lines.append(f"  DIF: {cur['dif']:.2f}  |  DEA: {cur['dea']:.2f}  |  BAR: {cur['bar']:.2f}")
    
    # 金叉/死叉
    if cur['dif'] > cur['dea']:
        lines.append(f"  信号: 🟢 金叉 (DIF > DEA)")
        # 距死叉还有多远
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
    
    # 2. 零轴位置
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
    
    # 3. 背离检查
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
    
    # 4. 综合判断
    lines.append(f"\n── 综合预测 ──")
    
    score = 0
    reasons = []
    
    # Bull factors
    if cur['dif'] > cur['dea']: score += 2; reasons.append('金叉')
    else: score -= 2; reasons.append('死叉')
    if cur['dif'] > 0: score += 1; reasons.append('零轴上')
    else: score -= 1; reasons.append('零轴下')
    if cur['regime'] == 'bull': score += 2; reasons.append('牛市')
    else: score -= 2; reasons.append('熊市')
    if cur['dif_slope_5d'] is not None:
        if cur['dif_slope_5d'] > 0: score += 1; reasons.append('DIF↑')
        else: score -= 1; reasons.append('DIF↓')
    
    # Recent divergence
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
    
    # 5. 近期关注点
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
    
    # 6. 结论 (放最后)
    lines.append(f"\n{'='*50}")
    n = max(1, min(10, abs(score)))
    if holding:
        if action == '卖出':
            lines.append(f"  {'❌' * n}  {action}")
        elif action in ('增持','拿住'):
            lines.append(f"  {'✅' * n}  {action}")
        else:
            lines.append(f"  {'⚠️' * n}  {action}")
    else:
        if action == '买入':
            lines.append(f"  {'✅' * n}  {action}")
        else:
            lines.append(f"  {'❌' * n}  {action}")
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
# 画图
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

# ═══════════════════════════
# HELP
# ═══════════════════════════
HELP = """
MACD 三维分析 + 背离回测工具
══════════════════════════

用法:
  ./run.py <股票代码>           出图+回测+预测 (默认)
  ./run.py chart   <股票代码>   仅出图
  ./run.py backtest <股票代码>  仅回测
  ./run.py predict <股票代码>   仅预测
  ./run.py help                 帮助

示例:
  ./run.py 603893            瑞芯微 (全部)
  ./run.py predict 000651    格力电器 (仅预测)
  ./run.py backtest 002371   北方华创 (仅回测)

输出:
  output/YYYYMMDD_CODE_NN/
    ├── report.txt             回测报告
    └── XXXX_macd_cycle.png     MACD图表

公式规则:
  买入: 底背离(熊市抄底) | 零轴上金叉(牛市追涨)
  卖出: 顶背离(牛市见顶) | 死叉(熊市止损)
  牛熊: 价格>MA60 且 MA60斜率↑ → 牛市
"""

# ═══════════════════════════
# MAIN
# ═══════════════════════════
if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] in ('help','-h','--help'):
        print(HELP)
        sys.exit(0)
    
    # 解析: ./run.py <代码>  → 默认全跑
    #       ./run.py chart/backtest/predict <代码>
    if sys.argv[1] in ('chart','backtest','predict','json'):
        mode, codes = sys.argv[1], sys.argv[2:]
    else:
        mode, codes = 'chart', sys.argv[1:]
    if not codes:
        print("请指定股票代码")
        sys.exit(1)
    
    if mode == 'json':
        # ── JSON批量输出：盯盘脚本集成用 ──
        import json as _json
        results = []
        for code in codes:
            try:
                data = fetch_kline(code)
                name = get_name(code)
                save_stock_name(code, name)
                dates = np.array([datetime.strptime(d['day'],'%Y-%m-%d') for d in data])
                closes = np.array([float(d['close']) for d in data])
                date_strs = [d['day'] for d in data]
                dif, dea, bar = calc_macd(closes)
                regimes, ma60 = detect_regime(closes)
                tops, bottoms = find_divergences(date_strs, closes, dif)
                
                i = len(dif)-1
                while i >= 0 and np.isnan(dif[i]): i -= 1
                
                # 最近背离
                recent_tops = [t for t in tops if (datetime.strptime(date_strs[-1],'%Y-%m-%d') - t['date']).days <= 120]
                recent_bottoms = [b for b in bottoms if (datetime.strptime(date_strs[-1],'%Y-%m-%d') - b['date']).days <= 120]
                
                dif_slope_5d = float(dif[i] - dif[max(0,i-5)]) if i>=5 else 0
                dif_slope_10d = float(dif[i] - dif[max(0,i-10)]) if i>=10 else 0
                
                golden = dif[i] > dea[i]
                if golden:
                    cross_dist = dif[i] - dea[i]
                    cross_days = int(cross_dist / abs(dif_slope_5d/5)) if dif_slope_5d < 0 and dif_slope_5d != 0 else -1
                else:
                    cross_dist = dea[i] - dif[i]
                    cross_days = int(cross_dist / (dif_slope_5d/5)) if dif_slope_5d > 0 else -1
                
                # 评分
                score = 0
                if regimes[i] == 'bull': score += 2
                else: score -= 2
                if golden: score += 2
                else: score -= 2
                if dif[i] > 0: score += 1
                else: score -= 1
                if recent_bottoms: score += 2
                if recent_tops: score -= 2
                
                advice = "买入" if score >= 3 else ("持有" if score >= 0 else "卖出")
                
                results.append({
                    "code": code, "name": name,
                    "date": date_strs[i], "close": float(closes[i]),
                    "dif": round(float(dif[i]),4), "dea": round(float(dea[i]),4),
                    "bar": round(float(bar[i]),4),
                    "golden": bool(golden),
                    "regime": "bull" if regimes[i]=='bull' else "bear",
                    "zero_above": bool(dif[i] > 0),
                    "dif_slope_5d": round(dif_slope_5d,4),
                    "recent_top_div": len(recent_tops),
                    "recent_bot_div": len(recent_bottoms),
                    "cross_days": cross_days,
                    "score": score,
                    "advice": advice,
                })
            except Exception as e:
                results.append({"code": code, "error": str(e)})
        print(_json.dumps(results, ensure_ascii=False))
        sys.exit(0)
    
    code = codes[0]  # 非json模式取第一个  # 默认全跑
    outdir = make_output_dir(code)
    print(f"输出: {outdir}\n")
    
    data = fetch_kline(code)  # 拉满~10年，新股自动返回实际数据
    name = get_name(code)
    save_stock_name(code, name)
    dates = np.array([datetime.strptime(d['day'],'%Y-%m-%d') for d in data])
    closes = np.array([float(d['close']) for d in data])
    date_strs = [d['day'] for d in data]
    
    dif, dea, bar = calc_macd(closes)
    regimes, ma60 = detect_regime(closes)
    tops, bottoms = find_divergences(date_strs, closes, dif)
    cycles = zero_line_cycles(date_strs, dif)
    trades = backtest(date_strs, closes, dif, dea, tops, bottoms, regimes)
    
    # ── 分红数据 ──
    dividends = fetch_dividends(code)
    if dividends:
        trades = enrich_trades_with_dividends(trades, dividends)
    
    lines = []
    lines.append("="*60)
    lines.append(f"  {name}({code}) MACD回测")
    lines.append("="*60)
    lines.append(f"数据: {data[0]['day']} ~ {data[-1]['day']} ({len(data)}K线)")
    lines.append(f"牛熊: {'🟢牛' if regimes[-1]=='bull' else '🔴熊'} | 背离: 顶{len(tops)} 底{len(bottoms)}")
    lines.append(f"\n── 交易 ({len(trades)}笔) ──")
    for t in trades:
        div_info = ""
        if t.get('dividend_total', 0) > 0:
            div_info = f" [分红{t['dividend_total']:.2f}元/股]"
        lines.append(f"  {'✅' if t['profit_pct']>0 else '❌'} {t['buy_date']}→{t['sell_date']} "
                    f"{t['buy_price']:.2f}→{t['sell_price']:.2f} 价差{t['profit_pct']:+.1f}%{div_info} [{t['hold_days']}天]")
    if trades:
        wins=[t for t in trades if t['profit_pct']>0]; total=sum(t['profit_pct'] for t in trades)
        total_div_pct = sum(t.get('dividend_yield_pct', 0) for t in trades)
        total_div_cash = sum(t.get('dividend_total', 0) for t in trades)
        total_return = total + total_div_pct
        lines.append(f"\n  胜率: {len(wins)}/{len(trades)}={len(wins)/len(trades)*100:.0f}%")
        lines.append(f"  价差收益: {total:+.1f}%")
        lines.append(f"  分红收益: {total_div_cash:.2f}元/股（折合{total_div_pct:+.1f}%）")
        lines.append(f"  总收益:   {total_return:+.1f}%")
    
    # ── 分红历史 ──
    if dividends:
        recent_divs = [d for d in dividends if d.get('ex_date', '') >= data[0]['day']]
        if recent_divs:
            lines.append(f"\n── 分红历史 ({len(recent_divs)}次) ──")
            for d in recent_divs:
                bonus_info = ""
                if d['bonus_share'] > 0: bonus_info += f" 送{d['bonus_share']:.0f}股"
                if d['transfer_share'] > 0: bonus_info += f" 转{d['transfer_share']:.0f}股"
                lines.append(f"  {d['ex_date']} 除权 | 10派{d['dividend_10']:.1f}元 | 每股{d['dividend_per_share']:.3f}元{bonus_info}")
    else:
        lines.append("  无交易")
    
    report = '\n'.join(lines)
    print(report)
    with open(os.path.join(outdir,'report.txt'),'w') as f: f.write(report)
    
    if mode == 'chart':
        path = plot_all(code, name, dates, closes, dif, dea, bar, cycles, tops, bottoms, trades, regimes, ma60, outdir)
        print(f"\n图表: {path}")
    
    if mode in ('chart', 'predict'):
        pred = predict(date_strs, closes, dif, dea, bar, regimes, tops, bottoms)
        pred_text = '\n'.join(pred)
        print(f"\n{pred_text}")
        with open(os.path.join(outdir,'predict.txt'),'w') as f: f.write(pred_text)
    
    print(f"\n完成 → {outdir}")
