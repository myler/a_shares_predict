#!/usr/bin/env python3
"""数据层：K线抓取、分红抓取、股票名称、多源降级"""

import sys, os, json, urllib.request, time, re
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

from db import (load_klines, save_klines, load_dividends, save_dividends,
                save_stock_name, load_stock_name)


def fetch_kline(code, days=2500, max_retries=3):
    """
    多源降级拉取不复权 K 线: 优先本地DB → 新浪 → 腾讯 → 东方财富。
    分红由 enrich_trades_with_dividends 单独计入，不能与前复权价格混用。
    首次拉全量并缓存，之后从DB秒读。
    """
    cached = load_klines(code)
    if cached:
        return cached

    sc = f'sh{code}' if code.startswith('6') else f'sz{code}'
    backoff = [3, 10, 30]

    sources = [
        ('新浪', f'https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={sc}&scale=240&ma=no&datalen={days}',
         {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://finance.sina.com.cn'}),
        ('腾讯(不复权)', f'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={sc},day,,,{days}',
         {'User-Agent': 'Mozilla/5.0'}),
        ('东方财富(不复权)', f'https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={1 if code.startswith("6") else 0}.{code}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt=101&fqt=0&end=20500101&lmt={days}',
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
                    klines = data.get('data', {}).get(sc, {}).get('day', [])
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

        continue

    print('\n'.join(errors), file=sys.stderr)
    raise RuntimeError(f'所有数据源失败 ({len(errors)}次尝试)。请检查网络连接。')


def get_name(code):
    """多源获取股票名称，失败返回代码本身"""
    cached_name = load_stock_name(code)
    if cached_name:
        return cached_name

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
                save_stock_name(code, result)
                return result
        except: continue
    return code


def fetch_dividends(code):
    """从新浪获取A股分红送配数据（优先本地DB缓存）
    返回: [{date, bonus_share, transfer_share, dividend_10, dividend_per_share, status, ex_date, record_date}, ...]
    """
    cached = load_dividends(code)
    if cached:
        return cached

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


def fetch_institution_participation(code):
    """拉取千股千评中的机构参与度（0-1之间），失败返回 None"""
    import urllib.request, json, re
    try:
        secid = f'1.{code}' if code.startswith('6') else f'0.{code}'
        url = f'https://push2.eastmoney.com/api/qt/stock/get?secid={secid}&fields=f162,f167,f43,f170,f116,f117'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://quote.eastmoney.com/'})
        raw = urllib.request.urlopen(req, timeout=10).read().decode('utf-8')
        data = json.loads(raw).get('data', {})
        if data:
            return {
                'institution_participation': data.get('f162', None),  # 机构参与度
                'composite_score': data.get('f167', None),  # 综合得分
                'main_cost': data.get('f170', None),  # 主力成本
                'attention_index': data.get('f117', None),  # 关注指数
            }
    except Exception:
        pass
    return None


def estimate_institution_proxy(closes, highs, lows, volumes):
    """回测用：从K线数据估算机构主导度代理（波动率+量能稳定性）"""
    n = len(closes)
    if n < 60:
        return 50.0
    
    i = n - 1
    # 1) 近期波动率（越低越像机构主导）
    returns = [(closes[j] - closes[j-1]) / closes[j-1] for j in range(max(1,i-60), i+1)]
    volatility = np.std(returns) * 100
    # 年化波动率 <20% → 机构型, 20-40% → 均衡, >40% → 散户型
    vol_score = 50
    if volatility < 15: vol_score += 25
    elif volatility < 25: vol_score += 15
    elif volatility < 35: vol_score += 5
    elif volatility > 50: vol_score -= 20
    elif volatility > 40: vol_score -= 10
    
    # 2) 量能稳定性（量比CV越低越像机构）
    vol_ma = np.mean(volumes[max(0,i-60):i+1])
    vol_std = np.std(volumes[max(0,i-60):i+1])
    vol_cv = (vol_std / vol_ma * 100) if vol_ma > 0 else 100
    # CV < 50% → 机构型, 50-100% → 均衡, >100% → 散户型
    vol_stab = 50
    if vol_cv < 40: vol_stab += 20
    elif vol_cv < 70: vol_stab += 10
    elif vol_cv > 120: vol_stab -= 15
    elif vol_cv > 100: vol_stab -= 8
    
    # 3) 日内振幅稳定性
    amplitudes = [(highs[j] - lows[j]) / closes[j] * 100 for j in range(max(0,i-60), i+1)]
    avg_amp = np.mean(amplitudes)
    amp_score = 50
    if avg_amp < 2: amp_score += 20
    elif avg_amp < 3: amp_score += 10
    elif avg_amp > 5: amp_score -= 15
    
    proxy = vol_score * 0.4 + vol_stab * 0.35 + amp_score * 0.25
    return round(proxy, 1)


def enrich_trades_with_dividends(trades, dividends):
    """为每笔交易计算现金分红、送转股和含分红总收益。

    回测按开盘成交：除权日开盘买入不享有本次权益，除权日开盘卖出仍享有
    前一交易日登记的权益。
    """
    for tr in trades:
        dlist = sorted(
            [d for d in dividends
             if d.get('ex_date') and tr['buy_date'] < d['ex_date'] <= tr['sell_date']],
            key=lambda item: item['ex_date'],
        )
        shares = 1.0
        dividend_total = 0.0
        for dividend in dlist:
            dividend_total += shares * dividend.get('dividend_per_share', 0)
            shares *= 1 + (
                dividend.get('bonus_share', 0) + dividend.get('transfer_share', 0)
            ) / 10

        tr['dividends'] = dlist
        tr['share_multiplier'] = shares
        tr['dividend_total'] = dividend_total
        tr['dividend_yield_pct'] = dividend_total / tr['buy_price'] * 100

        commission_rate = tr.get('commission_rate', 0)
        stamp_duty_rate = tr.get('stamp_duty_rate', 0)
        buy_cost = tr['buy_price'] * (1 + commission_rate)
        sale_proceeds = tr['sell_price'] * shares * (
            1 - commission_rate - stamp_duty_rate)
        tr['total_return_pct'] = (sale_proceeds + dividend_total) / buy_cost * 100 - 100
    return trades
