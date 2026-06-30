#!/usr/bin/env python3
"""数据层：K线抓取、分红抓取、股票名称、多源降级"""

import sys, os, json, urllib.request, time, re

ROOT = os.path.dirname(os.path.abspath(__file__))

from db import load_klines, save_klines, load_dividends, save_dividends, save_stock_name


def fetch_kline(code, days=2500, max_retries=3):
    """
    多源降级拉取K线: 优先本地DB → 新浪 → 腾讯 → 东方财富
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

        continue

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
        tr['dividend_yield_pct'] = tr['dividend_total'] / tr['buy_price'] * 100
        tr['total_return_pct'] = tr['profit_pct'] + tr['dividend_yield_pct']
    return trades
