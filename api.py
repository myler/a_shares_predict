#!/usr/bin/env python3
"""JSON API层 — 供微信小程序 / 外部分析调用，纯 JSON 输入输出"""

import sys
from datetime import datetime
import numpy as np

from fetcher import (fetch_kline, get_name, fetch_dividends,
                     enrich_trades_with_dividends, fetch_financial_summaries)
from db import save_stock_name
from fundamentals import screen_value_quality
from engine import (calc_macd, detect_regime, find_divergences,
                    backtest, predict, backtest_multifactor, predict_multifactor,
                    backtest_comprehensive, predict_comprehensive)

# ── 简易 JSON API（无需 Flask，用标准库 http.server）──
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs


def analyze_macd(code, holding=False):
    """MACD 策略分析 → JSON"""
    data = fetch_kline(code)
    name = get_name(code)
    save_stock_name(code, name)

    dates = np.array([datetime.strptime(d['day'], '%Y-%m-%d') for d in data])
    closes = np.array([float(d['close']) for d in data])
    date_strs = [d['day'] for d in data]

    dif, dea, bar = calc_macd(closes)
    regimes, ma60 = detect_regime(closes)
    tops, bottoms = find_divergences(date_strs, closes, dif)
    trades = backtest(date_strs, closes, dif, dea, tops, bottoms, regimes)
    pred = predict(date_strs, closes, dif, dea, bar, regimes, tops, bottoms,
                   holding=holding)

    return {
        'code': code, 'name': name,
        'strategy': 'macd',
        'data_range': {'start': data[0]['day'], 'end': data[-1]['day'],
                       'count': len(data)},
        'prediction': pred,
        'backtest': {
            'trades': len(trades),
            'wins': len([t for t in trades if t['profit_pct'] > 0]),
            'total_return_pct': round(sum(t['profit_pct'] for t in trades), 2) if trades else 0,
            'trades_detail': trades,
        },
        'regime': 'bull' if regimes[-1] == 'bull' else 'bear',
    }


def analyze_multifactor(code, holding=False):
    """多因子共振策略分析 → JSON"""
    data = fetch_kline(code)
    name = get_name(code)
    save_stock_name(code, name)

    dates = [d['day'] for d in data]
    closes = np.array([float(d['close']) for d in data])
    highs = np.array([float(d['high']) for d in data])
    lows = np.array([float(d['low']) for d in data])
    vols = np.array([float(d['volume']) for d in data])

    trades = backtest_multifactor(dates, closes, highs, lows, vols)
    pred = predict_multifactor(dates, closes, highs, lows, vols, holding=holding)

    return {
        'code': code, 'name': name,
        'strategy': 'multifactor',
        'data_range': {'start': dates[0], 'end': dates[-1], 'count': len(data)},
        'prediction': pred,
        'backtest': {
            'trades': len(trades),
            'wins': len([t for t in trades if t['profit_pct'] > 0]),
            'total_return_pct': round(sum(t['profit_pct'] for t in trades), 2) if trades else 0,
            'trades_detail': trades,
        },
    }


def analyze_comprehensive(code, holding=False, calc_dividend=False):
    """融合策略分析 → JSON"""
    data = fetch_kline(code)
    name = get_name(code)
    save_stock_name(code, name)

    dates = [d['day'] for d in data]
    closes = np.array([float(d['close']) for d in data])
    highs = np.array([float(d['high']) for d in data])
    lows = np.array([float(d['low']) for d in data])
    vols = np.array([float(d['volume']) for d in data])

    trades = backtest_comprehensive(dates, closes, highs, lows, vols)
    pred = predict_comprehensive(dates, closes, highs, lows, vols, holding=holding)

    dividends = []
    if calc_dividend:
        dividends_data = fetch_dividends(code)
        if dividends_data:
            trades = enrich_trades_with_dividends(trades, dividends_data)
            dividends = [{
                'ex_date': d['ex_date'],
                'dividend_per_share': d['dividend_per_share'],
                'dividend_10': d['dividend_10'],
            } for d in dividends_data if d.get('ex_date', '') >= dates[0]]

    result = {
        'code': code, 'name': name,
        'strategy': 'comprehensive',
        'data_range': {'start': dates[0], 'end': dates[-1], 'count': len(data)},
        'prediction': pred,
        'backtest': {
            'trades': len(trades),
            'wins': len([t for t in trades if t['profit_pct'] > 0]),
            'total_return_pct': round(sum(t['profit_pct'] for t in trades), 2) if trades else 0,
            'trades_detail': trades,
        },
    }

    if dividends:
        result['dividends'] = dividends

    return result


def analyze_buyhold(code, calc_dividend=False):
    """长线持有分析 → JSON"""
    data = fetch_kline(code)
    name = get_name(code)
    save_stock_name(code, name)

    first_date = data[0]['day']
    first_close = float(data[0]['close'])
    last_date = data[-1]['day']
    last_close = float(data[-1]['close'])
    closes = np.array([float(d['close']) for d in data])

    d0 = datetime.strptime(first_date, '%Y-%m-%d').date()
    d1 = datetime.strptime(last_date, '%Y-%m-%d').date()
    years = max((d1 - d0).days / 365.25, 0.01)

    price_return = (last_close - first_close) / first_close * 100

    total_div = 0
    dividends = []
    if calc_dividend:
        dividends_data = fetch_dividends(code)
        total_div = sum(d['dividend_per_share'] for d in dividends_data
                        if d.get('ex_date', '') >= first_date)
        dividends = [{
            'ex_date': d['ex_date'],
            'dividend_per_share': d['dividend_per_share'],
            'dividend_10': d['dividend_10'],
        } for d in dividends_data if d.get('ex_date', '') >= first_date]

    div_yield = total_div / first_close * 100
    total_return = price_return + div_yield

    return {
        'code': code, 'name': name,
        'strategy': 'buyhold',
        'buy_date': first_date,
        'buy_price': round(first_close, 2),
        'current_date': last_date,
        'current_price': round(last_close, 2),
        'hold_years': round(years, 1),
        'price_return_pct': round(price_return, 1),
        'dividend_yield_pct': round(div_yield, 1),
        'dividend_total_per_share': round(total_div, 2),
        'total_return_pct': round(total_return, 1),
        'dividends': dividends,
    }


def analyze_value(code):
    """巴芒财务质量初筛 → JSON；不返回交易建议或 DCF 目标价。"""
    rows = fetch_financial_summaries(code)
    research = screen_value_quality(rows)
    name = research['company'] or get_name(code)
    if name and name != code:
        save_stock_name(code, name)
    return {
        'code': code,
        'name': name,
        'strategy': 'value',
        'research': research,
    }


# ═══════════════════════════
# HTTP API Handler
# ═══════════════════════════
class APIHandler(BaseHTTPRequestHandler):
    """简易 JSON API，无需 Flask"""

    def _json_response(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, default=str).encode())

    def _error(self, msg, status=400):
        self._json_response({'error': msg}, status)

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        path = parsed.path.rstrip('/')

        if path == '/':
            self._json_response({
                'service': 'A股策略分析 API',
                'version': '2.0',
                'endpoints': {
                    '/api/analyze?code=XXXXXX': '融合策略分析 (默认)',
                    '/api/analyze?code=XXXXXX&strategy=macd': 'MACD策略分析',
                    '/api/analyze?code=XXXXXX&strategy=multi': '多因子共振分析',
                    '/api/analyze?code=XXXXXX&strategy=buyhold': '长线持有分析',
                    '/api/analyze?code=XXXXXX&strategy=value': '巴芒财务质量初筛（非交易信号）',
                    '/api/analyze?code=XXXXXX&holding=1': '已持仓模式',
                    '/api/analyze?code=XXXXXX&dividend=1': '含分红计算',
                },
            })
            return

        if path == '/api/analyze':
            code = params.get('code', [''])[0].strip()
            if not code or not code.isdigit() or len(code) != 6:
                return self._error('请提供6位股票代码')

            strategy = params.get('strategy', ['comprehensive'])[0]
            holding = params.get('holding', ['0'])[0] == '1'
            calc_dividend = params.get('dividend', ['0'])[0] == '1'

            try:
                if strategy == 'macd':
                    result = analyze_macd(code, holding)
                elif strategy == 'multi':
                    result = analyze_multifactor(code, holding)
                elif strategy == 'buyhold':
                    result = analyze_buyhold(code, calc_dividend)
                elif strategy == 'value':
                    result = analyze_value(code)
                else:
                    result = analyze_comprehensive(code, holding, calc_dividend)
                self._json_response(result)
            except Exception as e:
                self._error(f'分析失败: {e}', 500)
            return

        if path == '/api/health':
            self._json_response({'status': 'ok', 'time': datetime.now().isoformat()})
            return

        self._error('Not Found', 404)


if __name__ == '__main__':
    PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8100
    print(f"A股策略 JSON API → http://localhost:{PORT}")
    print(f"示例: http://localhost:{PORT}/api/analyze?code=000651")
    print("Ctrl+C 停止\n")
    server = HTTPServer(('0.0.0.0', PORT), APIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
        server.server_close()
