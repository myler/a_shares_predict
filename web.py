#!/usr/bin/env python3
"""Web界面 — 轻量HTTP服务"""

import sys, os, json, base64, io, urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timedelta

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── 导入数据层和引擎 ──
from fetcher import (fetch_kline, get_name, fetch_dividends,
                     enrich_trades_with_dividends, fetch_financial_summaries)
from db import save_stock_name
from fundamentals import screen_value_quality
from engine import (calc_macd, detect_regime, find_divergences, backtest,
                    backtest_multifactor, backtest_comprehensive,
                    predict, format_predict,
                    predict_multifactor, format_predict_multifactor,
                    predict_comprehensive, format_predict_comprehensive,
                    calc_bollinger, calc_obv, calc_rsi, calc_kdj, calc_wr,
                    summarize_trades)
from plotting import make_chart, make_comprehensive_chart, plot_multifactor
from scan_composite import run_scan

import numpy as np

# ── 加载HTML模板 ──
TEMPLATE_PATH = os.path.join(ROOT, 'templates', 'page.html')
with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
    PAGE = f.read()


# ═══════════════════════════
# Handler
# ═══════════════════════════
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200); self.send_header('Content-type','text/html; charset=utf-8'); self.end_headers()
            self.wfile.write(PAGE.encode())
            return

        if self.path.startswith('/analyze'):
            qs = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(qs)
            code = params.get('code', [''])[0].strip()
            holding = params.get('holding', ['0'])[0] == '1'
            calc_dividend = params.get('dividend', ['0'])[0] == '1'
            strategy = params.get('strategy', ['macd'])[0]

            if not code or not code.isdigit() or len(code) != 6:
                self.send_response(400); self.send_header('Content-type','text/html; charset=utf-8'); self.end_headers()
                self.wfile.write('请输入6位股票代码'.encode()); return

            try:
                if strategy == 'buyhold':
                    html = self.run_buyhold_analysis(code, calc_dividend)
                elif strategy == 'value':
                    html = self.run_value_analysis(code)
                elif strategy == 'multi':
                    html = self.run_multifactor_analysis(code, holding, calc_dividend)
                elif strategy == 'comprehensive':
                    html = self.run_comprehensive_analysis(code, holding, calc_dividend)
                else:
                    html = self.run_analysis(code, holding, calc_dividend)
                self.send_response(200); self.send_header('Content-type','text/html; charset=utf-8'); self.end_headers()
                self.wfile.write(html.encode())
            except Exception as e:
                self.send_response(500); self.send_header('Content-type','text/html; charset=utf-8'); self.end_headers()
                self.wfile.write(f'<div class=error>分析失败: {e}</div>'.encode())
            return

        if self.path.startswith('/scan'):
            qs = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(qs)
            try:
                n = int(params.get('n', ['20'])[0])
                max_pe = float(params.get('max_pe', ['100'])[0])
                min_price = float(params.get('min_price', ['5'])[0])
            except ValueError:
                n, max_pe, min_price = 20, 100.0, 5.0
            try:
                html = self.run_scan_html(n, max_pe, min_price)
                self.send_response(200); self.send_header('Content-type','text/html; charset=utf-8'); self.end_headers()
                self.wfile.write(html.encode())
            except Exception as e:
                self.send_response(500); self.send_header('Content-type','text/html; charset=utf-8'); self.end_headers()
                self.wfile.write(f'<div style="color:#c62828;padding:16px">海选失败: {e}</div>'.encode())
            return

        self.send_response(404); self.end_headers()

    def run_scan_html(self, n, max_pe, min_price):
        top, stats = run_scan(n, max_pe, min_price)
        rows_html = ''
        for i, r in enumerate(top, 1):
            pct = r.get('pct', 0) or 0
            pct_color = '#2e7d32' if pct >= 0 else '#c62828'
            rows_html += (
                f'<tr><td>{i}</td>'
                f'<td>{r["name"]} <span style="color:#999">({r["code"]})</span></td>'
                f'<td><b>{r["score"]:.1f}</b></td>'
                f'<td>{r["close"]:.2f}</td>'
                f'<td style="color:{pct_color}">{pct:+.2f}%</td>'
                f'<td>{r["pe"]:.0f}</td>'
                f'<td>{r["mcap"]:.0f}亿</td></tr>')
        latest = top[0]['date'] if top else '—'
        return f"""<table class="scan-table">
<tr><th>#</th><th>股票</th><th>得分</th><th>现价</th><th>涨跌</th><th>PE</th><th>市值</th></tr>
{rows_html}
</table>
<div class="scan-meta">共 {stats['passed']} 只通过过滤（{stats['total_codes']} 只有效因子）| 耗时 {stats['elapsed']:.0f}秒 | 数据日期 {latest}</div>"""

    def run_analysis(self, code, holding, calc_dividend=False):
        data = fetch_kline(code)
        name = get_name(code)
        save_stock_name(code, name)
        dates = np.array([datetime.strptime(d['day'],'%Y-%m-%d') for d in data])
        closes = np.array([float(d['close']) for d in data])
        date_strs = [d['day'] for d in data]

        dif, dea, bar = calc_macd(closes)
        regimes, ma60 = detect_regime(closes)
        tops, bottoms = find_divergences(date_strs, closes, dif)
        trades = backtest(date_strs, closes, dif, dea, tops, bottoms, regimes)
        pred_data = predict(date_strs, closes, dif, dea, bar, regimes, tops, bottoms, holding)
        pred_lines = format_predict(pred_data, holding)

        dividends = []
        if calc_dividend:
            dividends = fetch_dividends(code)
            if dividends:
                trades = enrich_trades_with_dividends(trades, dividends)

        conclusion_html = ''
        action = pred_data.get('action', '')
        score = pred_data.get('score', 0)
        if action in ('买入', '增持', '拿住'):
            conclusion_html = f'<div class="conclusion buy">{"✅" * min(10, max(1, abs(score)))} {action}</div>'
        elif action == '卖出':
            conclusion_html = f'<div class="conclusion sell">{"❌" * min(10, max(1, abs(score)))} {action}</div>'
        else:
            conclusion_html = f'<div class="conclusion warn">{"⚠️" * min(10, max(1, abs(score)))} {action}</div>'

        wins = [t for t in trades if t['profit_pct'] > 0]
        total_pnl = sum(t['profit_pct'] for t in trades) if trades else 0
        regime_label = '🟢 牛市' if regimes[-1] == 'bull' else '🔴 熊市'

        d0 = datetime.strptime(data[0]['day'], '%Y-%m-%d').date()
        d1 = datetime.strptime(data[-1]['day'], '%Y-%m-%d').date()
        years = max((d1 - d0).days / 365.25, 0.01)

        overview = f"""
        <table class="overview">
          <tr><th colspan="2">{name} ({code})</th></tr>
          <tr><td>数据范围</td><td>{data[0]['day']} ~ {data[-1]['day']}（{years:.1f}年）</td></tr>
          <tr><td>K线数量</td><td>{len(data)} 根</td></tr>
          <tr><td>当前牛熊</td><td>{regime_label}</td></tr>
          <tr><td>背离信号</td><td>顶 {len(tops)} 次 · 底 {len(bottoms)} 次</td></tr>
          <tr><td>回测交易</td><td>{len(trades)} 笔 · 胜率 {len(wins)/len(trades)*100:.0f}%</td></tr>"""
        if calc_dividend and dividends:
            total_div_pct = sum(t.get('dividend_yield_pct', 0) for t in trades)
            total_div_cash = sum(t.get('dividend_total', 0) for t in trades)
            total_return = total_pnl + total_div_pct
            div_share = total_div_pct / total_return * 100 if total_return > 0 else 0
            one_year_ago = d1 - timedelta(days=365)
            recent_div = sum(d['dividend_per_share'] for d in dividends if d.get('ex_date','') >= one_year_ago.isoformat())
            latest_div_yield = recent_div / closes[-1] * 100
            overview += f"""
          <tr style="background:#e8f5e9"><td>价差收益</td><td style="font-weight:bold">{total_pnl:+.1f}%</td></tr>
          <tr style="background:#e8f5e9"><td>分红收益</td><td style="font-weight:bold">{total_div_cash:.2f}元/股（{total_div_pct:+.1f}%）</td></tr>
          <tr style="background:#c8e6c9"><td>总收益</td><td style="font-weight:bold;font-size:1.1em">{total_return:+.1f}%</td></tr>
          <tr><td>分红占比</td><td>{div_share:.0f}%（分红贡献了总收益的{div_share:.0f}%）</td></tr>
          <tr><td>近12月股息率</td><td>{latest_div_yield:.1f}%（近12月分红{recent_div:.2f}÷现价{closes[-1]:.2f}）</td></tr>"""
        overview += """
        </table>"""

        pred_table = self.make_pred_table(pred_data)
        top_section = f'<div class="top-row"><div>{overview}</div><div>{pred_table}</div></div>'

        trade_rows = ''
        show_div_column = calc_dividend and dividends and any(t.get('dividend_total', 0) > 0 for t in trades)
        for t in trades:
            tr_class = 'win' if t['profit_pct'] > 0 else 'loss'
            if show_div_column:
                div_total = t.get('dividend_total', 0)
                if div_total > 0:
                    div_cell = f'<td>{div_total:.2f}元/股</td><td class="pnl">{t["total_return_pct"]:+.1f}%</td>'
                else:
                    div_cell = f'<td>-</td><td class="pnl">{t["profit_pct"]:+.1f}%</td>'
                trade_rows += f'<tr class="{tr_class}"><td>{t["buy_date"]}</td><td>{t["sell_date"]}</td><td>{t["buy_price"]:.2f}</td><td>{t["sell_price"]:.2f}</td><td class="pnl">{t["profit_pct"]:+.1f}%</td>{div_cell}<td>{t["hold_days"]}天</td><td class="reason">{t["buy_reason"]}→{t["sell_reason"]}</td></tr>'
            else:
                trade_rows += f'<tr class="{tr_class}"><td>{t["buy_date"]}</td><td>{t["sell_date"]}</td><td>{t["buy_price"]:.2f}</td><td>{t["sell_price"]:.2f}</td><td class="pnl">{t["profit_pct"]:+.1f}%</td><td>{t["hold_days"]}天</td><td class="reason">{t["buy_reason"]}→{t["sell_reason"]}</td></tr>'

        div_header = '<th>分红</th><th>总收益</th>' if show_div_column else ''
        trade_table = f"""
        <h3>📈 量化回测触发日</h3>
        <table class="trades">
          <tr><th>买入日</th><th>卖出日</th><th>买入价</th><th>卖出价</th><th>价差</th>{div_header}<th>持仓</th><th>触发</th></tr>
          {trade_rows}
        </table>""" if trades else '<h3>📈 量化回测触发日</h3><p>无交易信号</p>'

        dividend_history = ''
        if calc_dividend and dividends:
            recent_divs = [d for d in dividends if d.get('ex_date', '') >= data[0]['day']]
            if recent_divs:
                yr_prices = {}
                for y in set(d['ex_date'][:4] for d in recent_divs):
                    yr_closes = [float(dd['close']) for dd in data if dd['day'][:4] == y]
                    yr_prices[y] = np.mean(yr_closes) if yr_closes else closes[-1]
                div_rows = ''
                for d in recent_divs:
                    y = d['ex_date'][:4]
                    yr_yield = d['dividend_per_share'] / yr_prices.get(y, closes[-1]) * 100
                    div_rows += f'<tr><td>{y}</td><td>{d["ex_date"]}</td><td>10派{d["dividend_10"]:.1f}元</td><td>{yr_yield:.1f}%</td></tr>'
                dividend_history = f"""
        <h3>💰 分红历史</h3>
        <table class="trades">
          <tr><th>年份</th><th>除权日</th><th>方案</th><th>股息率</th></tr>
          {div_rows}
        </table>"""

        img_b64 = make_chart(code, name, dates, closes, dif, dea, bar, tops, bottoms, trades, regimes, ma60)

        return f"""
        <div class="result">
          {conclusion_html}
          {top_section}
          {dividend_history}
          {trade_table}
          <img src="data:image/png;base64,{img_b64}" alt="MACD Chart" loading="lazy">
        </div>"""

    def run_buyhold_analysis(self, code, calc_dividend=False):
        data = fetch_kline(code)
        name = get_name(code)
        save_stock_name(code, name)
        closes = np.array([float(d['close']) for d in data])

        first_date = data[0]['day']
        first_close = closes[0]
        last_date = data[-1]['day']
        last_close = closes[-1]

        d0 = datetime.strptime(first_date, '%Y-%m-%d').date()
        d1 = datetime.strptime(last_date, '%Y-%m-%d').date()
        years = max((d1 - d0).days / 365.25, 0.01)

        price_return = (last_close - first_close) / first_close * 100

        dividends = []
        total_div = 0
        if calc_dividend:
            dividends = fetch_dividends(code)
            total_div = sum(d['dividend_per_share'] for d in dividends if d.get('ex_date','') >= first_date)
        div_yield = total_div / first_close * 100
        total_return = price_return + div_yield

        recent_div = 0
        if calc_dividend and dividends:
            one_year_ago = d1 - timedelta(days=365)
            recent_div = sum(d['dividend_per_share'] for d in dividends if d.get('ex_date','') >= one_year_ago.isoformat())
        latest_div_yield = recent_div / last_close * 100 if recent_div > 0 else 0

        overview = f"""
        <table class="overview">
          <tr><th colspan="2">{name} ({code}) — 长线持有</th></tr>
          <tr><td>买入日期</td><td>{first_date}</td></tr>
          <tr><td>买入价</td><td>{first_close:.2f} 元</td></tr>
          <tr><td>当前日期</td><td>{last_date}</td></tr>
          <tr><td>当前价</td><td>{last_close:.2f} 元</td></tr>
          <tr><td>持有时间</td><td>{years:.1f} 年</td></tr>
          <tr style="background:#e8f5e9"><td>价差收益</td><td style="font-weight:bold">{price_return:+.1f}%</td></tr>"""
        if calc_dividend and dividends:
            overview += f"""
          <tr style="background:#e8f5e9"><td>分红收益</td><td style="font-weight:bold">{total_div:.2f}元/股（{div_yield:+.1f}%）</td></tr>"""
        overview += f"""
          <tr style="background:#c8e6c9"><td>总收益</td><td style="font-weight:bold;font-size:1.1em">{total_return:+.1f}%</td></tr>"""
        if calc_dividend and dividends:
            overview += f"""
          <tr><td>近12月股息率</td><td>{latest_div_yield:.1f}%（近12月分红{recent_div:.2f}÷现价{last_close:.2f}）</td></tr>"""
        overview += """
        </table>"""

        dividend_history = ''
        if calc_dividend and dividends:
            recent_divs = [d for d in dividends if d.get('ex_date','') >= first_date]
            if recent_divs:
                yr_prices = {}
                for y in set(d['ex_date'][:4] for d in recent_divs):
                    yr_closes = [float(dd['close']) for dd in data if dd['day'][:4] == y]
                    yr_prices[y] = np.mean(yr_closes) if yr_closes else last_close
                div_rows = ''
                for d in recent_divs:
                    y = d['ex_date'][:4]
                    yr_yield = d['dividend_per_share'] / yr_prices.get(y, last_close) * 100
                    div_rows += f'<tr><td>{y}</td><td>{d["ex_date"]}</td><td>10派{d["dividend_10"]:.1f}元</td><td>{yr_yield:.1f}%</td></tr>'
                dividend_history = f"""
        <h3>💰 分红历史</h3>
        <table class="trades">
          <tr><th>年份</th><th>除权日</th><th>方案</th><th>股息率</th></tr>
          {div_rows}
        </table>"""

        dates = np.array([datetime.strptime(d['day'],'%Y-%m-%d') for d in data])
        # buyhold chart with font setup
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.font_manager as _fm
        import matplotlib.pyplot as _plt
        import matplotlib.dates as _mdates
        _font_paths = [
            os.path.join(ROOT, 'wqy-zenhei.ttf'),
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        ]
        for _fp in _font_paths:
            if os.path.exists(_fp):
                _fm.fontManager.addfont(_fp)
                try:
                    _plt.rcParams['font.family'] = _fm.FontProperties(fname=_fp).get_name()
                except Exception:
                    pass
                break
        _plt.rcParams['axes.unicode_minus'] = False
        fig = _plt.figure(figsize=(18, 6))
        ax = _plt.subplot(1, 1, 1)
        ax.plot(dates, closes, color='#1565C0', linewidth=1.5, label='收盘价')
        ax.axhline(y=first_close, color='#FF6F00', linewidth=1, linestyle='--', alpha=0.7, label=f'买入价 {first_close:.2f}')
        ax.set_title(f'{name}({code}) — 长线持有', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(_mdates.DateFormatter('%Y-%m'))
        _plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
        _plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        _plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        return f"""
        <div class="result">
          {overview}
          {dividend_history}
          <img src="data:image/png;base64,{img_b64}" alt="BuyHold Chart" loading="lazy">
        </div>"""

    def run_value_analysis(self, code):
        """渲染独立的巴芒财务质量初筛，不产生交易信号或估值目标价。"""
        rows = fetch_financial_summaries(code)
        research = screen_value_quality(rows)
        name = research['company'] or get_name(code)
        if name and name != code:
            save_stock_name(code, name)

        styles = {
            'go': ('buy', '通过财务质量初筛'),
            'watch': ('warn', '观察并补充核验'),
            'no_go': ('sell', '暂不进入价值候选池'),
            'insufficient': ('warn', '数据不足'),
            'not_applicable': ('warn', '方法不适用'),
        }
        css_class, headline = styles[research['status']]
        conclusion_html = (
            f'<div class="conclusion {css_class}">{headline}（非买卖信号）</div>'
        )

        def amount(value):
            return '—' if value is None else f'{value / 1e8:.2f}亿元'

        def percent(value):
            return '—' if value is None else f'{value:.1f}%'

        def ratio(value):
            return '—' if value is None else f'{value:.2f}'

        displayed_reports = research['annual_reports'][:5]
        report_rows = ''.join(
            f'<tr><td>{report["report_name"] or report["report_date"]}</td>'
            f'<td>{report["notice_date"] or "—"}</td>'
            f'<td>{amount(report["revenue"])}</td>'
            f'<td>{amount(report["core_profit"])}</td>'
            f'<td>{amount(report["operating_cash_flow"])}</td>'
            f'<td>{percent(report["roe"])}</td><td>{percent(report["roic"])}</td>'
            f'<td>{percent(report["debt_ratio"])}</td>'
            f'<td>{ratio(report["current_ratio"])}</td></tr>'
            for report in displayed_reports
        )
        report_table = f"""
        <h3>最近五份已披露年报摘要</h3>
        <table class="trades">
          <tr><th>报告期</th><th>披露日</th><th>营收</th><th>扣非归母利润</th>
              <th>经营现金流</th><th>ROE</th><th>ROIC</th><th>资产负债率</th><th>流动比率</th></tr>
          {report_rows}
        </table>""" if report_rows else '<p>未取得可用年报摘要。</p>'

        check_style = {
            'pass': '#2e7d32', 'watch': '#ef6c00', 'fail': '#c62828',
            'unavailable': '#666',
        }
        check_label = {
            'pass': '通过', 'watch': '需核验', 'fail': '未通过',
            'unavailable': '数据不足',
        }
        check_rows = ''.join(
            f'<tr><td>{check["label"]}</td><td>{check["detail"]}</td>'
            f'<td style="color:{check_style[check["status"]]};font-weight:bold">'
            f'{check_label[check["status"]]}</td></tr>'
            for check in research['checks']
        )
        checks_table = f"""
        <table class="overview">
          <tr><th colspan="3">财务质量初筛</th></tr>
          <tr><th>项目</th><th>依据</th><th>结论</th></tr>
          {check_rows}
        </table>"""
        limitations = ''.join(f'<li>{item}</li>' for item in research['limitations'])
        overview = f"""
        <table class="overview">
          <tr><th colspan="2">{name} ({code}) — 巴芒基本面研究</th></tr>
          <tr><td>研究结论</td><td style="font-weight:bold">{research['status_label']}</td></tr>
          <tr><td>当前范围</td><td>{research['scope']}</td></tr>
          <tr><td>数据来源</td><td>{research['source']}；以年报摘要为初筛依据</td></tr>
          <tr><td>结论说明</td><td>{research['conclusion']}</td></tr>
        </table>"""
        limitations_html = f"""
        <h3>尚未覆盖的研究项</h3>
        <ul style="line-height:1.8;color:#555">{limitations}</ul>"""
        return f"""
        <div class="result">
          {conclusion_html}
          {overview}
          <br>{checks_table}
          {report_table}
          {limitations_html}
        </div>"""

    def run_multifactor_analysis(self, code, holding, calc_dividend=False):
        """多因子共振策略：RSI + KDJ + Bollinger + WR"""
        data = fetch_kline(code)
        name = get_name(code)
        save_stock_name(code, name)
        dates = [d['day'] for d in data]
        closes = np.array([float(d['close']) for d in data])
        highs = np.array([float(d['high']) for d in data])
        lows = np.array([float(d['low']) for d in data])
        vols = np.array([float(d['volume']) for d in data])

        trades = backtest_multifactor(dates, closes, highs, lows, vols)
        pred_data = predict_multifactor(dates, closes, highs, lows, vols, holding=holding)
        dividends = []
        if calc_dividend:
            dividends = fetch_dividends(code)
            if dividends:
                trades = enrich_trades_with_dividends(trades, dividends)

        wins = [t for t in trades if t['profit_pct'] > 0]
        total_pnl = sum(t['profit_pct'] for t in trades) if trades else 0
        d0 = datetime.strptime(dates[0], '%Y-%m-%d').date()
        d1 = datetime.strptime(dates[-1], '%Y-%m-%d').date()
        years = max((d1 - d0).days / 365.25, 0.01)

        overview = f"""
        <table class="overview">
          <tr><th colspan="2">{name} ({code}) — 多因子共振</th></tr>
          <tr><td>数据范围</td><td>{dates[0]} ~ {dates[-1]}（{years:.1f}年）</td></tr>
          <tr><td>K线数量</td><td>{len(data)} 根</td></tr>
          <tr><td>回测交易</td><td>{len(trades)} 笔 · 胜率 {len(wins)/len(trades)*100:.0f}%</td></tr>"""
        if calc_dividend and dividends:
            total_div_pct = sum(t.get('dividend_yield_pct', 0) for t in trades)
            total_div_cash = sum(t.get('dividend_total', 0) for t in trades)
            total_return = total_pnl + total_div_pct
            overview += f"""
          <tr style="background:#e8f5e9"><td>价差收益</td><td style="font-weight:bold">{total_pnl:+.1f}%</td></tr>
          <tr style="background:#e8f5e9"><td>分红收益</td><td style="font-weight:bold">{total_div_cash:.2f}元/股（{total_div_pct:+.1f}%）</td></tr>
          <tr style="background:#c8e6c9"><td>总收益</td><td style="font-weight:bold;font-size:1.1em">{total_return:+.1f}%</td></tr>"""
        else:
            overview += f"""
          <tr style="background:#c8e6c9"><td>总收益</td><td style="font-weight:bold;font-size:1.1em">{total_pnl:+.1f}%</td></tr>"""
        overview += """
        </table>"""

        # ── 结论横幅 ──
        conclusion_html = ''
        action = pred_data.get('action', '')
        score = pred_data.get('score', 0)
        if action in ('买入', '增持', '拿住'):
            conclusion_html = f'<div class="conclusion buy">{"✅" * min(10, max(1, abs(score)))} {action}</div>'
        elif action == '卖出':
            conclusion_html = f'<div class="conclusion sell">{"❌" * min(10, max(1, abs(score)))} {action}</div>'
        else:
            conclusion_html = f'<div class="conclusion warn">{"⚠️" * min(10, max(1, abs(score)))} {action}</div>'

        pred_table = self.make_pred_table(pred_data)
        top_section = f'<div class="top-row"><div>{overview}</div><div>{pred_table}</div></div>'

        trade_rows = ''
        show_div = calc_dividend and dividends and any(t.get('dividend_total', 0) > 0 for t in trades)
        for t in trades:
            tr_class = 'win' if t['profit_pct'] > 0 else 'loss'
            if show_div:
                div_total = t.get('dividend_total', 0)
                div_cell = f'<td>{div_total:.2f}元/股</td><td class="pnl">{t["total_return_pct"]:+.1f}%</td>' if div_total > 0 else f'<td>-</td><td class="pnl">{t["profit_pct"]:+.1f}%</td>'
                trade_rows += f'<tr class="{tr_class}"><td>{t["buy_date"]}</td><td>{t["sell_date"]}</td><td>{t["buy_price"]:.2f}</td><td>{t["sell_price"]:.2f}</td><td class="pnl">{t["profit_pct"]:+.1f}%</td>{div_cell}<td>{t["hold_days"]}天</td><td class="reason">{t["buy_reason"]}→{t["sell_reason"]}</td></tr>'
            else:
                trade_rows += f'<tr class="{tr_class}"><td>{t["buy_date"]}</td><td>{t["sell_date"]}</td><td>{t["buy_price"]:.2f}</td><td>{t["sell_price"]:.2f}</td><td class="pnl">{t["profit_pct"]:+.1f}%</td><td>{t["hold_days"]}天</td><td class="reason">{t["buy_reason"]}→{t["sell_reason"]}</td></tr>'

        div_header = '<th>分红</th><th>总收益</th>' if show_div else ''
        trade_table = f"""
        <h3>📈 多因子共振交易</h3>
        <table class="trades">
          <tr><th>买入日</th><th>卖出日</th><th>买入价</th><th>卖出价</th><th>价差</th>{div_header}<th>持仓</th><th>触发</th></tr>
          {trade_rows}
        </table>""" if trades else '<p>无交易信号</p>'

        # 分红历史
        dividend_history = ''
        if calc_dividend and dividends:
            recent_divs = [d for d in dividends if d.get('ex_date', '') >= dates[0]]
            if recent_divs:
                yr_prices = {}
                for y in set(d['ex_date'][:4] for d in recent_divs):
                    yr_closes = [float(dd['close']) for dd in data if dd['day'][:4] == y]
                    yr_prices[y] = np.mean(yr_closes) if yr_closes else closes[-1]
                div_rows = ''
                for d in recent_divs:
                    y = d['ex_date'][:4]
                    yr_yield = d['dividend_per_share'] / yr_prices.get(y, closes[-1]) * 100
                    div_rows += f'<tr><td>{y}</td><td>{d["ex_date"]}</td><td>10派{d["dividend_10"]:.1f}元</td><td>{yr_yield:.1f}%</td></tr>'
                dividend_history = f"""
        <h3>💰 分红历史</h3>
        <table class="trades">
          <tr><th>年份</th><th>除权日</th><th>方案</th><th>股息率</th></tr>
          {div_rows}
        </table>"""

        # 图表
        dates_dt = np.array([datetime.strptime(d, '%Y-%m-%d') for d in dates])
        img_b64 = plot_multifactor(code, name, dates_dt, closes, highs, lows, trades)

        return f"""
        <div class="result">
          {conclusion_html}
          {top_section}
          {dividend_history}
          {trade_table}
          <img src="data:image/png;base64,{img_b64}" alt="MultiFactor Chart" loading="lazy">
        </div>"""

    def run_comprehensive_analysis(self, code, holding, calc_dividend=False):
        """渲染融合策略；评分和门禁只读取 engine 的结构化结果。"""
        data = fetch_kline(code)
        name = get_name(code)
        save_stock_name(code, name)
        dates = [row['day'] for row in data]
        opens = np.array([float(row['open']) for row in data])
        closes = np.array([float(row['close']) for row in data])
        highs = np.array([float(row['high']) for row in data])
        lows = np.array([float(row['low']) for row in data])
        volumes = np.array([float(row['volume']) for row in data])

        trades = backtest_comprehensive(
            dates, closes, highs, lows, volumes, opens=opens)
        prediction = predict_comprehensive(
            dates, closes, highs, lows, volumes, holding=holding, opens=opens)
        if 'error' in prediction:
            return f'<div class="error">{prediction["error"]}</div>'

        dividends = []
        if calc_dividend:
            dividends = fetch_dividends(code)
            if dividends:
                trades = enrich_trades_with_dividends(trades, dividends)

        summary = summarize_trades(trades)
        d0 = datetime.strptime(dates[0], '%Y-%m-%d').date()
        d1 = datetime.strptime(dates[-1], '%Y-%m-%d').date()
        years = max((d1 - d0).days / 365.25, 0.01)
        scores = prediction['scores']
        score_level = lambda value: '🟢' if value >= 60 else ('🟡' if value >= 45 else '🔴')

        overview = f"""
        <table class="overview">
          <tr><th colspan="2">{name} ({code}) — 融合策略</th></tr>
          <tr><td>数据范围</td><td>{dates[0]} ~ {dates[-1]}（{years:.1f}年）</td></tr>
          <tr><td>K线数量</td><td>{len(data)} 根</td></tr>
          <tr><td>回测交易</td><td>{summary['trades']} 笔 · 胜率 {summary['win_rate_pct']:.0f}%</td></tr>
          <tr style="background:#e8f5e9"><td>价差复利收益</td><td style="font-weight:bold">{summary['price_return_pct']:+.1f}%</td></tr>"""
        if calc_dividend and dividends:
            total_div_cash = sum(trade.get('dividend_total', 0) for trade in trades)
            overview += f"""
          <tr style="background:#e8f5e9"><td>各笔分红合计</td><td style="font-weight:bold">{total_div_cash:.2f}元/股</td></tr>
          <tr style="background:#c8e6c9"><td>含分红复利总收益</td><td style="font-weight:bold;font-size:1.1em">{summary['total_return_pct']:+.1f}%</td></tr>"""
        else:
            overview += f"""
          <tr style="background:#c8e6c9"><td>复利总收益</td><td style="font-weight:bold;font-size:1.1em">{summary['total_return_pct']:+.1f}%</td></tr>"""
        overview += '</table>'

        def detail_table(title, items, color):
            rows = ''.join(
                f'<tr><td>{item["rule"]}</td><td style="color:{color};font-weight:bold">{item["adj"]}</td></tr>'
                for item in items)
            return f'<table class="overview"><tr><th colspan="2">{title}</th></tr>{rows}</table>'

        breakdown = prediction['breakdown']
        analysis_html = f"""
        <h3>🔍 分析过程</h3>
        {detail_table('MACD核心评分 (40%)', breakdown['macd'], '#1565c0')}
        <br>
        {detail_table('多因子评分 (30%)', breakdown['multifactor'], '#ef6c00')}
        <br>
        {detail_table('价格位置/趋势 (15%)', breakdown['fundamental'], '#2e7d32')}
        <br>
        {detail_table('量能 (15%)', breakdown['game'], '#6a1b9a')}
        """

        gate_rows = ''
        if prediction['gates']['details']:
            for gate in prediction['gates']['details']:
                value = gate.get('value')
                rendered = f'{value:+.0%}' if gate['gate'] == 'OBV净量能流' else str(value)
                gate_rows += f'<tr><td>{gate["gate"]}</td><td>{rendered}</td><td style="color:#c62828">否决</td></tr>'
        else:
            gate_rows = '<tr><td colspan="3" style="color:#2e7d32;font-weight:bold">全部通过</td></tr>'

        score_card = f"""
        <table class="overview">
          <tr><th colspan="3">📊 融合策略四维评分</th></tr>
          <tr style="background:#e3f2fd"><td><b>MACD核心 (40%)</b></td><td style="font-weight:bold;font-size:1.2em">{score_level(scores['macd']['score'])} {scores['macd']['score']:.1f}</td><td style="font-size:11px;color:#888">DIF{scores['macd']['dif']:.2f}/DEA{scores['macd']['dea']:.2f} BAR{scores['macd']['bar']:.2f}</td></tr>
          <tr style="background:#fff3e0"><td><b>多因子 (30%)</b></td><td style="font-weight:bold;font-size:1.2em">{score_level(scores['multifactor']['score'])} {scores['multifactor']['score']:.1f}</td><td style="font-size:11px;color:#888">RSI{scores['multifactor']['rsi']:.0f} K{scores['multifactor']['k']:.0f}/D{scores['multifactor']['d']:.0f}/J{scores['multifactor']['j']:.0f} WR{scores['multifactor']['wr']:.0f}</td></tr>
          <tr style="background:#e8f5e9"><td><b>{scores['fundamental']['label']} (15%)</b></td><td style="font-weight:bold;font-size:1.2em">{score_level(scores['fundamental']['score'])} {scores['fundamental']['score']:.1f}</td><td style="font-size:11px;color:#888">价格序列代理，非财务基本面</td></tr>
          <tr style="background:#f3e5f5"><td><b>量能 (15%)</b></td><td style="font-weight:bold;font-size:1.2em">{score_level(scores['game']['score'])} {scores['game']['score']:.1f}</td><td style="font-size:11px;color:#888">OBV 5日净量能流 {scores['game']['obv_flow']:+.0%}</td></tr>
                    <tr style="background:#f5f5f5"><td><b>四维基础分 S</b></td><td style="font-weight:bold;font-size:1.2em">{score_level(prediction['base_composite'])} {prediction['base_composite']:.1f}</td><td>M{scores['macd']['score']:.0f}×0.40+F{scores['multifactor']['score']:.0f}×0.30+价{scores['fundamental']['score']:.0f}×0.15+量{scores['game']['score']:.0f}×0.15</td></tr>
                    <tr style="background:#fff8e1"><td><b>辅助共识修正</b></td><td style="font-weight:bold;font-size:1.2em">{prediction['auxiliary_adjustment']:+.1f}</td><td>A={prediction['auxiliary_consensus']['consensus_score']:+.0f}，β={prediction['auxiliary_beta']:.2f}</td></tr>
                    <tr style="background:#f5f5f5"><td><b>有效评分 S*</b></td><td style="font-weight:bold;font-size:1.4em">{score_level(prediction['composite'])} {prediction['composite']:.1f}</td><td>S* = S − βA</td></tr>
        </table>
        <br>
        <table class="overview">
          <tr><th colspan="3">质量门禁</th></tr>
          {gate_rows}
        </table>"""

        conclusion_html = ''
        action = prediction['action']
        score = prediction['composite']
        if prediction['signal'] == '门禁否决':
            conclusion_html = '<div class="conclusion sell">🔴 门禁否决 → 观望</div>'
        elif action in ('买入', '增持', '拿住'):
            conclusion_html = f'<div class="conclusion buy">{"✅" * min(10, max(1, int(score / 10)))} {action}</div>'
        elif action == '卖出':
            conclusion_html = f'<div class="conclusion sell">{"❌" * min(10, max(1, int(score / 10)))} {action}</div>'
        else:
            conclusion_html = f'<div class="conclusion warn">{"⚠️" * min(10, max(1, int(score / 10)))} {action}</div>'

        aux = prediction.get('auxiliary_consensus', {})
        aux_html = ''
        if aux.get('votes'):
            vote_rows = ''
            for vote in aux['votes']:
                color = '#2e7d32' if vote['vote'] > 0 else ('#c62828' if vote['vote'] < 0 else '#888')
                label = '看多' if vote['vote'] > 0 else ('看空' if vote['vote'] < 0 else '中性')
                value = vote['value']
                if isinstance(value, dict):
                    value = ', '.join(f'{key}={item}' for key, item in value.items())
                vote_rows += f'<tr><td>{vote["name"]}</td><td style="font-family:monospace">{value}</td><td style="color:{color};font-weight:bold">{label}</td></tr>'
            aux_html = f"""
            <br>
            <table class="overview">
              <tr><th colspan="3">辅助指标投票面板 ({aux['total_indicators']}个指标)</th></tr>
              <tr><td colspan="3" style="text-align:center;font-weight:bold">共识度 {aux['consensus_score']:+.0f} ({aux['consensus_pct']:.0f}%看多) | 修正 {prediction['auxiliary_adjustment']:+.1f} | 看多{aux['bullish_count']} 看空{aux['bearish_count']} 中性{aux['neutral_count']}</td></tr>
              {vote_rows}
            </table>"""

        show_div = calc_dividend and dividends and any(trade.get('dividend_total', 0) > 0 for trade in trades)
        trade_rows = ''
        for trade in trades:
            row_class = 'win' if trade['profit_pct'] > 0 else 'loss'
            signals = f'信号 {trade.get("buy_signal_date", trade["buy_date"])} → {trade.get("sell_signal_date", trade["sell_date"])}'
            dividend_cell = ''
            if show_div:
                dividend_cell = f'<td>{trade.get("dividend_total", 0):.2f}元/股</td><td class="pnl">{trade.get("total_return_pct", trade["profit_pct"]):+.1f}%</td>'
            trade_rows += f'<tr class="{row_class}"><td>{trade["buy_date"]}</td><td>{trade["sell_date"]}</td><td>{trade["buy_price"]:.2f}</td><td>{trade["sell_price"]:.2f}</td><td class="pnl">{trade["profit_pct"]:+.1f}%</td>{dividend_cell}<td>{trade["hold_days"]}天</td><td class="reason">{signals}<br>{trade["buy_reason"]} → {trade["sell_reason"]}</td></tr>'

        dividend_header = '<th>分红</th><th>含分红</th>' if show_div else ''
        trade_table = f"""
        <h3>📈 综合策略交易记录</h3>
        <table class="trades">
          <tr><th>买入日</th><th>卖出日</th><th>买入价</th><th>卖出价</th><th>净价差</th>{dividend_header}<th>持仓</th><th>触发</th></tr>
          {trade_rows}
        </table>""" if trades else '<h3>📈 综合策略交易记录</h3><p>无已完成交易</p>'

        dates_dt = np.array([datetime.strptime(date, '%Y-%m-%d') for date in dates])
        image = make_comprehensive_chart(code, name, dates_dt, closes, highs, lows, volumes, trades)
        return f"""
        <div class="result">
          {conclusion_html}
          {overview}
          <br>{score_card}
          {aux_html}
          {analysis_html}
          {trade_table}
          <img src="data:image/png;base64,{image}" alt="Comprehensive Chart" loading="lazy">
        </div>"""

    def _run_comprehensive_analysis_legacy(self, code, holding, calc_dividend=False):
        """综合策略：三面量化评分 (技术30% + 博弈45% + 基本面25%)"""
        import engine as eng
        data = fetch_kline(code)
        name = get_name(code)
        save_stock_name(code, name)
        dates = [d['day'] for d in data]
        closes = np.array([float(d['close']) for d in data])
        highs = np.array([float(d['high']) for d in data])
        lows = np.array([float(d['low']) for d in data])
        vols = np.array([float(d['volume']) for d in data])

        trades = backtest_comprehensive(dates, closes, highs, lows, vols)
        pred_data = predict_comprehensive(dates, closes, highs, lows, vols, holding=holding)

        dividends = []
        if calc_dividend:
            dividends = fetch_dividends(code)
            if dividends:
                trades = enrich_trades_with_dividends(trades, dividends)

        wins = [t for t in trades if t['profit_pct'] > 0]
        total_pnl = sum(t['profit_pct'] for t in trades) if trades else 0
        d0 = datetime.strptime(dates[0], '%Y-%m-%d').date()
        d1 = datetime.strptime(dates[-1], '%Y-%m-%d').date()
        years = max((d1 - d0).days / 365.25, 0.01)

        # 当前评分
        tech_score, tech_dim, patterns = eng.score_technical(closes, highs, lows, vols)
        game_score, game_dim = eng.score_game_theory(closes, vols, highs, lows)
        fund_score, fund_dim = eng.score_fundamental(closes, vols)
        
        # 融合策略四维评分（回测同款）
        dif, dea, bar = calc_macd(closes)
        rsi_arr = calc_rsi(closes)
        k_arr, d_arr, j_arr = calc_kdj(highs, lows, closes)
        bb_u, bb_m, bb_l = calc_bollinger(closes)
        wr_arr = calc_wr(highs, lows, closes)
        obv_full = calc_obv(closes, vols)
        i = len(closes) - 1
        
        macd_score = 50
        if dif[i] > dea[i]: macd_score += 15
        else: macd_score -= 15
        if dif[i] > 0: macd_score += 12
        else: macd_score -= 8
        if i >= 5 and dif[i] > dif[i-5]: macd_score += 8
        else: macd_score -= 5
        if i >= 3 and bar[i] > bar[i-3]: macd_score += 5
        
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
        bb_pos = (closes[i] - bb_l[i]) / (bb_u[i] - bb_l[i]) * 100 if not np.isnan(bb_u[i]) and bb_u[i] != bb_l[i] else 50
        if bb_pos < 10: mf_score += 12
        elif bb_pos > 90: mf_score -= 8
        if wv > 80: mf_score += 8
        
        game_score2 = 50
        obv_ma20 = np.mean(obv_full[max(0,i-20):i]) if i >= 20 else np.mean(obv_full[:i])
        obv_ma5 = np.mean(obv_full[max(0,i-4):i+1])
        obv_ratio = obv_full[i] / obv_ma20 if obv_ma20 > 0 else 1
        if obv_ma5 > obv_ma20 * 1.08: game_score2 += 12
        elif obv_ma5 < obv_ma20 * 0.92: game_score2 -= 12
        
        fund_score2 = 50
        if i >= 249:
            h250 = np.max(closes[i-249:i+1])
            l250 = np.min(closes[i-249:i+1])
            pos250 = (closes[i] - l250) / (h250 - l250) * 100 if h250 != l250 else 50
            if pos250 < 25: fund_score2 += 20
            elif pos250 < 40: fund_score2 += 10
            elif pos250 > 80: fund_score2 -= 15
        
        composite = macd_score * 0.40 + mf_score * 0.30 + fund_score2 * 0.15 + game_score2 * 0.15

        overview = f"""
        <table class="overview">
          <tr><th colspan="2">{name} ({code}) — 综合策略</th></tr>
          <tr><td>数据范围</td><td>{dates[0]} ~ {dates[-1]}（{years:.1f}年）</td></tr>
          <tr><td>K线数量</td><td>{len(data)} 根</td></tr>
          <tr><td>回测交易</td><td>{len(trades)} 笔 · 胜率 {len(wins)/len(trades)*100:.0f}%</td></tr>"""
        if calc_dividend and dividends:
            total_div_pct = sum(t.get('dividend_yield_pct', 0) for t in trades)
            total_div_cash = sum(t.get('dividend_total', 0) for t in trades)
            total_return = total_pnl + total_div_pct
            overview += f"""
          <tr style="background:#e8f5e9"><td>价差收益</td><td style="font-weight:bold">{total_pnl:+.1f}%</td></tr>
          <tr style="background:#e8f5e9"><td>分红收益</td><td style="font-weight:bold">{total_div_cash:.2f}元/股（{total_div_pct:+.1f}%）</td></tr>
          <tr style="background:#c8e6c9"><td>总收益</td><td style="font-weight:bold;font-size:1.1em">{total_return:+.1f}%</td></tr>"""
        else:
            overview += f"""
          <tr style="background:#c8e6c9"><td>总收益</td><td style="font-weight:bold;font-size:1.1em">{total_pnl:+.1f}%</td></tr>"""

        # 三面评分卡
        dim_colors = {'趋势':'🔵','动量':'🟠','量能':'🟢','通道/波动':'🟣','形态/结构':'🔴','筹码':'⚪'}
        score_level = lambda s: '🟢' if s >= 60 else ('🟡' if s >= 45 else '🔴')
        tech_rows = ''.join(f'<tr><td>{dim_colors.get(k,"")} {k}</td><td>{v:.1f}</td></tr>' for k, v in tech_dim.items())
        game_rows = ''.join(f'<tr><td>{k}</td><td>{v:.1f}</td></tr>' for k, v in game_dim.items())
        fund_rows = ''.join(f'<tr><td>{k}</td><td>{v:.1f}</td></tr>' for k, v in fund_dim.items())

        # ── 机构参与度（优先真实数据，失败降级代理）──
        inst_data = None
        try:
            from fetcher import fetch_institution_participation, estimate_institution_proxy
            inst_data = fetch_institution_participation(code)
        except: pass
        
        if inst_data and inst_data.get('institution_participation') is not None:
            inst_val = inst_data['institution_participation'] * 100
            if inst_val > 50: inst_label = f'🏛️ 机构主导 ({inst_val:.0f}%)'
            elif inst_val > 30: inst_label = f'🤝 均衡型 ({inst_val:.0f}%)'
            else: inst_label = f'👤 散户活跃 ({inst_val:.0f}%)'
            inst_source = '千股千评'
        else:
            # 代理：用波动率估算
            from fetcher import estimate_institution_proxy
            inst_proxy = estimate_institution_proxy(closes, highs, lows, vols)
            if inst_proxy > 65: inst_label = '🏛️ 机构主导 (代理)'
            elif inst_proxy > 45: inst_label = '🤝 均衡型 (代理)'
            else: inst_label = '👤 散户活跃 (代理)'
            inst_source = '波动率代理'
            inst_val = inst_proxy

        # ── 博弈反弹信号 ──
        reb_signal = ''
        if i >= 5:
            ret_5d_val = (closes[i] - closes[max(0,i-5)]) / closes[max(0,i-5)] * 100
            if ret_5d_val < -3 and (inst_val > 30):
                reb_signal = f'<br><span style="color:#2e7d32;font-size:11px">⚡ 下跌{ret_5d_val:+.1f}% + 机构参与{inst_val:.0f}% → 博弈反弹信号</span>'
        analysis_html = f"""<h3>🔍 分析过程</h3>
        <table class="overview">
          <tr><th colspan="3">MACD核心评分 (40%)</th></tr>
          <tr><td>DIF vs DEA</td><td style="font-family:monospace">{dif[i]:.2f} {">" if dif[i]>dea[i] else "<"} {dea[i]:.2f} → {"金叉" if dif[i]>dea[i] else "死叉"}</td><td style="color:{'#2e7d32' if dif[i]>dea[i] else '#c62828'}">{'+15' if dif[i]>dea[i] else '−15'}</td></tr>
          <tr><td>零轴位置</td><td style="font-family:monospace">DIF={dif[i]:.2f} {">0 做多区" if dif[i]>0 else "<0 做空区"}</td><td style="color:{'#2e7d32' if dif[i]>0 else '#c62828'}">{'+12' if dif[i]>0 else '−8'}</td></tr>
          <tr><td>DIF 5日斜率</td><td style="font-family:monospace">{dif[i]-dif[max(0,i-5)]:+.2f}</td><td style="color:{'#2e7d32' if i>=5 and dif[i]>dif[i-5] else '#c62828'}">{'+8' if i>=5 and dif[i]>dif[i-5] else '−5'}</td></tr>
          <tr><td>BAR 3日趋势</td><td style="font-family:monospace">{bar[i]:+.2f} (3日前:{bar[max(0,i-3)]:+.2f})</td><td style="color:{'#2e7d32' if i>=3 and bar[i]>bar[i-3] else '#888'}">{'+5' if i>=3 and bar[i]>bar[i-3] else '0'}</td></tr>"""

        # 底背离/顶背离
        if i >= 30:
            rlo = np.min(closes[max(0,i-30):i])
            if closes[i] > rlo * 1.03 and dif[i] > dif[max(0,i-30)]:
                analysis_html += f'<tr><td>底背离 30日</td><td style="font-family:monospace">价{closes[i]:.2f}>低{rlo:.2f} DIF↑</td><td style="color:#2e7d32">+10</td></tr>'
            rhi = np.max(closes[max(0,i-30):i])
            if closes[i] >= rhi * 0.98 and dif[i] < dif[max(0,i-30)] * 0.9:
                analysis_html += f'<tr><td>🔴 顶背离 30日</td><td style="font-family:monospace">价{closes[i]:.2f}≈高{rhi:.2f} DIF↓</td><td style="color:#c62828">−15</td></tr>'
        
        analysis_html += f"""<tr style="background:#e3f2fd"><td><b>MACD小计</b></td><td></td><td style="font-weight:bold;font-size:1.1em">{macd_score:.0f}</td></tr>
        </table>
        <br>
        <table class="overview">
          <tr><th colspan="3">多因子评分 (30%)</th></tr>
          <tr><td>RSI(14)</td><td style="font-family:monospace">{rv:.0f}</td><td style="color:{'#2e7d32' if 30<=rv<=65 else ('#2e7d32' if rv<30 else ('#c62828' if rv>80 else '#888'))}">{'+10 (30-65)' if 30<=rv<=65 else ('+15 (<30超卖)' if rv<30 else ('−15 (>80超买)' if rv>80 else '−8 (>70偏强)' if rv>70 else '0'))}</td></tr>
          <tr><td>KDJ</td><td style="font-family:monospace">K={kv:.0f} D={dv:.0f} J={jv:.0f}</td><td style="color:{'#2e7d32' if kv>dv else '#c62828'}">{'+10 金叉' if kv>dv else '−8 死叉'}{' +8(J<0)' if jv<0 else ''}{' −8(J>100)' if jv>100 else ''}</td></tr>"""

        bb_pos_val = (closes[i] - bb_l[i]) / (bb_u[i] - bb_l[i]) * 100 if not np.isnan(bb_u[i]) and bb_u[i] != bb_l[i] else 50
        analysis_html += f"""<tr><td>布林带</td><td style="font-family:monospace">位置 {bb_pos_val:.0f}%（下{bb_l[i]:.2f}/上{bb_u[i]:.2f}）</td><td style="color:{'#2e7d32' if bb_pos_val<10 else ('#c62828' if bb_pos_val>90 else '#888')}">{'+12 下轨' if bb_pos_val<10 else ('−8 上轨' if bb_pos_val>90 else '0')}</td></tr>
          <tr><td>WR(10)</td><td style="font-family:monospace">{wv:.0f}</td><td style="color:{'#2e7d32' if wv>80 else ('#c62828' if wv<20 else '#888')}">{'+8' if wv>80 else ('−8' if wv<20 else '0')}</td></tr>
          <tr style="background:#fff3e0"><td><b>多因子小计</b></td><td></td><td style="font-weight:bold;font-size:1.1em">{mf_score:.0f}</td></tr>
        </table>
        <br>
        <table class="overview">
          <tr><th colspan="3">基本面 & 量能</th></tr>
          <tr><td>250日估值</td><td style="font-family:monospace">{"位置 "+str(int(pos250))+"%" if i>=249 else "数据不足"}</td><td style="color:{'#2e7d32' if i>=249 and pos250<40 else ('#c62828' if i>=249 and pos250>80 else '#888')}">{'+20 低估' if i>=249 and pos250<25 else ('+10 偏低' if i>=249 and pos250<40 else ('−15 高估' if i>=249 and pos250>80 else '0'))}</td></tr>
          <tr><td>基本面 (15%)</td><td></td><td style="font-weight:bold">{fund_score2:.0f}</td></tr>
          <tr><td>OBV比值</td><td style="font-family:monospace">MA₅/MA₂₀ = {obv_ratio:.2f}</td><td style="color:{'#2e7d32' if obv_ratio>1.08 else ('#c62828' if obv_ratio<0.92 else '#888')}">{'+12' if obv_ratio>1.08 else ('−12' if obv_ratio<0.92 else '0')}</td></tr>
          <tr><td>量能 (15%)</td><td></td><td style="font-weight:bold">{game_score2:.0f}</td></tr>
          <tr><td>机构参与</td><td style="font-size:12px">{inst_label} <small>({inst_source})</small>{reb_signal}</td><td></td></tr>
        </table>
        <br>
        <table class="overview">
          <tr><th colspan="3">综合 & 门禁</th></tr>
          <tr><td><b>S = M×0.40 + F×0.30 + V×0.15 + Q×0.15</b></td><td style="font-family:monospace">{macd_score:.0f}×0.40 + {mf_score:.0f}×0.30 + {fund_score2:.0f}×0.15 + {game_score2:.0f}×0.15</td><td style="font-weight:bold;font-size:1.2em">{score_level(composite)} {composite:.1f}</td></tr>
          <tr><td>OBV门禁</td><td style="font-family:monospace">{obv_ratio:.2f} {"≥" if obv_ratio>=0.90 else "<"} 0.90</td><td style="color:{'#2e7d32' if obv_ratio>=0.90 else '#c62828'}">{'✅ 通过' if obv_ratio>=0.90 else '🔴 否决'}</td></tr>
          <tr><td>RSI门禁</td><td style="font-family:monospace">{rv:.0f} {"≤" if rv<=92 else ">"} 92</td><td style="color:{'#2e7d32' if rv<=92 else '#c62828'}">{'✅ 通过' if rv<=92 else '🔴 否决'}</td></tr>
          <tr><td>双弱门禁</td><td style="font-family:monospace">M={macd_score:.0f} F={mf_score:.0f}</td><td style="color:{'#2e7d32' if not (macd_score<35 and mf_score<40) else '#c62828'}">{'✅ 通过' if not (macd_score<35 and mf_score<40) else '🔴 否决'}</td></tr>"""

        # 信号
        gates_pass = obv_ratio >= 0.90 and rv <= 92 and not (macd_score < 35 and mf_score < 40)
        if gates_pass:
            if composite >= 70: sig='🟢 强烈看多'; act='买入'
            elif composite >= 65: sig='🟢 偏多'; act='买入'
            elif composite >= 50: sig='🟡 中性'; act='观望'
            elif composite >= 40: sig='🟠 偏空'; act='不买'
            else: sig='🔴 看空'; act='不买'
        else:
            sig='🔴 门禁否决'; act='观望'
        sig_color = '#2e7d32' if '🟢' in sig else ('#c62828' if '🔴' in sig else '#f57f17')
        analysis_html += f"""<tr style="background:#f5f5f5"><td><b>信号</b></td><td></td><td style="font-weight:bold;font-size:1.2em;color:{sig_color}">{sig} → {act}</td></tr>
        </table>"""

        # 结论横幅
        conclusion_html = ''
        action = pred_data.get('action', '')
        score = pred_data.get('composite', 50)
        if pred_data.get('signal') == '门禁否决':
            conclusion_html = f'<div class="conclusion sell">🔴 门禁否决 → 观望</div>'
        elif action in ('买入', '增持', '拿住'):
            n = min(10, max(1, int(score / 10)))
            conclusion_html = f'<div class="conclusion buy">{"✅" * n} {action}</div>'
        elif action == '卖出':
            conclusion_html = f'<div class="conclusion sell">{"❌" * min(10, max(1, int(score / 10)))} {action}</div>'
        else:
            conclusion_html = f'<div class="conclusion warn">{"⚠️" * min(10, max(1, int(score / 10)))} {action}</div>'

        # ── 单列布局 ──
        overview = f"""
        {overview}
        <br>
        <table class="overview">
          <tr><th colspan="3">📊 融合策略四维评分</th></tr>
          <tr style="background:#e3f2fd"><td><b>MACD核心 (40%)</b></td><td style="font-weight:bold;font-size:1.2em">{score_level(macd_score)} {macd_score:.1f}</td><td style="font-size:11px;color:#888">DIF{dif[i]:.2f}/DEA{dea[i]:.2f} BAR{bar[i]:.2f}</td></tr>
          <tr style="background:#fff3e0"><td><b>多因子 (30%)</b></td><td style="font-weight:bold;font-size:1.2em">{score_level(mf_score)} {mf_score:.1f}</td><td style="font-size:11px;color:#888">RSI{rv:.0f} K{kv:.0f}/D{dv:.0f}/J{jv:.0f} WR{wv:.0f}</td></tr>
          <tr style="background:#e8f5e9"><td><b>基本面 (15%)</b></td><td style="font-weight:bold;font-size:1.2em">{score_level(fund_score2)} {fund_score2:.1f}</td><td style="font-size:11px;color:#888"><table>{fund_rows}</table></td></tr>
          <tr style="background:#f3e5f5"><td><b>量能 (15%)</b></td><td style="font-weight:bold;font-size:1.2em">{score_level(game_score2)} {game_score2:.1f}</td><td style="font-size:11px;color:#888">OBV比值{obv_ratio:.2f}<br>{inst_label}</td></tr>
          <tr style="background:#f5f5f5"><td><b>综合加权</b></td><td style="font-weight:bold;font-size:1.4em">{score_level(composite)} {composite:.1f}</td><td>M{macd_score:.0f}×0.40+F{mf_score:.0f}×0.30+基{fund_score2:.0f}×0.15+量{game_score2:.0f}×0.15</td></tr>
        </table>"""

        # ── 辅助指标投票面板 ──
        aux = pred_data.get('auxiliary_consensus', {})
        if aux and aux.get('votes'):
            n_total = aux.get('total_indicators', len(aux['votes']))
            bull_c = aux.get('bullish_count', 0)
            bear_c = aux.get('bearish_count', 0)
            neutral_c = aux.get('neutral_count', 0)
            cons_score = aux.get('consensus_score', 0)
            cons_pct = aux.get('consensus_pct', 50)
            if cons_score > 20:
                cons_color = '#2e7d32'; cons_icon = '🟢'
            elif cons_score > 5:
                cons_color = '#f57f17'; cons_icon = '🟡'
            elif cons_score > -5:
                cons_color = '#888'; cons_icon = '⚪'
            elif cons_score > -20:
                cons_color = '#e65100'; cons_icon = '🟠'
            else:
                cons_color = '#c62828'; cons_icon = '🔴'
            
            vote_rows = ''
            for v in aux['votes']:
                if v['vote'] == 1:
                    vc = '#2e7d32'; vs = '📈 ' + v.get('signal', '看多')
                elif v['vote'] == -1:
                    vc = '#c62828'; vs = '📉 ' + v.get('signal', '看空')
                else:
                    vc = '#888'; vs = '➖ ' + v.get('signal', '中性')
                val_display = v.get('value', 0)
                if isinstance(val_display, dict):
                    val_display = ', '.join(f'{k}={va}' for k, va in val_display.items())
                vote_rows += f'<tr><td style="font-size:11px">{v["name"]}</td><td style="font-family:monospace;font-size:10px">{val_display}</td><td style="color:{vc};font-weight:bold">{vs}</td></tr>'

            overview += f"""
        <br>
        <table class="overview">
          <tr><th colspan="3">🗳️ 辅助指标投票面板 ({n_total}个指标)</th></tr>
          <tr style="background:#f5f5f5">
            <td colspan="3" style="text-align:center;font-weight:bold;font-size:1.1em;color:{cons_color}">
              {cons_icon} 共识度: {cons_score:+.0f} ({cons_pct:.0f}%看多) | 
              📈{bull_c}看多 📉{bear_c}看空 ➖{neutral_c}中性
            </td>
          </tr>
          {vote_rows}
        </table>"""

        trade_rows = ''
        show_div = calc_dividend and dividends and any(t.get('dividend_total', 0) > 0 for t in trades)
        for t in trades:
            tr_class = 'win' if t['profit_pct'] > 0 else 'loss'
            if show_div:
                div_total = t.get('dividend_total', 0)
                div_cell = f'<td>{div_total:.2f}元/股</td><td class="pnl">{t["total_return_pct"]:+.1f}%</td>' if div_total > 0 else f'<td>-</td><td class="pnl">{t["profit_pct"]:+.1f}%</td>'
                trade_rows += f'<tr class="{tr_class}"><td>{t["buy_date"]}</td><td>{t["sell_date"]}</td><td>{t["buy_price"]:.2f}</td><td>{t["sell_price"]:.2f}</td><td class="pnl">{t["profit_pct"]:+.1f}%</td>{div_cell}<td>{t["hold_days"]}天</td><td class="reason">{t["buy_reason"]}→{t["sell_reason"]}</td></tr>'
            else:
                trade_rows += f'<tr class="{tr_class}"><td>{t["buy_date"]}</td><td>{t["sell_date"]}</td><td>{t["buy_price"]:.2f}</td><td>{t["sell_price"]:.2f}</td><td class="pnl">{t["profit_pct"]:+.1f}%</td><td>{t["hold_days"]}天</td><td class="reason">{t["buy_reason"]}→{t["sell_reason"]}</td></tr>'

        div_header = '<th>分红</th><th>总收益</th>' if show_div else ''
        trade_table = f"""
        <h3>📈 综合策略交易记录</h3>
        <table class="trades">
          <tr><th>买入日</th><th>卖出日</th><th>买入价</th><th>卖出价</th><th>价差</th>{div_header}<th>持仓</th><th>触发</th></tr>
          {trade_rows}
        </table>""" if trades else '<h3>📈 综合策略交易记录</h3><p>无交易信号</p>'

        # Chart
        dates_dt = np.array([datetime.strptime(d, '%Y-%m-%d') for d in dates])
        img_b64 = make_comprehensive_chart(code, name, dates_dt, closes, highs, lows, vols, trades)

        return f"""
        <div class="result">
          {conclusion_html}
          {overview}
          {analysis_html}
          {trade_table}
          <img src="data:image/png;base64,{img_b64}" alt="Comprehensive Chart" loading="lazy">
        </div>"""



    def log_message(self, format, *args):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {args[0]}", flush=True)

    def make_pred_table(self, pred):
        """从结构化预测 dict 或文本行列表生成 HTML 表格"""
        html = '<table class="overview"><tr><th colspan="2">🔮 预测分析</th></tr>'

        # If it's a dict (structured output), extract key fields
        if isinstance(pred, dict):
            if 'error' in pred:
                html += f'<tr><td colspan="2">{pred["error"]}</td></tr>'
            else:
                strategy = pred.get('strategy', '')
                html += f'<tr><td>日期</td><td>{pred["date"]}</td></tr>'
                html += f'<tr><td>收盘</td><td>{pred["close"]:.2f}</td></tr>'
                if strategy == 'macd':
                    html += f'<tr><td>DIF/DEA/BAR</td><td>{pred["dif"]:.2f} / {pred["dea"]:.2f} / {pred["bar"]:.2f}</td></tr>'
                    html += f'<tr><td>牛熊</td><td>{"🟢牛市" if pred["regime"]=="bull" else "🔴熊市"}</td></tr>'
                    html += f'<tr><td>金叉/死叉</td><td>{"🟢金叉" if pred["golden_cross"] else "🔴死叉"}</td></tr>'
                    if pred.get('cross_prediction'):
                        html += f'<tr><td>交叉预测</td><td>{pred["cross_prediction"]["type"]}: ~{pred["cross_prediction"]["days"]}天</td></tr>'
                elif strategy == 'multifactor':
                    ind = pred['indicators']
                    html += f'<tr><td>RSI</td><td>{ind["rsi"]:.1f}</td></tr>'
                    html += f'<tr><td>KDJ</td><td>K={ind["kdj"]["k"]:.1f} D={ind["kdj"]["d"]:.1f} J={ind["kdj"]["j"]:.1f}</td></tr>'
                    html += f'<tr><td>WR</td><td>{ind["wr"]:.1f}</td></tr>'
                elif strategy == 'comprehensive':
                    s = pred['scores']
                    html += f'<tr><td>MACD核心</td><td>{s["macd"]["score"]:.1f} (×0.40)</td></tr>'
                    html += f'<tr><td>多因子</td><td>{s["multifactor"]["score"]:.1f} (×0.30)</td></tr>'
                    html += f'<tr><td>基本面</td><td>{s["fundamental"]["score"]:.1f} (×0.15)</td></tr>'
                    html += f'<tr><td>量能</td><td>{s["game"]["score"]:.1f} (×0.15)</td></tr>'
                    html += f'<tr><td>门禁</td><td>{"✅通过" if pred["gates"]["passed"] else "🔴否决"}</td></tr>'

                # Common fields
                html += f'<tr style="background:#e8f5e9"><td><b>评分</b></td><td><b>{pred.get("score") or pred.get("composite"):.0f}</b></td></tr>'
                signal = pred.get('signal', '')
                icon = {'强烈看多': '🟢', '偏多': '🟢', '中性': '🟡', '偏空': '🟠', '看空': '🔴', '门禁否决': '🔴'}.get(signal, '⚪')
                html += f'<tr style="background:#fff3e0"><td><b>信号</b></td><td><b>{icon} {signal}</b></td></tr>'
                html += f'<tr style="background:#ffebee"><td><b>建议</b></td><td><b>{pred["action"]}</b></td></tr>'
        else:
            # Legacy: list of formatted text lines
            for line in pred:
                s = line.strip()
                if not s or s.startswith('==') or s.startswith('──'): continue
                if ':' in s:
                    k, v = s.split(':', 1)
                    k = k.strip(); v = v.strip()
                    if k and v: html += f'<tr><td>{k}</td><td>{v}</td></tr>'

        html += '</table>'
        return html


# ═══════════════════════════
# MAIN
# ═══════════════════════════
if __name__ == '__main__':
    import socket
    print(f"A股策略分析 Web → http://localhost:{PORT}")
    print("Ctrl+C 停止\n")
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(('0.0.0.0', PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
        server.server_close()
