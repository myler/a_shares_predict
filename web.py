#!/usr/bin/env python3
"""Web界面 — 轻量HTTP服务"""

import sys, os, json, base64, io, urllib.parse
import inspect
from html import escape
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timedelta

PORT = 8080

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
                    summarize_trades)
from plotting import make_chart, make_comprehensive_chart, plot_multifactor
from scan_composite import run_scan
import pareto_web

import numpy as np

# ── 加载HTML模板 ──
TEMPLATE_PATH = os.path.join(ROOT, 'templates', 'page.html')
with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
    PAGE_TEMPLATE = f.read()


def render_page():
    """首页不读数据；新旧策略说明分别取对应模型默认值。"""
    defaults = inspect.signature(backtest_comprehensive).parameters
    config = {name: defaults[name].default for name in (
        'buy_score', 'stop_loss_pct', 'take_profit_pct',
        'profit_protect_pct', 'profit_protect_score', 'auxiliary_beta',
        'commission_rate', 'stamp_duty_rate',
    )}
    # HTML is a JSON string, not a template literal; prevent script termination
    # even if a future model label contains externally supplied text.
    pareto_formula = json.dumps(pareto_web.render_methodology()).replace('<', '\\u003c').replace('>', '\\u003e')
    return (PAGE_TEMPLATE.replace('__COMPREHENSIVE_CONFIG__', json.dumps(config))
            .replace('__PARETO_FORMULA__', pareto_formula))


PAGE = render_page()


# ═══════════════════════════
# Handler
# ═══════════════════════════
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ('/', '/index.html'):
            self.send_response(200); self.send_header('Content-type','text/html; charset=utf-8'); self.end_headers()
            self.wfile.write(PAGE.encode())
            return

        if parsed.path == '/analyze':
            qs = parsed.query
            params = urllib.parse.parse_qs(qs)
            code = params.get('code', [''])[0].strip()
            holding = params.get('holding', ['0'])[0] == '1'
            calc_dividend = params.get('dividend', ['0'])[0] == '1'
            strategy = params.get('strategy', ['pareto'])[0]

            if not code.isascii() or not code.isdigit() or len(code) != 6:
                self.send_response(400); self.send_header('Content-type','text/html; charset=utf-8'); self.end_headers()
                self.wfile.write('请输入6位股票代码'.encode()); return

            if strategy not in ('pareto', 'comprehensive', 'buyhold', 'value', 'macd', 'multi'):
                self.send_response(400); self.send_header('Content-type','text/html; charset=utf-8'); self.end_headers()
                self.wfile.write('不支持的策略'.encode()); return

            try:
                if strategy == 'pareto':
                    html = self.run_pareto_analysis(code)
                elif strategy == 'buyhold':
                    html = self.run_buyhold_analysis(code, calc_dividend)
                elif strategy == 'value':
                    html = self.run_value_analysis(code)
                elif strategy == 'multi':
                    html = self.run_multifactor_analysis(code, holding, calc_dividend)
                elif strategy == 'comprehensive':
                    html = self.run_comprehensive_analysis(code, holding, calc_dividend)
                elif strategy == 'macd':
                    html = self.run_analysis(code, holding, calc_dividend)
                self.send_response(200); self.send_header('Content-type','text/html; charset=utf-8'); self.end_headers()
                self.wfile.write(html.encode())
            except Exception as e:
                self.send_response(500); self.send_header('Content-type','text/html; charset=utf-8'); self.end_headers()
                self.wfile.write(f'<div class=error>分析失败: {escape(str(e))}</div>'.encode())
            return

        if parsed.path == '/pareto-summary':
            params = urllib.parse.parse_qs(parsed.query)
            try:
                n = max(1, min(100, int(params.get('n', ['5'])[0])))
            except ValueError:
                n = 5
            # Only n is accepted. RUN_DIR is service configuration, not a URL path.
            try:
                html = pareto_web.render_population(n)
                self.send_response(200); self.send_header('Content-type','text/html; charset=utf-8'); self.end_headers()
                self.wfile.write(html.encode())
            except Exception as e:
                self.send_response(500); self.send_header('Content-type','text/html; charset=utf-8'); self.end_headers()
                self.wfile.write(f'<div class=error>读取多维回测失败: {escape(str(e))}</div>'.encode())
            return

        if parsed.path == '/scan':
            qs = parsed.query
            params = urllib.parse.parse_qs(qs)
            try:
                n = int(params.get('n', ['5'])[0])
                max_pe = float(params.get('max_pe', ['100'])[0])
                min_price = float(params.get('min_price', ['5'])[0])
            except ValueError:
                n, max_pe, min_price = 5, 100.0, 5.0
            try:
                html = self.run_scan_html(n, max_pe, min_price)
                self.send_response(200); self.send_header('Content-type','text/html; charset=utf-8'); self.end_headers()
                self.wfile.write(html.encode())
            except Exception as e:
                self.send_response(500); self.send_header('Content-type','text/html; charset=utf-8'); self.end_headers()
                self.wfile.write(f'<div style="color:#c62828;padding:16px">旧加权海选失败: {escape(str(e))}</div>'.encode())
            return

        self.send_response(404); self.end_headers()

    def run_pareto_analysis(self, code):
        """Read completed local artifacts; never fall back to the weighted engine."""
        return pareto_web.render_stock_report(code)

    def run_scan_html(self, n, max_pe, min_price):
        top, stats = run_scan(n, max_pe, min_price)
        rows_html = ''
        for i, r in enumerate(top, 1):
            pct = r.get('pct', 0) or 0
            pct_color = '#2e7d32' if pct >= 0 else '#c62828'
            rows_html += (
                f'<tr><td>{i}</td>'
                f'<td>{escape(str(r["name"]))} <span style="color:#999">({escape(str(r["code"]))})</span></td>'
                f'<td><b>{r["score"]:.1f}</b></td>'
                f'<td>{r["close"]:.2f}</td>'
                f'<td style="color:{pct_color}">{pct:+.2f}%</td>'
                f'<td>{r["pe"]:.0f}</td>'
                f'<td>{r["mcap"]:.0f}亿</td>'
                f'<td>{escape(str(r["date"]))}</td></tr>')
        return (
            '<h2>旧加权海选（非帕累托）</h2>'
            '<p class="method-note">横截面因子加权相对分 Top-N，不是每日 Pareto34 前沿或独立账户回测；此旧入口可能联网取报价。</p>'
            '<table class="scan-table">'
            '<tr><th>#</th><th>股票</th><th>加权相对分</th><th>现价</th><th>涨跌</th><th>PE</th><th>市值</th><th>因子日期</th></tr>'
            + rows_html + '</table>'
            + f'<div class="scan-meta">共 {stats["passed"]} 只通过过滤（{stats["total_codes"]} 只有效因子）| 耗时 {stats["elapsed"]:.0f}秒 | '
            '因子按各股缓存末日计算，未统一日期；现价来自本次报价。相对分不是融合 S*。</div>'
        )

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
                    <p class="hint">当前建议仅按评分和是否持仓映射；“增持/减持”不代表分仓执行，也未结合实际买入成本。</p>
          {overview}
                    <p class="hint">回测为收盘信号、下一交易日开盘成交；汇总仅含已完成交易，不计期末未平仓。权息风控和不可成交约束尚不完整，非完整账户收益。</p>
          <br>{score_card}
          {aux_html}
          {analysis_html}
          {trade_table}
          <img src="data:image/png;base64,{image}" alt="Comprehensive Chart" loading="lazy">
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
                    html += f'<tr><td>价格位置/趋势</td><td>{s["fundamental"]["score"]:.1f} (×0.15)</td></tr>'
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
    PORT = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    print(f"A股策略分析 Web → http://localhost:{PORT}")
    print("Ctrl+C 停止\n")
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(('0.0.0.0', PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
        server.server_close()
