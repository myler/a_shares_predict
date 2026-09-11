#!/usr/bin/env python3
"""Web界面 — 轻量HTTP服务"""

import sys, os, json, base64, io, urllib.parse
from html import escape
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timedelta

PORT = 8080

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── 独立分析入口与长线/财务共享数据 ──
from fetcher import (fetch_kline, get_name, fetch_dividends,
                     fetch_financial_summaries)
from db import save_stock_name
from fundamentals import screen_value_quality
import fusion_web
import pareto_web

import numpy as np

# ── 加载HTML模板 ──
TEMPLATE_PATH = os.path.join(ROOT, 'templates', 'page.html')
with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
    PAGE_TEMPLATE = f.read()


def render_page():
    """首页仅展示模型默认公式，不读取或修改历史运行的冻结参数。"""
    # HTML is a JSON string, not a template literal; prevent script termination
    # even if a future model label contains externally supplied text.
    pareto_formula = json.dumps(pareto_web.render_methodology()).replace('<', '\\u003c').replace('>', '\\u003e')
    return PAGE_TEMPLATE.replace('__PARETO_FORMULA__', pareto_formula)


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

            if strategy not in ('pareto', 'buyhold', 'value'):
                self.send_response(400); self.send_header('Content-type','text/html; charset=utf-8'); self.end_headers()
                self.wfile.write('不支持的策略'.encode()); return

            try:
                if strategy == 'pareto':
                    html = self.run_pareto_analysis(code, holding, calc_dividend)
                elif strategy == 'buyhold':
                    html = self.run_buyhold_analysis(code, calc_dividend)
                elif strategy == 'value':
                    html = self.run_value_analysis(code)
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
                self.wfile.write(f'<div class=error>读取融合策略回测失败: {escape(str(e))}</div>'.encode())
            return

        self.send_response(404); self.end_headers()

    def run_pareto_analysis(self, code, holding=False, calc_dividend=False):
        """刷新本股并分析；历史总览仍由 pareto_web 独立只读渲染。"""
        return fusion_web.render_analysis(code, holding, calc_dividend)

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

    def log_message(self, format, *args):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {args[0]}", flush=True)


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
