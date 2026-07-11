#!/usr/bin/env python3
"""Web界面 — 轻量HTTP服务"""

import sys, os, json, base64, io, urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timedelta

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── 导入数据层和引擎 ──
from fetcher import fetch_kline, get_name, fetch_dividends, enrich_trades_with_dividends
from db import save_stock_name
from engine import (calc_macd, detect_regime, find_divergences, backtest, predict,
                    backtest_multifactor, predict_multifactor, plot_multifactor,
                    backtest_comprehensive, predict_comprehensive, calc_bollinger, calc_obv)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm

# ── 字体 ──
font_candidates = [
    os.path.join(ROOT, 'wqy-zenhei.ttf'),
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
]
font_ok = False
for fp in font_candidates:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
        try: plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
        except: pass
        font_ok = True
        break
if not font_ok:
    import warnings; warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False

# ── 加载HTML模板 ──
TEMPLATE_PATH = os.path.join(ROOT, 'templates', 'page.html')
with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
    PAGE = f.read()


# ═══════════════════════════
# Web图表生成 (返回base64)
# ═══════════════════════════
def make_chart(code, name, dates, closes, dif, dea, bar, tops, bottoms, trades, regimes, ma60):
    fig = plt.figure(figsize=(18, 12))
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(dates, closes, color='#1565C0', linewidth=1, alpha=0.7, label='收盘价')
    ax1.plot(dates, ma60, color='#FF6F00', linewidth=1.5, alpha=0.6, label='MA60')

    in_bull, bs = False, 0
    bull_added = False
    for i in range(len(regimes)):
        if regimes[i] == 'bull' and not in_bull: bs = i; in_bull = True
        elif regimes[i] == 'bear' and in_bull:
            ax1.axvspan(dates[bs], dates[i-1], alpha=0.12, color='#e8f5e9')
            in_bull = False; bull_added = True
    if in_bull: ax1.axvspan(dates[bs], dates[-1], alpha=0.12, color='#e8f5e9'); bull_added = True

    for t in tops: ax1.scatter(t['date'], t['price'], color='red', s=80, marker='v', zorder=5)
    for b in bottoms: ax1.scatter(b['date'], b['price'], color='green', s=80, marker='^', zorder=5)
    dt_idx = {d.strftime('%Y-%m-%d'): i for i, d in enumerate(dates)}
    for tr in trades:
        bi = dt_idx.get(tr['buy_date']); si = dt_idx.get(tr['sell_date'])
        if bi is not None: ax1.scatter(dates[bi], closes[bi], color='lime', s=120, marker='o', zorder=6, edgecolors='black')
        if si is not None: ax1.scatter(dates[si], closes[si], color='orange', s=120, marker='s', zorder=6, edgecolors='black')

    h1, l1 = ax1.get_legend_handles_labels()
    h1 += [
        mlines.Line2D([],[],color='lime',marker='o',linestyle='',markersize=8,markeredgecolor='black',label='买入'),
        mlines.Line2D([],[],color='orange',marker='s',linestyle='',markersize=8,markeredgecolor='black',label='卖出'),
        mlines.Line2D([],[],color='red',marker='v',linestyle='',markersize=8,label='顶背离'),
        mlines.Line2D([],[],color='green',marker='^',linestyle='',markersize=8,label='底背离'),
    ]
    if bull_added: h1.append(mpatches.Patch(color='#e8f5e9',alpha=0.5,label='牛市'))
    ax1.legend(handles=h1, loc='upper left', fontsize=8, ncol=2)
    ax1.set_title(f'{name}({code})', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3); ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

    ax2 = plt.subplot(3, 1, (2, 3))
    colors = ['#ef5350' if v >= 0 else '#26a69a' for v in bar]
    ax2.bar(dates, bar, color=colors, width=0.8, alpha=0.75, label='BAR')
    ax2.plot(dates, dif, color='#FFB300', linewidth=1.5, label='DIF金线')
    ax2.plot(dates, dea, color='#212121', linewidth=1.5, label='DEA黑线')
    ax2.axhline(y=0, color='#9e9e9e', linewidth=1, label='零轴')
    for t in tops: ax2.scatter(t['date'], t['dif'], color='red', s=60, marker='v', zorder=5)
    for b in bottoms: ax2.scatter(b['date'], b['dif'], color='green', s=60, marker='^', zorder=5)
    for tr in trades:
        bi = dt_idx.get(tr['buy_date']); si = dt_idx.get(tr['sell_date'])
        if bi is not None: ax2.scatter(dates[bi], dif[bi], color='lime', s=100, marker='o', zorder=6, edgecolors='black')
        if si is not None: ax2.scatter(dates[si], dif[si], color='orange', s=100, marker='s', zorder=6, edgecolors='black')

    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(handles=h2 + [
        mlines.Line2D([],[],color='lime',marker='o',linestyle='',markersize=8,markeredgecolor='black',label='买入'),
        mlines.Line2D([],[],color='orange',marker='s',linestyle='',markersize=8,markeredgecolor='black',label='卖出'),
    ], loc='upper left', fontsize=8, ncol=2)
    ax2.set_title('MACD (12,26,9)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3); ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()


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

        self.send_response(404); self.end_headers()

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
        pred = predict(date_strs, closes, dif, dea, bar, regimes, tops, bottoms, holding)

        dividends = []
        if calc_dividend:
            dividends = fetch_dividends(code)
            if dividends:
                trades = enrich_trades_with_dividends(trades, dividends)

        conclusion_html = ''
        for line in pred:
            s = line.strip()
            if s.startswith('✅') or s.startswith('❌') or s.startswith('⚠'):
                cls = 'buy' if '✅' in s else ('sell' if '❌' in s else 'warn')
                conclusion_html = f'<div class="conclusion {cls}">{s.replace(" ","&nbsp;")}</div>'

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

        pred_table = self.make_pred_table(pred)
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

        fig = plt.figure(figsize=(18, 6))
        ax = plt.subplot(1, 1, 1)
        dates = np.array([datetime.strptime(d['day'],'%Y-%m-%d') for d in data])
        ax.plot(dates, closes, color='#1565C0', linewidth=1.5, label='收盘价')
        ax.axhline(y=first_close, color='#FF6F00', linewidth=1, linestyle='--', alpha=0.7, label=f'买入价 {first_close:.2f}')
        ax.set_title(f'{name}({code}) — 长线持有', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        return f"""
        <div class="result">
          {overview}
          {dividend_history}
          <img src="data:image/png;base64,{img_b64}" alt="BuyHold Chart" loading="lazy">
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
        pred = predict_multifactor(dates, closes, highs, lows, vols, holding=holding)
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
        for line in pred:
            s = line.strip()
            if s.startswith('✅') or s.startswith('❌') or s.startswith('⚠'):
                cls = 'buy' if '✅' in s else ('sell' if '❌' in s else 'warn')
                conclusion_html = f'<div class="conclusion {cls}">{s.replace(" ","&nbsp;")}</div>'

        pred_table = self.make_pred_table(pred)
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
        pred = predict_comprehensive(dates, closes, highs, lows, vols, holding=holding)

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
        composite = tech_score * 0.30 + game_score * 0.45 + fund_score * 0.25

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

        overview += f"""
        </table>
        <br>
        <table class="overview">
          <tr><th colspan="3">📊 三面评分卡</th></tr>
          <tr style="background:#e3f2fd"><td><b>技术面 (30%)</b></td><td style="font-weight:bold;font-size:1.2em">{score_level(tech_score)} {tech_score:.1f}</td><td style="font-size:11px;color:#888"><table>{tech_rows}</table></td></tr>
          <tr style="background:#fff3e0"><td><b>博弈面 (45%)</b></td><td style="font-weight:bold;font-size:1.2em">{score_level(game_score)} {game_score:.1f}</td><td style="font-size:11px;color:#888"><table>{game_rows}</table></td></tr>
          <tr style="background:#e8f5e9"><td><b>基本面 (25%)</b></td><td style="font-weight:bold;font-size:1.2em">{score_level(fund_score)} {fund_score:.1f}</td><td style="font-size:11px;color:#888"><table>{fund_rows}</table></td></tr>
          <tr style="background:#f5f5f5"><td><b>综合加权</b></td><td style="font-weight:bold;font-size:1.4em">{score_level(composite)} {composite:.1f}</td><td>技{tech_score:.0f}×0.30 + 博{game_score:.0f}×0.45 + 基{fund_score:.0f}×0.25</td></tr>
        </table>"""

        # 结论横幅
        conclusion_html = ''
        for line in pred:
            s = line.strip()
            if s.startswith('✅') or s.startswith('❌') or s.startswith('⚠'):
                cls = 'buy' if '✅' in s else ('sell' if '❌' in s else 'warn')
                conclusion_html = f'<div class="conclusion {cls}">{s.replace(" ","&nbsp;")}</div>'

        pred_table = self.make_pred_table(pred)
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
        <h3>📈 综合策略交易记录</h3>
        <table class="trades">
          <tr><th>买入日</th><th>卖出日</th><th>买入价</th><th>卖出价</th><th>价差</th>{div_header}<th>持仓</th><th>触发</th></tr>
          {trade_rows}
        </table>""" if trades else '<h3>📈 综合策略交易记录</h3><p>无交易信号</p>'

        # Chart
        dates_dt = np.array([datetime.strptime(d, '%Y-%m-%d') for d in dates])
        img_b64 = self.make_comprehensive_chart(code, name, dates_dt, closes, highs, lows, vols, trades)

        return f"""
        <div class="result">
          {conclusion_html}
          {top_section}
          {trade_table}
          <img src="data:image/png;base64,{img_b64}" alt="Comprehensive Chart" loading="lazy">
        </div>"""

    def make_comprehensive_chart(self, code, name, dates, closes, highs, lows, vols, trades):
        """综合策略图表：价格+MACD+OBV（优化：降DPI+降采样）"""
        fig = plt.figure(figsize=(14, 8))
        import matplotlib.lines as mlines

        # 降采样：超过1000个点就每隔N个取一个
        step = max(1, len(dates) // 800)
        d_dates = dates[::step]
        d_closes = closes[::step]
        
        dif, dea, bar = calc_macd(closes)
        obv = calc_obv(closes, vols)
        d_dif = dif[::step]; d_dea = dea[::step]; d_bar = bar[::step]
        d_obv = obv[::step]
        
        # 重新映射买卖点索引到降采样后的位置
        dt_idx = {}
        for i, d in enumerate(dates):
            if i % step == 0:
                dt_idx[d.strftime('%Y-%m-%d')] = i // step

        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(d_dates, d_closes, color='#1565C0', linewidth=1, alpha=0.8, label='收盘价')
        ma20 = np.array([np.mean(closes[max(0,i-19):i+1]) for i in range(0, len(closes), step)])
        ma60 = np.array([np.mean(closes[max(0,i-59):i+1]) for i in range(0, len(closes), step)])
        ax1.plot(d_dates, ma20, color='#FF6F00', linewidth=1, alpha=0.6, label='MA20')
        ax1.plot(d_dates, ma60, color='#E91E63', linewidth=1, alpha=0.5, label='MA60')
        for t in trades:
            bi = dt_idx.get(t['buy_date']); si = dt_idx.get(t['sell_date'])
            if bi is not None: ax1.scatter(d_dates[bi], d_closes[bi], color='lime', s=100, marker='o', zorder=6, edgecolors='black')
            if si is not None: ax1.scatter(d_dates[si], d_closes[si], color='orange', s=100, marker='s', zorder=6, edgecolors='black')
        h1, _ = ax1.get_legend_handles_labels()
        h1 += [mlines.Line2D([],[],color='lime',marker='o',linestyle='',markersize=8,markeredgecolor='black',label='买入'),
               mlines.Line2D([],[],color='orange',marker='s',linestyle='',markersize=8,markeredgecolor='black',label='卖出')]
        ax1.legend(handles=h1, loc='upper left', fontsize=7, ncol=2)
        ax1.set_title(f'{name}({code}) — 三合资本综合策略', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3); ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

        ax2 = plt.subplot(3, 1, 2)
        colors_bar = ['#ef5350' if v >= 0 else '#26a69a' for v in d_bar]
        ax2.bar(d_dates, d_bar, color=colors_bar, width=0.8, alpha=0.75, label='BAR')
        ax2.plot(d_dates, d_dif, color='#FFB300', linewidth=1.5, label='DIF')
        ax2.plot(d_dates, d_dea, color='#212121', linewidth=1.5, label='DEA')
        ax2.axhline(y=0, color='#9e9e9e', linewidth=1, label='零轴')
        for t in trades:
            bi = dt_idx.get(t['buy_date']); si = dt_idx.get(t['sell_date'])
            if bi is not None: ax2.scatter(d_dates[bi], d_dif[bi], color='lime', s=80, marker='o', zorder=6, edgecolors='black')
            if si is not None: ax2.scatter(d_dates[si], d_dif[si], color='orange', s=80, marker='s', zorder=6, edgecolors='black')
        ax2.legend(loc='upper left', fontsize=7, ncol=2)
        ax2.set_title('MACD (12,26,9)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3); ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

        ax3 = plt.subplot(3, 1, 3)
        ax3.plot(d_dates, d_obv, color='#7B1FA2', linewidth=1, label='OBV')
        obv_ma20_arr = np.array([np.mean(obv[max(0,i-19):i+1]) for i in range(0, len(obv), step)])
        ax3.plot(d_dates, obv_ma20_arr, color='#FF6F00', linewidth=0.8, alpha=0.6, linestyle='--', label='OBV MA20')
        for t in trades:
            bi = dt_idx.get(t['buy_date']); si = dt_idx.get(t['sell_date'])
            if bi is not None: ax3.scatter(d_dates[bi], d_obv[bi], color='lime', s=80, marker='o', zorder=6, edgecolors='black')
            if si is not None: ax3.scatter(d_dates[si], d_obv[si], color='orange', s=80, marker='s', zorder=6, edgecolors='black')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.set_title('OBV 能量潮 (量价背离检测)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3); ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=90, bbox_inches='tight')
        plt.close()
        return base64.b64encode(buf.getvalue()).decode()

    def log_message(self, format, *args):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {args[0]}", flush=True)

    def make_pred_table(self, pred_lines):
        html = '<table class="overview"><tr><th colspan="2">🔮 预测分析</th></tr>'
        for line in pred_lines:
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
    print(f"A股策略分析 Web → http://localhost:{PORT}")
    print("Ctrl+C 停止\n")
    server = HTTPServer(('0.0.0.0', PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
        server.server_close()
