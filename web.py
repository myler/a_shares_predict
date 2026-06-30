#!/usr/bin/env python3
"""
MACD Web 界面
用法: ./web.py [端口]    默认 8080
     nohup ./web.py > web.log 2>&1 &
"""
import sys, os, json, base64, io, urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

# ── 复用 run.py 的分析逻辑 ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import (
    fetch_kline, get_name, calc_macd, detect_regime,
    find_divergences, zero_line_cycles, backtest, predict,
    fetch_dividends, enrich_trades_with_dividends, save_stock_name,
)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── 字体 ──
ROOT = os.path.dirname(os.path.abspath(__file__))
import matplotlib.font_manager as fm

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

# ═══════════════════════════
# 图表生成 (返回 base64)
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

    # Legend with all markers
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
# HTML 模板
# ═══════════════════════════
PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>A股预测</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect x='4' y='4' width='3' height='24' fill='%23ef5350'/><rect x='10' y='10' width='3' height='18' fill='%234caf50'/><rect x='16' y='7' width='3' height='21' fill='%234caf50'/><rect x='22' y='13' width='3' height='15' fill='%23ef5350'/><rect x='28' y='16' width='3' height='12' fill='%234caf50'/></svg>">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,system-ui,sans-serif;background:#f5f5f5;color:#333;padding:20px}
h1{font-size:20px;margin-bottom:16px}
form{display:flex;flex-direction:column;gap:8px;margin-bottom:20px;max-width:400px}
input{width:100%;padding:10px 14px;font-size:16px;border:2px solid #ddd;border-radius:6px;outline:none}
input:focus{border-color:#1565C0}
button{padding:10px 24px;font-size:16px;background:#1565C0;color:#fff;border:none;border-radius:6px;cursor:pointer}
button:hover{background:#0D47A1}
button:disabled{background:#90CAF9;cursor:wait}
label.holding{display:flex;align-items:center;gap:8px;font-size:15px;cursor:pointer;padding:4px 0}
label.holding input{width:auto}
.result{background:#fff;border-radius:8px;padding:20px;box-shadow:0 1px 3px rgba(0,0,0,.1)}
.result h2{font-size:18px;margin-bottom:12px}
.result img{max-width:100%;margin-top:16px;border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,.12)}
.error{background:#fff0f0;color:#c62828;padding:16px;border-radius:6px;border:1px solid #ffcdd2}
.loading{text-align:center;padding:40px;color:#666}
.footer{margin-top:20px;font-size:12px;color:#999;text-align:center}
.conclusion{padding:14px 20px;margin:10px 0;border-radius:6px;font-size:20px;font-weight:bold;text-align:center;letter-spacing:4px}
.buy{background:#e8f5e9;color:#2e7d32;border:2px solid #4caf50}
.sell{background:#fce4ec;color:#c62828;border:2px solid #ef5350}
.warn{background:#fff8e1;color:#f57f17;border:2px solid #ffc107}
.top-row{display:flex;gap:20px;margin-bottom:16px}
.top-row>*{flex:1;min-width:0}
table{border-collapse:collapse;width:100%;margin:0;font-size:13px}
table.overview td{padding:8px 12px;border-bottom:1px solid #eee}
table.overview td:first-child{font-weight:bold;color:#666;width:120px}
table.overview th{background:#1565C0;color:#fff;padding:10px;text-align:center;font-size:15px;border-radius:6px 6px 0 0}
table.trades th{background:#f5f5f5;padding:8px 10px;text-align:left;border-bottom:2px solid #ddd}
table.trades td{padding:8px 10px;border-bottom:1px solid #f0f0f0}
table.trades tr.win{background:#f1f8e9}
table.trades tr.loss{background:#fff3f0}
td.pnl{font-weight:bold}
tr.win td.pnl{color:#2e7d32}
tr.loss td.pnl{color:#c62828}
td.reason{color:#888;font-size:12px}
table.pred{width:100%}
table.pred th{background:#f5f5f5;padding:6px 10px;text-align:left;border-bottom:2px solid #ddd;font-size:12px}
table.pred td{padding:6px 10px;border-bottom:1px solid #f5f5f5}
table.pred td:first-child{font-weight:bold;color:#555;width:120px}
h3{margin:16px 0 8px;font-size:16px}
h4{margin:0 0 4px;font-size:13px;color:#555}
.strategy{background:#fff;border-radius:8px;padding:16px 20px;margin-top:20px;box-shadow:0 1px 3px rgba(0,0,0,.1)}
.strategy h3{font-size:15px;margin-bottom:8px}
.strategy pre{background:#fafafa;padding:12px;border-radius:4px;font-size:12px;line-height:1.8;overflow-x:auto}
</style>
</head>
<body>
<h1>📊 A股预测</h1>
<form onsubmit="analyze(event)">
  <input id="code" type="text" placeholder="输入股票代码，如 603893" autofocus required>
  <div style="margin:8px 0">
    <select id="strategy" onchange="onStrategyChange()" style="padding:8px 12px;font-size:16px;border:2px solid #ddd;border-radius:6px;background:#fff">
      <option value="macd">MACD择时策略</option>
      <option value="buyhold">长线持有策略</option>
    </select>
  </div>
  <label class="holding">
    <input type="checkbox" id="holding" onchange="toggleDividend()"> 已持仓
  </label>
  <label class="holding" id="divLabel" style="display:none">
    <input type="checkbox" id="calcDividend"> 计算分红（含每股分红收益）
  </label>
  <button id="btn" type="submit">分析</button>
</form>
<div id="result"></div>
<div class="strategy">
  <h3>📐 策略公式</h3>
  <pre>买入信号 = 零轴上金叉(DIF>0 ∧ DIF↑ ∧ 确认2天) ∨ 底背离(股价↓ DIF↑)
卖出信号 = 顶背离(股价↑ DIF↓) ∨ (熊市 ∧ 死叉)
牛熊判定 = 价格>MA60 ∧ MA60(10日斜率)>0 → 牛，连续5日确认
评分    = 金叉+2 死叉-2 | 零轴上+1 零轴下-1 | 牛+2 熊-2 | DIF↑+1 DIF↓-1 | 顶背离-3
结论    = 评分≥2→买入/拿住 | 评分≥0→中性/减持 | 评分≥-2→不买/减持 | 评分&lt;-2→卖出</pre>
</div>
<script>
function getStrategy(){
  return document.getElementById('strategy').value;
}
function onStrategyChange(){
  const s=getStrategy();
  const holding=document.getElementById('holding');
  if(s==='buyhold'){
    // 长线持有必须已持仓
    holding.checked=true;
    holding.disabled=true;
    document.getElementById('divLabel').style.display='flex';
  }else{
    holding.disabled=false;
    toggleDividend();
  }
}
function toggleDividend(){
  const holding=document.getElementById('holding').checked;
  document.getElementById('divLabel').style.display=holding?'flex':'none';
  if(!holding)document.getElementById('calcDividend').checked=false;
}
async function analyze(e){
  e.preventDefault();
  const code=document.getElementById('code').value.trim();
  const holding=document.getElementById('holding').checked;
  const calcDividend=document.getElementById('calcDividend').checked;
  const strategy=getStrategy();
  if(!code)return;
  if(strategy==='buyhold' && !holding){
    alert('长线持有策略需要先勾选"已持仓"');
    return;
  }
  const btn=document.getElementById('btn');
  const result=document.getElementById('result');
  btn.disabled=true;btn.textContent='分析中...';
  result.innerHTML='<div class=loading>⏳ 正在拉取数据并计算...</div>';
  try{
    const url='/analyze?code='+encodeURIComponent(code)+'&holding='+(holding?'1':'0')+'&dividend='+(calcDividend?'1':'0')+'&strategy='+strategy;
    const resp=await fetch(url);
    if(!resp.ok){const t=await resp.text();result.innerHTML='<div class=error>'+t+'</div>'}
    else{result.innerHTML=await resp.text()}
  }catch(err){result.innerHTML='<div class=error>请求失败: '+err.message+'</div>'}
  btn.disabled=false;btn.textContent='分析'
}
</script>
</body>
</html>"""

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
        
        # ── 分红数据 ──
        dividends = []
        if calc_dividend:
            dividends = fetch_dividends(code)
            if dividends:
                trades = enrich_trades_with_dividends(trades, dividends)
        
        # ── 结论 ──
        conclusion_html = ''
        for line in pred:
            s = line.strip()
            if s.startswith('✅') or s.startswith('❌') or s.startswith('⚠'):
                cls = 'buy' if '✅' in s else ('sell' if '❌' in s else 'warn')
                conclusion_html = f'<div class="conclusion {cls}">{s.replace(" ","&nbsp;")}</div>'
        
        # ── 概览表格 ──
        wins = [t for t in trades if t['profit_pct'] > 0]
        total_pnl = sum(t['profit_pct'] for t in trades) if trades else 0
        regime_label = '🟢 牛市' if regimes[-1] == 'bull' else '🔴 熊市'
        
        # 年化计算
        from datetime import date as dt_date
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
            # 最近12个月股息率（滚动）
            from datetime import timedelta
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
        
        # ── 预测表格 ──
        pred_table = self.make_pred_table(pred)
        
        # ── 概览+预测 并排 ──
        top_section = f'<div class="top-row"><div>{overview}</div><div>{pred_table}</div></div>'
        
        # ── 交易明细表 ──
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
        
        # ── 分红历史 ──
        dividend_history = ''
        if calc_dividend and dividends:
            recent_divs = [d for d in dividends if d.get('ex_date', '') >= data[0]['day']]
            if recent_divs:
                # 找每年均价用于算股息率
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
        
        # ── 图表 ──
        img_b64 = make_chart(code, name, dates, closes, dif, dea, bar, tops, bottoms, trades, regimes, ma60)
        
        return f"""
        <div class="result">
          {conclusion_html}
          {top_section}
          {dividend_history}
          {trade_table}
          <img src="data:image/png;base64,{img_b64}" alt="MACD Chart" loading="lazy">
        </div>"""
    
    def make_pred_table(self, pred_lines):
        """把预测纯文本转成表格"""
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
    
    def run_buyhold_analysis(self, code, calc_dividend=False):
        """长线持有策略：从数据第一天买入持有至今"""
        data = fetch_kline(code)
        name = get_name(code)
        save_stock_name(code, name)
        closes = np.array([float(d['close']) for d in data])
        
        first_date = data[0]['day']
        first_close = closes[0]
        last_date = data[-1]['day']
        last_close = closes[-1]
        
        from datetime import date as dt_date, timedelta
        d0 = datetime.strptime(first_date, '%Y-%m-%d').date()
        d1 = datetime.strptime(last_date, '%Y-%m-%d').date()
        years = max((d1 - d0).days / 365.25, 0.01)
        
        price_return = (last_close - first_close) / first_close * 100
        
        # 分红
        dividends = []
        total_div = 0
        if calc_dividend:
            dividends = fetch_dividends(code)
            total_div = sum(d['dividend_per_share'] for d in dividends if d.get('ex_date','') >= first_date)
        div_yield = total_div / first_close * 100
        total_return = price_return + div_yield
        
        # 现价股息率
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
        
        # 分红历史
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
        
        # 简化图表：只有收盘价
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
    
    def log_message(self, format, *args):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {args[0]}", flush=True)

# ═══════════════════════════
# MAIN
# ═══════════════════════════
if __name__ == '__main__':
    print(f"MACD Web 启动 → http://localhost:{PORT}")
    print("Ctrl+C 停止\n")
    server = HTTPServer(('0.0.0.0', PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
        server.server_close()
