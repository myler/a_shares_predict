/**
 * 主分析页 — 融合策略分析 + 长线持有
 * 全部本地计算，无后端依赖
 */

const fetcher = require('../../utils/fetcher');
const engine = require('../../utils/engine');

Page({
  data: {
    code: '',
    strategy: 'comprehensive',
    holding: false,
    dividend: false,
    chartTab: 0,
    tradesOpen: false,
    formulaOpen: false,
    dimOpen: { macd: false, mf: false, fund: false, game: false },
    loading: false,
    error: '',
    result: null,
  },

  // ── 交互事件 ──
  onCodeInput(e) { this.setData({ code: e.detail.value }); },
  onStrategy(e) { this.setData({ strategy: e.currentTarget.dataset.s }); },

  onToggleFormula() { this.setData({ formulaOpen: !this.data.formulaOpen }); },

  onToggleDim(e) {
    const dim = e.currentTarget.dataset.dim;
    const key = `dimOpen.${dim}`;
    this.setData({ [key]: !this.data.dimOpen[dim] });
  },

  onToggleHolding() {
    const v = !this.data.holding;
    this.setData({ holding: v });
    if (this.data.result) this.recalc();
  },

  onToggleDividend() {
    const v = !this.data.dividend;
    this.setData({ dividend: v });
    if (this.data.result) this.recalc();
  },

  onChartTab(e) {
    const t = +e.currentTarget.dataset.t;
    this.setData({ chartTab: t }, () => {
      if (this._rawData) this.drawAllCharts(this._rawData);
    });
  },

  onToggleTrades() {
    this.setData({ tradesOpen: !this.data.tradesOpen });
  },

  // ── 主分析 ──
  async onAnalyze() {
    const { code, strategy, holding, dividend } = this.data;
    if (!code || code.length !== 6) {
      wx.showToast({ title: '请输入6位代码', icon: 'none' });
      return;
    }

    this.setData({ loading: true, error: '', result: null, tradesOpen: false, chartTab: 0 });

    try {
      const data = await fetcher.fetchKline(code);
      const name = await fetcher.getStockName(code);

      if (!data || data.length < 60) {
        throw new Error('K线数据不足（需要至少60个交易日）');
      }

      const raw = {
        code, name,
        dates: data.map(d => d.day),
        closes: data.map(d => d.close),
        highs: data.map(d => d.high),
        lows: data.map(d => d.low),
        volumes: data.map(d => d.volume),
      };
      this._rawData = raw;

      // 分红数据
      let dividends = [];
      if (dividend) {
        try { dividends = await fetcher.fetchDividends(code); } catch (e) {}
      }

      const result = this.buildResult(raw, strategy, holding, dividends);
      this.setData({ loading: false, result }, () => {
        this.drawAllCharts(raw);
      });

    } catch (e) {
      console.error('分析失败:', e);
      this.setData({ loading: false, error: e.message || '分析失败', result: null });
    }
  },

  recalc() {
    if (!this._rawData) return;
    const result = this.buildResult(this._rawData, this.data.strategy,
                                     this.data.holding, this._dividendsCache || []);
    this.setData({ result });
  },

  // ── 构建结果 ──
  buildResult(raw, strategy, holding, dividends) {
    const { code, name, dates, closes, highs, lows, volumes } = raw;

    if (strategy === 'buyhold') {
      const bhr = engine.analyzeBuyhold(dates, closes, dividends);
      bhr.strategy = 'buyhold';
      bhr.code = code; bhr.name = name;
      bhr.conclusion = '📊 长线持有回测';
      bhr.cls = 'buy';
      return bhr;
    }

    // 融合策略
    const pred = engine.predictComprehensive(dates, closes, highs, lows, volumes, holding);
    let trades = engine.backtestComprehensive(dates, closes, highs, lows, volumes);

    // 分红增强
    if (dividends.length > 0 && trades.length > 0) {
      trades = this.enrichTrades(trades, dividends, closes);
    }

    const wins = trades.filter(t => t.profit_pct > 0);
    const totalPnl = trades.reduce((s, t) => s + t.profit_pct, 0);
    const totalDiv = trades.reduce((s, t) => s + (t.dividend_total || 0), 0);

    // 结论横幅
    let cls = 'warn';
    if (pred.action === '买入' || pred.action === '增持' || pred.action === '拿住') cls = 'buy';
    else if (pred.action === '卖出') cls = 'sell';

    const n = Math.max(1, Math.min(10, Math.floor(pred.composite / 10)));
    const marks = pred.action === '买入' ? '✅'.repeat(n)
      : pred.action === '卖出' ? '❌'.repeat(n)
      : '⚠️'.repeat(Math.max(1, n));

    // 机构参与度解读
    let instLabel = '';
    if (pred.institution_proxy) {
      const ip = pred.institution_proxy;
      const v = ip.vol60d || 20;
      if (v < 20) instLabel = '🏛️ 机构主导 (低波动)';
      else if (v < 30) instLabel = '🤝 均衡型';
      else instLabel = '👤 散户活跃 (高波动)';
      if (ip.type === '机构吸筹' || ip.type === '强吸筹') {
        instLabel += ' ⚡' + ip.type + '信号';
      }
    }

    // 近12月股息率
    let latestDivYield = 0;
    let recentDivs = [];
    if (dividends.length > 0) {
      const lastDate = new Date(dates[dates.length - 1]);
      const oneYearAgo = new Date(lastDate.getTime() - 365 * 86400000);
      recentDivs = dividends.filter(d => new Date(d.ex_date) >= oneYearAgo);
      const recentTotal = recentDivs.reduce((s, d) => s + (d.dividend_per_share || 0), 0);
      latestDivYield = +(recentTotal / closes[closes.length - 1] * 100).toFixed(1);
    }

    // 总收益（含分红）
    let totalReturnPct = totalPnl;
    if (totalDiv > 0) {
      const divYieldPct = trades.reduce((s, t) => s + (t.dividend_yield_pct || 0), 0);
      totalReturnPct = totalPnl + divYieldPct;
    }

    return {
      ...pred,
      strategy: 'comprehensive',
      conclusion: `${marks} ${pred.action}`,
      cls,
      trades: trades.slice(-30),
      winRate: trades.length > 0 ? Math.round(wins.length / trades.length * 100) : 0,
      totalPnl: +totalPnl.toFixed(1),
      totalDiv: +totalDiv.toFixed(2),
      totalReturnPct: +totalReturnPct.toFixed(1),
      name, code,
      dividends: recentDivs,
      latest_div_yield: latestDivYield,
      institution_label: instLabel,
    };
  },

  // ── 分红增强回测 ──
  enrichTrades(trades, dividends, closes) {
    return trades.map(t => {
      const buyDate = new Date(t.buy_date);
      const sellDate = new Date(t.sell_date);
      let divTotal = 0;
      for (const d of dividends) {
        const exDate = new Date(d.ex_date);
        if (exDate >= buyDate && exDate <= sellDate) {
          divTotal += d.dividend_per_share || 0;
        }
      }
      if (divTotal > 0) {
        const divYieldPct = +(divTotal / t.buy_price * 100).toFixed(1);
        return {
          ...t,
          dividend_total: +divTotal.toFixed(3),
          dividend_yield_pct: divYieldPct,
          total_return_pct: +(t.profit_pct + divYieldPct).toFixed(1),
        };
      }
      return t;
    });
  },

  // ═══════════════════════════
  // Canvas 图表绘制
  // ═══════════════════════════
  drawAllCharts(raw) {
    const { closes, highs, lows, volumes, dates } = raw;
    const trades = (this.data.result && this.data.result.trades) || [];

    // 价量图 (价格+MA20/MA60+买卖点)
    this.drawPriceChart(closes, dates, trades);
    // MACD 图
    this.drawMACDChart(closes, dates, trades);
    // OBV 图
    this.drawOBVChart(closes, volumes, dates, trades);
  },

  drawPriceChart(closes, dates, trades) {
    const query = wx.createSelectorQuery();
    query.select('#priceChart').fields({ node: true, size: true }).exec(res => {
      if (!res[0] || !closes.length) return;
      const canvas = res[0].node, ctx = canvas.getContext('2d');
      const dpr = wx.getSystemInfoSync().pixelRatio;
      const w = res[0].width, h = res[0].height;
      canvas.width = w * dpr; canvas.height = h * dpr;
      ctx.scale(dpr, dpr);

      const step = Math.max(1, Math.floor(closes.length / 300));
      const sd = []; for (let i = 0; i < closes.length; i += step) sd.push(closes[i]);

      const maxP = Math.max(...sd), minP = Math.min(...sd);
      const range = maxP - minP || 1, pad = range * 0.1;
      const toX = i => (i / (sd.length - 1)) * (w - 20) + 10;
      const toY = v => h - 10 - ((v - (minP - pad)) / (range + pad * 2)) * (h - 20);

      // 背景网格
      ctx.strokeStyle = '#f0f0f0'; ctx.lineWidth = 0.5;
      for (let i = 0; i < 5; i++) {
        const y = 10 + (h - 20) * i / 4;
        ctx.beginPath(); ctx.moveTo(10, y); ctx.lineTo(w - 10, y); ctx.stroke();
      }

      // 价格线
      ctx.strokeStyle = '#1565C0'; ctx.lineWidth = 1.5;
      ctx.beginPath();
      sd.forEach((v, i) => { i === 0 ? ctx.moveTo(toX(i), toY(v)) : ctx.lineTo(toX(i), toY(v)); });
      ctx.stroke();

      // MA20
      if (closes.length >= 20) {
        const ma = []; for (let i = 19; i < closes.length; i++) ma.push(engine.arrMean(closes.slice(i - 19, i + 1)));
        ctx.strokeStyle = '#FF6F00'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
        ctx.beginPath();
        for (let i = 0; i < ma.length; i += step) {
          const x = toX(Math.floor(i / step)), y = toY(ma[i]);
          i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke(); ctx.setLineDash([]);
      }

      // MA60
      if (closes.length >= 60) {
        const ma = []; for (let i = 59; i < closes.length; i++) ma.push(engine.arrMean(closes.slice(i - 59, i + 1)));
        ctx.strokeStyle = '#E91E63'; ctx.lineWidth = 0.8; ctx.setLineDash([2, 3]);
        ctx.beginPath();
        for (let i = 0; i < ma.length; i += step) {
          const x = toX(Math.floor(i / step)), y = toY(ma[i]);
          i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke(); ctx.setLineDash([]);
      }

      // 买卖点
      trades.forEach(t => {
        const bi = dates.indexOf(t.buy_date);
        if (bi >= 0 && Math.floor(bi / step) < sd.length) {
          ctx.fillStyle = t.profit_pct > 0 ? '#2e7d32' : '#c62828';
          ctx.beginPath(); ctx.arc(toX(Math.floor(bi / step)), toY(closes[bi]), 4, 0, Math.PI * 2); ctx.fill();
        }
        const si = dates.indexOf(t.sell_date);
        if (si >= 0 && Math.floor(si / step) < sd.length) {
          ctx.fillStyle = '#FF6F00';
          ctx.fillRect(toX(Math.floor(si / step)) - 3, toY(closes[si]) - 3, 6, 6);
        }
      });

      // 标签
      ctx.fillStyle = '#1565C0'; ctx.font = '11px sans-serif';
      ctx.fillText('价格', 10, 14);
      ctx.fillStyle = '#FF6F00'; ctx.fillText('MA20', 50, 14);
      ctx.fillStyle = '#E91E63'; ctx.fillText('MA60', 95, 14);
    });
  },

  drawMACDChart(closes, dates, trades) {
    const query = wx.createSelectorQuery();
    query.select('#macdChart').fields({ node: true, size: true }).exec(res => {
      if (!res[0] || !closes.length) return;
      const canvas = res[0].node, ctx = canvas.getContext('2d');
      const dpr = wx.getSystemInfoSync().pixelRatio;
      const w = res[0].width, h = res[0].height;
      canvas.width = w * dpr; canvas.height = h * dpr;
      ctx.scale(dpr, dpr);

      const { dif, dea, bar } = engine.calcMACD(closes);
      const step = Math.max(1, Math.floor(closes.length / 300));

      const sd = [], dd = [], ed = [], bd = [];
      for (let i = 0; i < closes.length; i += step) {
        sd.push(closes[i]); dd.push(dif[i] || 0); ed.push(dea[i] || 0); bd.push(bar[i] || 0);
      }

      const allVals = [...dd.filter(v => !isNaN(v)), ...ed.filter(v => !isNaN(v)), ...bd.filter(v => !isNaN(v))];
      const maxV = Math.max(...allVals), minV = Math.min(...allVals);
      const range = maxV - minV || 1, pad = range * 0.1;
      const toX = i => (i / (sd.length - 1)) * (w - 20) + 10;
      const toY = v => h / 2 - ((v - (minV - pad)) / (range + pad * 2)) * (h - 20) + 5;

      // 零轴
      const zeroY = toY(0);
      ctx.strokeStyle = '#9e9e9e'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(10, zeroY); ctx.lineTo(w - 10, zeroY); ctx.stroke();

      // BAR
      const barW = (w - 20) / sd.length * 0.8;
      sd.forEach((_, i) => {
        const v = bd[i];
        if (isNaN(v)) return;
        ctx.fillStyle = v >= 0 ? '#ef5350' : '#26a69a';
        const y = toY(v), y0 = zeroY;
        ctx.fillRect(toX(i) - barW / 2, Math.min(y, y0), barW, Math.abs(y - y0));
      });

      // DIF + DEA
      [{ arr: dd, color: '#FFB300', label: 'DIF' },
       { arr: ed, color: '#212121', label: 'DEA' }].forEach(({ arr, color, label }) => {
        ctx.strokeStyle = color; ctx.lineWidth = 1.3;
        ctx.beginPath();
        let started = false;
        arr.forEach((v, i) => {
          if (isNaN(v)) return;
          const x = toX(i), y = toY(v);
          if (!started) { ctx.moveTo(x, y); started = true; }
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      });

      // 买卖点 (只标在 DIF 线上)
      trades.forEach(t => {
        const bi = dates.indexOf(t.buy_date);
        if (bi >= 0 && Math.floor(bi / step) < dd.length) {
          const v = dif[bi]; if (isNaN(v)) return;
          ctx.fillStyle = t.profit_pct > 0 ? '#2e7d32' : '#c62828';
          ctx.beginPath(); ctx.arc(toX(Math.floor(bi / step)), toY(v), 3, 0, Math.PI * 2); ctx.fill();
        }
      });

      ctx.fillStyle = '#FFB300'; ctx.font = '11px sans-serif';
      ctx.fillText('DIF', 10, 14);
      ctx.fillStyle = '#212121'; ctx.fillText('DEA', 42, 14);
    });
  },

  drawOBVChart(closes, volumes, dates, trades) {
    const query = wx.createSelectorQuery();
    query.select('#obvChart').fields({ node: true, size: true }).exec(res => {
      if (!res[0] || !closes.length) return;
      const canvas = res[0].node, ctx = canvas.getContext('2d');
      const dpr = wx.getSystemInfoSync().pixelRatio;
      const w = res[0].width, h = res[0].height;
      canvas.width = w * dpr; canvas.height = h * dpr;
      ctx.scale(dpr, dpr);

      const obv = engine.calcOBV(closes, volumes);
      const step = Math.max(1, Math.floor(closes.length / 300));
      const sd = []; for (let i = 0; i < obv.length; i += step) sd.push(obv[i]);

      const maxV = Math.max(...sd), minV = Math.min(...sd);
      const range = maxV - minV || 1, pad = range * 0.05;
      const toX = i => (i / (sd.length - 1)) * (w - 20) + 10;
      const toY = v => h - 10 - ((v - (minV - pad)) / (range + pad * 2)) * (h - 20);

      // OBV 线
      ctx.strokeStyle = '#7B1FA2'; ctx.lineWidth = 1.5;
      ctx.beginPath();
      sd.forEach((v, i) => { i === 0 ? ctx.moveTo(toX(i), toY(v)) : ctx.lineTo(toX(i), toY(v)); });
      ctx.stroke();

      // OBV MA20
      if (obv.length >= 20) {
        const ma = []; for (let i = 19; i < obv.length; i++) ma.push(engine.arrMean(obv.slice(i - 19, i + 1)));
        ctx.strokeStyle = '#FF6F00'; ctx.lineWidth = 0.8; ctx.setLineDash([4, 4]);
        ctx.beginPath();
        for (let i = 0; i < ma.length; i += step) {
          const x = toX(Math.floor(i / step)), y = toY(ma[i]);
          i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke(); ctx.setLineDash([]);
      }

      // 买卖点
      trades.forEach(t => {
        const bi = dates.indexOf(t.buy_date);
        if (bi >= 0 && Math.floor(bi / step) < sd.length) {
          ctx.fillStyle = t.profit_pct > 0 ? '#2e7d32' : '#c62828';
          ctx.beginPath(); ctx.arc(toX(Math.floor(bi / step)), toY(obv[bi]), 3, 0, Math.PI * 2); ctx.fill();
        }
        const si = dates.indexOf(t.sell_date);
        if (si >= 0 && Math.floor(si / step) < sd.length) {
          ctx.fillStyle = '#FF6F00';
          ctx.fillRect(toX(Math.floor(si / step)) - 3, toY(obv[si]) - 3, 6, 6);
        }
      });

      ctx.fillStyle = '#7B1FA2'; ctx.font = '11px sans-serif';
      ctx.fillText('OBV', 10, 14);
      ctx.fillStyle = '#FF6F00'; ctx.fillText('MA20', 45, 14);
    });
  },
});
