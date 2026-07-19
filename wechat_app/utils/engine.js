/**
 * 三合资本 · 核心引擎 (JavaScript 版) v2
 * 
 * 从 engine.py 一对一翻译，纯 JS 无依赖。
 * v2: 评分函数返回明细 breakdown
 */

// ═══════════════════════════
// Array 工具
// ═══════════════════════════

function arrMean(arr) {
  if (arr.length === 0) return NaN;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function arrStd(arr) {
  if (arr.length === 0) return NaN;
  const mean = arrMean(arr);
  return Math.sqrt(arrMean(arr.map(v => (v - mean) ** 2)));
}

function arrMax(arr) { return Math.max(...arr); }
function arrMin(arr) { return Math.min(...arr); }


// ═══════════════════════════
// EMA / MACD
// ═══════════════════════════

function ema(data, period) {
  const n = data.length;
  const result = new Array(n).fill(NaN);
  if (n < period) return result;
  result[period - 1] = arrMean(data.slice(0, period));
  const m = 2 / (period + 1);
  for (let i = period; i < n; i++) result[i] = data[i] * m + result[i - 1] * (1 - m);
  return result;
}

function calcMACD(closes, fast = 12, slow = 26, signal = 9) {
  const ef = ema(closes, fast);
  const es = ema(closes, slow);
  const dif = ef.map((v, i) => v - es[i]);
  const dea = new Array(closes.length).fill(NaN);
  const slowStart = slow - 1;
  if (closes.length > slowStart) {
    const difSlice = dif.slice(slowStart);
    const deaSlice = ema(difSlice, signal);
    for (let i = 0; i < deaSlice.length; i++) dea[slowStart + i] = deaSlice[i];
  }
  const bar = dif.map((v, i) => (v - dea[i]) * 2);
  return { dif, dea, bar };
}


// ═══════════════════════════
// 技术指标
// ═══════════════════════════

function calcRSI(closes, period = 14) {
  const n = closes.length;
  const rsi = new Array(n).fill(NaN);
  if (n < period + 1) return rsi;
  const gains = [0], losses = [0];
  for (let i = 1; i < n; i++) {
    const diff = closes[i] - closes[i - 1];
    gains.push(Math.max(diff, 0));
    losses.push(Math.max(-diff, 0));
  }
  let avgGain = arrMean(gains.slice(1, period + 1));
  let avgLoss = arrMean(losses.slice(1, period + 1));
  rsi[period] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
  for (let i = period + 1; i < n; i++) {
    avgGain = (avgGain * (period - 1) + gains[i]) / period;
    avgLoss = (avgLoss * (period - 1) + losses[i]) / period;
    rsi[i] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
  }
  return rsi;
}

function calcKDJ(highs, lows, closes, n = 9, m1 = 3, m2 = 3) {
  const len = closes.length;
  const k = new Array(len).fill(NaN);
  const d = new Array(len).fill(NaN);
  const j = new Array(len).fill(NaN);
  if (len < n) return { k, d, j };
  for (let i = n - 1; i < len; i++) {
    const hh = arrMax(highs.slice(i - n + 1, i + 1));
    const ll = arrMin(lows.slice(i - n + 1, i + 1));
    const rsv = hh !== ll ? (closes[i] - ll) / (hh - ll) * 100 : 50;
    if (i === n - 1) { k[i] = 50; d[i] = 50; }
    else {
      k[i] = rsv / m1 + k[i - 1] * (m1 - 1) / m1;
      d[i] = k[i] / m2 + d[i - 1] * (m2 - 1) / m2;
    }
    j[i] = 3 * k[i] - 2 * d[i];
  }
  return { k, d, j };
}

function calcBollinger(closes, period = 20, nbdev = 2) {
  const n = closes.length;
  const upper = new Array(n).fill(NaN);
  const middle = new Array(n).fill(NaN);
  const lower = new Array(n).fill(NaN);
  if (n < period) return { upper, middle, lower };
  for (let i = period - 1; i < n; i++) {
    const w = closes.slice(i - period + 1, i + 1);
    middle[i] = arrMean(w);
    const std = arrStd(w);
    upper[i] = middle[i] + nbdev * std;
    lower[i] = middle[i] - nbdev * std;
  }
  return { upper, middle, lower };
}

function calcWR(highs, lows, closes, period = 10) {
  const n = closes.length;
  const wr = new Array(n).fill(NaN);
  if (n < period) return wr;
  for (let i = period - 1; i < n; i++) {
    const hh = arrMax(highs.slice(i - period + 1, i + 1));
    const ll = arrMin(lows.slice(i - period + 1, i + 1));
    wr[i] = hh !== ll ? (hh - closes[i]) / (hh - ll) * 100 : 50;
  }
  return wr;
}

function calcOBV(closes, volumes) {
  const n = closes.length;
  const obv = new Array(n).fill(0);
  obv[0] = volumes[0];
  for (let i = 1; i < n; i++) {
    if (closes[i] > closes[i - 1]) obv[i] = obv[i - 1] + volumes[i];
    else if (closes[i] < closes[i - 1]) obv[i] = obv[i - 1] - volumes[i];
    else obv[i] = obv[i - 1];
  }
  return obv;
}


// ═══════════════════════════
// 综合策略四维评分 (v2: 含明细)
// ═══════════════════════════

function scoreComprehensive(closes, highs, lows, volumes,
                            dif, dea, bar, rsiArr, kArr, dArr, jArr,
                            bbU, bbL, wrArr, obvFull, i) {
  // ── MACD核心 40% ──
  const macdItems = [{ rule: '基准分', adj: '50' }];
  let macdScore = 50;

  if (dif[i] > dea[i]) { macdScore += 15; macdItems.push({ rule: 'DIF > DEA 金叉', adj: '+15' }); }
  else { macdScore -= 15; macdItems.push({ rule: 'DIF < DEA 死叉', adj: '−15' }); }

  if (dif[i] > 0) { macdScore += 12; macdItems.push({ rule: 'DIF 零轴上 (做多区)', adj: '+12' }); }
  else { macdScore -= 8; macdItems.push({ rule: 'DIF 零轴下 (做空区)', adj: '−8' }); }

  if (i >= 5 && dif[i] > dif[i - 5]) {
    macdScore += 8;
    macdItems.push({ rule: `DIF 5日斜率 ↑ (${(dif[i]-dif[i-5]).toFixed(2)})`, adj: '+8' });
  } else {
    macdScore -= 5; macdItems.push({ rule: 'DIF 5日斜率 ↓', adj: '−5' });
  }

  if (i >= 3 && bar[i] > bar[i - 3]) {
    macdScore += 5;
    macdItems.push({ rule: `BAR 3日趋势 ↑ (${(bar[i]-bar[i-3]).toFixed(2)})`, adj: '+5' });
  } else {
    macdItems.push({ rule: 'BAR 3日无上升趋势', adj: '0' });
  }

  if (i >= 30) {
    const slice30 = closes.slice(Math.max(0, i - 30), i);
    const min30 = arrMin(slice30);
    if (closes[i] > min30 * 1.03 && dif[i] > dif[Math.max(0, i - 30)]) {
      macdScore += 10; macdItems.push({ rule: '底背离 30日 (价>低点 DIF↑)', adj: '+10' });
    } else macdItems.push({ rule: '无底背离信号', adj: '0' });

    const max30 = arrMax(slice30);
    if (closes[i] >= max30 * 0.98 && dif[i] < dif[Math.max(0, i - 30)] * 0.9) {
      macdScore -= 15; macdItems.push({ rule: '顶背离 30日 (价≈高点 DIF↓)', adj: '−15' });
    } else macdItems.push({ rule: '无顶背离信号', adj: '0' });
  }

  // ── 多因子 30% ──
  const mfItems = [{ rule: '基准分', adj: '50' }];
  let mfScore = 50;
  const rv = !isNaN(rsiArr[i]) ? rsiArr[i] : 50;
  const kv = !isNaN(kArr[i]) ? kArr[i] : 50;
  const dv = !isNaN(dArr[i]) ? dArr[i] : 50;
  const jv = !isNaN(jArr[i]) ? jArr[i] : 50;
  const wv = !isNaN(wrArr[i]) ? wrArr[i] : 50;

  if (rv >= 30 && rv <= 65) {
    mfScore += 10; mfItems.push({ rule: `RSI=${Math.round(rv)} 在健康区间30-65`, adj: '+10' });
  } else if (rv < 30) {
    mfScore += 15; mfItems.push({ rule: `RSI=${Math.round(rv)} 超卖 (<30)`, adj: '+15' });
  } else if (rv > 80) {
    mfScore -= 15; mfItems.push({ rule: `RSI=${Math.round(rv)} 超买 (>80)`, adj: '−15' });
  } else if (rv > 70) {
    mfScore -= 8; mfItems.push({ rule: `RSI=${Math.round(rv)} 偏强 (>70)`, adj: '−8' });
  } else {
    mfItems.push({ rule: `RSI=${Math.round(rv)} 中性`, adj: '0' });
  }

  if (kv > dv) {
    mfScore += 10; mfItems.push({ rule: `KDJ 金叉 (K=${Math.round(kv)}>D=${Math.round(dv)})`, adj: '+10' });
  } else if (kv < dv) {
    mfScore -= 8; mfItems.push({ rule: `KDJ 死叉 (K=${Math.round(kv)}<D=${Math.round(dv)})`, adj: '−8' });
  } else {
    mfItems.push({ rule: 'KDJ 无交叉', adj: '0' });
  }

  if (jv < 0) { mfScore += 8; mfItems.push({ rule: `J=${Math.round(jv)}<0 超卖`, adj: '+8' }); }
  else if (jv > 100) { mfScore -= 8; mfItems.push({ rule: `J=${Math.round(jv)}>100 超买`, adj: '−8' }); }
  else { mfItems.push({ rule: `J=${Math.round(jv)} 正常`, adj: '0' }); }

  const bbPos = (!isNaN(bbU[i]) && bbU[i] !== bbL[i])
    ? (closes[i] - bbL[i]) / (bbU[i] - bbL[i]) * 100 : 50;
  if (bbPos < 10) {
    mfScore += 12; mfItems.push({ rule: `布林位置${Math.round(bbPos)}% 下轨 (<10%)`, adj: '+12' });
  } else if (bbPos > 90) {
    mfScore -= 8; mfItems.push({ rule: `布林位置${Math.round(bbPos)}% 上轨 (>90%)`, adj: '−8' });
  } else {
    mfItems.push({ rule: `布林位置${Math.round(bbPos)}% 中轨区间`, adj: '0' });
  }

  if (wv > 80) { mfScore += 8; mfItems.push({ rule: `WR=${Math.round(wv)} 超卖 (>80)`, adj: '+8' }); }
  else if (wv < 20) { mfScore -= 8; mfItems.push({ rule: `WR=${Math.round(wv)} 超买 (<20)`, adj: '−8' }); }
  else { mfItems.push({ rule: `WR=${Math.round(wv)} 中性`, adj: '0' }); }

  // ── 基本面 15% ──
  const fundItems = [{ rule: '基准分', adj: '50' }];
  let fundScore = 50;

  if (i >= 249) {
    const h250 = arrMax(closes.slice(i - 249, i + 1));
    const l250 = arrMin(closes.slice(i - 249, i + 1));
    const pos250 = h250 !== l250 ? (closes[i] - l250) / (h250 - l250) * 100 : 50;
    if (pos250 < 25) {
      fundScore += 20; fundItems.push({ rule: `250日位置${Math.round(pos250)}% 低估 (<25%)`, adj: '+20' });
    } else if (pos250 < 40) {
      fundScore += 10; fundItems.push({ rule: `250日位置${Math.round(pos250)}% 偏低 (<40%)`, adj: '+10' });
    } else if (pos250 > 80) {
      fundScore -= 15; fundItems.push({ rule: `250日位置${Math.round(pos250)}% 高估 (>80%)`, adj: '−15' });
    } else {
      fundItems.push({ rule: `250日位置${Math.round(pos250)}% 适中`, adj: '0' });
    }
  } else fundItems.push({ rule: '250日数据不足', adj: '0' });

  if (i >= 250) {
    const yoy = (closes[i] - closes[i - 250]) / closes[i - 250] * 100;
    if (yoy > 20) {
      fundScore += 10; fundItems.push({ rule: `年涨幅${yoy.toFixed(0)}% 高增长 (>20%)`, adj: '+10' });
    } else if (yoy < -20) {
      fundScore -= 10; fundItems.push({ rule: `年涨幅${yoy.toFixed(0)}% 衰退 (<−20%)`, adj: '−10' });
    } else fundItems.push({ rule: `年涨幅${yoy.toFixed(0)}% 正常`, adj: '0' });
  } else fundItems.push({ rule: '年增长数据不足', adj: '0' });

  // ── 量能 15% ──
  const gameItems = [{ rule: '基准分', adj: '50' }];
  let gameScore = 50;

  const obvSlice = i >= 20 ? obvFull.slice(Math.max(0, i - 20), i) : obvFull.slice(0, i);
  const obvMA20 = obvSlice.length > 0 ? arrMean(obvSlice) : 1;
  const obvMA5 = arrMean(obvFull.slice(Math.max(0, i - 4), i + 1));
  const obvRatio = obvMA20 > 0 ? obvFull[i] / obvMA20 : 1;

  if (obvMA5 > obvMA20 * 1.08) {
    gameScore += 12; gameItems.push({ rule: `OBV 短期上行 (MA5/MA20=${obvRatio.toFixed(2)})`, adj: '+12' });
  } else if (obvMA5 < obvMA20 * 0.92) {
    gameScore -= 12; gameItems.push({ rule: `OBV 短期下行 (MA5/MA20=${obvRatio.toFixed(2)})`, adj: '−12' });
  } else {
    gameItems.push({ rule: `OBV 趋势平稳 (MA5/MA20=${obvRatio.toFixed(2)})`, adj: '0' });
  }

  const chg5 = closes[i] - closes[Math.max(0, i - 5)];
  const obvChg5 = obvFull[i] - obvFull[Math.max(0, i - 5)];
  if (chg5 > 0 && obvChg5 < 0) {
    gameScore -= 18; gameItems.push({ rule: `量价背离 (价涨${chg5.toFixed(2)} OBV跌)`, adj: '−18' });
  } else if (chg5 < 0 && obvChg5 > 0) {
    gameScore += 12; gameItems.push({ rule: `底部吸筹 (价跌${chg5.toFixed(2)} OBV涨)`, adj: '+12' });
  } else {
    gameItems.push({ rule: '无量价背离', adj: '0' });
  }

  // 机构参与度代理
  let instAgent = null;
  if (i >= 60) {
    const ret5d = (closes[i] - closes[Math.max(0, i - 5)]) / closes[Math.max(0, i - 5)] * 100;
    const ret20d = (closes[i] - closes[Math.max(0, i - 20)]) / closes[Math.max(0, i - 20)] * 100;
    const returns = [];
    for (let j = Math.max(1, i - 60); j <= i; j++)
      returns.push((closes[j] - closes[j - 1]) / closes[j - 1]);
    const vol60d = arrStd(returns) * 100;

    if (ret5d < -3 && vol60d < 25) {
      instAgent = { type: '机构吸筹', strength: 1, ret5d, vol60d };
      gameScore += 10;
      gameItems.push({ rule: `下跌${ret5d.toFixed(0)}%+低波${vol60d.toFixed(0)}%→机构吸筹`, adj: '+10' });
    } else if (ret20d < -10 && vol60d < 25) {
      instAgent = { type: '强吸筹', strength: 2, ret20d, vol60d };
      gameScore += 15;
      gameItems.push({ rule: `深跌${ret20d.toFixed(0)}%+低波${vol60d.toFixed(0)}%→强吸筹`, adj: '+15' });
    } else if (ret5d < -3 && vol60d > 40) {
      instAgent = { type: '散户恐慌', strength: -1, ret5d, vol60d };
      gameScore -= 8;
      gameItems.push({ rule: `下跌${ret5d.toFixed(0)}%+高波${vol60d.toFixed(0)}%→恐慌`, adj: '−8' });
    } else {
      gameItems.push({ rule: '无特殊机构信号', adj: '0' });
    }
  } else {
    gameItems.push({ rule: '60日数据不足，无机构代理', adj: '0' });
  }

  const composite = macdScore * 0.40 + mfScore * 0.30 + fundScore * 0.15 + gameScore * 0.15;
  const gatesPass = obvRatio >= 0.90 && rv <= 92 && !(macdScore < 35 && mfScore < 40);

  return {
    macdScore, mfScore, fundScore, gameScore, obvRatio,
    composite, gatesPass,
    details: { rv, kv, dv, jv, wv, bbPos, instAgent },
    breakdown: {
      macd: macdItems,
      multifactor: mfItems,
      fundamental: fundItems,
      game: gameItems,
    },
  };
}


// ═══════════════════════════
// 融合策略回测
// ═══════════════════════════

function backtestComprehensive(dates, closes, highs, lows, volumes) {
  const n = closes.length;
  if (n < 60) return [];
  const { dif, dea, bar } = calcMACD(closes);
  const rsiArr = calcRSI(closes);
  const { k: kArr, d: dArr, j: jArr } = calcKDJ(highs, lows, closes);
  const { upper: bbU, lower: bbL } = calcBollinger(closes);
  const wrArr = calcWR(highs, lows, closes);
  const obvFull = calcOBV(closes, volumes);
  const trades = []; let pos = null;

  for (let i = 60; i < n; i++) {
    if (isNaN(dif[i])) continue;
    const sc = scoreComprehensive(closes, highs, lows, volumes,
      dif, dea, bar, rsiArr, kArr, dArr, jArr, bbU, bbL, wrArr, obvFull, i);
    if (pos === null) {
      if (sc.composite >= 65 && sc.gatesPass) {
        pos = { bd: dates[i], bp: closes[i], bi: i,
          macd: +sc.macdScore.toFixed(1), mf: +sc.mfScore.toFixed(1),
          fund: +sc.fundScore.toFixed(1), game: +sc.gameScore.toFixed(1),
          comp: +sc.composite.toFixed(1) };
      }
    } else {
      const pnl = (closes[i] - pos.bp) / pos.bp * 100;
      let sell = false, reason = '';
      if (pnl < -8) { sell = true; reason = `止损${pnl.toFixed(1)}%`; }
      else if (pnl > 25) { sell = true; reason = `止盈+${pnl.toFixed(1)}%`; }
      else if (sc.composite < 35) { sell = true; reason = `综合分${sc.composite.toFixed(0)}<35`; }
      else if (pnl > 12 && sc.composite < 50) { sell = true; reason = `获利回吐+${pnl.toFixed(1)}%`; }
      if (!sell && i >= 30) {
        const rh = arrMax(closes.slice(Math.max(0, i - 30), i));
        if (closes[i] >= rh * 0.98 && dif[i] < dif[Math.max(0, i - 30)] * 0.85) {
          sell = true; reason = '顶背离';
        }
      }
      if (sell) {
        trades.push({
          buy_date: pos.bd, sell_date: dates[i],
          buy_price: pos.bp, sell_price: closes[i],
          profit_pct: +pnl.toFixed(2), hold_days: i - pos.bi,
          buy_reason: `融合${pos.comp.toFixed(0)}(M${pos.macd.toFixed(0)}/F${pos.mf.toFixed(0)}/基${pos.fund.toFixed(0)}/量${pos.game.toFixed(0)})`,
          sell_reason: reason,
        });
        pos = null;
      }
    }
  }
  return trades;
}


// ═══════════════════════════
// 融合策略预测
// ═══════════════════════════

function predictComprehensive(dates, closes, highs, lows, volumes, holding = false) {
  const n = closes.length;
  if (n < 60) return { error: '数据不足，需要至少60根K线' };
  const { dif, dea, bar } = calcMACD(closes);
  const rsiArr = calcRSI(closes);
  const { k: kArr, d: dArr, j: jArr } = calcKDJ(highs, lows, closes);
  const { upper: bbU, middle: bbM, lower: bbL } = calcBollinger(closes);
  const wrArr = calcWR(highs, lows, closes);
  const obvFull = calcOBV(closes, volumes);
  const i = n - 1;

  const sc = scoreComprehensive(closes, highs, lows, volumes,
    dif, dea, bar, rsiArr, kArr, dArr, jArr, bbU, bbL, wrArr, obvFull, i);

  // 信号映射
  let signal, action;
  if (sc.gatesPass) {
    if (sc.composite >= 70) { signal = '强烈看多'; action = holding ? '增持' : '买入'; }
    else if (sc.composite >= 65) { signal = '偏多'; action = holding ? '拿住' : '买入'; }
    else if (sc.composite >= 50) { signal = '中性'; action = holding ? '减持' : '观望'; }
    else if (sc.composite >= 40) { signal = '偏空'; action = holding ? '减持' : '不买'; }
    else { signal = '看空'; action = holding ? '卖出' : '不买'; }
  } else { signal = '门禁否决'; action = '观望'; }

  const gateDetails = [];
  if (sc.obvRatio < 0.90) gateDetails.push({ gate: 'OBV', passed: false });
  if (sc.details.rv > 92) gateDetails.push({ gate: 'RSI', passed: false });
  if (sc.macdScore < 35 && sc.mfScore < 40) gateDetails.push({ gate: '双弱', passed: false });

  const ma20 = arrMean(closes.slice(-20));
  const ma60 = arrMean(closes.slice(-60));

  return {
    strategy: 'comprehensive',
    date: dates[i],
    close: +closes[i].toFixed(2),
    holding,
    scores: {
      macd: { weight: 0.40, score: +sc.macdScore.toFixed(1),
              dif: +dif[i].toFixed(2), dea: +dea[i].toFixed(2), bar: +bar[i].toFixed(2) },
      multifactor: { weight: 0.30, score: +sc.mfScore.toFixed(1),
                     rsi: Math.round(sc.details.rv),
                     k: Math.round(sc.details.kv), d: Math.round(sc.details.dv),
                     j: Math.round(sc.details.jv), wr: Math.round(sc.details.wv) },
      fundamental: { weight: 0.15, score: +sc.fundScore.toFixed(1) },
      game: { weight: 0.15, score: +sc.gameScore.toFixed(1), obv_ratio: +sc.obvRatio.toFixed(2) },
    },
    composite: +sc.composite.toFixed(1),
    gates: { passed: sc.gatesPass, details: gateDetails },
    signal, action,
    key_levels: {
      ma20: +ma20.toFixed(2), ma60: +ma60.toFixed(2),
      bb_upper: +bbU[i].toFixed(2), bb_lower: +bbL[i].toFixed(2),
    },
    institution_proxy: sc.details.instAgent,
    breakdown: sc.breakdown,
  };
}


// ═══════════════════════════
// 长线持有
// ═══════════════════════════

function analyzeBuyhold(dates, closes, dividends = []) {
  const firstClose = closes[0], lastClose = closes[closes.length - 1];
  const years = Math.max((new Date(dates[dates.length - 1]) - new Date(dates[0])) / (365.25 * 86400000), 0.01);
  const priceReturn = (lastClose - firstClose) / firstClose * 100;
  const totalDiv = dividends.reduce((s, d) => s + (d.dividend_per_share || 0), 0);
  const divYield = totalDiv / firstClose * 100;
  return {
    strategy: 'buyhold',
    buy_date: dates[0], buy_price: +firstClose.toFixed(2),
    current_date: dates[dates.length - 1], current_price: +lastClose.toFixed(2),
    hold_years: +years.toFixed(1),
    price_return_pct: +priceReturn.toFixed(1),
    dividend_yield_pct: +divYield.toFixed(1),
    total_return_pct: +(priceReturn + divYield).toFixed(1),
  };
}


// ═══════════════════════════
// 导出
// ═══════════════════════════
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    ema, calcMACD, calcRSI, calcKDJ, calcBollinger, calcWR, calcOBV,
    scoreComprehensive, backtestComprehensive, predictComprehensive,
    analyzeBuyhold,
    arrMean, arrStd, arrMax, arrMin,
  };
}
