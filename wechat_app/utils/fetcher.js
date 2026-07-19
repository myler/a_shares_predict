/**
 * 数据获取层 — K线抓取 + 本地缓存
 *
 * 多源降级: 腾讯(前复权) → 东方财富(前复权) → 新浪
 * 缓存策略: wx.Storage 本地存 K线，首次拉取后缓存 24 小时
 */

const CACHE_PREFIX = 'kline_';
const CACHE_TTL_MS = 24 * 60 * 60 * 1000; // 24 小时

/**
 * 从缓存读取 K线
 */
function loadFromCache(code) {
  try {
    const raw = wx.getStorageSync(CACHE_PREFIX + code);
    if (!raw) return null;
    const cached = JSON.parse(raw);
    if (Date.now() - cached.ts > CACHE_TTL_MS) return null; // 过期
    return cached.data;
  } catch (e) {
    return null;
  }
}

/**
 * 保存 K线到缓存
 */
function saveToCache(code, data) {
  try {
    wx.setStorageSync(CACHE_PREFIX + code, JSON.stringify({ ts: Date.now(), data }));
  } catch (e) {
    // 存储满则忽略
  }
}

/**
 * 格式化代码为市场前缀 (sh/sz)
 */
function marketCode(code) {
  return (code.startsWith('6') || code.startsWith('5')) ? `sh${code}` : `sz${code}`;
}

/**
 * 请求封装 (返回 Promise)
 */
function fetchJSON(url) {
  return new Promise((resolve, reject) => {
    wx.request({
      url,
      method: 'GET',
      timeout: 10000,
      success(res) {
        if (res.statusCode === 200) resolve(res.data);
        else reject(new Error(`HTTP ${res.statusCode}`));
      },
      fail(err) { reject(err); },
    });
  });
}

/**
 * 腾讯数据源 (前复权，最优先)
 */
async function fetchTencent(code) {
  const sc = marketCode(code);
  const url = `https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=${sc},day,,,2500,qfq`;
  const raw = await fetchJSON(url);
  const klines = raw?.data?.[sc]?.qfqday || raw?.data?.[sc]?.day || [];
  if (!klines.length) throw new Error('腾讯: 空数据');

  return klines.map(r => {
    if (Array.isArray(r)) {
      return { day: r[0], open: +r[1], high: +r[2], low: +r[3], close: +r[4], volume: +r[5] };
    }
    return { day: r.date, open: +r.open, high: +r.high, low: +r.low, close: +r.close, volume: +r.volume };
  });
}

/**
 * 东方财富数据源 (前复权，备用)
 */
async function fetchEastMoney(code) {
  const market = (code.startsWith('6') || code.startsWith('5')) ? 1 : 0;
  const url = `https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=${market}.${code}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57&klt=101&fqt=1&end=20500101&lmt=2500`;
  const raw = await fetchJSON(url);
  const klines = raw?.data?.klines || [];
  if (!klines.length) throw new Error('东方财富: 空数据');

  return klines.map(line => {
    const parts = line.split(',');
    return {
      day: parts[0],
      open: +parts[1], high: +parts[3], low: +parts[4],
      close: +parts[2], volume: +parts[5],
    };
  });
}

/**
 * 多源降级抓取 K线
 */
async function fetchKline(code) {
  // 1. 先读缓存
  const cached = loadFromCache(code);
  if (cached) return cached;

  // 2. 多源降级
  const sources = [
    { name: '腾讯', fn: () => fetchTencent(code) },
    { name: '东方财富', fn: () => fetchEastMoney(code) },
  ];

  for (const src of sources) {
    try {
      const data = await src.fn();
      if (data && data.length > 0) {
        saveToCache(code, data);
        return data;
      }
    } catch (e) {
      console.warn(`${src.name} 失败:`, e.message);
    }
  }

  throw new Error(`无法获取 ${code} 的K线数据`);
}

/**
 * 获取股票名称 (从腾讯接口查)
 */
async function getStockName(code) {
  try {
    const sc = marketCode(code);
    const url = `https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=${sc},day,,,5,qfq`;
    const raw = await fetchJSON(url);
    return raw?.data?.[sc]?.qt?.[sc]?.[1] || code;
  } catch (e) {
    return code;
  }
}

/**
 * 完整分析流程
 */
async function analyze(code, holding = false) {
  const data = await fetchKline(code);
  const name = await getStockName(code);

  const dates = data.map(d => d.day);
  const closes = data.map(d => d.close);
  const highs = data.map(d => d.high);
  const lows = data.map(d => d.low);
  const volumes = data.map(d => d.volume);

  // 引入引擎（小程序环境）
  // const engine = require('./engine');  // 或 import

  return { code, name, dates, closes, highs, lows, volumes, data };
}

/**
 * 获取分红数据 (东方财富接口)
 */
async function fetchDividends(code) {
  const market = (code.startsWith('6') || code.startsWith('5')) ? 1 : 0;
  const url = `https://datacenter.eastmoney.com/securities/api/data/v1/get?reportName=RPT_F10_FINANCE_MAINFINADATA&columns=ALL&filter=(SECURITY_CODE=%22${code}%22)&pageNumber=1&pageSize=20&sortColumns=NOTICE_DATE&sortTypes=-1`;
  
  try {
    const raw = await fetchJSON(url);
    const items = raw?.result?.data || [];
    return items.map(item => ({
      ex_date: item.EX_DIVIDEND_DATE ? item.EX_DIVIDEND_DATE.slice(0, 10) : '',
      dividend_10: +(item.BONUS_IT_RATIO || 0),       // 10派X元
      dividend_per_share: +(item.BONUS_IT_RATIO || 0) / 10,
      bonus_share: +(item.BONUS_SHARE_RATIO || 0),      // 送股
      transfer_share: +(item.TRANSFER_SHARE_RATIO || 0), // 转增
    })).filter(d => d.ex_date);
  } catch (e) {
    console.warn('分红获取失败:', e.message);
    return [];
  }
}

module.exports = {
  fetchKline, getStockName, fetchDividends, analyze,
  loadFromCache, saveToCache,
};
