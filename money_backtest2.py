#!/usr/bin/env python3
"""money_backtest2.py: 总策略全市场资金回测（复用 engine 完整 S* 评分）。

总策略 = 四维 S(M×0.40+F×0.30+P×0.15+Q×0.15) − 0.08×A(18共识)
          + G(龙虎榜龙头+15) + Z(资金流rank×10)

买卖（沿用 engine.backtest_comprehensive 规则）：
  买入 S*≥69 且过门禁；卖出 止损-12%/止盈+15%/S*<35/获利回吐/顶背离/破颈线
每支独立 100 万复利，期末清仓，输出总盈亏金额。
"""
import sys, os, sqlite3, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_cache.db")
CAPITAL = 1_000_000
BUY_SCORE = engine.COMPREHENSIVE_BUY_SCORE          # 69
STOP = engine.COMPREHENSIVE_STOP_LOSS_PCT           # -12
TAKE = engine.COMPREHENSIVE_TAKE_PROFIT_PCT         # 15
PROTECT_PCT = engine.COMPREHENSIVE_PROFIT_PROTECT_PCT   # 12
PROTECT_SCORE = engine.COMPREHENSIVE_PROFIT_PROTECT_SCORE  # 50
BETA = engine.COMPREHENSIVE_AUXILIARY_CONTRARIAN_BETA  # 0.08
G_LHB = 15.0   # 龙虎榜龙头加分
Z_SCALE = 10.0  # 资金流 rank 加分上限
START_DATE = "2025-11-01"   # 交易起始（资金流数据覆盖范围）


def build_g_map(lhb, k):
    """龙虎榜龙头信号 → {(code, date): 15}。连板>=2 + 净买>15% + 换手<12.5%"""
    lhb = lhb.drop_duplicates(subset=["code", "list_date"], keep="first")
    lhb = lhb[lhb["market_turnover"] > 3e7]
    lhb["nbr"] = lhb["net_buy_ratio"].clip(-100, 100)
    # 连板高度（klines 算）
    k2 = k.sort_values(["code", "date"])
    k2["pct"] = k2.groupby("code")["close"].pct_change() * 100
    zt = (k2["pct"] >= 9.5)
    k2["height"] = k2.groupby("code")["pct"].transform(_streak)
    hm = dict(zip(zip(k2["code"], k2["date"]), k2["height"]))
    lhb["height"] = [hm.get((c, d), 0) for c, d in
                     zip(lhb["code"], lhb["list_date"])]
    hi = lhb[(lhb["height"] >= 2) & (lhb["nbr"] > 15) &
             (lhb["turnover_rate"] < 12.5)]
    return {(c, d): G_LHB for c, d in zip(hi["code"], hi["list_date"])}


def build_z_map(ff):
    """资金流净流入每日横截面 rank → {(code, date): rank*10}"""
    ff = ff[ff["main_net"].notna()]
    ff["rank"] = ff.groupby("date")["main_net"].rank(pct=True)
    return {(c, d): r * Z_SCALE for c, d, r in
            zip(ff["code"], ff["date"], ff["rank"])}


def _streak(s):
    idx = s.index
    s = s.to_numpy()
    out = np.zeros(len(s), dtype=int)
    run = 0
    for i in range(len(s)):
        run = run + 1 if s[i] else 0
        out[i] = run
    return pd.Series(out, index=idx)


def backtest_one(code, dates, closes, highs, lows, volumes, opens, g_map, z_map):
    """单只股票回测，返回每笔交易的 profit_pct（%）。"""
    n = len(closes)
    if n < 60:
        return []
    closes = np.asarray(closes, float)
    highs = np.asarray(highs, float)
    lows = np.asarray(lows, float)
    volumes = np.asarray(volumes, float)
    opens = np.asarray(opens, float)

    dif, dea, bar = engine.calc_macd(closes)
    rsi_arr = engine.calc_rsi(closes)
    k_arr, d_arr, j_arr = engine.calc_kdj(highs, lows, closes)
    bb_u, bb_m, bb_l = engine.calc_bollinger(closes)
    wr_arr = engine.calc_wr(highs, lows, closes)
    obv_full = engine.calc_obv(closes, volumes)
    aux = engine.calc_auxiliary_consensus_series(closes, highs, lows, volumes, opens)

    trades = []
    pos = None
    pending = None
    for i in range(60, n):
        if pending == "buy":
            pos = {"bp": opens[i], "bi": i}
            pending = None
        elif pending == "sell" and pos is not None:
            pnl = engine.calc_net_trade_return(pos["bp"], opens[i], 0.0003, 0.0005)
            trades.append(pnl)
            pos = None
            pending = None

        if np.isnan(dif[i]):
            continue
        macd_score, mf_score, fund_score, game_score, obv_flow, composite, gates_pass, _, _ = \
            engine._score_comprehensive(closes, highs, lows, volumes,
                                        dif, dea, bar, rsi_arr, k_arr, d_arr, j_arr,
                                        bb_u, bb_l, wr_arr, obv_full, i)
        composite = engine.apply_auxiliary_consensus_adjustment(
            composite, aux[i], BETA)
        g = g_map.get((code, dates[i]), 0.0)
        z = z_map.get((code, dates[i]), 0.0)
        s_star = composite + g + z

        if pos is None:
            if (pending is None and i + 1 < n and s_star >= BUY_SCORE
                    and gates_pass and dates[i] >= START_DATE):
                pending = "buy"
        else:
            pnl = engine.calc_net_trade_return(pos["bp"], closes[i], 0.0003, 0.0005)
            sell = False
            if pnl < STOP:
                sell = True
            elif pnl > TAKE:
                sell = True
            elif s_star < 35:
                sell = True
            elif pnl > PROTECT_PCT and s_star < PROTECT_SCORE:
                sell = True
            if not sell:
                _, top_div = engine._recent_price_divergence(closes, dif, i)
                if top_div:
                    sell = True
            if not sell and i - pos["bi"] > 10 and (i - pos["bi"]) % 5 == 0:
                if engine._quick_pattern_sell(closes, highs, i):
                    sell = True
            if sell and i + 1 < n:
                pending = "sell"

    # 期末清仓
    if pos is not None and pos["bi"] < n - 1:
        pnl = engine.calc_net_trade_return(pos["bp"], opens[-1], 0.0003, 0.0005)
        trades.append(pnl)
    return trades


def main():
    ap_lim = int(os.environ.get("LIMIT", "0"))
    db = sqlite3.connect(DB)
    ff = pd.read_sql(
        "SELECT code,date,main_net FROM fund_flow WHERE main_net IS NOT NULL "
        "AND date >= '2025-11-01'", db)
    lhb = pd.read_sql(
        "SELECT code,list_date,net_buy_ratio,turnover_rate,market_turnover FROM lhb", db)
    codes = [r[0] for r in db.execute(
        "SELECT DISTINCT code FROM klines WHERE code LIKE '6%' OR code LIKE '0%'").fetchall()]
    db.close()

    # 加载 klines（只加载回测区间 + warmup 250 日）
    db = sqlite3.connect(DB)
    k = pd.read_sql(
        "SELECT code,date,open,high,low,close,volume FROM klines "
        "WHERE date >= '2024-11-01' ORDER BY code,date", db)
    db.close()

    g_map = build_g_map(lhb, k)
    z_map = build_z_map(ff)
    print(f"G 龙虎榜信号 {len(g_map)} 个 | Z 资金流映射 {len(z_map)} 条")

    if ap_lim:
        codes = codes[:ap_lim]

    t0 = time.time()
    results = {}
    n_trades = 0
    for idx, code in enumerate(codes, 1):
        sub = k[k["code"] == code]
        if len(sub) < 60:
            results[code] = CAPITAL
            continue
        dates = sub["date"].tolist()
        closes = sub["close"].tolist()
        highs = sub["high"].tolist()
        lows = sub["low"].tolist()
        volumes = sub["volume"].tolist()
        opens = sub["open"].tolist()
        trades = backtest_one(code, dates, closes, highs, lows, volumes, opens, g_map, z_map)
        final = CAPITAL
        for p in trades:
            final *= (1 + p / 100)
        results[code] = final
        n_trades += len(trades)
        if idx % 200 == 0:
            el = time.time() - t0
            print(f"  进度 {idx}/{len(codes)} | 交易 {n_trades} 笔 | {el:.0f}s "
                  f"ETA {el/idx*(len(codes)-idx):.0f}s", flush=True)

    total_final = sum(results.values())
    total_cost = CAPITAL * len(results)
    pnl = total_final - total_cost
    print(f"\n=== 总策略资金回测结果 ===")
    print(f"回测 2025-11 ~ 2026-09-08 | 股票 {len(results)} 只 | 交易 {n_trades} 笔")
    print(f"总本金: {total_cost/1e8:.2f} 亿")
    print(f"总期末金额: {total_final/1e8:.4f} 亿")
    print(f"总盈亏: {pnl/1e4:+,.0f} 万元 ({pnl/1e8:+.4f} 亿) | 收益率 {pnl/total_cost*100:+.3f}%")
    print(f"耗时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
