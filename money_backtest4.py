#!/usr/bin/env python3
"""money_backtest4.py: 总策略 v4 全市场资金回测。

核心评分 S* = 4主指标(M×0.40+F×0.30+P×0.15+Q×0.15) − 0.08×18副指标共识 + 资金流rank×10

双路径买入，双交易方式：
  路径1 技术达标股(S*≥69): 稳健建仓 首仓25%→加仓(S*≥75→50%,≥80→75%,≥85→100%)
        减仓 S*<35/炸板→50%；清仓 移动止盈(回撤8%)/止损-8%/跌破5日线
  路径2 龙虎榜龙头(连板≥2+净买>15%+缩量): 直接满仓100万，持有2日卖出

每支独立100万现金，期末清仓，输出总盈亏金额。
"""
import sys, os, sqlite3, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_cache.db")
CAPITAL = 1_000_000
START_DATE = "2016-09-08"
BETA = engine.COMPREHENSIVE_AUXILIARY_CONTRARIAN_BETA  # 0.08
Z_SCALE = 10.0
BUY1 = 69.0    # 路径1 买入阈值
STOP = 0.08
TRAIL = 0.08
S_EXIT = 35.0


def _streak(s):
    """连续涨停天数（涨幅>=9.5%视为涨停；停牌NaN中断连板）"""
    idx = s.index
    s = s.to_numpy()
    out = np.zeros(len(s), dtype=int)
    run = 0
    for i in range(len(s)):
        if s[i] >= 9.5:
            run += 1
        else:
            run = 0
        out[i] = run
    return pd.Series(out, index=idx)


def build_lhb_signals(lhb, k):
    lhb = lhb.drop_duplicates(subset=["code", "list_date"], keep="first")
    lhb = lhb[lhb["market_turnover"] > 3e7]
    lhb["nbr"] = lhb["net_buy_ratio"].clip(-100, 100)
    k2 = k.sort_values(["code", "date"])
    k2["pct"] = k2.groupby("code")["close"].pct_change() * 100
    k2["height"] = k2.groupby("code")["pct"].transform(_streak)
    hm = dict(zip(zip(k2["code"], k2["date"]), k2["height"]))
    lhb["height"] = [hm.get((c, d), 0) for c, d in
                     zip(lhb["code"], lhb["list_date"])]
    hi = lhb[(lhb["height"] >= 2) & (lhb["nbr"] > 15) &
             (lhb["turnover_rate"] < 12.5)]
    return set(zip(hi["code"], hi["list_date"]))


def build_z_map(ff):
    ff = ff[ff["main_net"].notna()]
    ff["rank"] = ff.groupby("date")["main_net"].rank(pct=True)
    return {(c, d): r * Z_SCALE for c, d, r in
            zip(ff["code"], ff["date"], ff["rank"])}


def backtest_one(code, dates, opens, highs, lows, closes, volumes,
                 lhb_signals, z_map):
    n = len(closes)
    if n < 60:
        return CAPITAL
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

    cash = CAPITAL
    shares = 0.0
    avg_cost = 0.0
    highest = 0.0
    pending = None        # ('buy', target_pct) / ('sell', target_pct) / ('buy_full', None)
    path2_sell_i = None   # 路径2 卖出索引（持有2日）

    for i in range(60, n):
        # 执行 pending（次日开盘）
        if pending is not None:
            act, target_pct = pending
            if act == "buy":
                want = CAPITAL * target_pct
                gap = want - shares * opens[i]
                if gap > 0 and cash > 0:
                    amt = min(cash, gap)
                    add = amt / opens[i]
                    avg_cost = (avg_cost * shares + amt) / (shares + add)
                    shares += add
                    cash -= amt
                    highest = opens[i]
            elif act == "buy_full":
                if cash > 0:
                    amt = cash
                    add = amt / opens[i]
                    avg_cost = opens[i]
                    shares += add
                    cash -= amt
                    highest = opens[i]
            else:  # sell
                want = CAPITAL * target_pct
                cur_val = shares * opens[i]
                gap = cur_val - want
                if gap > 0:
                    sell_sh = gap / opens[i]
                    cash += sell_sh * opens[i]
                    shares -= sell_sh
                    if shares <= 0:
                        shares = 0.0
                        avg_cost = 0.0
                        highest = 0.0
            pending = None

        # 路径2 到期卖出
        if path2_sell_i is not None and i >= path2_sell_i:
            cash += shares * opens[i]
            shares = 0.0
            avg_cost = 0.0
            highest = 0.0
            path2_sell_i = None

        if np.isnan(dif[i]):
            continue

        macd_score, mf_score, fund_score, game_score, obv_flow, composite, gates_pass, _, _ = \
            engine._score_comprehensive(closes, highs, lows, volumes,
                                        dif, dea, bar, rsi_arr, k_arr, d_arr, j_arr,
                                        bb_u, bb_l, wr_arr, obv_full, i)
        composite = engine.apply_auxiliary_consensus_adjustment(composite, aux[i], BETA)
        z = z_map.get((code, dates[i]), 0.0)
        s_star = composite + z

        is_lhb = (code, dates[i]) in lhb_signals

        if shares == 0 and dates[i] >= START_DATE:
            # 空仓：路径2 优先（龙虎榜满仓快进快出），否则路径1 首仓
            if is_lhb and gates_pass:
                pending = ("buy_full", None)
                path2_sell_i = i + 2
            elif s_star >= BUY1 and gates_pass:
                pending = ("buy", 0.25)

        elif shares > 0:
            if path2_sell_i is None:
                # 路径1 持仓：加仓/减仓/清仓
                # 加仓（S*分数增强）
                if s_star >= 85:
                    target = 1.0
                elif s_star >= 80:
                    target = 0.75
                elif s_star >= 75:
                    target = 0.50
                else:
                    target = 0.25
                cur_pct = shares * opens[i] / CAPITAL
                if target > cur_pct + 1e-9 and pending is None:
                    pending = ("buy", target)

                # 减仓：炸板 或 S*<35
                limit_price = round(closes[i - 1] * 1.1, 2)
                zhakai = highs[i] >= limit_price and closes[i] < limit_price
                if (zhakai or s_star < S_EXIT) and pending is None:
                    if cur_pct > 0.51:
                        pending = ("sell", 0.50)

                # 清仓：移动止盈/止损/跌破5日线
                highest = max(highest, closes[i])
                ma5 = np.mean(closes[max(0, i - 4):i + 1])
                trail_hit = highest > 0 and closes[i] <= highest * (1 - TRAIL)
                stop_hit = closes[i] <= avg_cost * (1 - STOP)
                ma5_hit = closes[i] < ma5
                if (trail_hit or stop_hit or ma5_hit) and pending is None:
                    pending = ("sell", 0.0)

    # 期末清仓
    if shares > 0:
        cash += shares * opens[-1]
    return cash


def main():
    ap_lim = int(os.environ.get("LIMIT", "0"))
    db = sqlite3.connect(DB)
    ff = pd.read_sql(
        "SELECT code,date,main_net FROM fund_flow WHERE main_net IS NOT NULL "
        "AND date >= '2025-01-01'", db)
    lhb = pd.read_sql(
        "SELECT code,list_date,net_buy_ratio,turnover_rate,market_turnover FROM lhb", db)
    codes = [r[0] for r in db.execute(
        "SELECT DISTINCT code FROM klines WHERE code LIKE '6%' OR code LIKE '0%'").fetchall()]
    db.close()

    db = sqlite3.connect(DB)
    k = pd.read_sql(
        "SELECT code,date,open,high,low,close,volume FROM klines "
        "WHERE date >= '2015-09-01' ORDER BY code,date", db)
    db.close()

    lhb_signals = build_lhb_signals(lhb, k)
    z_map = build_z_map(ff)
    print(f"龙虎榜信号 {len(lhb_signals)} 个 | 资金流映射 {len(z_map)} 条")

    if ap_lim:
        codes = codes[:ap_lim]

    t0 = time.time()
    results = {}
    n_triggered = 0
    for idx, code in enumerate(codes, 1):
        sub = k[k["code"] == code]
        if len(sub) < 60:
            results[code] = CAPITAL
            continue
        dates = sub["date"].tolist()
        final = backtest_one(
            code, dates, sub["open"].tolist(), sub["high"].tolist(),
            sub["low"].tolist(), sub["close"].tolist(), sub["volume"].tolist(),
            lhb_signals, z_map)
        results[code] = final
        if final != CAPITAL:
            n_triggered += 1
        if idx % 200 == 0:
            el = time.time() - t0
            print(f"  进度 {idx}/{len(codes)} | 触发交易 {n_triggered} 只 | {el:.0f}s "
                  f"ETA {el/idx*(len(codes)-idx):.0f}s", flush=True)

    total_final = sum(results.values())
    total_cost = CAPITAL * len(results)
    pnl = total_final - total_cost
    print(f"\n=== 总策略 v4 资金回测结果 ===")
    print(f"回测 {START_DATE} ~ 2026-09-08 | 股票 {len(results)} 只 | 触发交易 {n_triggered} 只")
    print(f"总本金: {total_cost/1e8:.2f} 亿")
    print(f"总期末金额: {total_final/1e8:.4f} 亿")
    print(f"总盈亏: {pnl/1e4:+,.0f} 万元 ({pnl/1e8:+.4f} 亿) | 收益率 {pnl/total_cost*100:+.3f}%")
    print(f"耗时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
