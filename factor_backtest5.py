#!/usr/bin/env python3
"""集成回测：横截面多因子候选池 + 个股 MACD 择时。目标突破 62%。

买 = MACD买入信号(底背离/零轴上金叉) 且 信号日该股在横截面 top-N 候选池内
卖 = MACD卖出信号(顶背离/死叉熊市)，不受候选池限制

对照：纯 MACD 择时（无横截面过滤，应复现 ~62% baseline）。
"""
import sys, os, argparse, sqlite3
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import calc_macd, detect_regime, find_divergences
from factors import compute_technical_factors, TECH_FACTORS, FACTOR_WEIGHTS

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_cache.db")
MAINBOARD = ("600", "601", "603", "605", "000", "001", "002", "003")


def gen_macd_signals(dates, closes, dif, dea, tops, bottoms, regimes,
                     cooldown=25):
    """复制 engine.backtest 的信号生成（不配对）。"""
    sigs = []
    for t in tops:
        sigs.append({'idx': t['idx'], 'action': 'sell', 'price': t['price'],
                     'reason': '顶背离'})
    for b in bottoms:
        sigs.append({'idx': b['idx'], 'action': 'buy', 'price': b['price'],
                     'reason': '底背离'})
    ls = -999
    for i in range(35, len(dif) - 1):
        if np.isnan(dif[i]) or np.isnan(dea[i]):
            continue
        if (dif[i - 1] <= dea[i - 1] and dif[i] > dea[i] and dif[i] > 0
                and dif[i] > dif[i - 3] and i - ls > cooldown
                and regimes[i] == 'bull'):
            if i + 2 < len(dif) and dif[i + 1] > dea[i + 1] \
                    and dif[i + 2] > dea[i + 2]:
                if not any(abs(s['idx'] - i) < 15
                           for s in sigs if s['action'] == 'buy'):
                    sigs.append({'idx': i, 'action': 'buy', 'price': closes[i],
                                 'reason': '零轴上金叉'})
        if dif[i - 1] >= dea[i - 1] and dif[i] < dea[i]:
            if regimes[i] == 'bear' and not any(
                    abs(s['idx'] - i) < 10 for s in sigs if s['action'] == 'sell'):
                sigs.append({'idx': i, 'action': 'sell', 'price': closes[i],
                             'reason': '死叉(熊市)'})
                ls = i
    sigs.sort(key=lambda x: x['idx'])
    return sigs


def pair_trades(sigs, reb_dates_sorted, pool_by_date, use_pool,
                buy_filter=None, max_hold=0, close_by_code=None):
    """配对买卖信号 → 交易列表。use_pool=True 时 buy 需在候选池内。
    buy_filter: 只保留该买入信号类型。max_hold>0 时持有超时强制平仓。"""
    trades, pos = [], None
    for s in sigs:
        if s['action'] == 'buy' and pos is None:
            if buy_filter and s['reason'] != buy_filter:
                continue
            if use_pool:
                rp = np.searchsorted(reb_dates_sorted, s['_date'], side="right") - 1
                if rp < 0 or s['_code'] not in pool_by_date.get(
                        reb_dates_sorted[rp], set()):
                    continue
            pos = {'bp': s['price'], 'bi': s['idx'], 'br': s['reason'],
                   '_code': s['_code'], '_date': s['_date']}
        elif s['action'] == 'sell' and pos is not None:
            if max_hold > 0 and s['idx'] - pos['bi'] > max_hold \
                    and close_by_code is not None:
                exit_price = close_by_code[pos['_code']][pos['bi'] + max_hold]
                pct = (exit_price - pos['bp']) / pos['bp'] * 100
                trades.append({'profit_pct': pct, 'buy_reason': pos['br'],
                               'sell_reason': '超时强平', 'hold_days': max_hold,
                               '_code': pos['_code'], 'buy_date': pos['_date']})
                pos = None
                continue
            pct = (s['price'] - pos['bp']) / pos['bp'] * 100
            trades.append({'profit_pct': pct, 'buy_reason': pos['br'],
                           'sell_reason': s['reason'],
                           'hold_days': s['idx'] - pos['bi'],
                           '_code': pos['_code'], 'buy_date': pos['_date']})
            pos = None
    return trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reb-interval", type=int, default=20)
    ap.add_argument("--top", type=int, default=100)
    args = ap.parse_args()

    db = sqlite3.connect(DB)
    cur = db.cursor()
    codes = [r[0] for r in cur.execute(
        "SELECT DISTINCT code FROM klines").fetchall()]
    codes = [c for c in codes if c.startswith(MAINBOARD)]
    if args.limit:
        codes = codes[:args.limit]

    cal = [r[0] for r in cur.execute(
        "SELECT date FROM klines WHERE code='000001' ORDER BY date").fetchall()]
    cal = cal[250:]
    reb_dates = cal[::args.reb_interval]
    reb_dates_sorted = np.array(reb_dates)
    print(f"{len(codes)} 只 | {len(reb_dates)} 调仓日 | 候选池 top{args.top}")

    factor_matrix = {f: [] for f in FACTOR_WEIGHTS}
    signals_by_code = {}
    close_by_code = {}
    valid_codes = []
    for code in codes:
        rec = cur.execute(
            "SELECT date,open,high,low,close,volume FROM klines "
            "WHERE code=? ORDER BY date", (code,)).fetchall()
        if len(rec) < 300:
            continue
        df = pd.DataFrame(rec, columns=["date", "open", "high", "low",
                                        "close", "volume"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df = df.set_index("date")
        dates = df.index.values
        closes = df["close"].values
        close_by_code[code] = closes

        # 横截面因子（reindex 到 reb_dates）
        f = compute_technical_factors(df)
        for fac in FACTOR_WEIGHTS:
            factor_matrix[fac].append(f[fac].reindex(reb_dates).values)

        # MACD 序列 + 信号
        dif, dea, bar = calc_macd(closes)
        regimes, _ = detect_regime(closes)
        tops, bottoms = find_divergences(dates, closes, dif)
        sigs = gen_macd_signals(dates, closes, dif, dea, tops, bottoms,
                                regimes)
        for s in sigs:
            s['_code'] = code
            s['_date'] = dates[s['idx']]
        signals_by_code[code] = sigs
        valid_codes.append(code)

    for fac in FACTOR_WEIGHTS:
        factor_matrix[fac] = np.array(factor_matrix[fac])  # (n_stocks, n_dates)

    # 每个 reb_date 的横截面排名（按 score 降序的全量 code 列表）
    ranked_pool = {}
    for di in range(len(reb_dates)):
        valid_mask = np.ones(len(valid_codes), bool)
        for fac in FACTOR_WEIGHTS:
            valid_mask &= ~np.isnan(factor_matrix[fac][:, di])
        idx_valid = np.where(valid_mask)[0]
        if len(idx_valid) < 50:
            continue
        scores_valid = np.zeros(len(idx_valid))
        for fac, w in FACTOR_WEIGHTS.items():
            x = factor_matrix[fac][idx_valid, di]
            r = pd.Series(x).rank(pct=True).values * 100.0
            scores_valid += w * r
        order_valid = np.argsort(-scores_valid)
        ranked_pool[reb_dates[di]] = [valid_codes[idx_valid[i]]
                                      for i in order_valid]

    def report(name, trades):
        if not trades:
            print(f"{name:26s} 无交易")
            return
        pct = np.array([t['profit_pct'] for t in trades])
        days = np.array([t['hold_days'] for t in trades])
        by_code = {}
        for t in trades:
            by_code.setdefault(t['_code'], []).append(t['profit_pct'])
        avg_wr = np.mean([(np.array(v) > 0).mean() for v in by_code.values()])
        print(f"{name:26s} {len(trades):5d}笔 | 全交易胜率 {(pct>0).mean()*100:5.1f}% "
              f"| 均胜率 {avg_wr*100:5.1f}% | 平均收益 {pct.mean():+6.2f}% "
              f"| 均持有 {days.mean():.0f}天")

    def run_all(pool, use_pool, buy_filter=None, max_hold=0):
        """每只股票独立配对（与 engine.backtest 语义一致），合并交易。"""
        trades = []
        for code in valid_codes:
            trades.extend(pair_trades(signals_by_code[code], reb_dates_sorted,
                                      pool, use_pool, buy_filter, max_hold,
                                      close_by_code))
        return trades

    base = run_all(None, use_pool=False)
    print(f"\n=== 集成回测对比 ===")
    report("纯 MACD 择时(基线)", base)
    pool200 = {d: set(codes_list[:200])
               for d, codes_list in ranked_pool.items()}
    report("全信号 top200", run_all(pool200, True))
    report("只底背离 top200", run_all(pool200, True, buy_filter='底背离'))
    for mh in [30, 40, 45, 50, 55, 60, 70, 80, 90]:
        report(f"只底背离 top200 强平{mh}日",
               run_all(pool200, True, buy_filter='底背离', max_hold=mh))

    # 分段稳健性验证（按每笔交易的买入日期分前后两半）
    print("\n=== 分段稳健性 (只底背离 top200 强平60日) ===")
    integ60 = run_all(pool200, True, buy_filter='底背离', max_hold=60)
    buy_dates = sorted(set(t['buy_date'] for t in integ60))
    mid = buy_dates[len(buy_dates) // 2]
    for label, sub in [('前半段', [t for t in integ60 if t['buy_date'] <= mid]),
                       ('后半段', [t for t in integ60 if t['buy_date'] > mid])]:
        report(f"  {label}", sub)


if __name__ == "__main__":
    main()
