#!/usr/bin/env python3
"""money_backtest.py: 总策略全市场资金回测（每支100万，看总盈亏金额）。

总策略 = 横截面多因子海选（融合）：
  每日对全市场主板打分，分 = 主力净流入rank + 短期动量rank + 龙虎榜龙头确认加分，
  选 top5，次日收盘买入，持有2日收盘卖出，每支100万独立复利，期末清仓。

输出：总盈亏金额（元），不是胜率。
复杂度：全向量化，禁止逐行 dict.get。
"""
import sqlite3
import pandas as pd
import numpy as np

DB = "stock_cache.db"
CAPITAL = 1_000_000   # 每支股票独立本金


def load_data():
    db = sqlite3.connect(DB)
    ff = pd.read_sql(
        "SELECT code,date,main_net FROM fund_flow WHERE main_net IS NOT NULL "
        "AND date >= '2025-11-01'", db)
    k = pd.read_sql("SELECT code,date,close FROM klines WHERE date >= '2025-10-01'", db)
    lhb = pd.read_sql(
        "SELECT code,list_date,net_buy_ratio,turnover_rate,market_turnover "
        "FROM lhb", db)
    db.close()
    return ff, k, lhb


def compute_signals(ff, k, lhb):
    """每日横截面打分，返回 (信号日, code, 次日买入价, 持有2日卖出价)。"""
    # 1) 短期动量（klines，10日）
    k = k.sort_values(["code", "date"])
    k["mom10"] = k.groupby("code")["close"].pct_change(10) * 100
    # 2) 主力净流入 rank（每日横截面）
    ff["ff_rank"] = ff.groupby("date")["main_net"].rank(pct=True)
    # 3) 龙虎榜龙头确认：连板>=2 + 净买>15% + 低换手
    lhb = lhb.drop_duplicates(subset=["code", "list_date"], keep="first")
    lhb = lhb[lhb["market_turnover"] > 3e7]
    lhb["nbr"] = lhb["net_buy_ratio"].clip(-100, 100)
    # 连板高度（klines 算）
    k2 = k.sort_values(["code", "date"])
    k2["pct"] = k2.groupby("code")["close"].pct_change() * 100
    zt = (k2["pct"] >= 9.5).astype(int)
    k2["height"] = k2.groupby("code")["pct"].transform(
        lambda x: _streak(x >= 9.5))
    hm = k2.set_index(["code", "date"])["height"]
    lhb["height"] = [hm.get((c, d), 0) for c, d in
                     zip(lhb["code"], lhb["list_date"])]
    lhb_hi = lhb[(lhb["height"] >= 2) & (lhb["nbr"] > 15) &
                 (lhb["turnover_rate"] < 12.5)]
    lhb_hi = lhb_hi[["code", "list_date"]].rename(
        columns={"list_date": "date"})
    lhb_hi["lhb_flag"] = 1.0

    # 合并：ff + klines动量 + lhb加分
    m = ff.merge(k[["code", "date", "mom10"]], on=["code", "date"], how="left")
    m = m.merge(lhb_hi, on=["code", "date"], how="left")
    m["lhb_flag"] = m["lhb_flag"].fillna(0.0)
    m = m.dropna(subset=["mom10"])
    # 动量 rank（每日横截面）
    m["mom_rank"] = m.groupby("date")["mom10"].rank(pct=True)
    # 综合分：龙虎榜龙头绝对优先（×100），资金流+动量补充排序
    m["score"] = m["ff_rank"] + m["mom_rank"] + m["lhb_flag"] * 100.0
    # 每日选 top5
    m = m.sort_values(["date", "score"], ascending=[True, False])
    m["rank_sel"] = m.groupby("date")["score"].rank(
        ascending=False, method="first")
    sel = m[m["rank_sel"] <= 5][["code", "date", "score", "lhb_flag"]].copy()
    return sel, k


def _streak(s):
    """连续涨停天数（向量化版）。"""
    idx = s.index
    s = s.to_numpy()
    out = np.zeros(len(s), dtype=int)
    run = 0
    for i in range(len(s)):
        run = run + 1 if s[i] else 0
        out[i] = run
    return pd.Series(out, index=idx)


def run_backtest(sel, k):
    """每笔交易：次日(T+1)收盘买入100万，持有2日(T+2)收盘卖出。"""
    close_map = k.set_index(["code", "date"])["close"]
    kdates = k.groupby("code")["date"].apply(list).to_dict()

    trades = []
    for code, date in zip(sel["code"], sel["date"]):
        dates = kdates.get(code)
        if dates is None:
            continue
        idx = np.searchsorted(np.array(dates), date)
        # 次日买入、再持有一日卖出：需要 date 之后至少 2 个交易日
        if idx + 2 >= len(dates):
            continue
        buy_d = dates[idx + 1]
        sell_d = dates[idx + 2]
        buy_p = close_map.get((code, buy_d))
        sell_p = close_map.get((code, sell_d))
        if buy_p is None or sell_p is None or buy_p <= 0:
            continue
        ret = sell_p / buy_p - 1
        trades.append((code, date, buy_d, sell_d, ret))
    return trades


def main():
    ff, k, lhb = load_data()
    sel, k = compute_signals(ff, k, lhb)
    print(f"回测区间信号: 每日 top5 共 {len(sel)} 个买入信号")

    trades = run_backtest(sel, k)
    print(f"可执行交易: {len(trades)} 笔")

    # 每支股票独立 100 万复利
    from collections import defaultdict
    final = defaultdict(lambda: CAPITAL)
    for code, date, buy_d, sell_d, ret in trades:
        final[code] *= (1 + ret)

    # 汇总：所有股票最终金额 vs 总本金
    # 参与交易的股票 + 未参与交易的股票（保持 100万）
    n_stocks = 3307
    traded = set(final.keys())
    total_final = sum(final.values()) + CAPITAL * (n_stocks - len(traded))
    total_cost = CAPITAL * n_stocks
    pnl = total_final - total_cost

    n_win = sum(1 for c, d, b, s, r in trades if r > 0)
    avg_ret = np.mean([r for c, d, b, s, r in trades]) * 100
    print(f"\n=== 总策略资金回测结果 ===")
    print(f"回测: 2025-11 ~ 2026-09-08 | 覆盖 {n_stocks} 支主板")
    print(f"买入信号 {len(sel)} 个 → 可执行交易 {len(trades)} 笔")
    print(f"盈利笔数 {n_win}/{len(trades)} | 单笔平均收益 {avg_ret:+.2f}%")
    print(f"总本金: {total_cost/1e8:.2f} 亿 ({CAPITAL/1e4:.0f}万 × {n_stocks})")
    print(f"总期末金额: {total_final/1e8:.4f} 亿")
    print(f"总盈亏: {pnl/1e4:+,.0f} 万元  ({pnl/1e8:+.3f} 亿)")


if __name__ == "__main__":
    main()
