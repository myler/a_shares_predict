#!/usr/bin/env python3
"""批量回测综合策略 — 100只股票"""

import sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetcher import fetch_kline
from db import load_stock_name
from engine import backtest_comprehensive, predict_comprehensive, summarize_trades

STOCKS = {
    # ── 芯片 (40) ──
    "芯片": [
        "603893", "603501", "002049", "002371", "002415", "603019",
        "603986", "600703", "600460", "688981", "688012", "688256",
        "688536", "300782", "002916", "603290", "688047", "688041",
        "002156", "603160", "688608", "688595", "300661", "603005",
        "688126", "688385", "300223", "002185", "688521", "688396",
        "688200", "688368", "300458", "603068", "688037",
        "603690", "300604", "688110", "002409",
    ],
    # ── 资源 (30) ──
    "资源": [
        "601899", "000933", "600362", "603799", "002460", "000630",
        "600489", "601600", "601069", "600547", "000426", "601168",
        "600111", "600259", "600219", "603993", "601088", "601898",
        "600188", "000983", "601225", "600985", "600392", "000831",
        "000878", "002738", "601958", "600497", "002155", "600988",
    ],
    # ── 消费 (30) ──
    "消费": [
        "000651", "000333", "600519", "000858", "002304", "600887",
        "601888", "300957", "603288", "600809", "000568", "002714",
        "600690", "000895", "300015", "600882", "300146", "603345",
        "002557", "600436", "300896", "688363", "603899", "002032",
        "600132", "000423", "600085", "603833", "002242", "300740",
    ],
    "金融": ["000001"],
}

def main():
    t0 = time.time()
    results = []
    total = sum(len(v) for v in STOCKS.values())
    done = 0
    
    for category, codes in STOCKS.items():
        for code in codes:
            done += 1
            try:
                data = fetch_kline(code)
                name = load_stock_name(code) or code
                dates = [d['day'] for d in data]
                opens = np.array([float(d['open']) for d in data])
                closes = np.array([float(d['close']) for d in data])
                highs = np.array([float(d['high']) for d in data])
                lows = np.array([float(d['low']) for d in data])
                vols = np.array([float(d['volume']) for d in data])
                
                trades = backtest_comprehensive(
                    dates, closes, highs, lows, vols, opens=opens)
                summary = summarize_trades(trades)
                
                # 当前评分
                pred = predict_comprehensive(
                    dates, closes, highs, lows, vols, holding=False, opens=opens)
                comp_score = pred.get('composite') if 'error' not in pred else None
                signal = pred.get('signal', '—') if 'error' not in pred else '—'
                score_display = f'{comp_score:.0f}' if comp_score is not None else '—'
                
                results.append({
                    'category': category, 'code': code, 'name': name,
                    'trades': summary['trades'],
                    'win_rate': round(summary['win_rate_pct'], 1),
                    'total_return_pct': round(summary['price_return_pct'], 1),
                    'score': round(comp_score, 1) if comp_score is not None else None,
                    'signal': signal,
                })
                
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f"[{done}/{total}] {code} {name:6s} {summary['trades']:2d}笔 胜率{summary['win_rate_pct']:5.1f}% 价差复利{summary['price_return_pct']:+6.1f}% 评分{score_display} {signal} | {elapsed:.0f}s elapsed ETA {eta:.0f}s", flush=True)
                
            except Exception as e:
                results.append({'category': category, 'code': code, 'name': 'ERR', 'trades': 0, 'win_rate': 0, 'total_return_pct': 0, 'score': None, 'signal': '⚠️'})
                print(f"[{done}/{total}] {code} ERROR: {e}", flush=True)
    
    # ── 汇总 ──
    print("\n" + "="*100)
    print(f"综合策略回测汇总 — {total}只股票")
    print("="*100)
    
    for cat in STOCKS:
        cat_results = [r for r in results if r['category'] == cat]
        cat_scores = [r['score'] for r in cat_results if r['score'] is not None]
        cat_pnls = [r['total_return_pct'] for r in cat_results if r['trades'] > 0]
        cat_wins = [r['win_rate'] for r in cat_results if r['trades'] > 0]
        
        print(f"\n## {cat} ({len(cat_results)}只)")
        print(f"{'代码':<8} {'名称':<8} {'交易':>4} {'胜率':>7} {'价差复利':>8} {'评分':>6} {'信号'}")
        print("-"*55)
        for r in sorted(cat_results, key=lambda x: x['score'] or 0, reverse=True):
            score_display = f"{r['score']:.1f}" if r['score'] is not None else '—'
            print(f"{r['code']:<8} {r['name']:<8} {r['trades']:>4} {r['win_rate']:>6.1f}% {r['total_return_pct']:>+7.1f}% {score_display:>6} {r['signal']}")
        
        buy_cnt = sum(1 for r in cat_results if r['signal'] in ('强烈看多', '偏多'))
        avg_score = np.mean(cat_scores) if cat_scores else 0
        avg_win = np.mean(cat_wins) if cat_wins else 0
        avg_pnl = np.mean(cat_pnls) if cat_pnls else 0
        print(f"\n  🟢买入{buy_cnt}只 | 均分{avg_score:.1f} | 均胜率{avg_win:.1f}% | 均价差复利{avg_pnl:+.1f}%")
    
    # 总榜 TOP 20
    print(f"\n{'='*100}")
    print("🏆 综合评分 TOP 20")
    print(f"{'='*100}")
    valid = [r for r in results if r['score'] is not None]
    for i, r in enumerate(sorted(valid, key=lambda x: x['score'], reverse=True)[:20], 1):
        print(f"  {i:2d}. {r['code']} {r['name']:<8s} [{r['category']}] 评分{r['score']:.1f} 胜率{r['win_rate']:.1f}% 价差复利{r['total_return_pct']:+.1f}% {r['signal']}")
    
    # 总统计
    all_pnls = [r['total_return_pct'] for r in results if r['trades'] > 0]
    all_wins = [r['win_rate'] for r in results if r['trades'] > 0]
    all_scores = [r['score'] for r in results if r['score'] is not None]
    buy_total = sum(1 for r in results if r['signal'] in ('强烈看多', '偏多'))
    print(f"\n  总计: {len(results)}只 | 🟢买入{buy_total}只 | 均分{np.mean(all_scores):.1f} | 均胜率{np.mean(all_wins):.1f}% | 均价差复利{np.mean(all_pnls):+.1f}%")
    print(f"  耗时: {time.time()-t0:.0f}s")

if __name__ == '__main__':
    main()
