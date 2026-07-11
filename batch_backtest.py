#!/usr/bin/env python3
"""批量回测综合策略 — 100只股票"""

import sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetcher import fetch_kline, get_name
from engine import backtest_comprehensive, predict_comprehensive

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
                name = get_name(code)
                dates = [d['day'] for d in data]
                closes = np.array([float(d['close']) for d in data])
                highs = np.array([float(d['high']) for d in data])
                lows = np.array([float(d['low']) for d in data])
                vols = np.array([float(d['volume']) for d in data])
                
                trades = backtest_comprehensive(dates, closes, highs, lows, vols)
                
                wins = [t for t in trades if t['profit_pct'] > 0]
                total_pnl = sum(t['profit_pct'] for t in trades)
                win_rate = len(wins)/len(trades)*100 if trades else 0
                
                # 当前评分
                pred = predict_comprehensive(dates, closes, highs, lows, vols, holding=False)
                comp_score = None
                for line in pred:
                    if '加权总分' in line:
                        try: comp_score = float(line.split(':')[1].strip().split()[0])
                        except: pass
                
                results.append({
                    'category': category, 'code': code, 'name': name,
                    'trades': len(trades), 'win_rate': round(win_rate, 1),
                    'total_pnl': round(total_pnl, 1),
                    'score': round(comp_score, 1) if comp_score else None,
                    'signal': '🟢' if comp_score and comp_score >= 60 else ('🟡' if comp_score and comp_score >= 45 else ('🔴' if comp_score else '—')),
                })
                
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f"[{done}/{total}] {code} {name:6s} {len(trades):2d}笔 胜率{win_rate:5.1f}% 收益{total_pnl:+6.1f}% 评分{comp_score:.0f}  | {elapsed:.0f}s elapsed ETA {eta:.0f}s", flush=True)
                
            except Exception as e:
                results.append({'category': category, 'code': code, 'name': 'ERR', 'trades': 0, 'win_rate': 0, 'total_pnl': 0, 'score': None, 'signal': '⚠️'})
                print(f"[{done}/{total}] {code} ERROR: {e}", flush=True)
    
    # ── 汇总 ──
    print("\n" + "="*100)
    print("综合策略回测汇总 — 100只股票")
    print("="*100)
    
    for cat in ["芯片", "资源", "消费"]:
        cat_results = [r for r in results if r['category'] == cat]
        cat_scores = [r['score'] for r in cat_results if r['score']]
        cat_pnls = [r['total_pnl'] for r in cat_results if r['trades'] > 0]
        cat_wins = [r['win_rate'] for r in cat_results if r['trades'] > 0]
        
        print(f"\n## {cat} ({len(cat_results)}只)")
        print(f"{'代码':<8} {'名称':<8} {'交易':>4} {'胜率':>7} {'收益':>8} {'评分':>6} {'信号'}")
        print("-"*55)
        for r in sorted(cat_results, key=lambda x: x['score'] or 0, reverse=True):
            print(f"{r['code']:<8} {r['name']:<8} {r['trades']:>4} {r['win_rate']:>6.1f}% {r['total_pnl']:>+7.1f}% {r['score'] or '—':>6} {r['signal']}")
        
        buy_cnt = sum(1 for r in cat_results if r['signal'] == '🟢')
        avg_score = np.mean(cat_scores) if cat_scores else 0
        avg_win = np.mean(cat_wins) if cat_wins else 0
        avg_pnl = np.mean(cat_pnls) if cat_pnls else 0
        print(f"\n  🟢买入{buy_cnt}只 | 均分{avg_score:.1f} | 均胜率{avg_win:.1f}% | 均收益{avg_pnl:+.1f}%")
    
    # 总榜 TOP 20
    print(f"\n{'='*100}")
    print("🏆 综合评分 TOP 20")
    print(f"{'='*100}")
    valid = [r for r in results if r['score']]
    for i, r in enumerate(sorted(valid, key=lambda x: x['score'], reverse=True)[:20], 1):
        print(f"  {i:2d}. {r['code']} {r['name']:<8s} [{r['category']}] 评分{r['score']:.1f} 胜率{r['win_rate']:.1f}% 收益{r['total_pnl']:+.1f}% {r['signal']}")
    
    # 总统计
    all_pnls = [r['total_pnl'] for r in results if r['trades'] > 0]
    all_wins = [r['win_rate'] for r in results if r['trades'] > 0]
    all_scores = [r['score'] for r in results if r['score']]
    buy_total = sum(1 for r in results if r['signal'] == '🟢')
    print(f"\n  总计: {len(results)}只 | 🟢买入{buy_total}只 | 均分{np.mean(all_scores):.1f} | 均胜率{np.mean(all_wins):.1f}% | 均收益{np.mean(all_pnls):+.1f}%")
    print(f"  耗时: {time.time()-t0:.0f}s")

if __name__ == '__main__':
    main()
