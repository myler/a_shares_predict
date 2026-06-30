#!/usr/bin/env python3
"""CLI入口 — 轻量参数解析+调度"""

import sys, os, json
from datetime import datetime
import numpy as np

from fetcher import fetch_kline, get_name, fetch_dividends, enrich_trades_with_dividends
from db import save_stock_name
from engine import (make_output_dir, calc_macd, detect_regime, find_divergences,
                    zero_line_cycles, backtest, predict, plot_all)

HELP = """
A股策略分析工具
══════════════════════════

用法:
  ./run_cli.py <股票代码>           出图+回测+预测 (默认)
  ./run_cli.py chart   <股票代码>   仅出图
  ./run_cli.py backtest <股票代码>  仅回测
  ./run_cli.py predict <股票代码>   仅预测
  ./run_cli.py help                 帮助

示例:
  ./run_cli.py 603893            瑞芯微 (全部)
  ./run_cli.py predict 000651    格力电器 (仅预测)
  ./run_cli.py backtest 002371   北方华创 (仅回测)

策略:
  MACD择时策略 — 牛市追涨、熊市空仓，背离信号触发买卖
"""


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] in ('help','-h','--help'):
        print(HELP)
        sys.exit(0)

    if sys.argv[1] in ('chart','backtest','predict','json'):
        mode, codes = sys.argv[1], sys.argv[2:]
    else:
        mode, codes = 'chart', sys.argv[1:]
    if not codes:
        print("请指定股票代码")
        sys.exit(1)

    if mode == 'json':
        import json as _json
        results = []
        for code in codes:
            try:
                data = fetch_kline(code)
                name = get_name(code)
                save_stock_name(code, name)
                dates = np.array([datetime.strptime(d['day'],'%Y-%m-%d') for d in data])
                closes = np.array([float(d['close']) for d in data])
                date_strs = [d['day'] for d in data]
                dif, dea, bar = calc_macd(closes)
                regimes, ma60 = detect_regime(closes)
                tops, bottoms = find_divergences(date_strs, closes, dif)
                i = len(dif)-1
                while i >= 0 and np.isnan(dif[i]): i -= 1
                recent_tops = [t for t in tops if (datetime.strptime(date_strs[-1],'%Y-%m-%d') - t['date']).days <= 120]
                recent_bottoms = [b for b in bottoms if (datetime.strptime(date_strs[-1],'%Y-%m-%d') - b['date']).days <= 120]
                dif_slope_5d = float(dif[i] - dif[max(0,i-5)]) if i>=5 else 0
                dif_slope_10d = float(dif[i] - dif[max(0,i-10)]) if i>=10 else 0
                golden = dif[i] > dea[i]
                cross_dist = dif[i] - dea[i] if golden else dea[i] - dif[i]
                cross_days = int(cross_dist / abs(dif_slope_5d/5)) if (golden and dif_slope_5d < 0) or (not golden and dif_slope_5d > 0) else -1
                score = 0
                if regimes[i] == 'bull': score += 2
                else: score -= 2
                if golden: score += 2
                else: score -= 2
                if dif[i] > 0: score += 1
                else: score -= 1
                if recent_bottoms: score += 2
                if recent_tops: score -= 2
                advice = "买入" if score >= 3 else ("持有" if score >= 0 else "卖出")
                results.append({
                    "code": code, "name": name,
                    "date": date_strs[i], "close": float(closes[i]),
                    "dif": round(float(dif[i]),4), "dea": round(float(dea[i]),4),
                    "bar": round(float(bar[i]),4),
                    "golden": bool(golden),
                    "regime": "bull" if regimes[i]=='bull' else "bear",
                    "zero_above": bool(dif[i] > 0),
                    "dif_slope_5d": round(dif_slope_5d,4),
                    "recent_top_div": len(recent_tops),
                    "recent_bot_div": len(recent_bottoms),
                    "cross_days": cross_days,
                    "score": score,
                    "advice": advice,
                })
            except Exception as e:
                results.append({"code": code, "error": str(e)})
        print(_json.dumps(results, ensure_ascii=False))
        sys.exit(0)

    code = codes[0]
    outdir = make_output_dir(code)
    print(f"输出: {outdir}\n")

    data = fetch_kline(code)
    name = get_name(code)
    save_stock_name(code, name)
    dates = np.array([datetime.strptime(d['day'],'%Y-%m-%d') for d in data])
    closes = np.array([float(d['close']) for d in data])
    date_strs = [d['day'] for d in data]

    dif, dea, bar = calc_macd(closes)
    regimes, ma60 = detect_regime(closes)
    tops, bottoms = find_divergences(date_strs, closes, dif)
    cycles = zero_line_cycles(date_strs, dif)
    trades = backtest(date_strs, closes, dif, dea, tops, bottoms, regimes)

    dividends = fetch_dividends(code)
    if dividends:
        trades = enrich_trades_with_dividends(trades, dividends)

    lines = []
    lines.append("="*60)
    lines.append(f"  {name}({code}) MACD回测")
    lines.append("="*60)
    lines.append(f"数据: {data[0]['day']} ~ {data[-1]['day']} ({len(data)}K线)")
    lines.append(f"牛熊: {'🟢牛' if regimes[-1]=='bull' else '🔴熊'} | 背离: 顶{len(tops)} 底{len(bottoms)}")
    lines.append(f"\n── 交易 ({len(trades)}笔) ──")
    for t in trades:
        div_info = ""
        if t.get('dividend_total', 0) > 0:
            div_info = f" [分红{t['dividend_total']:.2f}元/股]"
        lines.append(f"  {'✅' if t['profit_pct']>0 else '❌'} {t['buy_date']}→{t['sell_date']} "
                    f"{t['buy_price']:.2f}→{t['sell_price']:.2f} 价差{t['profit_pct']:+.1f}%{div_info} [{t['hold_days']}天]")
    if trades:
        wins=[t for t in trades if t['profit_pct']>0]; total=sum(t['profit_pct'] for t in trades)
        total_div_pct = sum(t.get('dividend_yield_pct', 0) for t in trades)
        total_div_cash = sum(t.get('dividend_total', 0) for t in trades)
        total_return = total + total_div_pct
        lines.append(f"\n  胜率: {len(wins)}/{len(trades)}={len(wins)/len(trades)*100:.0f}%")
        lines.append(f"  价差收益: {total:+.1f}%")
        lines.append(f"  分红收益: {total_div_cash:.2f}元/股（折合{total_div_pct:+.1f}%）")
        lines.append(f"  总收益:   {total_return:+.1f}%")

    if dividends:
        recent_divs = [d for d in dividends if d.get('ex_date', '') >= data[0]['day']]
        if recent_divs:
            lines.append(f"\n── 分红历史 ({len(recent_divs)}次) ──")
            for d in recent_divs:
                bonus_info = ""
                if d['bonus_share'] > 0: bonus_info += f" 送{d['bonus_share']:.0f}股"
                if d['transfer_share'] > 0: bonus_info += f" 转{d['transfer_share']:.0f}股"
                lines.append(f"  {d['ex_date']} 除权 | 10派{d['dividend_10']:.1f}元 | 每股{d['dividend_per_share']:.3f}元{bonus_info}")

    report = '\n'.join(lines)
    print(report)
    with open(os.path.join(outdir,'report.txt'),'w') as f: f.write(report)

    if mode == 'chart':
        path = plot_all(code, name, dates, closes, dif, dea, bar, cycles, tops, bottoms, trades, regimes, ma60, outdir)
        print(f"\n图表: {path}")

    if mode in ('chart', 'predict'):
        pred = predict(date_strs, closes, dif, dea, bar, regimes, tops, bottoms)
        pred_text = '\n'.join(pred)
        print(f"\n{pred_text}")
        with open(os.path.join(outdir,'predict.txt'),'w') as f: f.write(pred_text)

    print(f"\n完成 → {outdir}")
