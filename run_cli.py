#!/usr/bin/env python3
"""CLI入口 — 轻量参数解析+调度"""

import difflib
import re
import sys, os, json
from datetime import datetime
import numpy as np

from fetcher import (fetch_kline, get_name, fetch_dividends,
                     enrich_trades_with_dividends, fetch_financial_summaries)
from db import save_stock_name
from fundamentals import screen_value_quality
from engine import (make_output_dir, calc_macd, detect_regime, find_divergences,
                    zero_line_cycles, backtest, predict, format_predict,
                    backtest_multifactor, predict_multifactor, format_predict_multifactor,
                    predict_comprehensive)
from plotting import plot_all

HELP = """
A股策略分析工具
══════════════════════════

用法:
  ./run_cli.py <股票代码>           出图+回测+预测 (默认)
  ./run_cli.py chart   <股票代码>   仅出图
  ./run_cli.py backtest <股票代码>  仅回测(MACD)
  ./run_cli.py multi   <股票代码>   多因子共振策略
    ./run_cli.py predict <股票代码>   融合策略预测（不回测、不出图）
    ./run_cli.py bmfund  <股票代码>   巴芒财务质量初筛（非交易信号）
  ./run_cli.py help                 帮助

示例:
  ./run_cli.py 603893            瑞芯微 (全部)
  ./run_cli.py predict 000651    格力电器 (仅预测)
    ./run_cli.py bmfund 601888     中国中免 (基本面研究)
  ./run_cli.py backtest 002371   北方华创 (仅回测)

策略:
  融合策略 — MACD择时、多因子、价格位置/趋势和量能综合评分
  巴芒基本面研究 — 非金融企业财务质量初筛，不输出买卖建议
"""

CLI_MODES = ('chart', 'backtest', 'predict', 'json', 'multi', 'bmfund')
SINGLE_CODE_MODES = frozenset(CLI_MODES) - {'json'}


def parse_cli_args(arguments):
    """验证命令行输入，避免把拼写错误的子命令误当成股票代码。"""
    if not arguments:
        raise ValueError('请指定子命令或6位股票代码')

    first = arguments[0]
    if first in CLI_MODES:
        mode, codes = first, arguments[1:]
    elif re.fullmatch(r'[0-9]{6}', first):
        mode, codes = 'chart', arguments
    else:
        suggestion = difflib.get_close_matches(first, CLI_MODES, n=1, cutoff=0.6)
        if suggestion:
            raise ValueError(
                f'未知子命令 "{first}"；是否想使用 "{suggestion[0]}"？')
        raise ValueError(
            f'未知子命令或无效股票代码 "{first}"；股票代码必须为6位数字')

    if not codes:
        raise ValueError(f'{mode} 需要一个6位股票代码')
    if mode in SINGLE_CODE_MODES and len(codes) != 1:
        raise ValueError(f'{mode} 一次只能分析一个6位股票代码')
    invalid_codes = [code for code in codes if re.fullmatch(r'[0-9]{6}', code) is None]
    if invalid_codes:
        raise ValueError(
            f'股票代码必须为6位数字：{", ".join(invalid_codes)}')
    return mode, codes


def format_comprehensive_cli_prediction(prediction, name, code, dates):
    if 'error' in prediction:
        return [f"{name}({code}) 融合策略", prediction['error']]

    scores = prediction['scores']
    lines = [
        '=' * 60,
        f"  {name}({code}) 融合策略",
        '=' * 60,
        f"数据: {dates[0]} ~ {dates[-1]} ({len(dates)}K线)",
        f"收盘: {prediction['close']:.2f} | 信号: {prediction['signal']} | 建议: {prediction['action']}",
        "\n── 融合策略四维评分 ──",
        f"  MACD核心 (40%): {scores['macd']['score']:.1f} 分 — "
        f"DIF{scores['macd']['dif']:.2f}/DEA{scores['macd']['dea']:.2f} "
        f"BAR{scores['macd']['bar']:.2f}",
        f"  多因子 (30%): {scores['multifactor']['score']:.1f} 分 — "
        f"RSI{scores['multifactor']['rsi']:.0f} "
        f"K{scores['multifactor']['k']:.0f}/D{scores['multifactor']['d']:.0f}/"
        f"J{scores['multifactor']['j']:.0f} WR{scores['multifactor']['wr']:.0f}",
        f"  {scores['fundamental']['label']} (15%): "
        f"{scores['fundamental']['score']:.1f} 分",
        f"  量能 (15%): {scores['game']['score']:.1f} 分 — "
        f"OBV 5日净量能流 {scores['game']['obv_flow']:+.0%}",
        f"  四维基础分 S: {prediction['base_composite']:.1f} "
        f"(M{scores['macd']['score']:.0f}×0.40 + "
        f"F{scores['multifactor']['score']:.0f}×0.30 + "
        f"价{scores['fundamental']['score']:.0f}×0.15 + "
        f"量{scores['game']['score']:.0f}×0.15)",
        f"  辅助共识修正: {prediction['auxiliary_adjustment']:+.1f} "
        f"(A={prediction['auxiliary_consensus']['consensus_score']:+.0f}, "
        f"β={prediction['auxiliary_beta']:.2f})",
        f"  有效评分 S*: {prediction['composite']:.1f}",
        "\n── 质量门禁 ──",
    ]

    if prediction['gates']['passed']:
        lines.append("  全部通过")
    else:
        for gate in prediction['gates']['details']:
            if gate['gate'] == 'OBV净量能流':
                lines.append(f"  OBV 5日净量能流过低 ({gate['value']:+.0%} < -60%) -> 否决")
            elif gate['gate'] == 'RSI':
                lines.append(f"  RSI极度超买 ({gate['value']:.0f} > 92) -> 否决")
            elif gate['gate'] == '双弱':
                lines.append("  双弱 (MACD<35 且 多因子<40) -> 否决")

    auxiliary = prediction['auxiliary_consensus']
    lines.extend([
        f"\n── 辅助指标投票面板 ({auxiliary['total_indicators']}个指标) ──",
        f"  共识度 {auxiliary['consensus_score']:+.0f} "
        f"({auxiliary['consensus_pct']:.0f}%看多) | 修正 {prediction['auxiliary_adjustment']:+.1f} | "
        f"看多{auxiliary['bullish_count']} 看空{auxiliary['bearish_count']} 中性{auxiliary['neutral_count']}",
    ])
    for vote in auxiliary['votes']:
        value = vote['value']
        if isinstance(value, dict):
            value = ', '.join(f'{key}={item}' for key, item in value.items())
        label = '看多' if vote['vote'] > 0 else ('看空' if vote['vote'] < 0 else '中性')
        lines.append(f"  {vote['name']}: {value} -> {label}")

    for title, key in (
        ('MACD核心评分 (40%)', 'macd'),
        ('多因子评分 (30%)', 'multifactor'),
        ('价格位置/趋势 (15%)', 'fundamental'),
        ('量能 (15%)', 'game'),
    ):
        lines.append(f"\n── {title} ──")
        for item in prediction['breakdown'][key]:
            lines.append(f"  {item['rule']}: {item['adj']}")

    return lines


def format_bmfund_cli_research(research, name, code):
    """格式化巴芒财务质量初筛；输出研究状态而非交易建议。"""
    status_labels = {
        'pass': '通过',
        'watch': '需核验',
        'fail': '未通过',
        'unavailable': '数据不足',
    }

    def amount(value):
        return '—' if value is None else f'{value / 1e8:.2f}亿元'

    def percent(value):
        return '—' if value is None else f'{value:.1f}%'

    def ratio(value):
        return '—' if value is None else f'{value:.2f}'

    lines = [
        '=' * 60,
        f'  {name}({code}) 巴芒基本面研究',
        '=' * 60,
        f"研究结论: {research['status_label']}（非买卖信号）",
        f"说明: {research['conclusion']}",
        f"数据源: {research['source']}；以年报摘要为初筛依据",
        f"范围: {research['scope']}",
        '\n── 财务质量初筛 ──',
    ]
    for check in research['checks']:
        lines.append(
            f"  [{status_labels[check['status']]}] {check['label']}: {check['detail']}"
        )

    reports = research['annual_reports'][:5]
    if reports:
        lines.append('\n── 最近五份已披露年报摘要 ──')
        for report in reports:
            lines.append(
                f"  {report['report_name'] or report['report_date']} | 披露 {report['notice_date'] or '—'} | "
                f"营收 {amount(report['revenue'])} | 扣非归母 {amount(report['core_profit'])} | "
                f"经营现金流 {amount(report['operating_cash_flow'])} | ROE {percent(report['roe'])} | "
                f"ROIC {percent(report['roic'])} | 资产负债率 {percent(report['debt_ratio'])} | "
                f"流动比率 {ratio(report['current_ratio'])}"
            )

    lines.append('\n── 尚未覆盖的研究项 ──')
    lines.extend(f'  - {item}' for item in research['limitations'])
    return lines


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] in ('help','-h','--help'):
        print(HELP)
        sys.exit(0)

    try:
        mode, codes = parse_cli_args(sys.argv[1:])
    except ValueError as error:
        print(f'参数错误: {error}', file=sys.stderr)
        print('使用 ./run_cli.py help 查看可用命令。', file=sys.stderr)
        sys.exit(2)

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
                pred = predict(date_strs, closes, dif, dea, bar, regimes, tops, bottoms)
                pred['code'] = code
                pred['name'] = name
                results.append(pred)
            except Exception as e:
                results.append({"code": code, "error": str(e)})
        print(_json.dumps(results, ensure_ascii=False, default=str))
        sys.exit(0)

    if mode == 'bmfund':
        code = codes[0]
        research = screen_value_quality(fetch_financial_summaries(code))
        name = research['company'] or get_name(code)
        if name and name != code:
            save_stock_name(code, name)
        print('\n'.join(format_bmfund_cli_research(research, name, code)))
        sys.exit(0)

    # ── 多因子共振策略 ──
    if mode == 'multi':
        code = codes[0]
        data = fetch_kline(code)
        name = get_name(code)
        save_stock_name(code, name)
        dates = [d['day'] for d in data]
        closes = np.array([float(d['close']) for d in data])
        highs = np.array([float(d['high']) for d in data])
        lows = np.array([float(d['low']) for d in data])
        vols = np.array([float(d['volume']) for d in data])

        trades = backtest_multifactor(dates, closes, highs, lows, vols)
        dividends = fetch_dividends(code)
        if dividends:
            trades = enrich_trades_with_dividends(trades, dividends)

        pred = format_predict_multifactor(
            predict_multifactor(dates, closes, highs, lows, vols))

        print(f"  {name}({code}) 多因子共振回测")
        print(f"  数据: {dates[0]} ~ {dates[-1]} ({len(data)}K线)")
        print(f"\n  交易 ({len(trades)}笔):")
        for t in trades:
            div_info = ""
            if t.get('dividend_total', 0) > 0:
                div_info = f" [分红{t['dividend_total']:.2f}元/股]"
            print(f"  {'✅' if t['profit_pct']>0 else '❌'} {t['buy_date']}→{t['sell_date']} "
                  f"{t['buy_price']:.2f}→{t['sell_price']:.2f} {t['profit_pct']:+.1f}%{div_info} [{t['hold_days']}天]")
        if trades:
            wins = [t for t in trades if t['profit_pct'] > 0]
            total = sum(t['profit_pct'] for t in trades)
            total_div = sum(t.get('dividend_yield_pct', 0) for t in trades)
            total_cash = sum(t.get('dividend_total', 0) for t in trades)
            print(f"\n  胜率: {len(wins)}/{len(trades)}={len(wins)/len(trades)*100:.0f}%")
            print(f"  价差收益: {total:+.1f}%")
            print(f"  分红收益: {total_cash:.2f}元/股（折合{total_div:+.1f}%）")
            print(f"  总收益:   {total+total_div:+.1f}%")
        sys.exit(0)

    if mode == 'predict':
        code = codes[0]
        data = fetch_kline(code)
        name = get_name(code)
        save_stock_name(code, name)
        dates = [row['day'] for row in data]
        opens = np.array([float(row['open']) for row in data])
        closes = np.array([float(row['close']) for row in data])
        highs = np.array([float(row['high']) for row in data])
        lows = np.array([float(row['low']) for row in data])
        volumes = np.array([float(row['volume']) for row in data])
        prediction = predict_comprehensive(
            dates, closes, highs, lows, volumes, opens=opens)
        print('\n'.join(format_comprehensive_cli_prediction(
            prediction, name, code, dates)))
        sys.exit(0)

    code = codes[0]  # 非json/multi模式取第一个
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
        pred = format_predict(predict(date_strs, closes, dif, dea, bar, regimes, tops, bottoms))
        pred_text = '\n'.join(pred)
        print(f"\n{pred_text}")
        with open(os.path.join(outdir,'predict.txt'),'w') as f: f.write(pred_text)

    print(f"\n完成 → {outdir}")
