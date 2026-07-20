#!/usr/bin/env python3
"""沪深主板融合策略基线：DB 持久化、可中断恢复、失败持续重试。"""

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np

from db import (
    claim_baseline_jobs,
    complete_baseline_job,
    create_baseline_run,
    fail_baseline_job,
    find_latest_baseline_run,
    find_latest_open_baseline_run,
    get_baseline_progress,
    get_next_baseline_retry_at,
    list_baseline_jobs,
    recover_running_baseline_jobs,
    requeue_baseline_jobs,
    save_baseline_universe,
    supersede_baseline_run,
    load_klines,
)
from engine import backtest_comprehensive, predict_comprehensive, summarize_trades
from fetcher import fetch_kline, fetch_mainboard_universe


UNIVERSE_NAME = 'cn_sh_sz_mainboard'
STRATEGY_VERSION = 'comprehensive_next_open_v4_auxiliary'
MIN_KLINES = 300
MAX_STALE_DAYS = 60


def _utc_date():
    return datetime.now(timezone.utc).date().isoformat()


def _new_run_id(as_of_date):
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return f'{UNIVERSE_NAME}_{as_of_date.replace("-", "")}_{timestamp}'


def _days_since(as_of_date, last_kline_date):
    return (datetime.strptime(as_of_date, '%Y-%m-%d').date() -
            datetime.strptime(last_kline_date, '%Y-%m-%d').date()).days


def initialize_run(run_id=None, as_of_date=None, force_new=False):
    """恢复最新未完成运行，或冻结一份新的主板股票池。"""
    as_of_date = as_of_date or _utc_date()
    if run_id is None and not force_new:
        run_id = find_latest_open_baseline_run(UNIVERSE_NAME)
    if run_id is None:
        run_id = _new_run_id(as_of_date)

    create_baseline_run(run_id, UNIVERSE_NAME, STRATEGY_VERSION, as_of_date)
    progress = get_baseline_progress(run_id)
    if progress['total_count'] == 0:
        universe = fetch_mainboard_universe()
        save_baseline_universe(run_id, universe)
        progress = get_baseline_progress(run_id)
        print(f'已冻结主板股票池: {len(universe)} 只 | run_id={run_id}', flush=True)

    recovered = recover_running_baseline_jobs(run_id)
    if recovered:
        print(f'已恢复 {recovered} 个异常中断任务到失败重试池', flush=True)
    return run_id


def process_job(run_id, job, as_of_date, retry_base_seconds, retry_max_seconds,
                max_attempts):
    """抓取单只股票并持久化回测摘要；异常只进入失败池，不丢任务。"""
    code = job['code']
    try:
        data = fetch_kline(
            code, max_retries=1, force_refresh=True,
            allow_stale_on_refresh=False)
        dates = [row['day'] for row in data]
        opens = np.array([float(row['open']) for row in data])
        closes = np.array([float(row['close']) for row in data])
        highs = np.array([float(row['high']) for row in data])
        lows = np.array([float(row['low']) for row in data])
        volumes = np.array([float(row['volume']) for row in data])
        kline_count = len(data)
        first_date = dates[0] if dates else None
        last_date = dates[-1] if dates else None

        if kline_count < MIN_KLINES:
            complete_baseline_job(
                run_id, code, 'insufficient_history', kline_count, first_date,
                last_date)
            return 'insufficient_history', None

        if _days_since(as_of_date, last_date) > MAX_STALE_DAYS:
            complete_baseline_job(
                run_id, code, 'stale_history', kline_count, first_date,
                last_date)
            return 'stale_history', None

        trades = backtest_comprehensive(
            dates, closes, highs, lows, volumes, opens=opens)
        summary = summarize_trades(trades)
        win_count = sum(trade['profit_pct'] > 0 for trade in trades)
        prediction = predict_comprehensive(
            dates, closes, highs, lows, volumes, opens=opens)
        complete_baseline_job(
            run_id, code, 'eligible', kline_count, first_date, last_date,
            trade_count=summary['trades'],
            win_count=win_count,
            win_rate_pct=round(summary['win_rate_pct'], 4),
            price_return_pct=round(summary['price_return_pct'], 4),
            composite_score=prediction.get('composite'),
            signal=prediction.get('signal'),
        )
        return 'eligible', summary
    except Exception as error:
        error_text = f'{type(error).__name__}: {error}'
        if job['attempts'] >= max_attempts:
            complete_baseline_job(
                run_id, code, 'fetch_failed', None, None, None,
                last_error=error_text)
            return 'fetch_failed', error_text
        retry_seconds = min(
            retry_max_seconds,
            retry_base_seconds * (2 ** min(job['attempts'] - 1, 10)),
        )
        fail_baseline_job(run_id, code, error, retry_seconds)
        return 'failed', error_text


def repair_success_metrics(run_id):
    """从已缓存 K 线重算成功池摘要，不发起任何网络请求。"""
    progress = get_baseline_progress(run_id)
    if not progress:
        raise ValueError(f'未知基线运行: {run_id}')
    if progress['strategy_version'] != STRATEGY_VERSION:
        raise ValueError(
            f'{run_id} 使用 {progress["strategy_version"]}；当前修复策略为 '
            f'{STRATEGY_VERSION}。跨版本比较请使用 --replay-cached。')
    as_of_date = progress['as_of_date']
    repaired = 0
    for job in list_baseline_jobs(run_id, 'success'):
        if job['result_status'] != 'eligible':
            continue
        data = load_klines(job['code'], end_date=as_of_date)
        if not data:
            raise RuntimeError(f'{job["code"]} 缺少本地 K 线缓存，无法离线重算')
        dates = [row['day'] for row in data]
        opens = np.array([float(row['open']) for row in data])
        closes = np.array([float(row['close']) for row in data])
        highs = np.array([float(row['high']) for row in data])
        lows = np.array([float(row['low']) for row in data])
        volumes = np.array([float(row['volume']) for row in data])
        trades = backtest_comprehensive(
            dates, closes, highs, lows, volumes, opens=opens)
        summary = summarize_trades(trades)
        prediction = predict_comprehensive(
            dates, closes, highs, lows, volumes, opens=opens)
        complete_baseline_job(
            run_id, job['code'], 'eligible', len(data), dates[0], dates[-1],
            trade_count=summary['trades'],
            win_count=sum(trade['profit_pct'] > 0 for trade in trades),
            win_rate_pct=summary['win_rate_pct'],
            price_return_pct=summary['price_return_pct'],
            composite_score=prediction.get('composite'),
            signal=prediction.get('signal'),
        )
        repaired += 1
    return repaired


def replay_cached_baseline(run_id):
    """只读 SQLite 缓存重放已完成基线，不下载也不写回摘要。"""
    progress = get_baseline_progress(run_id)
    if not progress:
        raise ValueError(f'未知基线运行: {run_id}')
    as_of_date = progress['as_of_date']
    started = time.perf_counter()
    eligible_jobs = [
        job for job in list_baseline_jobs(run_id, 'success')
        if job['result_status'] == 'eligible'
    ]
    stock_win_rates = []
    stock_price_returns = []
    total_trades = 0
    total_wins = 0
    missing_cached_count = 0

    for job in eligible_jobs:
        data = load_klines(job['code'], end_date=as_of_date)
        if not data:
            missing_cached_count += 1
            continue
        dates = [row['day'] for row in data]
        opens = np.array([float(row['open']) for row in data])
        closes = np.array([float(row['close']) for row in data])
        highs = np.array([float(row['high']) for row in data])
        lows = np.array([float(row['low']) for row in data])
        volumes = np.array([float(row['volume']) for row in data])
        summary = summarize_trades(backtest_comprehensive(
            dates, closes, highs, lows, volumes, opens=opens))
        stock_win_rates.append(float(summary['win_rate_pct']))
        stock_price_returns.append(float(summary['price_return_pct']))
        total_trades += int(summary['trades'])
        total_wins += int(summary['wins'])

    replayed_count = len(stock_win_rates)
    return {
        'run_id': run_id,
        'as_of_date': as_of_date,
        'strategy_version': STRATEGY_VERSION,
        'source_strategy_version': progress['strategy_version'],
        'eligible_count': len(eligible_jobs),
        'replayed_count': replayed_count,
        'missing_cached_count': missing_cached_count,
        'mean_stock_win_rate_pct': (
            float(np.mean(stock_win_rates)) if stock_win_rates else 0.0),
        'mean_stock_price_return_pct': (
            float(np.mean(stock_price_returns)) if stock_price_returns else 0.0),
        'total_trade_count': total_trades,
        'total_win_count': total_wins,
        'global_trade_win_rate_pct': (
            total_wins / total_trades * 100 if total_trades else 0.0),
        'elapsed_seconds': round(time.perf_counter() - started, 3),
    }


def _print_progress(run_id):
    progress = get_baseline_progress(run_id)
    print(json.dumps(progress, ensure_ascii=False, sort_keys=True), flush=True)
    return progress


def run_worker(run_id, batch_size, max_jobs, request_delay,
               retry_base_seconds, retry_max_seconds, max_attempts,
               until_complete):
    processed = 0
    while True:
        progress = get_baseline_progress(run_id)
        if progress['status'] == 'completed':
            _print_progress(run_id)
            return 0
        if max_jobs and processed >= max_jobs:
            _print_progress(run_id)
            return 0

        remaining = max_jobs - processed if max_jobs else batch_size
        jobs = claim_baseline_jobs(run_id, min(batch_size, remaining))
        if not jobs:
            _print_progress(run_id)
            if not until_complete:
                return 0
            retry_at = get_next_baseline_retry_at(run_id)
            if retry_at:
                print(f'无可立即处理任务，失败池将在 {retry_at} 后重试', flush=True)
            else:
                print('无可立即处理任务，等待任务状态更新', flush=True)
            time.sleep(15)
            continue

        for job in jobs:
            status, detail = process_job(
                run_id, job, progress['as_of_date'], retry_base_seconds,
                retry_max_seconds, max_attempts)
            processed += 1
            if status == 'eligible':
                print(
                    f'[{processed}] {job["code"]} 成功 | '
                    f'{detail["trades"]}笔 | 胜率{detail["win_rate_pct"]:.1f}% | '
                    f'价差复利{detail["price_return_pct"]:+.1f}%',
                    flush=True,
                )
            elif status == 'insufficient_history':
                print(f'[{processed}] {job["code"]} 成功 | 数据不足 {MIN_KLINES} 根', flush=True)
            elif status == 'stale_history':
                print(f'[{processed}] {job["code"]} 成功 | 最近日线超过 {MAX_STALE_DAYS} 天', flush=True)
            elif status == 'fetch_failed':
                print(
                    f'[{processed}] {job["code"]} 已归档 | 连续 {max_attempts} 次抓取失败 | {detail}',
                    flush=True,
                )
            else:
                print(f'[{processed}] {job["code"]} 失败 | {detail}', flush=True)
            if request_delay:
                time.sleep(request_delay)


def main():
    parser = argparse.ArgumentParser(
        description='持续抓取沪深主板 K 线并建立融合策略基线')
    parser.add_argument('--run-id', help='指定已有 run_id；省略时自动恢复最近未完成任务')
    parser.add_argument('--as-of-date', help='首次冻结池的日期，格式 YYYY-MM-DD')
    parser.add_argument('--new-run', action='store_true', help='忽略可恢复运行并冻结新股票池')
    parser.add_argument('--init-only', action='store_true', help='仅创建/恢复任务池，不处理股票')
    parser.add_argument('--status', action='store_true', help='显示当前进度，不处理股票')
    parser.add_argument('--show-failed', type=int, default=0,
                        help='状态查询时显示前 N 个失败任务')
    parser.add_argument('--repair-success-metrics', action='store_true',
                        help='直接读取 SQLite 缓存重算成功池指标，不下载数据')
    parser.add_argument('--replay-cached', action='store_true',
                        help='只读 SQLite 缓存重放已完成基线，不下载也不写回摘要')
    parser.add_argument('--requeue-success', action='store_true',
                        help='将成功池重新置为待处理，用于强制刷新全部 K 线')
    parser.add_argument('--requeue-failed', action='store_true',
                        help='将失败池立即重新置为待处理，用于修复 worker 后重试')
    parser.add_argument('--supersede', action='store_true',
                        help='将指定运行标为 superseded，不再作为活动基线恢复')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='每次原子领取的任务数，默认 1')
    parser.add_argument('--max-jobs', type=int, default=0,
                        help='本次最多处理的股票数；0 表示处理全部当前可领取任务')
    parser.add_argument('--until-complete', action='store_true',
                        help='持续重试失败任务，直到失败池为 0 且运行完成')
    parser.add_argument('--request-delay', type=float, default=0.2,
                        help='每只股票完成后等待秒数，默认 0.2')
    parser.add_argument('--retry-base-seconds', type=int, default=30,
                        help='首次失败重试延迟秒数，之后指数退避')
    parser.add_argument('--retry-max-seconds', type=int, default=1800,
                        help='单次失败最大重试延迟秒数')
    parser.add_argument('--max-attempts', type=int, default=12,
                        help='单只股票达到此抓取尝试次数后归档为 fetch_failed，默认 12')
    args = parser.parse_args()

    if args.status:
        run_id = (args.run_id or find_latest_open_baseline_run(UNIVERSE_NAME)
                  or find_latest_baseline_run(UNIVERSE_NAME))
        if not run_id:
            print('没有主板基线任务', flush=True)
            return 0
        _print_progress(run_id)
        if args.show_failed:
            print(json.dumps(
                list_baseline_jobs(run_id, 'failed', args.show_failed),
                ensure_ascii=False, indent=2), flush=True)
        return 0

    if args.replay_cached:
        run_id = args.run_id or find_latest_baseline_run(UNIVERSE_NAME)
        if not run_id:
            print('没有主板基线任务', flush=True)
            return 0
        print(json.dumps(
            replay_cached_baseline(run_id), ensure_ascii=False, sort_keys=True),
            flush=True)
        return 0

    if args.supersede:
        if not args.run_id:
            parser.error('--supersede 需要 --run-id')
        supersede_baseline_run(args.run_id)
        print(f'已标记基线运行为 superseded: {args.run_id}', flush=True)
        _print_progress(args.run_id)
        return 0

    run_id = initialize_run(args.run_id, args.as_of_date, args.new_run)
    if args.requeue_success:
        count = requeue_baseline_jobs(run_id, 'success')
        print(f'已将 {count} 个成功任务重新放回待处理池', flush=True)
        _print_progress(run_id)
        return 0
    if args.requeue_failed:
        count = requeue_baseline_jobs(run_id, 'failed')
        print(f'已将 {count} 个失败任务重新放回待处理池', flush=True)
        _print_progress(run_id)
        return 0
    if args.repair_success_metrics:
        repaired = repair_success_metrics(run_id)
        print(f'已回填 {repaired} 个成功任务的指标', flush=True)
        _print_progress(run_id)
        return 0
    _print_progress(run_id)
    if args.init_only:
        return 0
    return run_worker(
        run_id, max(1, args.batch_size), max(0, args.max_jobs),
        max(0.0, args.request_delay), max(1, args.retry_base_seconds),
        max(1, args.retry_max_seconds), max(1, args.max_attempts),
        args.until_complete,
    )


if __name__ == '__main__':
    raise SystemExit(main())