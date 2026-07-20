"""mainboard_baseline.py 的可恢复任务测试。"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mainboard_baseline import (
    process_job,
    repair_success_metrics,
    replay_cached_baseline,
)


class TestBaselineWorker(unittest.TestCase):
    def test_repair_refuses_to_overwrite_different_strategy_version(self):
        with patch('mainboard_baseline.get_baseline_progress', return_value={
                'as_of_date': '2026-07-19',
                'strategy_version': 'comprehensive_next_open_v2',
        }), patch('mainboard_baseline.list_baseline_jobs') as jobs:
            with self.assertRaisesRegex(ValueError, '跨版本比较'):
                repair_success_metrics('test_run')

        jobs.assert_not_called()

    def test_replay_cached_baseline_reads_cache_without_fetching(self):
        job = {'code': '600000', 'result_status': 'eligible'}
        data = [{
            'day': '2026-01-01', 'open': '10', 'high': '11',
            'low': '9', 'close': '10', 'volume': '1000',
        }]
        summary = {
            'trades': 1, 'wins': 1, 'win_rate_pct': 100.0,
            'price_return_pct': 2.0,
        }

        with patch('mainboard_baseline.list_baseline_jobs', return_value=[job]), \
             patch('mainboard_baseline.get_baseline_progress', return_value={
                 'as_of_date': '2026-07-19',
                 'strategy_version': 'comprehensive_next_open_v2',
             }), \
             patch('mainboard_baseline.load_klines', return_value=data) as load, \
             patch('mainboard_baseline.fetch_kline') as fetch, \
             patch('mainboard_baseline.backtest_comprehensive', return_value=[{
                 'profit_pct': 2.0,
             }]), \
             patch('mainboard_baseline.summarize_trades', return_value=summary):
            replay = replay_cached_baseline('test_run')

        load.assert_called_once_with('600000', end_date='2026-07-19')
        fetch.assert_not_called()
        self.assertEqual(replay['as_of_date'], '2026-07-19')
        self.assertEqual(replay['source_strategy_version'],
                 'comprehensive_next_open_v2')
        self.assertEqual(replay['replayed_count'], 1)
        self.assertEqual(replay['global_trade_win_rate_pct'], 100.0)

    def test_repair_success_metrics_reads_cache_without_fetching(self):
        job = {'code': '600000', 'result_status': 'eligible'}
        data = [{
            'day': '2026-01-01', 'open': '10', 'high': '11',
            'low': '9', 'close': '10', 'volume': '1000',
        }]
        summary = {
            'trades': 1, 'wins': 1, 'win_rate_pct': 100.0,
            'price_return_pct': 2.0,
        }

        with patch('mainboard_baseline.list_baseline_jobs', return_value=[job]), \
             patch('mainboard_baseline.get_baseline_progress', return_value={
                 'as_of_date': '2026-07-19',
                 'strategy_version': 'comprehensive_next_open_v3',
             }), \
             patch('mainboard_baseline.load_klines', return_value=data) as load, \
             patch('mainboard_baseline.fetch_kline') as fetch, \
             patch('mainboard_baseline.backtest_comprehensive', return_value=[{
                 'profit_pct': 2.0,
             }]), \
             patch('mainboard_baseline.summarize_trades', return_value=summary), \
             patch('mainboard_baseline.predict_comprehensive', return_value={
                 'composite': 69.0, 'signal': '偏多',
             }), \
             patch('mainboard_baseline.complete_baseline_job') as complete:
            repaired = repair_success_metrics('test_run')

        self.assertEqual(repaired, 1)
        load.assert_called_once_with('600000', end_date='2026-07-19')
        fetch.assert_not_called()
        self.assertEqual(complete.call_args.args[0:3],
                         ('test_run', '600000', 'eligible'))

    def test_stale_history_is_completed_exclusion(self):
        data = [{
            'day': '2026-01-01', 'open': '10', 'high': '11',
            'low': '9', 'close': '10', 'volume': '1000',
        }] * 300
        job = {'code': '600000', 'attempts': 1}

        with patch('mainboard_baseline.fetch_kline', return_value=data) as fetch, \
             patch('mainboard_baseline.complete_baseline_job') as complete:
            status, detail = process_job(
                'test_run', job, '2026-07-19', 30, 1800, 12)

        self.assertEqual(status, 'stale_history')
        self.assertIsNone(detail)
        fetch.assert_called_once_with(
            '600000', max_retries=1, force_refresh=True,
            allow_stale_on_refresh=False)
        self.assertEqual(complete.call_args.args[2], 'stale_history')

    def test_max_attempts_archives_permanent_fetch_failure(self):
        job = {'code': '600000', 'attempts': 3}
        with patch('mainboard_baseline.fetch_kline',
                   side_effect=RuntimeError('all sources empty')) as fetch, \
             patch('mainboard_baseline.complete_baseline_job') as complete, \
             patch('mainboard_baseline.fail_baseline_job') as fail:
            status, detail = process_job(
                'test_run', job, '2026-07-19', 30, 1800, 3)

        self.assertEqual(status, 'fetch_failed')
        self.assertIn('RuntimeError', detail)
        fetch.assert_called_once()
        fail.assert_not_called()
        self.assertEqual(complete.call_args.args[2], 'fetch_failed')
        self.assertIn('RuntimeError', complete.call_args.kwargs['last_error'])


if __name__ == '__main__':
    unittest.main(verbosity=2)