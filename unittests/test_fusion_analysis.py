"""Fresh-analysis contracts using synthetic 280-bar histories and frozen peers.

No production database, network, historical replay, or real report is accessed.
The shared indicator/action math is exercised, not replaced by all-green votes.
Only deliberately malformed risk/indicator outputs are mocked in edge tests.
"""

from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import call, patch
from zipfile import BadZipFile

import numpy as np

import fusion_analysis as analysis
from live_data import RefreshError
from pareto_strategy import RiskConfig, STRATEGY_VERSION


CODE, PEER, OTHER = '000001', '600000', '600001'
TODAY = '2026-09-10'


def fresh_fixture(n=280, latest=TODAY, checked_at=TODAY + 'T16:00:00+08:00'):
    """Slow trend with small pullbacks: warmed RSI, not an artificial RSI=100."""
    dates = np.busday_offset(np.datetime64(latest), np.arange(1 - n, 1)).astype('U10')
    close = 20. + .01 * np.arange(n) + .16 * np.sin(np.arange(n) * .8)
    rows = [dict(day=str(day), open=float(c), high=float(c + .2), low=float(c - .2),
                 close=float(c), volume=float(100_000 + i * 10))
            for i, (day, c) in enumerate(zip(dates, close))]
    return dict(rows=rows, name='合成股票', checked_at=checked_at,
                provider='sina', latest_date=latest, date=latest,
                refresh_id='synthetic-refresh', evidence_dir='/not/a/real/evidence/path',
                warnings=['synthetic provider warning'],
                actions=dict(code=CODE, status='ok', rights_issue_present=False,
                             events=[], retrieved_at=checked_at,
                             metadata=dict(coverage_start='2015-01-01', coverage_end=TODAY,
                                           quote_date=latest, coverage_matches_quote=True,
                                           pagination_complete=True, live_recommendation_blocked=False,
                                           cash_payment_date_verified=False, cash_tax_applied=False,
                                           supplier_history_completeness_guaranteed=False)))


class TestFusionAnalysis(unittest.TestCase):
    def guard(self, target, **kwargs):
        patcher = patch(target, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.folder = self.root / 'report'
        self.folder.mkdir()
        self.blockers = []
        for target in ('socket.create_connection', 'socket.socket.connect',
                       'socket.socket.connect_ex', 'socket.getaddrinfo', 'urllib.request.urlopen',
                       'sqlite3.connect', 'db.get_db', 'pareto_backtest.run',
                       'pareto_backtest.prepare_stock', 'pareto_backtest.compute_fronts',
                       'pareto_backtest.replay_stock', 'pareto_backtest.replay_population'):
            self.blockers.append(self.guard(target, side_effect=AssertionError('offline analysis only')))
        # Assert blockers were not even attempted if a future broad catch hides one.
        self.addCleanup(self.assert_no_forbidden_calls)
        self.guard('pareto_web.RUN_DIR', new=self.folder)
        self.guard('live_data.SNAPSHOT_ROOT', new=self.root / 'refresh')
        self.fresh = fresh_fixture()
        self.refresh = self.guard('live_data.refresh_stock', side_effect=lambda code: deepcopy(self.fresh))
        self.manifest = dict(start='2020-01-02', requested_end=TODAY, effective_end=TODAY,
                             calendar=[row['day'] for row in self.fresh['rows']],
                             universe=[dict(code=CODE), dict(code=PEER)])
        self.risk = RiskConfig(annual_vol_target=.10)
        self.summary = dict(selected_risk=dict(annual_vol_target=.10))
        self.config = object()
        self.original_load_run = analysis.pareto_web._load_run
        self.load_run = self.guard('pareto_web._load_run', side_effect=lambda: (
            self.folder, self.manifest, self.summary, self.risk, self.config))
        # Deliberately incompatible with a personal holding account: never consumed.
        self.account = dict(code=CODE, final_cash=0., shares=999_999., risk_blocked=True,
                            current_weight=.99, cooldown_until=999999, high_water=1e30)
        self.load_account = self.guard('pareto_web._load_account', return_value=self.account)
        self.write_json(f'ledgers/{CODE}.json', [dict(day='2026-09-08', event='exit',
                                                    reason='final_five_days')])
        self.write_peer()

    def assert_no_forbidden_calls(self):
        for blocker in self.blockers:
            blocker.assert_not_called()

    def write_json(self, relative, payload):
        path = self.folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding='utf-8')

    def current_vector(self):
        raw = np.array([[row[key] for key in analysis.live_data.FIELDS] for row in self.fresh['rows']])
        return analysis.make_objectives(*raw.T)['votes'][-1]

    def write_peer(self, code=PEER, *, dates=None, vector=None, eligible=True, gate=True, **meta):
        dates = np.array(dates if dates is not None else
                         [row['day'] for row in self.fresh['rows']], dtype='U10')
        vector = self.current_vector() if vector is None else vector
        n = len(dates)
        payload = dict(code=code, feature_version=STRATEGY_VERSION,
                       action_status='ok', rights_issue_present=False, feature_error=None)
        payload.update(meta)
        self.write_json(f'features/{code}.json', payload)
        np.savez(self.folder / f'features/{code}.npz', dates=dates,
                 votes=np.tile(np.asarray(vector, dtype=np.int8), (n, 1)),
                 eligible=(np.arange(n) >= 250) & eligible, gates=np.full(n, gate, dtype=bool))

    def rewrite_archive(self, **changes):
        path = self.folder / f'features/{PEER}.npz'
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        arrays.update(changes)
        np.savez(path, **arrays)

    def compare(self, day=None, code=CODE, vector=None):
        return analysis.compare_same_day(code, day or self.fresh['latest_date'],
                                         self.current_vector() if vector is None else vector,
                                         dict(manifest=self.manifest, folder=self.folder))

    def test_every_call_refreshes_first_then_history_then_math(self):
        order = []
        refresh, load, math = self.refresh.side_effect, self.load_run.side_effect, analysis.make_objectives
        self.refresh.side_effect = lambda code: (order.append('refresh'), refresh(code))[1]
        self.load_run.side_effect = lambda: (order.append('history'), load())[1]
        with patch.object(analysis, 'make_objectives', side_effect=lambda *args: (
                order.append('math'), math(*args))[1]):
            analysis.analyze_stock(CODE)
            analysis.analyze_stock(CODE, holding=True)
        self.assertEqual(order, ['refresh', 'history', 'math'] * 2)
        self.assertEqual(self.refresh.call_args_list, [call(CODE), call(CODE)])

    def test_failed_refresh_never_loads_history_or_returns_previous_success(self):
        success = analysis.analyze_stock(CODE)
        self.assertEqual(success['css'], 'buy')
        self.load_run.reset_mock()
        error = RefreshError('synthetic failure', refresh_id='failed-refresh')
        self.refresh.side_effect = error
        with patch.object(analysis, 'historical_context') as history:
            for _ in range(2):
                with self.assertRaises(RefreshError) as raised:
                    analysis.analyze_stock(CODE)
                self.assertIs(raised.exception, error)
            history.assert_not_called()
        self.load_run.assert_not_called()

    def test_invalid_code_and_nonboolean_holding_never_refresh(self):
        for code in (None, 1, '00001', '000001 ', '../001', '３００００１', '300001', '688001'):
            with self.subTest(code=code), self.assertRaises(ValueError):
                analysis.analyze_stock(code)
        for holding in ('False', 0, 1, None, {}, []):
            with self.subTest(holding=holding), self.assertRaises(ValueError):
                analysis.analyze_stock(CODE, holding=holding)
        self.refresh.assert_not_called()

    def test_real_280_bar_math_matches_shared_indicator_and_risk_functions(self):
        result = analysis.analyze_stock(CODE)
        expected = analysis.make_objectives(*result['raw'].T)
        stats = analysis.risk_statistics(result['raw'][:, 3], self.risk.risk_window, self.risk.es_confidence)
        self.assertEqual(result['votes'].shape, (280, 34))
        np.testing.assert_array_equal(result['votes'], expected['votes'])
        self.assertEqual(result['rsi'], expected['rsi'][-1])
        self.assertLess(result['rsi'], 92)
        self.assertTrue(result['eligible'] and result['gate'] and result['verified_actions'])
        self.assertEqual(result['annual_vol'], stats['annual_vol'][-1])
        self.assertEqual(result['daily_es'], stats['daily_es'][-1])
        self.assertEqual(result['conditional_cap'], analysis.risk_target(
            1, True, stats['annual_vol'][-1], stats['daily_es'][-1], self.risk))
        self.assertIn('synthetic provider warning', result['warnings'])

    def test_fresh_rows_and_latest_date_not_cut_by_backtest_effective_end(self):
        self.manifest['effective_end'] = '2026-09-08'
        self.manifest['calendar'] = self.manifest['calendar'][:-2]
        result = analysis.analyze_stock(CODE)
        self.assertEqual(result['dates'][-1], TODAY)
        self.assertEqual(len(result['dates']), 280)
        self.assertEqual(result['raw'][-1, 3], self.fresh['rows'][-1]['close'])
        self.assertEqual(result['comparison']['status'], 'unknown')
        self.assertEqual(result['history']['manifest']['effective_end'], '2026-09-08')
        self.load_account.assert_called_once_with(self.folder, CODE, self.config, '2026-09-08')
        self.assertEqual(result['action'], '暂不下买卖结论')

    def test_final_five_days_and_old_position_risk_do_not_control_current_holding(self):
        original = deepcopy(self.account)
        flat = analysis.analyze_stock(CODE, holding=False)
        held = analysis.analyze_stock(CODE, holding=True)
        self.assertEqual(flat['action'], '建仓候选')
        self.assertEqual(held['action'], '持有候选')
        self.assertEqual(flat['conditional_cap'], held['conditional_cap'])
        self.assertFalse(flat['holding'])
        self.assertTrue(held['holding'])
        self.assertEqual(self.account, original)
        for result in (flat, held):
            self.assertNotIn('current_weight', result)
            self.assertNotIn('target_weight', result)
            self.assertNotIn('quantity', result)
            self.assertIn('无法计算实际加减仓', ''.join(result['warnings']))

    def test_latest_online_response_is_only_asof_candidate_not_today(self):
        self.fresh = fresh_fixture(latest='2026-09-08')
        self.write_peer()
        for holding, label in ((False, '建仓候选'), (True, '持有候选')):
            result = analysis.analyze_stock(CODE, holding=holding)
            self.assertEqual(result['action'], '截至2026-09-08的' + label)
            self.assertIn('不是核验日2026-09-10行情', ''.join(result['warnings']))
            self.assertEqual(result['comparison']['date'], '2026-09-08')

    def test_before_1500_yesterday_is_valid_completed_diagnostic(self):
        self.fresh = fresh_fixture(latest='2026-09-09', checked_at=TODAY + 'T14:59:59+08:00')
        self.write_peer()
        result = analysis.analyze_stock(CODE)
        self.assertTrue(result['eligible'] and result['verified_actions'])
        self.assertEqual(result['action'], '截至2026-09-09的建仓候选')
        self.assertIn('15:00前', ''.join(result['warnings']))

    def test_clock_converts_to_shanghai_and_1500_boundary(self):
        # UTC and a negative offset may have a different date than Shanghai.
        for clock in (TODAY + 'T07:00:00+00:00', '2026-09-09T23:00:00-08:00'):
            self.fresh['checked_at'] = clock
            result = analysis.analyze_stock(CODE)
            self.assertEqual(result['action'], '建仓候选')
        self.fresh['checked_at'] = TODAY + 'T06:59:59+00:00'
        with self.assertRaisesRegex(ValueError, '暂不下结论'):
            analysis.analyze_stock(CODE)

    def test_empty_bad_nonfinite_rows_and_dates_fail_friendly_before_history(self):
        baseline = deepcopy(self.fresh)
        variants = []
        for field, value in (('day', TODAY + 'T16:00:00+08:00'), ('day', '2026-02-30'),
                             ('day', '20260910'), ('day', '2026-9-10'),
                             ('close', float('nan')), ('open', float('inf')),
                             ('open', 0.), ('high', 1.), ('low', 99.), ('volume', -1.),
                             ('volume', True), ('open', 'secret-untrusted-value')):
            changed = deepcopy(baseline)
            changed['rows'][-1][field] = value
            variants.append(changed)
        variants += [dict(baseline, rows=[]), dict(baseline, rows=list(reversed(baseline['rows']))),
                     dict(baseline, rows=baseline['rows'] + [baseline['rows'][-1]]),
                     dict(baseline, latest_date='2026-09-09'), dict(baseline, date='2026-09-09'),
                     dict(baseline, checked_at=TODAY + 'T16:00:00'),
                     dict(baseline, checked_at='2026-09-09T16:00:00+08:00')]
        for fresh in variants:
            with self.subTest(last=fresh['rows'][-1:] if fresh['rows'] else []):
                self.fresh = fresh
                with self.assertRaisesRegex(ValueError, '本次刷新日线或时间字段无效') as error:
                    analysis.analyze_stock(CODE)
                self.assertNotIn('secret-untrusted-value', str(error.exception))
        self.load_run.assert_not_called()

    def test_error_unknown_rights_and_explicit_blocks_keep_raw_diagnostics(self):
        original = deepcopy(self.fresh['actions'])
        variants = [dict(original, status='error'), dict(original, rights_issue_present=None),
                    dict(original, rights_issue_present=True), dict(original, rights_issue_present=0),
                    dict(original, code=OTHER)]
        for value in (True, None, 0, 'false'):
            variants.append(dict(original, metadata=dict(original['metadata'], live_recommendation_blocked=value)))
        variants.append(dict(original, metadata={}))
        for actions in variants:
            with self.subTest(actions=actions):
                self.fresh['actions'] = actions
                with patch.object(analysis, 'compare_same_day') as compare:
                    result = analysis.analyze_stock(CODE, holding=True)
                compare.assert_not_called()
                self.assertFalse(result['verified_actions'])
                self.assertEqual(result['action'], '暂不下买卖结论')
                self.assertEqual(result['conditional_cap'], 0)
                self.assertEqual(result['votes'].shape, (280, 34))
                self.assertTrue(np.isfinite(result['rsi']))
                np.testing.assert_array_equal(result['adjusted'], result['raw'])

    def test_explicit_coverage_mismatch_blocks_even_with_unblocked_flag(self):
        original = deepcopy(self.fresh['actions']['metadata'])
        for field, value in (('coverage_start', '2026-01-01'), ('coverage_end', '2026-09-09'),
                             ('coverage_end', '2026-09-11'), ('coverage_start', 'bad-date'),
                             ('quote_date', '2026-09-09'), ('quote_date', None),
                             ('coverage_matches_quote', False), ('coverage_matches_quote', 1),
                             ('pagination_complete', False)):
            with self.subTest(field=field, value=value):
                self.fresh['actions']['metadata'] = dict(original, **{field: value})
                result = analysis.analyze_stock(CODE)
                self.assertFalse(result['verified_actions'])
                self.assertEqual(result['css'], 'warn')
                self.assertEqual(result['conditional_cap'], 0)

    def test_optional_coverage_absence_does_not_invent_a_failure(self):
        self.fresh['actions']['metadata'] = dict(live_recommendation_blocked=False)
        result = analysis.analyze_stock(CODE)
        self.assertTrue(result['verified_actions'])
        self.assertEqual(result['css'], 'buy')

    def test_bad_events_and_impossible_reference_keep_raw_diagnostics(self):
        original = dict(ex_date=self.fresh['rows'][-5]['day'], cash_per_share=.2, share_multiplier=1.)
        for changes in (dict(cash_per_share=-1), dict(cash_per_share=float('nan')),
                        dict(cash_per_share=10000), dict(share_multiplier=0),
                        dict(share_multiplier=float('inf')), dict(share_multiplier=True),
                        dict(ex_date='2026-09-11'), dict(ex_date='2026-02-30')):
            with self.subTest(changes=changes):
                self.fresh['actions']['events'] = [dict(original, **changes)]
                result = analysis.analyze_stock(CODE)
                self.assertFalse(result['verified_actions'])
                self.assertEqual(result['css'], 'warn')
                np.testing.assert_array_equal(result['adjusted'], result['raw'])

    def test_250_rows_and_rsi_warmup_never_establish_f1(self):
        for n in (1, 13, 250):
            with self.subTest(n=n):
                self.fresh = fresh_fixture(n=n)
                with patch.object(analysis, 'compare_same_day') as compare:
                    result = analysis.analyze_stock(CODE)
                compare.assert_not_called()
                self.assertFalse(result['eligible'])
                self.assertEqual(result['css'], 'warn')
                self.assertEqual(result['conditional_cap'], 0.)
                if n < 14:
                    self.assertTrue(np.isnan(result['rsi']))
        self.fresh = fresh_fixture(n=251)
        self.assertTrue(analysis.analyze_stock(CODE)['eligible'])

    def test_zero_volume_fails_eligibility_not_a_sell_or_buy(self):
        self.fresh['rows'][-1]['volume'] = 0.
        result = analysis.analyze_stock(CODE, holding=True)
        self.assertFalse(result['eligible'])
        self.assertEqual(result['css'], 'warn')
        self.assertEqual(result['conditional_cap'], 0.)

    def test_rsi_gate_failure_gives_only_unheld_no_buy_or_held_exit_signal(self):
        for i, row in enumerate(self.fresh['rows']):
            close = 20. + .01 * i
            row.update(open=close, close=close, high=close + .2, low=close - .2)
        for holding, label in ((False, '不买入'), (True, '退出信号')):
            result = analysis.analyze_stock(CODE, holding=holding)
            self.assertTrue(result['eligible'])
            self.assertEqual(result['rsi'], 100.)
            self.assertFalse(result['gate'])
            self.assertEqual(result['action'], label)
            self.assertEqual(result['conditional_cap'], 0.)

    def test_obv_gate_is_separate_from_rsi_gate(self):
        # A large down-day volume then low-volume up days: RSI passes, flow fails.
        self.fresh['rows'][-4]['volume'] = 1e10
        # Find a recent down day within the five-day OBV window deterministically.
        for i in range(len(self.fresh['rows']) - 5, len(self.fresh['rows'])):
            if self.fresh['rows'][i]['close'] < self.fresh['rows'][i - 1]['close']:
                self.fresh['rows'][i]['volume'] = 1e12
                break
        result = analysis.analyze_stock(CODE)
        self.assertLessEqual(result['rsi'], 92)
        self.assertLess(result['obv_flow'], -.6)
        self.assertFalse(result['gate'])
        self.assertEqual(result['action'], '不买入')

    def test_nonfinite_or_zero_risk_cannot_confirm_candidate(self):
        for vol, es in ((float('nan'), .02), (.2, float('inf'))):
            stats = dict(annual_vol=np.full(280, vol), daily_es=np.full(280, es))
            with self.subTest(vol=vol, es=es), patch.object(analysis, 'risk_statistics', return_value=stats):
                result = analysis.analyze_stock(CODE)
            self.assertEqual(result['conditional_cap'], 0)
            self.assertNotEqual(result['css'], 'buy')
        with patch.object(analysis, 'risk_target', return_value=0.):
            self.assertNotEqual(analysis.analyze_stock(CODE)['css'], 'buy')

    def test_unexplained_25_percent_gap_quarantines_all_later_signals(self):
        j = 260
        baseline = deepcopy(self.fresh)
        for factor, eligible in ((1.25, True), (1.251, False), (.749, False)):
            self.fresh = deepcopy(baseline)
            row = self.fresh['rows'][j]
            row['open'] = self.fresh['rows'][j - 1]['close'] * factor
            row['high'] = max(row['high'], row['open'])
            row['low'] = min(row['low'], row['open'])
            with self.subTest(factor=factor):
                result = analysis.analyze_stock(CODE)
                self.assertEqual(result['eligible'], eligible)
                if not eligible:
                    self.assertEqual(result['css'], 'warn')
                    self.assertIn('超过25%', ''.join(result['warnings']))

    def test_ex_rights_reference_explains_split_and_keeps_raw_prices(self):
        j, cash, multiplier = 261, .2, 2.
        baseline = deepcopy(self.fresh)
        original_raw = np.array([[r[k] for k in analysis.live_data.FIELDS] for r in baseline['rows']])
        reference = (original_raw[j - 1, 3] - cash) / multiplier
        price_scale = reference / original_raw[j - 1, 3]
        for row in self.fresh['rows'][j:]:
            for key in analysis.live_data.FIELDS[:-1]:
                row[key] *= price_scale
            row['volume'] *= multiplier
        # Event falls on a missing session/weekend, maps to the next actual bar.
        prior = self.fresh['rows'][j - 1]['day']
        ex_day = str(np.datetime64(prior) + np.timedelta64(1, 'D'))
        self.assertNotIn(ex_day, [row['day'] for row in self.fresh['rows']])
        self.fresh['actions']['events'] = [dict(ex_date=ex_day, cash_per_share=cash,
                                              share_multiplier=multiplier)]
        result = analysis.analyze_stock(CODE)
        self.assertTrue(result['eligible'] and result['verified_actions'])
        np.testing.assert_array_equal(result['adjusted'][:j], original_raw[:j])
        np.testing.assert_allclose(result['adjusted'], original_raw)
        self.assertLess(result['raw'][j, 3], original_raw[j, 3] * .6)
        self.assertNotIn('超过25%', ''.join(result['warnings']))
        self.fresh['actions']['events'] = []
        unexplained = analysis.analyze_stock(CODE)
        self.assertFalse(unexplained['eligible'])

    def test_report_selected_risk_is_used_and_clearly_marked(self):
        # Use a restrictive risk target so the max-weight cap cannot hide selection.
        self.risk = RiskConfig(annual_vol_target=.001)
        result = analysis.analyze_stock(CODE)
        self.assertLess(result['conditional_cap'], 1.)
        self.assertEqual(result['history']['risk_source'], 'frozen_selected')
        self.assertIn('冻结报告所选值', ''.join(result['warnings']))
        self.assertIn('年化波动目标0.10%', ''.join(result['warnings']))
        self.assertEqual(result['history']['risk'], self.risk)

    def test_missing_report_defaults_to_15_percent_without_old_analysis(self):
        self.load_run.side_effect = self.original_load_run  # Empty temporary run.
        result = analysis.analyze_stock(CODE)
        self.assertIsNone(result['history']['manifest'])
        self.assertEqual(result['history']['risk'].annual_vol_target, .15)
        self.assertEqual(result['history']['risk_source'], 'model_default')
        self.assertEqual(result['comparison']['status'], 'unknown')
        self.assertEqual(result['action'], '暂不下买卖结论')
        self.assertIn('默认风险参数：年化波动目标15%', ''.join(result['warnings']))
        self.load_account.assert_not_called()

    def test_missing_account_or_ledger_preserves_report_dates_peers_and_selected_risk(self):
        for missing in ('account', 'ledger'):
            with self.subTest(missing=missing):
                self.load_account.side_effect = OSError('missing') if missing == 'account' else None
                if missing == 'ledger':
                    (self.folder / f'ledgers/{CODE}.json').unlink()
                result = analysis.analyze_stock(CODE)
                self.assertIsNone(result['history']['account'])
                self.assertEqual(result['history']['ledger'], [])
                self.assertIs(result['history']['manifest'], self.manifest)
                self.assertIs(result['history']['risk'], self.risk)
                self.assertEqual(result['action'], '建仓候选')
                self.assertIn('该股历史账户或账本不完整', result['history']['error'])

    def test_outside_frozen_universe_is_not_wrong_code_or_missing_current_data(self):
        self.manifest['universe'] = [dict(code=PEER)]
        result = analysis.analyze_stock(CODE)
        self.refresh.assert_called_once_with(CODE)
        self.load_account.assert_not_called()
        self.assertTrue(result['eligible'])
        self.assertFalse(result['comparison']['in_universe'])
        self.assertEqual(result['comparison']['expected'], 1)
        self.assertEqual(result['comparison']['status'], 'front1')
        self.assertIn('不代表全市场F1', result['comparison']['coverage'])
        self.assertIn('不代表代码无效', result['history']['error'])

    def test_equal_vectors_do_not_dominate(self):
        result = self.compare()
        self.assertEqual(result['status'], 'front1')
        self.assertEqual(result['compared'], 1)
        self.assertEqual(result['dominators'], [])

    def test_strict_dominance_requires_all_coordinates_and_one_strict(self):
        vector = np.zeros(34, dtype=np.int8)
        strict = vector.copy()
        strict[4] = 1
        self.write_peer(vector=strict)
        result = self.compare(vector=vector)
        self.assertEqual(result['status'], 'dominated')
        self.assertEqual(result['dominators'], [PEER])
        incomparable = strict.copy()
        incomparable[5] = -1
        self.write_peer(vector=incomparable)
        self.assertEqual(self.compare(vector=vector)['status'], 'front1')

    def test_dominated_current_stock_is_not_buy_for_either_holding_flag(self):
        self.write_peer(vector=np.ones(34, dtype=np.int8))
        for holding, action in ((False, '不买入'), (True, '退出信号')):
            result = analysis.analyze_stock(CODE, holding=holding)
            self.assertEqual(result['action'], action)
            self.assertEqual(result['comparison']['status'], 'dominated')

    def test_self_is_excluded_even_if_own_frozen_file_is_absent(self):
        self.assertFalse((self.folder / f'features/{CODE}.npz').exists())
        result = self.compare()
        self.assertEqual((result['expected'], result['compared']), (1, 1))
        self.assertEqual(result['status'], 'front1')

    def test_no_peer_or_only_self_never_fabricates_one_stock_f1(self):
        self.manifest['universe'] = [dict(code=CODE)]
        result = self.compare()
        self.assertEqual(result['status'], 'unknown')
        self.assertEqual(result['compared'], 0)

    def test_ineligible_gate_failed_and_known_quarantined_peers_are_not_compared(self):
        for kwargs in (dict(eligible=False), dict(gate=False), dict(action_status='error'),
                       dict(rights_issue_present=True), dict(feature_error='invalid prices')):
            with self.subTest(kwargs=kwargs):
                self.write_peer(vector=np.ones(34, dtype=np.int8), **kwargs)
                result = self.compare()
                self.assertEqual(result['status'], 'unknown')
                self.assertEqual(result['compared'], 0)

    def test_unknown_rights_peer_cannot_enable_positive_f1(self):
        self.manifest['universe'].append(dict(code=OTHER))
        self.write_peer(OTHER)
        for rights in (None, 0, 'false'):
            with self.subTest(rights=rights):
                self.write_peer(rights_issue_present=rights)
                result = self.compare()
                self.assertEqual(result['status'], 'unknown')
                self.assertEqual(result['compared'], 1)
                self.assertIn('不完整', result['coverage'])

    def test_missing_peer_file_prevents_positive_but_real_dominator_still_rejects(self):
        self.manifest['universe'].append(dict(code=OTHER))
        for relative in (f'features/{OTHER}.json', f'features/{OTHER}.npz'):
            self.write_peer(OTHER)
            (self.folder / relative).unlink()
            with self.subTest(relative=relative):
                self.assertEqual(self.compare()['status'], 'unknown')
                self.write_peer(vector=np.ones(34, dtype=np.int8))
                self.assertEqual(self.compare()['status'], 'dominated')
                self.write_peer()

    def test_no_matching_date_does_not_use_previous_or_future_peer_row(self):
        dates = self.manifest['calendar']
        self.write_peer(dates=dates[:-1], vector=np.ones(34, dtype=np.int8))
        self.assertEqual(self.compare()['status'], 'unknown')
        self.write_peer(vector=np.ones(34, dtype=np.int8))
        with np.load(self.folder / f'features/{PEER}.npz') as data:
            original = {key: data[key] for key in data.files}
        # An absent middle date must not searchsorted into a later observation.
        removed = len(dates) - 3
        self.rewrite_archive(**{key: np.delete(value, removed, axis=0) for key, value in original.items()})
        result = self.compare(day=dates[removed])
        self.assertEqual(result['status'], 'unknown')
        self.assertEqual(result['compared'], 0)

    def test_date_after_calendar_or_effective_end_never_uses_latest_front(self):
        for boundary in ('calendar', 'effective_end'):
            with self.subTest(boundary=boundary):
                old = deepcopy(self.manifest)
                if boundary == 'calendar':
                    self.manifest['calendar'] = self.manifest['calendar'][:-1]
                else:
                    self.manifest['effective_end'] = '2026-09-09'
                with patch.object(analysis.pareto_web, '_json', side_effect=AssertionError('no date fallback')):
                    result = self.compare()
                self.assertEqual(result['status'], 'unknown')
                self.assertEqual(result['compared'], 0)
                self.manifest = old

    def test_malformed_peer_masks_vectors_dates_and_onepoint_warmup_fail_closed(self):
        dates = np.array(self.manifest['calendar'])
        variants = [dict(eligible=np.full(280, 'true')), dict(gates=np.ones(280)),
                    dict(votes=np.ones((280, 33))), dict(votes=np.full((280, 34), float('nan'))),
                    dict(votes=np.full((280, 34), 2)), dict(dates=dates[::-1]),
                    dict(dates=np.repeat(dates[-1], 280)),
                    dict(dates=np.append(dates[:-1], '2026-09-11')),
                    dict(dates=np.array([TODAY]), votes=np.zeros((1, 34)),
                         eligible=np.array([True]), gates=np.array([True]))]
        for changes in variants:
            with self.subTest(keys=list(changes)):
                self.write_peer()
                self.rewrite_archive(**changes)
                result = self.compare()
                self.assertEqual(result['status'], 'unknown')
                self.assertEqual(result['compared'], 0)
                self.assertIn('不完整', result['coverage'])

    def test_corrupt_archive_and_incompatible_metadata_do_not_confirm_positive(self):
        for error in (ValueError('bad archive'), EOFError('truncated'), BadZipFile('bad CRC')):
            with self.subTest(error=type(error)), patch.object(analysis.np, 'load', side_effect=error):
                self.assertEqual(self.compare()['status'], 'unknown')
        for metadata in (dict(code=OTHER), dict(feature_version='wrong-version')):
            self.write_json(f'features/{PEER}.json', dict(action_status='ok', rights_issue_present=False,
                                                         feature_error=None, **metadata))
            self.assertEqual(self.compare()['status'], 'unknown')

    def test_malformed_metadata_remains_unknown_with_other_valid_peers(self):
        self.manifest['universe'].append(dict(code=OTHER))
        self.write_peer(OTHER)
        for metadata in (None, [], 'bad metadata', {}):
            with self.subTest(metadata=metadata):
                self.write_json(f'features/{PEER}.json', metadata)
                result = self.compare()
                self.assertEqual(result['status'], 'unknown')
                self.assertEqual(result['compared'], 1)

    def test_invalid_current_vector_is_rejected(self):
        for vector in (np.zeros(33), np.full(34, float('nan')), np.full(34, 2),
                       np.zeros(34, dtype=bool), np.zeros(34, dtype=complex)):
            with self.subTest(shape=vector.shape), self.assertRaises(ValueError):
                self.compare(vector=vector)

    def test_renderer_handles_invalid_fresh_rows_without_old_chart_or_history(self):
        # This is an integration check in this suite, not execution of Web's suite.
        import fusion_web
        self.fresh['rows'][-1]['close'] = float('nan')
        with patch.object(fusion_web, 'render_chart') as chart:
            html = fusion_web.render_analysis(CODE)
        self.assertIn('分析失败，暂不下结论', html)
        self.assertNotIn('<img', html)
        chart.assert_not_called()
        self.load_run.assert_not_called()

    def test_renderer_refresh_failure_never_loads_history_or_chart(self):
        import fusion_web
        self.refresh.side_effect = RefreshError('secret-body', refresh_id='failed-refresh')
        with patch.object(fusion_web, 'render_chart') as chart:
            html = fusion_web.render_analysis(CODE)
        self.assertIn('更新失败，暂不下结论', html)
        self.assertNotIn('secret-body', html)
        self.assertNotIn('<img', html)
        chart.assert_not_called()
        self.load_run.assert_not_called()


if __name__ == '__main__':
    unittest.main()