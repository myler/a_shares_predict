"""Offline contracts for daily atoms, Pareto fronts and risk/cost-based sizing."""

from dataclasses import FrozenInstanceError
import unittest
from unittest.mock import patch

import numpy as np

import engine
import pareto_strategy as strategy


def market(n=320, seed=7):
    rng = np.random.default_rng(seed)
    closes = 30 * np.exp(np.cumsum(rng.normal(0, 0.018, n)))
    opens = closes * np.exp(rng.normal(0, 0.009, n))
    highs = np.maximum(opens, closes) * (1 + rng.uniform(0.001, 0.02, n))
    lows = np.minimum(opens, closes) * (1 - rng.uniform(0.001, 0.02, n))
    volumes = rng.uniform(100, 10000, n)
    return opens, highs, lows, closes, volumes


def reference_layers(votes, max_layers=4):
    """Deliberately unpacked all-dimension strict dominance oracle."""
    votes = np.asarray(votes)
    result = np.full(len(votes), 5, dtype=np.int32)
    remaining = list(range(len(votes)))
    for layer in range(1, max_layers + 1):
        front = [i for i in remaining if not any(
            np.all(votes[j] >= votes[i]) and np.any(votes[j] > votes[i])
            for j in remaining)]
        result[front] = layer
        removed = set(front)
        remaining = [i for i in remaining if i not in removed]
        if not remaining:
            break
    return result


class TestAuxiliaryVotes(unittest.TestCase):
    def test_panel_columns_and_legacy_consensus_match(self):
        for seed in (7, 42):
            opens, highs, lows, closes, volumes = market(280, seed)
            for real_opens in (opens, None):
                votes = engine.calc_auxiliary_votes_series(
                    closes, highs, lows, volumes, real_opens)
                consensus = engine.calc_auxiliary_consensus_series(
                    closes, highs, lows, volumes, real_opens)
                self.assertEqual(votes.shape, (280, 18))
                self.assertEqual(votes.dtype, np.int8)
                np.testing.assert_array_equal(votes[:59], 0)
                np.testing.assert_array_equal(
                    consensus, np.round(votes.sum(axis=1) / 18 * 100, 1))
                for n in (1, 59, 60, 61, 75, 100, 180, 250, 280):
                    with self.subTest(seed=seed, real_opens=real_opens is not None, n=n):
                        panel = engine.calc_auxiliary_consensus(
                            closes[:n], highs[:n], lows[:n], volumes[:n],
                            None if real_opens is None else real_opens[:n])
                        self.assertEqual([v['name'] for v in panel['votes']],
                                         engine._INDICATOR_NAMES)
                        np.testing.assert_array_equal(
                            votes[n - 1], [v['vote'] for v in panel['votes']])
                        self.assertEqual(consensus[n - 1], panel['consensus_score'])

    def test_consensus_delegates_without_reversing_or_changing_rounding(self):
        # Every possible net vote count, not just market-generated samples.
        votes = np.zeros((37, 18), dtype=np.int8)
        for row, total in enumerate(range(-18, 19)):
            votes[row, :abs(total)] = np.sign(total)
        with patch.object(engine, 'calc_auxiliary_votes_series', return_value=votes) as calc:
            actual = engine.calc_auxiliary_consensus_series([], [], [], [], opens=[])
        calc.assert_called_once_with([], [], [], [], opens=[])
        np.testing.assert_array_equal(
            actual, [round(total / 18 * 100, 1) for total in range(-18, 19)])
        self.assertEqual(actual.dtype, np.float64)

    def test_short_empty_and_opens_length_compatibility(self):
        for n in (0, 1, 14, 26, 59):
            opens, highs, lows, closes, volumes = market(n)
            votes = engine.calc_auxiliary_votes_series(closes, highs, lows, volumes, opens)
            self.assertEqual(votes.shape, (n, 18))
            np.testing.assert_array_equal(votes, 0)
        opens, highs, lows, closes, volumes = market(60)
        with self.assertRaises(ValueError):
            engine.calc_auxiliary_votes_series(closes, highs, lows, volumes, opens[:-1])


class TestMakeObjectives(unittest.TestCase):
    def test_contract_and_causal_prefixes_all_34_columns(self):
        data = market()
        result = strategy.make_objectives(*data)
        self.assertEqual(strategy.STRATEGY_VERSION, 'pareto_daily_atoms_risk_v1')
        self.assertEqual(len(strategy.OBJECTIVE_NAMES), 34)
        self.assertEqual(len(set(strategy.OBJECTIVE_NAMES)), 34)
        self.assertEqual([sum(name.startswith(prefix) for name in strategy.OBJECTIVE_NAMES)
                          for prefix in ('M_', 'F_', 'P_', 'Q_', 'A_')], [6, 5, 2, 3, 18])
        self.assertEqual(set(result), {'votes', 'eligible', 'gate_pass', 'rsi', 'obv_flow'})
        self.assertEqual(result['votes'].shape, (320, 34))
        self.assertEqual(result['votes'].dtype, np.int8)
        self.assertTrue(np.isin(result['votes'], [-1, 0, 1]).all())
        for key, dtype in (('eligible', np.bool_), ('gate_pass', np.bool_),
                           ('rsi', np.float64), ('obv_flow', np.float64)):
            self.assertEqual(result[key].shape, (320,))
            self.assertEqual(result[key].dtype, dtype)
        np.testing.assert_array_equal(result['eligible'][:250], False)
        np.testing.assert_array_equal(result['eligible'][250:], True)
        for n in (0, 1, 14, 26, 59, 60, 61, 150, 249, 250, 251, 300):
            with self.subTest(n=n):
                prefix = strategy.make_objectives(*(values[:n] for values in data))
                for key in result:
                    np.testing.assert_array_equal(prefix[key], result[key][:n], err_msg=key)
        # An appended, radically different future must not change any prior row.
        future = market(80, 123)
        extended = strategy.make_objectives(*(np.r_[a, b * 2] for a, b in zip(data, future)))
        for key in result:
            np.testing.assert_array_equal(extended[key][:320], result[key], err_msg=key)

    def test_core_atoms_match_original_rule_signs_not_weighted_scores(self):
        for seed in (1, 7, 42):
            opens, highs, lows, closes, volumes = market(320, seed)
            result = strategy.make_objectives(opens, highs, lows, closes, volumes)
            dif, dea, bar = engine.calc_macd(closes)
            rsi = engine.calc_rsi(closes)
            k, d, j = engine.calc_kdj(highs, lows, closes)
            bb_u, _, bb_l = engine.calc_bollinger(closes)
            wr = engine.calc_wr(highs, lows, closes)
            obv = engine.calc_obv(closes, volumes)
            for i in (0, 5, 14, 33, 59, 60, 61, 248, 249, *range(250, 320)):
                original = engine._score_comprehensive(
                    closes, highs, lows, volumes, dif, dea, bar, rsi, k, d, j,
                    bb_u, bb_l, wr, obv, i)
                atoms = [int(np.sign(float(item['adj'].replace('−', '-'))))
                         for group in ('macd', 'multifactor', 'fundamental', 'game')
                         for item in original[-1][group][1:]]
                np.testing.assert_array_equal(result['votes'][i, :16], atoms,
                                              err_msg=f'seed={seed}, i={i}')
                self.assertEqual(result['obv_flow'][i], original[4])
            np.testing.assert_array_equal(
                result['votes'][:, 16:],
                engine.calc_auxiliary_votes_series(closes, highs, lows, volumes, opens))

    def test_real_opens_drive_ar_without_touching_core(self):
        n = 270
        closes = np.full(n, 10.)
        highs, lows, volumes = np.full(n, 11.), np.full(n, 9.), np.ones(n)
        low_open = strategy.make_objectives(np.full(n, 9.1), highs, lows, closes, volumes)
        high_open = strategy.make_objectives(np.full(n, 10.9), highs, lows, closes, volumes)
        ar = strategy.OBJECTIVE_NAMES.index('A_AR')
        self.assertEqual(low_open['votes'][-1, ar], -1)
        self.assertEqual(high_open['votes'][-1, ar], 1)
        np.testing.assert_array_equal(low_open['votes'][:, :16], high_open['votes'][:, :16])
        with self.assertRaises(ValueError):
            strategy.make_objectives(None, highs, lows, closes, volumes)

    def test_gate_is_independent_of_warmup_volume_and_double_weak_scores(self):
        n = 260
        closes = np.full(n, 10.)
        volumes = np.ones(n)
        volumes[-1] = 0
        with patch.object(strategy, 'calc_macd', return_value=(
                np.full(n, -1.), np.zeros(n), np.full(n, -2.))), \
                patch.object(strategy, 'calc_rsi', return_value=np.full(n, 68.)), \
                patch.object(strategy, 'calc_kdj', return_value=(
                    np.full(n, 10.), np.full(n, 20.), np.full(n, 110.))), \
                patch.object(strategy, 'calc_wr', return_value=np.full(n, 10.)), \
                patch.object(strategy, 'calc_bollinger', return_value=(
                    np.full(n, 10.), np.full(n, 5.), np.zeros(n))):
            result = strategy.make_objectives(closes, closes + 1, closes - 1, closes, volumes)
        self.assertTrue(result['gate_pass'].all())
        self.assertFalse(result['eligible'][249])
        self.assertTrue(result['eligible'][250])
        self.assertFalse(result['eligible'][-1])
        # Preserve the legacy top rule: constant negative DIF still satisfies
        # dif[i] < dif[high_idx] * .9. Do not silently "fix" that rule here.
        np.testing.assert_array_equal(result['votes'][-1, :11],
                                      [-1, -1, -1, 0, 0, -1, 0, -1, -1, -1, -1])

    def test_rsi_obv_thresholds_and_nan_eligibility(self):
        n = 270
        rsi = np.full(n, 50.)
        rsi[-11:] = [0, 29.9, 30, 65, 65.1, 70, 70.1, 80, 92, 92.1, np.nan]
        flows = np.zeros(n)
        flows[250:257] = [-0.6001, -0.6, -0.4001, -0.4, 0, 0.4, 0.4001]
        with patch.object(strategy, 'calc_rsi', return_value=rsi), \
                patch.object(strategy, 'calc_obv_flow', side_effect=flows):
            result = strategy.make_objectives(*market(n))
        np.testing.assert_array_equal(result['votes'][-11:, 6],
                                      [1, 1, 1, 1, 0, 0, -1, -1, -1, -1, 1])
        np.testing.assert_array_equal(result['votes'][250:257, 13], [-1, -1, -1, 0, 0, 0, 1])
        np.testing.assert_array_equal(result['gate_pass'], (rsi <= 92) & (flows >= -0.6))
        self.assertFalse(result['eligible'][-1])
        self.assertFalse(result['gate_pass'][-1])

    def test_constant_and_all_zero_volume_are_valid_data(self):
        n = 260
        closes = np.full(n, 10.)
        result = strategy.make_objectives(closes, closes + 1, closes - 1,
                                          closes, np.zeros(n))
        self.assertFalse(result['eligible'].any())
        np.testing.assert_array_equal(result['obv_flow'], 0.)
        np.testing.assert_array_equal(result['rsi'][14:], 50.)

    def test_invalid_ohlcv(self):
        data = market(60)
        for column in range(5):
            for bad in (np.nan, np.inf, -np.inf, -1.):
                args = [values.copy() for values in data]
                args[column][10] = bad
                with self.subTest(column=column, bad=bad), self.assertRaises(ValueError):
                    strategy.make_objectives(*args)
            for bad_array in (data[column][:-1], data[column][:, None],
                              np.full(60, '10'), np.ones(60, dtype=complex), 10.):
                args = list(data)
                args[column] = bad_array
                with self.subTest(column=column, shape=np.shape(bad_array)), \
                        self.assertRaises(ValueError):
                    strategy.make_objectives(*args)
        for column, value in ((0, 0), (1, 0), (2, 0), (3, 0),
                              (0, data[1][10] + 1), (3, data[2][10] - 1),
                              (1, data[2][10] - 1), (2, data[1][10] + 1)):
            args = [values.copy() for values in data]
            args[column][10] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                strategy.make_objectives(*args)


class TestDailyFrontLayers(unittest.TestCase):
    def test_conflicts_do_not_cancel_and_neutral_is_dominated(self):
        np.testing.assert_array_equal(strategy.daily_front_layers([[1, -1], [-1, 1]]), [1, 1])
        np.testing.assert_array_equal(strategy.daily_front_layers([[0, 0], [1, 0]]), [2, 1])
        np.testing.assert_array_equal(
            strategy.daily_front_layers([[1, -1], [-1, 1], [0, 0]]), [1, 1, 1])

    def test_chain_duplicate_permutation_and_layer_cap(self):
        chain = np.array([[1, 1], [1, 0], [0, 0], [0, -1], [-1, -1]], dtype=np.int8)
        votes = np.repeat(chain, [3, 1, 2, 4, 2], axis=0)
        expected = np.repeat([1, 2, 3, 4, 5], [3, 1, 2, 4, 2])
        np.testing.assert_array_equal(strategy.daily_front_layers(votes), expected)
        permutation = np.random.default_rng(3).permutation(len(votes))
        np.testing.assert_array_equal(strategy.daily_front_layers(votes[permutation]),
                                      expected[permutation])
        for cap in range(1, 5):
            np.testing.assert_array_equal(strategy.daily_front_layers(votes, max_layers=cap),
                                          np.where(expected <= cap, expected, 5))
        self.assertEqual(strategy.daily_front_layers(votes).dtype, np.int32)

    def test_random_against_bruteforce_and_chunk_boundaries(self):
        rng = np.random.default_rng(123)
        for dimensions in (1, 2, 3, 5, 16, 34, 64):
            for trial in range(10):
                votes = rng.integers(-1, 2, (int(rng.integers(1, 45)), dimensions), dtype=np.int8)
                votes = np.concatenate((votes, votes[:3]))
                cap = trial % 4 + 1
                expected = reference_layers(votes, cap)
                for chunk in (1, 7, 32, 256):
                    with self.subTest(dimensions=dimensions, trial=trial, chunk=chunk), \
                            patch.object(strategy, '_PAIRWISE_BLOCK_ROWS', chunk):
                        np.testing.assert_array_equal(
                            strategy.daily_front_layers(votes, cap), expected)
                # Non-contiguous column views and finite ternary floats are accepted.
                np.testing.assert_array_equal(strategy.daily_front_layers(votes[:, ::-1].astype(float), cap),
                                              expected)

    def test_high_bits_and_extremes_are_not_lost(self):
        votes = np.zeros((5, 64), dtype=np.int8)
        votes[0, 63] = 1
        votes[1, 33] = 1
        votes[3] = -1
        votes[4] = 1
        np.testing.assert_array_equal(strategy.daily_front_layers(votes), [2, 2, 3, 4, 1])

    def test_3300_stock_antichain_keeps_every_nondominated_stock(self):
        rng = np.random.default_rng(99)
        half = rng.integers(-1, 2, (3300, 17), dtype=np.int8)
        votes = np.column_stack((half, -half))
        layers = strategy.daily_front_layers(votes)
        np.testing.assert_array_equal(layers, np.ones(3300, dtype=np.int32))
        targets = [strategy.risk_target(layer, True, .3, .01) for layer in layers]
        np.testing.assert_array_equal(targets, np.full(3300, .5))
        self.assertEqual(sum(targets), 1650.)  # Independent NAVs, no shared cash budget.

    def test_empty_singleton_and_all_identical(self):
        result = strategy.daily_front_layers(np.empty((0, 34), dtype=np.int8))
        self.assertEqual(result.shape, (0,))
        self.assertEqual(result.dtype, np.int32)
        np.testing.assert_array_equal(strategy.daily_front_layers([[-1] * 34]), [1])
        np.testing.assert_array_equal(strategy.daily_front_layers(np.zeros((37, 34))), 1)

    def test_invalid_inputs(self):
        for votes in ([], [1, 0], 1, [[[1]]], np.empty((3, 0)), np.zeros((1, 65)),
                      [[2]], [[-2]], [[0.5]], [[np.nan]], [[np.inf]], [[1j]], [['1']], [[True]]):
            with self.subTest(votes=votes), self.assertRaises(ValueError):
                strategy.daily_front_layers(votes)
        for cap in (0, -1, 5, 1.5, True, np.bool_(False), '4', None):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                strategy.daily_front_layers([[0]], cap)


class TestRiskConfig(unittest.TestCase):
    def test_defaults_and_immutability(self):
        config = strategy.RiskConfig()
        self.assertEqual(config.annual_vol_target, .15)
        self.assertEqual(config.es_budget, .02)
        self.assertEqual(config.risk_window, 63)
        self.assertEqual(config.es_confidence, .95)
        self.assertEqual(config.tracking_penalty, 5.)
        self.assertEqual(config.max_weight, 1.)
        with self.assertRaises(FrozenInstanceError):
            config.max_weight = .5
        config = strategy.RiskConfig(risk_window=np.int64(2), es_confidence=np.float64(.8),
                                     annual_vol_target=2., es_budget=1., max_weight=.8)
        self.assertIsInstance(config.risk_window, int)
        self.assertIsInstance(config.es_confidence, float)

    def test_finite_numeric_ranges(self):
        for name in ('annual_vol_target', 'es_budget', 'tracking_penalty', 'max_weight'):
            for bad in (np.nan, np.inf, -np.inf, 0, -1, True, np.bool_(False),
                        '.15', None, [1.], 1j):
                with self.subTest(name=name, bad=bad), self.assertRaises(ValueError):
                    strategy.RiskConfig(**{name: bad})
        for name in ('es_budget', 'max_weight'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                strategy.RiskConfig(**{name: 1.001})
        for bad in (np.nan, np.inf, -np.inf, 0, 1, -.1, 1.1, True, '.95', [.95], None):
            with self.subTest(confidence=bad), self.assertRaises(ValueError):
                strategy.RiskConfig(es_confidence=bad)
        for bad in (0, 1, -1, 2., True, np.bool_(False), '63', np.nan, np.inf, [63], None):
            with self.subTest(window=bad), self.assertRaises(ValueError):
                strategy.RiskConfig(risk_window=bad)


class TestRiskStatistics(unittest.TestCase):
    def test_rolling_sample_volatility_and_tail_against_sort_oracle(self):
        closes = market(150)[3]
        returns = np.diff(closes) / closes[:-1]
        for window in (2, 4, 17, 63):
            for confidence in (.1, .6, .95, .999):
                with self.subTest(window=window, confidence=confidence):
                    result = strategy.risk_statistics(closes, window, confidence)
                    self.assertEqual(set(result), {'annual_vol', 'daily_es'})
                    expected_vol, expected_es = [], []
                    count = int(np.ceil(window * (1 - confidence)))
                    for i in range(window, len(closes)):
                        sample = returns[i - window:i]
                        expected_vol.append(np.std(sample, ddof=1) * np.sqrt(252))
                        expected_es.append(np.sort(np.maximum(0, -sample))[-count:].mean())
                    for key in result:
                        self.assertEqual(result[key].dtype, np.float64)
                        self.assertEqual(result[key].shape, closes.shape)
                        self.assertTrue(np.isnan(result[key][:window]).all())
                    np.testing.assert_allclose(result['annual_vol'][window:], expected_vol)
                    np.testing.assert_allclose(result['daily_es'][window:], expected_es)

    def test_tail_rounding_includes_zero_losses_not_only_negative_days(self):
        # Four returns: -.08, +.02, +.03, +.01. ceil(4*.4)=2 includes one zero.
        returns = np.array([-.08, .02, .03, .01])
        closes = 10 * np.r_[1., np.cumprod(1 + returns)]
        result = strategy.risk_statistics(closes, window=4, confidence=.6)
        self.assertAlmostEqual(result['daily_es'][4], .04)
        self.assertAlmostEqual(result['annual_vol'][4], returns.std(ddof=1) * np.sqrt(252))
        worst = strategy.risk_statistics(closes, window=4, confidence=.99)
        self.assertAlmostEqual(worst['daily_es'][4], .08)

    def test_empty_short_constant_positive_and_scale_invariance(self):
        for n in (0, 1, 2, 62, 63):
            result = strategy.risk_statistics(np.full(n, 10.))
            for values in result.values():
                self.assertEqual(values.shape, (n,))
                self.assertTrue(np.isnan(values).all())
        result = strategy.risk_statistics(np.full(70, 10.))
        for values in result.values():
            np.testing.assert_array_equal(values[63:], 0.)
        rising = strategy.risk_statistics(np.arange(10., 90.))
        np.testing.assert_array_equal(rising['daily_es'][63:], 0.)
        closes = market(150)[3]
        original = strategy.risk_statistics(closes)
        scaled = strategy.risk_statistics(closes * 16)
        for key in original:
            np.testing.assert_allclose(original[key], scaled[key])
        strided = strategy.risk_statistics(np.repeat(closes, 2)[::2])
        for key in original:
            np.testing.assert_array_equal(original[key], strided[key])

    def test_causal_prefixes_and_current_close_alignment(self):
        closes = market(150)[3]
        for window in (2, 17, 63):
            result = strategy.risk_statistics(closes, window)
            for n in (0, 1, window, window + 1, 100, 150):
                prefix = strategy.risk_statistics(closes[:n], window)
                for key in result:
                    np.testing.assert_array_equal(prefix[key], result[key][:n])
            extended = strategy.risk_statistics(np.r_[closes, market(80, 42)[3] * 100], window)
            for key in result:
                np.testing.assert_array_equal(extended[key][:150], result[key])
        crash = np.r_[np.full(63, 100.), 50.]
        result = strategy.risk_statistics(crash)
        self.assertTrue(np.isnan(result['daily_es'][:63]).all())
        # Current close supplies the first complete 63-return window: four worst losses.
        self.assertAlmostEqual(result['daily_es'][63], .5 / 4)
        self.assertGreater(result['annual_vol'][63], 0.)

    def test_invalid_inputs(self):
        for closes in ([0, 1], [-1, 1], [np.nan], [np.inf], [-np.inf],
                       [[1., 2.]], 10., None, ['1', '2'], [1j], [True, False]):
            with self.subTest(closes=closes), self.assertRaises(ValueError):
                strategy.risk_statistics(closes)
        for window in (0, 1, -1, 2., True, '63', [63], np.nan, np.inf):
            with self.subTest(window=window), self.assertRaises(ValueError):
                strategy.risk_statistics([1, 2, 3], window=window)
        for confidence in (0, 1, -.1, 1.1, np.nan, np.inf, True, '.95', [.95]):
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                strategy.risk_statistics([1, 2, 3], confidence=confidence)


class TestRiskTarget(unittest.TestCase):
    def test_inverse_volatility_and_es_without_rank_sizing(self):
        self.assertAlmostEqual(strategy.risk_target(1, True, .3, 0), .5)
        self.assertAlmostEqual(strategy.risk_target(1, True, .6, 0), .25)
        self.assertAlmostEqual(strategy.risk_target(1, True, 0, .04), .5)
        self.assertAlmostEqual(strategy.risk_target(1, True, 0, .08), .25)
        self.assertAlmostEqual(strategy.risk_target(1, True, .4, .08), .25)
        self.assertAlmostEqual(strategy.risk_target(1, True, .4, .01), .375)
        for layer in (-1, 0, 2, 3, 4, 5, 100):
            self.assertEqual(strategy.risk_target(layer, True, .1, .01), 0.)

    def test_gate_missing_risk_and_bounds(self):
        self.assertEqual(strategy.risk_target(1, False, .1, .01), 0.)
        for bad in (np.nan, np.inf, -np.inf):
            self.assertEqual(strategy.risk_target(1, True, bad, .01), 0.)
            self.assertEqual(strategy.risk_target(1, True, .1, bad), 0.)
        config = strategy.RiskConfig(max_weight=.65)
        self.assertEqual(strategy.risk_target(1, True, 0, 0, config), .65)
        self.assertEqual(strategy.risk_target(1, True, 0, 0), 1.)
        for vol in (0, 1e-12, .1, .2, 2., 1e100):
            for es in (0, 1e-12, .01, .1, 1.):
                weight = strategy.risk_target(np.int32(1), np.bool_(True), vol, es, config)
                self.assertIsInstance(weight, float)
                self.assertGreaterEqual(weight, 0.)
                self.assertLessEqual(weight, config.max_weight)

    def test_configuration_changes_risk_budgets_not_front_membership(self):
        config = strategy.RiskConfig(annual_vol_target=.2, es_budget=.03)
        self.assertAlmostEqual(strategy.risk_target(1, True, .4, .02, config), .5)
        self.assertAlmostEqual(strategy.risk_target(1, True, .1, .06, config), .5)
        self.assertEqual(strategy.risk_target(2, True, .1, .01, config), 0.)

    def test_invalid_scalar_inputs(self):
        for layer in (True, 1., np.nan, np.inf, '1', [1], None):
            with self.subTest(layer=layer), self.assertRaises(ValueError):
                strategy.risk_target(layer, True, .2, .01)
        for gate in (1, 0, 'yes', None, [True], np.nan):
            with self.subTest(gate=gate), self.assertRaises(ValueError):
                strategy.risk_target(1, gate, .2, .01)
        for index in (2, 3):
            for bad in (-.1, True, '.2', [.2], 1j, None):
                args = [1, True, .2, .01]
                args[index] = bad
                with self.subTest(index=index, bad=bad), self.assertRaises(ValueError):
                    strategy.risk_target(*args)
        with self.assertRaises(ValueError):
            strategy.risk_target(1, True, .2, .01, config=None)


class TestOptimalRebalanceWeight(unittest.TestCase):
    def test_asymmetric_no_trade_zone_and_soft_threshold(self):
        # lambda*vol^2=.2; buy band=.01/.2=.05, sell band=.02/.2=.10.
        for desired in (.41, .449, .35, .301, .4):
            self.assertAlmostEqual(strategy.optimal_rebalance_weight(.4, desired, .2, .01, .02),
                                   .4)
        self.assertAlmostEqual(strategy.optimal_rebalance_weight(.4, .6, .2, .01, .02), .55)
        self.assertAlmostEqual(strategy.optimal_rebalance_weight(.4, .2, .2, .01, .02), .3)
        # Opposite-side costs must not affect the chosen side of the trade.
        self.assertAlmostEqual(strategy.optimal_rebalance_weight(.4, .6, .2, .01, 1.), .55)
        self.assertAlmostEqual(strategy.optimal_rebalance_weight(.4, .2, .2, 1., .02), .3)

    def test_zero_cost_bounds_and_appreciated_holdings_above_cap(self):
        config = strategy.RiskConfig(max_weight=.8)
        for current in (0., .4, .8, 1.2, 1e100):
            for desired in (0., .3, .8, 1.5):
                target = strategy.optimal_rebalance_weight(current, desired, .2, 0, 0, config)
                self.assertIsInstance(target, float)
                self.assertAlmostEqual(target, min(desired, config.max_weight))
        self.assertAlmostEqual(strategy.optimal_rebalance_weight(1.2, .7, .2, 0, .04), .9)
        self.assertEqual(strategy.optimal_rebalance_weight(1.2, 1.2, .2, 1, 1, config), .8)
        self.assertEqual(strategy.optimal_rebalance_weight(0., 1.5, .2, 0, 0, config), .8)

    def test_force_exit_and_zero_desired_override_costs(self):
        for current in (0., .2, .9, 1.3):
            for vol in (0., 1e-12, .2):
                self.assertEqual(strategy.optimal_rebalance_weight(current, .8, vol, 1, 1,
                                                                   force_exit=True), 0.)
                self.assertEqual(strategy.optimal_rebalance_weight(current, 0., vol, 1, 1), 0.)
        desired = strategy.risk_target(1, False, .2, .01)
        self.assertEqual(strategy.optimal_rebalance_weight(.8, desired, .2, .01, .02), 0.)

    def test_variance_floor_and_tracking_penalty(self):
        self.assertAlmostEqual(strategy.optimal_rebalance_weight(.4, .8, 0., 0., 0.), .8)
        self.assertAlmostEqual(strategy.optimal_rebalance_weight(.4, .8, 0., .001, .001), .4)
        for vol in (0., 1e-12, 1e-4):
            self.assertAlmostEqual(strategy.optimal_rebalance_weight(.4, .8, vol, 5e-9, 5e-9), .7)
        config = strategy.RiskConfig(tracking_penalty=10.)
        self.assertAlmostEqual(strategy.optimal_rebalance_weight(.4, .6, .2, .01, .02, config),
                               .575)
        self.assertAlmostEqual(strategy.optimal_rebalance_weight(.4, .6, .4, .01, .02), .5875)

    def test_analytic_solution_against_brute_grid(self):
        rng = np.random.default_rng(456)
        for trial in range(60):
            cap = float(rng.uniform(.2, 1.))
            config = strategy.RiskConfig(max_weight=cap,
                                         tracking_penalty=float(rng.uniform(.2, 15.)))
            current = float(rng.uniform(0, 1.5))  # Includes drift above max_weight.
            desired = float(rng.uniform(.01, 1.5))  # Zero is a separate hard constraint.
            vol = 0. if trial % 10 == 0 else float(rng.uniform(.02, 1.))
            buy, sell = rng.uniform(0, .06, 2)
            if trial % 7 == 0:
                buy = sell = 0.
            actual = strategy.optimal_rebalance_weight(current, desired, vol, buy, sell, config)
            grid = np.linspace(0, cap, 20001)
            curvature = config.tracking_penalty * max(vol * vol, 1e-8)

            def objective(weight):
                return (0.5 * curvature * (weight - desired) ** 2
                        + buy * np.maximum(weight - current, 0)
                        + sell * np.maximum(current - weight, 0))

            costs = objective(grid)
            best = grid[np.argmin(costs)]
            with self.subTest(trial=trial):
                self.assertGreaterEqual(actual, 0.)
                self.assertLessEqual(actual, cap)
                self.assertLessEqual(abs(actual - best), cap / 20000 + 1e-12)
                self.assertLessEqual(objective(actual), costs.min() + 1e-12)

    def test_strict_finite_validation_even_for_exit(self):
        for index in range(5):
            for bad in (np.nan, np.inf, -np.inf, -.1, True, '.1', [.1], 1j, None):
                for forced in (False, True):
                    args = [.4, .6, .2, .001, .002]
                    args[index] = bad
                    with self.subTest(index=index, bad=bad, forced=forced), \
                            self.assertRaises(ValueError):
                        strategy.optimal_rebalance_weight(*args, force_exit=forced)
        for buy, sell in ((1.001, 0.), (0., 1.001)):
            with self.assertRaises(ValueError):
                strategy.optimal_rebalance_weight(.4, .6, .2, buy, sell)
        for forced in (1, 'yes', None, [True], np.nan):
            with self.subTest(forced=forced), self.assertRaises(ValueError):
                strategy.optimal_rebalance_weight(.4, .6, .2, 0, 0, force_exit=forced)
        with self.assertRaises(ValueError):
            strategy.optimal_rebalance_weight(.4, .6, .2, 0, 0, config=None)


if __name__ == '__main__':
    unittest.main()