"""Offline runner contracts using disposable SQLite/NPZ/JSON/front snapshots.

Only orchestration (process execution and network access) is replaced. Replay,
accounting, risk sizing and aggregation use the real implementations. Regression
tests enforce continuous phase accounting and conservative data-quality handling.
No production cache is opened; SQLite connections outside the temporary root
and all network connections fail immediately. This module starts no workers.
"""

from concurrent.futures import Future
from contextlib import closing, redirect_stderr, redirect_stdout
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit
from urllib.request import url2pathname

import numpy as np

import pareto_backtest as runner
from capital_account import AccountConfig
from pareto_strategy import OBJECTIVE_NAMES, STRATEGY_VERSION, RiskConfig


PRINCIPAL = 1_000_000.
FILL_EVENTS = {'build', 'add', 'reduce', 'exit'}


def sessions(start='2024-01-02', count=18):
    """Synthetic weekday calendar, not a claim about exchange holidays."""
    return np.busday_offset(start, np.arange(count), roll='forward').astype('U10')


def ohlcv(prices, volumes=1000.):
    prices = np.asarray(prices, dtype=float)
    return np.column_stack((prices, prices * 1.01, prices * .99, prices,
                            np.broadcast_to(volumes, prices.shape)))


def action(day, cash=0., multiplier=1.):
    return dict(ex_date=str(day), cash_per_share=cash, share_multiplier=multiplier)


def fills(ledger):
    return [row for row in ledger if row['event'] in FILL_EVENTS]


def ledger_nav(ledger, through, price):
    """Independent cash/share arithmetic; no Account or runner summary calls."""
    cash, shares = PRINCIPAL, 0.
    for row in ledger:
        if row['day'] > through:
            continue
        if row['event'] == 'action':
            cash += row['gross']
            shares += row['quantity']
        elif row['event'] in FILL_EVENTS:
            direction = 1 if row['side'] == 'buy' else -1
            cash -= direction * row['gross'] + row['fee']
            shares += direction * row['quantity']
    return cash + shares * price


def summary_record(code, cash, residual=0., **changes):
    """Complete aggregate input without guessing optional runner fields."""
    record = dict(code=code, initial_capital=PRINCIPAL, final_cash=cash,
                  residual_value=residual, pnl=cash + residual - PRINCIPAL,
                  liquidation_complete=residual == 0., fees=0., cash_dividends=0.,
                  action_status='ok', rights_issue_present=False, feature_error=None,
                  unexplained_gap_count=0, stale=False, late_start=False,
                  build_count=0, add_count=0, reduce_count=0, exit_count=0)
    record.update(changes)
    return record


class InlineExecutor:
    """ProcessPoolExecutor protocol with real Futures but no child processes."""

    def __init__(self, max_workers=None):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def map(self, function, iterable):
        return map(function, iterable)

    def submit(self, function, *args, **kwargs):
        future = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except Exception as error:
            future.set_exception(error)
        return future


class SnapshotCase(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory(prefix='pareto-runner-test-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.fixture_count = 0
        real_connect = sqlite3.connect

        def temporary_connect(database, *args, **kwargs):
            location = os.fspath(database)
            if location.startswith('file:'):
                parsed = urlsplit(location)
                location = url2pathname(('//' + parsed.netloc if parsed.netloc else '')
                                       + parsed.path)
            try:
                Path(location).resolve().relative_to(self.root)
            except ValueError as error:
                raise AssertionError('non-fixture SQLite access forbidden') from error
            return real_connect(database, *args, **kwargs)

        self.enterContext(patch('sqlite3.connect', side_effect=temporary_connect))
        for target in ('socket.socket.connect', 'socket.socket.connect_ex',
                       'socket.create_connection', 'socket.getaddrinfo',
                       'urllib.request.urlopen', 'action_snapshot.collect_action_snapshot'):
            self.enterContext(patch(target, side_effect=AssertionError('network forbidden')))
        self.enterContext(patch.object(runner, 'ProcessPoolExecutor', InlineExecutor))

    @staticmethod
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')

    def fixture(self, calendar=None, stocks=None):
        """Build every runner artifact; prepared=False exercises prepare_stock.

        Stock-level dates can omit common sessions (suspension/stale coverage).
        All supplied risk/eligibility vectors are deliberately synthetic, so
        small replay tests need not fabricate 250 warm-up observations.
        """
        calendar = np.asarray(sessions() if calendar is None else calendar, dtype='U10')
        stocks = [dict(code='600000')] if stocks is None else stocks
        self.fixture_count += 1
        folder = self.root / f'run-{self.fixture_count}'
        (folder / 'features').mkdir(parents=True)
        (folder / 'actions').mkdir()
        universe, columns = [], []
        with closing(sqlite3.connect(folder / 'source.sqlite')) as db, db:
            db.executescript('''
                CREATE TABLE stocks (code TEXT PRIMARY KEY, name TEXT);
                CREATE TABLE klines (
                    code TEXT, date TEXT, open REAL, high REAL, low REAL,
                    close REAL, volume REAL, PRIMARY KEY (code, date));
                CREATE TABLE dividends (
                    code TEXT, ex_date TEXT, dividend_per_share REAL,
                    bonus_share REAL, transfer_share REAL);
            ''')
            for spec in stocks:
                code = spec['code']
                name = spec.get('name', 'synthetic-' + code)
                dates = np.asarray(spec.get('dates', calendar), dtype='U10')
                n = len(dates)
                raw = np.asarray(spec.get('raw', ohlcv(np.full(n, 10.))), dtype=float)
                history = dict(code=code, status='ok', rights_issue_present=False,
                               events=spec.get('events', []))
                history.update(spec.get('history', {}))
                self.write_json(folder / 'actions' / (code + '.json'), history)
                db.execute('INSERT INTO stocks VALUES (?, ?)', (code, name))
                db.executemany('INSERT INTO klines VALUES (?, ?, ?, ?, ?, ?, ?)',
                               [(code, str(day), *[float(v) for v in row])
                                for day, row in zip(dates, raw)])
                universe.append(dict(code=code, name=name,
                                     first_date=str(dates[0]) if n else None,
                                     last_date=str(dates[-1]) if n else None))
                columns.append(np.broadcast_to(spec.get('layers', 1), (len(calendar),)))
                if not spec.get('prepared', True):
                    continue
                event_cash, event_mult = np.zeros(n), np.ones(n)
                for i, (cash, mult) in runner.aligned_actions(history['events'], dates).items():
                    event_cash[i], event_mult[i] = cash, mult
                data = dict(dates=dates, raw=raw,
                            votes=np.asarray(spec.get('votes', np.zeros((n, 34))), dtype=np.int8),
                            eligible=np.broadcast_to(spec.get('eligible', True), (n,)),
                            gates=np.broadcast_to(spec.get('gates', True), (n,)),
                            annual_vol=np.broadcast_to(spec.get('annual_vol', .5), (n,)),
                            daily_es=np.broadcast_to(spec.get('daily_es', .01), (n,)),
                            event_cash=event_cash, event_mult=event_mult)
                np.savez_compressed(folder / 'features' / (code + '.npz'), **data)
                meta = dict(code=code, name=name, bars=n, feature_version=STRATEGY_VERSION,
                            first_date=str(dates[0]) if n else None,
                            last_date=str(dates[-1]) if n else None,
                            late_start=bool(not n or dates[0] > calendar[0]),
                            stale=bool(not n or dates[-1] < calendar[-1]),
                            feature_error=None, unexplained_gap_count=0,
                            action_status=history['status'], action_count=len(history['events']),
                            rights_issue_present=history['rights_issue_present'])
                meta.update(spec.get('meta', {}))
                self.write_json(folder / 'features' / (code + '.json'), meta)
        manifest = dict(version=STRATEGY_VERSION, start=str(calendar[0]),
                        requested_end=str(calendar[-1]), effective_end=str(calendar[-1]),
                        calendar=calendar.tolist(), universe=universe,
                        objective_names=list(OBJECTIVE_NAMES),
                        risk_defaults=asdict(RiskConfig()), account_defaults=asdict(AccountConfig()),
                        candidate_vol_targets=list(runner.CANDIDATE_VOL_TARGETS),
                        train_end=runner.TRAIN_END, validation_end=runner.VALIDATION_END,
                        limitations=['synthetic fixtures, no production data'])
        self.write_json(folder / 'manifest.json', manifest)
        np.save(folder / 'fronts.npy', np.column_stack(columns).astype(np.int8))
        return folder

    @staticmethod
    def replay(folder, code='600000', col=0, end=None):
        return runner.replay_stock(folder, code, col, RiskConfig(), end, record_ledger=True)

    @staticmethod
    def run_fixture(folder):
        manifest = runner.read_json(folder / 'manifest.json')
        with redirect_stdout(io.StringIO()):
            return runner.run(folder, folder / 'source.sqlite', manifest['start'],
                              manifest['requested_end'], workers=1, fetch_actions=False)


class TestCausalPrices(unittest.TestCase):
    def test_split_cash_and_combined_ex_date_continuity(self):
        for cash, multiplier in ((0., 2.), (2., 1.), (2., 2.)):
            with self.subTest(cash=cash, multiplier=multiplier):
                reference = (10. - cash) / multiplier
                raw = ohlcv([10., 10., reference, reference],
                            [1000., 1000., 1000. * multiplier, 1000. * multiplier])
                original = raw.copy()
                adjusted = runner.causal_adjusted_prices(raw, {2: (cash, multiplier)})
                np.testing.assert_allclose(adjusted, ohlcv(np.full(4, 10.)))
                np.testing.assert_array_equal(adjusted[:2], raw[:2])
                np.testing.assert_array_equal(raw, original)
                self.assertFalse(np.shares_memory(raw, adjusted))

    def test_future_actions_and_prices_never_rewrite_any_earlier_prefix(self):
        raw = ohlcv([10., 10., 4., 4., 4., 1.8, 1.8],
                    [1000., 1000., 2000., 2000., 2000., 4000., 4000.])
        actions = {2: (2., 2.), 5: (.4, 2.)}
        full = runner.causal_adjusted_prices(raw, actions)
        for length in range(len(raw) + 1):
            with self.subTest(length=length):
                prefix = runner.causal_adjusted_prices(
                    raw[:length], {i: value for i, value in actions.items() if i < length})
                np.testing.assert_array_equal(prefix, full[:length])
        changed = raw.copy()
        changed[5:, :4] *= 3
        alternative = runner.causal_adjusted_prices(changed, {2: (2., 2.), 5: (1., 3.)})
        np.testing.assert_array_equal(alternative[:5], full[:5])

    def test_nonpositive_ex_rights_reference_is_rejected(self):
        for cash, mult in ((10., 1.), (11., 2.), (0., -1.)):
            with self.subTest(cash=cash, mult=mult), self.assertRaises(ValueError):
                runner.causal_adjusted_prices(ohlcv([10., 5.]), {1: (cash, mult)})

    def test_aligned_actions_ignore_first_bar_and_compose_missing_bar_events(self):
        dates = np.array(['2024-01-05', '2024-01-09', '2024-01-10'], dtype='U10')
        events = [action('2024-01-11', 99.), action('2024-01-08', .5, 1.5),
                  action('2023-12-29', 99.), action(dates[0], 99.),
                  action('2024-01-06', 1., 2.), action(dates[2], .25)]
        self.assertEqual(runner.aligned_actions(events, dates),
                         {1: (2., 3.), 2: (.25, 1.)})
        self.assertEqual(runner.aligned_actions(events, np.array([], dtype='U10')), {})


class TestAggregate(unittest.TestCase):
    def test_exact_three_accounts_include_loser_and_all_principal(self):
        records = [summary_record('600000', 1_100_000.),
                   summary_record('000001', 500_000.),
                   summary_record('600001', 5_000_000.)]
        result = runner.aggregate(records)
        self.assertEqual(result['account_count'], 3)
        self.assertEqual(result['initial_capital'], 3_000_000.)
        self.assertEqual(result['final_cash'], 6_600_000.)
        self.assertEqual(result['final_equity'], 6_600_000.)
        self.assertEqual(result['pnl'], 3_600_000.)
        self.assertEqual(result['cash_minus_all_principal'], 3_600_000.)
        self.assertEqual(result['residual_marked_value'], 0.)
        self.assertEqual(result['profitable_accounts'], 2)
        self.assertEqual(result['losing_accounts'], 1)
        self.assertEqual(result['fully_liquidated_count'], 3)

    def test_residual_is_not_cash_and_fees_dividends_are_not_added_twice(self):
        records = [summary_record('600000', 800_000., 250_000., fees=123.,
                                  cash_dividends=456., stale=True, build_count=1),
                   summary_record('000001', PRINCIPAL, feature_error='bad OHLC',
                                  action_status='cache_unverified', rights_issue_present=True,
                                  unexplained_gap_count=2, late_start=True)]
        result = runner.aggregate(records)
        self.assertEqual(result['final_cash'], 1_800_000.)
        self.assertEqual(result['final_equity'], 2_050_000.)
        self.assertEqual(result['pnl'], 50_000.)
        self.assertEqual(result['cash_minus_all_principal'], -200_000.)
        self.assertEqual(result['fees'], 123.)
        self.assertEqual(result['cash_dividends'], 456.)
        for key in ('fully_liquidated_count', 'feature_errors', 'action_fetch_failures',
                    'rights_issue_accounts', 'unexplained_gap_accounts', 'stale_accounts',
                    'late_start_accounts', 'build_count'):
            self.assertEqual(result[key], 1, key)


class TestReplay(SnapshotCase):
    def test_dynamic_zero_to_f1_volatility_build_add_reduce_exit_next_open(self):
        calendar = sessions()
        layers = np.full(len(calendar), 5, dtype=np.int8)
        layers[1:4] = 1
        vol = np.full(len(calendar), .5)
        vol[1:4] = [.6, .2, .75]
        folder = self.fixture(calendar, [dict(code='600000', layers=layers, annual_vol=vol)])
        summary, curve, ledger = self.replay(folder)
        orders = fills(ledger)
        self.assertEqual([r['event'] for r in orders], ['build', 'add', 'reduce', 'exit'])
        self.assertEqual([r['day'] for r in orders], calendar[2:6].tolist())
        self.assertEqual([r['signal_day'] for r in orders], calendar[1:5].tolist())
        self.assertTrue(all(r['raw_price'] == 10. for r in orders))
        self.assertGreater(orders[1]['target_weight'], orders[0]['target_weight'])
        self.assertLess(orders[2]['target_weight'], orders[0]['target_weight'])
        self.assertAlmostEqual(orders[2]['target_weight'], .2)
        self.assertEqual(orders[3]['target_weight'], 0.)
        for event in FILL_EVENTS:
            self.assertEqual(summary[event + '_count'], 1)
        np.testing.assert_array_equal(curve[:2], PRINCIPAL)
        self.assertEqual(curve[-1], summary['final_cash'])
        self.assertTrue(summary['liquidation_complete'])

    def test_ex_date_first_buyer_cannot_receive_old_holder_entitlements(self):
        calendar = sessions()
        folder = self.fixture(calendar, [dict(
            code='600000', raw=ohlcv([10.] + [4.] * (len(calendar) - 1)),
            events=[action(calendar[0], 9., 10.), action(calendar[1], 2., 2.)])])
        summary, _, ledger = self.replay(folder)
        actions = [row for row in ledger if row['event'] == 'action']
        self.assertEqual(len(actions), 1)  # No pre-history entitlement at index zero.
        self.assertEqual(actions[0]['day'], calendar[1])
        self.assertEqual(actions[0]['eligible_shares'], 0.)
        self.assertEqual(actions[0]['quantity'], 0.)
        self.assertEqual(actions[0]['gross'], 0.)
        self.assertEqual(summary['cash_dividends'], 0.)
        buy = fills(ledger)[0]
        self.assertEqual((buy['event'], buy['day'], buy['raw_price']),
                         ('build', calendar[1], 4.))
        self.assertLess(ledger.index(actions[0]), ledger.index(buy))
        self.assertEqual(buy['shares'], buy['quantity'])

    def test_final_five_sessions_override_pending_new_buy(self):
        calendar = sessions()
        deadline = len(calendar) - 5
        layers = np.full(len(calendar), 5, dtype=np.int8)
        layers[deadline - 1:] = 1  # First BUY would execute at the deadline start.
        folder = self.fixture(calendar, [dict(code='600000', layers=layers)])
        summary, curve, ledger = self.replay(folder)
        self.assertEqual(fills(ledger), [])
        self.assertEqual(summary['final_cash'], PRINCIPAL)
        np.testing.assert_array_equal(curve, PRINCIPAL)

    def test_scheduled_exit_respects_t_plus_one_and_never_reopens(self):
        calendar = sessions()
        folder = self.fixture(calendar)
        summary, _, ledger = self.replay(folder)
        orders = fills(ledger)
        buys = [row for row in orders if row['side'] == 'buy']
        sells = [row for row in orders if row['side'] == 'sell']
        self.assertTrue(buys)
        self.assertEqual(len(sells), 1)
        self.assertEqual(sells[0]['day'], calendar[-5])
        self.assertEqual(sells[0]['reason'], 'scheduled_deadline')
        self.assertIsNone(sells[0]['signal_day'])
        self.assertTrue(all(row['day'] < sells[0]['day'] for row in buys))
        self.assertTrue(summary['liquidation_complete'])
        short = self.fixture(sessions(count=5))
        self.assertEqual(fills(self.replay(short)[2]), [])

    def test_same_day_close_risk_signal_waits_for_next_day_sell(self):
        calendar = sessions()
        raw = ohlcv([10., 6.] + [6.] * (len(calendar) - 2))
        raw[1, 0], raw[1, 1] = 10., 10.1  # Collapse AFTER the buy, not an open gap.
        folder = self.fixture(calendar, [dict(code='600000', raw=raw, annual_vol=.2)])
        summary, _, ledger = self.replay(folder)
        orders = fills(ledger)
        self.assertEqual([row['event'] for row in orders], ['build', 'exit'])
        self.assertEqual([row['day'] for row in orders], calendar[1:3].tolist())
        self.assertEqual(orders[1]['signal_day'], calendar[1])
        self.assertEqual(orders[1]['reason'], 'risk_exit')
        self.assertGreaterEqual(summary['max_drawdown_observed'], .15)

    def test_stale_last_date_is_not_a_foreknown_liquidation_deadline(self):
        calendar = sessions()
        stale = self.fixture(calendar, [dict(code='600000', dates=calendar[:5])])
        complete = self.fixture(calendar)
        summary, curve, ledger = self.replay(stale)
        _, complete_curve, complete_ledger = self.replay(complete)
        self.assertTrue(summary['stale'])
        self.assertEqual(summary['last_date'], calendar[4])
        self.assertFalse(summary['liquidation_complete'])
        self.assertGreater(summary['residual_value'], 0.)
        self.assertFalse(any(row['side'] == 'sell' for row in fills(ledger)))
        self.assertEqual(ledger, [r for r in complete_ledger if r['day'] <= calendar[4]])
        np.testing.assert_array_equal(curve[:5], complete_curve[:5])
        np.testing.assert_array_equal(curve[4:], curve[4])
        self.assertAlmostEqual(curve[-1], summary['final_cash'] + summary['residual_value'])

    def test_suspension_discards_stale_buy_but_next_fresh_signal_can_buy(self):
        calendar = sessions()
        dates = np.delete(calendar, 1)
        folder = self.fixture(calendar, [dict(code='600000', dates=dates)])
        _, curve, ledger = self.replay(folder)
        first = fills(ledger)[0]
        self.assertEqual(first['day'], calendar[3])
        self.assertEqual(first['signal_day'], calendar[2])
        np.testing.assert_array_equal(curve[:3], PRINCIPAL)

    def test_open_gap_buy_block_does_not_invent_a_fill(self):
        calendar = sessions()
        folder = self.fixture(calendar, [dict(
            code='600000', raw=ohlcv([10.] + [10.6] * (len(calendar) - 1)))])
        _, _, ledger = self.replay(folder)
        blocked = [r for r in ledger if r.get('block_reason') == 'block_buy']
        self.assertEqual(len(blocked), 1)
        self.assertEqual(blocked[0]['day'], calendar[1])
        self.assertEqual(blocked[0]['cash'], PRINCIPAL)
        self.assertEqual(blocked[0]['shares'], 0.)
        self.assertEqual(fills(ledger)[0]['day'], calendar[2])

    def test_blocked_exit_retries_at_actual_later_price(self):
        calendar = sessions()
        layers = np.full(len(calendar), 5, dtype=np.int8)
        layers[:2] = 1
        raw = ohlcv([10., 10., 10., 9., 8.1] + [8.1] * (len(calendar) - 5))
        folder = self.fixture(calendar, [dict(code='600000', raw=raw, layers=layers)])
        summary, _, ledger = self.replay(folder)
        blocked = [r for r in ledger if r.get('block_reason') == 'block_sell']
        self.assertEqual([row['day'] for row in blocked], calendar[3:5].tolist())
        self.assertEqual(blocked[0]['cash'], blocked[1]['cash'])
        self.assertEqual(blocked[0]['shares'], blocked[1]['shares'])
        sell = fills(ledger)[-1]
        self.assertEqual((sell['event'], sell['day'], sell['raw_price']),
                         ('exit', calendar[5], 8.1))
        self.assertTrue(summary['liquidation_complete'])

    def test_gap_blocked_deadline_keeps_residual_not_fictitious_cash(self):
        calendar = sessions()
        prices = 10. * .94 ** np.maximum(np.arange(len(calendar)) - 2, 0)
        layers = np.full(len(calendar), 5, dtype=np.int8)
        layers[:2] = 1
        folder = self.fixture(calendar, [dict(code='600000', raw=ohlcv(prices), layers=layers)])
        summary, curve, ledger = self.replay(folder)
        orders = fills(ledger)
        self.assertEqual([r['event'] for r in orders], ['build'])
        self.assertEqual(summary['final_cash'], orders[0]['cash'])
        self.assertEqual(summary['shares'], orders[0]['quantity'])
        self.assertAlmostEqual(summary['residual_value'], summary['shares'] * prices[-1])
        self.assertFalse(summary['liquidation_complete'])
        self.assertEqual(summary['exit_count'], 0)
        self.assertTrue(any(r.get('block_reason') == 'block_sell' and
                            r['reason'] == 'deadline_close' for r in ledger))
        self.assertAlmostEqual(curve[-1], summary['final_cash'] + summary['residual_value'])
        self.assertAlmostEqual(runner.aggregate([summary])['final_equity'], curve[-1])

    def test_train_nav_uses_exact_cutoff_mark_not_future_or_final_cash(self):
        calendar = sessions('2021-12-20', 35)
        prices = 10. * 1.01 ** np.arange(len(calendar))
        folder = self.fixture(calendar, [dict(code='600000', raw=ohlcv(prices))])
        summary, curve, ledger = self.replay(folder)
        cutoff = int(np.searchsorted(calendar, runner.TRAIN_END, side='right')) - 1
        expected = ledger_nav(ledger, str(calendar[cutoff]), prices[cutoff])
        self.assertAlmostEqual(summary['train_nav'], expected)
        self.assertEqual(summary['train_nav'], curve[cutoff])
        self.assertNotEqual(summary['train_nav'], curve[cutoff + 1])
        self.assertNotEqual(summary['train_nav'], summary['final_cash'])
        prices[cutoff + 1:] *= 1.02
        future = self.fixture(calendar, [dict(code='600000', raw=ohlcv(prices))])
        other, other_curve, _ = self.replay(future)
        self.assertEqual(other['train_nav'], summary['train_nav'])
        np.testing.assert_array_equal(other_curve[:cutoff + 1], curve[:cutoff + 1])

    def test_cutoffs_before_calendar_keep_original_principal(self):
        summary, _, _ = self.replay(self.fixture())
        self.assertEqual(summary['train_nav'], PRINCIPAL)
        self.assertEqual(summary['validation_nav'], PRINCIPAL)

    def test_cash_dividend_split_fees_and_slippage_reconcile_exactly_once(self):
        calendar = sessions()
        layers = np.full(len(calendar), 5, dtype=np.int8)
        layers[:3] = 1
        folder = self.fixture(calendar, [dict(
            code='600000', layers=layers, raw=ohlcv([10.] * 3 + [4.] * (len(calendar) - 3)),
            events=[action(calendar[3], 2., 2.)])])
        summary, curve, ledger = self.replay(folder)
        orders = fills(ledger)
        events = [row for row in ledger if row['event'] == 'action']
        self.assertEqual(len(events), 1)
        self.assertGreater(events[0]['eligible_shares'], 0.)
        self.assertEqual(events[0]['gross'], events[0]['eligible_shares'] * 2.)
        self.assertEqual(summary['cash_dividends'], events[0]['gross'])
        fee_total, slippage = 0., 0.
        for row in orders:
            commission = max(5., row['gross'] * .0003)
            stamp = 0. if row['side'] == 'buy' else row['gross'] * .0005
            transfer = row['gross'] * .00001
            self.assertAlmostEqual(row['fee'], commission + stamp + transfer)
            fee_total += commission + stamp + transfer
            slippage += abs(row['price'] - row['raw_price']) * row['quantity']
        self.assertTrue(summary['liquidation_complete'])
        self.assertAlmostEqual(summary['fees'], fee_total)
        self.assertAlmostEqual(summary['pnl'], -fee_total - slippage)
        self.assertAlmostEqual(summary['final_cash'], PRINCIPAL - fee_total - slippage)
        self.assertAlmostEqual(curve[-1], ledger_nav(ledger, calendar[-1], 4.))
        self.assertAlmostEqual(runner.aggregate([summary])['final_equity'], curve[-1])


class TestSnapshotRunner(SnapshotCase):
    def test_snapshot_backup_is_read_only_and_filters_mainboard(self):
        source_folder = self.fixture(stocks=[dict(code='600000'), dict(code='300001')])
        source = source_folder / 'source.sqlite'
        before = source.read_bytes()
        with closing(runner.connect_ro(source)) as db:
            self.assertEqual(db.execute('PRAGMA query_only').fetchone()[0], 1)
            with self.assertRaises(sqlite3.OperationalError):
                db.execute("DELETE FROM klines WHERE code='600000'")
        destination = self.root / 'backup'
        manifest = runner.create_run(destination, source, str(sessions()[0]), '2024-12-31')
        self.assertEqual([r['code'] for r in manifest['universe']], ['600000'])
        self.assertEqual(manifest['effective_end'], sessions()[-1])
        self.assertEqual(manifest['requested_end'], '2024-12-31')
        self.assertEqual(manifest['source_path'], str(source.resolve()))
        self.assertEqual(manifest['universe_limit'], 0)
        self.assertEqual(manifest['feature_start'], '2015-01-01')
        self.assertEqual(manifest['action_coverage_start'], '2015-01-01')
        self.assertEqual(manifest['source_sha256'],
                         hashlib.sha256((destination / 'source.sqlite').read_bytes()).hexdigest())
        self.assertEqual(source.read_bytes(), before)
        with closing(runner.connect_ro(destination / 'source.sqlite')) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM klines').fetchone()[0], 36)

    def test_bad_ohlc_and_nan_source_cache_stock_keep_principal_and_error_flag(self):
        calendar = sessions(count=270)
        for bad in (9., np.nan):  # high below open, or SQLite NULL -> NumPy NaN.
            with self.subTest(bad=bad):
                raw = ohlcv(np.full(len(calendar), 10.))
                raw[260, 1] = bad
                folder = self.fixture(calendar, [dict(code='600000', raw=raw, prepared=False)])
                before = (folder / 'source.sqlite').read_bytes()
                meta = runner.prepare_stock((str(folder), '600000', 'bad-input',
                                             str(calendar[0]), str(calendar[-1])))
                self.assertTrue(meta['feature_error'])
                with np.load(folder / 'features' / '600000.npz') as data:
                    self.assertFalse(data['eligible'].any())
                    self.assertFalse(data['gates'].any())
                    self.assertTrue(np.isnan(data['annual_vol']).all())
                with redirect_stdout(io.StringIO()):
                    runner.compute_fronts(folder, ['600000'], calendar)
                summary, curve, ledger = self.replay(folder)
                self.assertEqual(fills(ledger), [])
                self.assertEqual(summary['final_cash'], PRINCIPAL)
                self.assertEqual(summary['pnl'], 0.)
                self.assertEqual(summary['residual_value'], 0.)
                self.assertEqual(summary['feature_error'], meta['feature_error'])
                self.assertEqual(runner.aggregate([summary])['feature_errors'], 1)
                np.testing.assert_array_equal(curve, PRINCIPAL)
                self.assertEqual((folder / 'source.sqlite').read_bytes(), before)
                # Resume must use the just-built temporary cache, not fetch data.
                self.assertEqual(runner.prepare_stock((str(folder), '600000', 'bad-input',
                                                      str(calendar[0]), str(calendar[-1]))), meta)

    def test_fronts_only_compare_same_date_eligible_gate_passing_stocks(self):
        calendar = sessions(count=6)
        strong = np.ones((6, 34), dtype=np.int8)
        weak = np.zeros((6, 34), dtype=np.int8)
        eligible, gates = np.ones(6, bool), np.ones(6, bool)
        eligible[1], gates[2] = False, False
        folder = self.fixture(calendar, [
            dict(code='600000', votes=strong, eligible=eligible, gates=gates),
            dict(code='000001', votes=weak),
        ])
        stats = runner.compute_fronts(folder, ['600000', '000001'], calendar)
        layers = np.load(folder / 'fronts.npy')
        np.testing.assert_array_equal(layers[[0, 3, 4, 5]], [[1, 5]] * 4)
        np.testing.assert_array_equal(layers[[1, 2]], [[5, 1]] * 2)
        self.assertEqual([row['eligible'] for row in stats], [2, 1, 1, 2, 2, 2])
        self.assertTrue(all(row['first_front'] == 1 for row in stats))

    def test_population_columns_independent_principal_and_json_artifacts(self):
        folder = self.fixture(stocks=[dict(code='600001'), dict(code='000001', layers=5)])
        summaries, curve = runner.replay_population(
            folder, ['600001', '000001'], .15, workers=1, ledgers=True)
        self.assertEqual([s['code'] for s in summaries], ['000001', '600001'])
        self.assertEqual(summaries[0]['final_cash'], PRINCIPAL)
        self.assertEqual(summaries[0]['build_count'], 0)
        self.assertGreater(summaries[1]['build_count'], 0)
        for code in ('600001', '000001'):
            self.assertEqual(runner.read_json(folder / 'accounts' / (code + '.json'))['code'], code)
            self.assertIsInstance(runner.read_json(folder / 'ledgers' / (code + '.json')), list)
        totals = runner.aggregate(summaries)
        self.assertEqual(totals['initial_capital'], 2 * PRINCIPAL)
        self.assertAlmostEqual(curve[-1], totals['final_equity'])
        np.testing.assert_array_equal(curve[0], 2 * PRINCIPAL)

    def test_run_offline_resumes_artifacts_and_selects_training_only_ties_low_risk(self):
        folder = self.fixture()
        effective_end = runner.read_json(folder / 'manifest.json')['effective_end']
        # Cached trials deliberately rank validation opposite to training.
        for vol, train, validation in ((.10, 100., -900.), (.15, 100., 900.),
                                       (.20, 99., 9_000_000.)):
            self.write_json(folder / 'trials' / f'vol_{vol:.2f}.json', dict(
                vol_target=vol, train_pnl=train, validation_pnl=validation,
                validation_end_nav=PRINCIPAL + train + validation,
            scope_end=effective_end))
        before = (folder / 'source.sqlite').read_bytes()
        result = self.run_fixture(folder)
        self.assertEqual(result['selected_risk']['annual_vol_target'], .10)
        selection = runner.read_json(folder / 'selection.json')
        self.assertEqual(selection['selected']['vol_target'], .10)
        self.assertEqual(result['totals']['account_count'], 1)
        self.assertEqual(result['totals']['initial_capital'], PRINCIPAL)
        for name in ('summary.json', 'accounts.csv', 'equity.csv', 'ledgers/600000.json'):
            self.assertTrue((folder / name).is_file(), name)
        self.assertEqual((folder / 'source.sqlite').read_bytes(), before)
        json.dumps(result, allow_nan=False)

    def test_run_computes_uncached_trial_pnl_from_real_cash_share_ledgers(self):
        calendar = sessions('2021-12-20', 550)
        folder = self.fixture(calendar)
        effective_end = runner.read_json(folder / 'manifest.json')['effective_end']
        result = self.run_fixture(folder)
        train_i = int(np.searchsorted(calendar, runner.TRAIN_END, side='right')) - 1
        validation_i = int(np.searchsorted(calendar, runner.VALIDATION_END, side='right')) - 1
        self.assertGreater(train_i, 0)
        self.assertLess(validation_i, len(calendar) - 5)
        for trial in result['trials']:
            with self.subTest(vol=trial['vol_target']):
                _, curve, ledger = runner.replay_stock(
                    folder, '600000', 0, RiskConfig(annual_vol_target=trial['vol_target']),
                    end=effective_end, record_ledger=True)
                train_nav = ledger_nav(ledger, calendar[train_i], 10.)
                validation_nav = ledger_nav(ledger, calendar[validation_i], 10.)
                self.assertAlmostEqual(trial['train_pnl'], train_nav - PRINCIPAL)
                self.assertAlmostEqual(trial['validation_pnl'], validation_nav - train_nav)
                self.assertAlmostEqual(trial['validation_end_nav'], validation_nav)
                self.assertAlmostEqual(curve[train_i], train_nav)
                self.assertAlmostEqual(curve[validation_i], validation_nav)
                self.assertEqual(trial['scope_end'], effective_end)
                exits = [row for row in fills(ledger) if row['event'] == 'exit']
                self.assertEqual([row['day'] for row in exits], [calendar[-5]])
        winner = max(result['trials'], key=lambda trial: (trial['train_pnl'], -trial['vol_target']))
        self.assertEqual(result['selected_risk']['annual_vol_target'], winner['vol_target'])
        self.assertAlmostEqual(sum(row['pnl'] for row in result['annual_results']),
                               result['totals']['pnl'])

    def test_unverified_actions_or_rights_never_buy_and_keep_all_principal(self):
        calendar = sessions()
        event_i = len(calendar) - 6
        spec = dict(code='600000', raw=ohlcv([10.] * event_i + [4.] * 6),
                    events=[action(calendar[event_i], 2., 2.)])
        # Positive control: the same F1/eligible/gate-passing cached signals buy
        # before the event and earn a real holder's entitlement when verified.
        control, _, control_ledger = self.replay(self.fixture(calendar, [spec]))
        self.assertTrue(any(row['side'] == 'buy' for row in fills(control_ledger)))
        self.assertGreater(control['cash_dividends'], 0.)
        summaries = []
        for status, rights in (('cache_unverified', False), ('error', False),
                               ('ok', True), ('cache_unverified', True)):
            with self.subTest(status=status, rights=rights):
                folder = self.fixture(calendar, [dict(spec, history=dict(
                    status=status, rights_issue_present=rights))])
                summary, curve, ledger = self.replay(folder)
                self.assertEqual(summary['action_status'], status)
                self.assertEqual(summary['rights_issue_present'], rights)
                self.assertEqual(fills(ledger), [])
                self.assertTrue(all(row['shares'] == 0. for row in ledger))
                self.assertEqual(summary['initial_capital'], PRINCIPAL)
                self.assertEqual(summary['final_cash'], PRINCIPAL)
                self.assertEqual(summary['train_nav'], PRINCIPAL)
                self.assertEqual(summary['validation_nav'], PRINCIPAL)
                self.assertTrue(summary['liquidation_complete'])
                for key in ('shares', 'residual_value', 'pnl', 'fees', 'cash_dividends',
                            'average_exposure', 'build_count', 'add_count',
                            'reduce_count', 'exit_count'):
                    self.assertEqual(summary[key], 0., key)
                np.testing.assert_array_equal(curve, PRINCIPAL)
                summaries.append(summary)
        totals = runner.aggregate(summaries)
        self.assertEqual(totals['account_count'], 4)
        self.assertEqual(totals['initial_capital'], 4 * PRINCIPAL)
        self.assertEqual(totals['final_cash'], 4 * PRINCIPAL)
        self.assertEqual(totals['final_equity'], 4 * PRINCIPAL)
        self.assertEqual(totals['fully_liquidated_count'], 4)
        self.assertEqual(totals['rights_issue_accounts'], 2)
        self.assertEqual(totals['action_fetch_failures'], 3)
        for key in ('pnl', 'cash_minus_all_principal', 'residual_marked_value',
                    'fees', 'cash_dividends', 'build_count', 'add_count',
                    'reduce_count', 'exit_count'):
            self.assertEqual(totals[key], 0., key)

    def test_changed_action_snapshot_invalidates_prepared_features(self):
        calendar = sessions(count=270)
        folder = self.fixture(calendar, [dict(code='600000', prepared=False)])
        job = (str(folder), '600000', 'synthetic-600000', str(calendar[0]), str(calendar[-1]))
        original_source = (folder / 'source.sqlite').read_bytes()
        previous = runner.prepare_stock(job)
        self.assertIsNone(previous['feature_error'])
        with np.load(folder / 'features' / '600000.npz') as data:
            self.assertTrue(data['eligible'].any())
            np.testing.assert_array_equal(data['event_cash'], 0.)
            np.testing.assert_array_equal(data['event_mult'], 1.)
        history = runner.read_json(folder / 'actions' / '600000.json')
        for change in (dict(events=[action(calendar[260], .25, 1.05)]),
                       dict(status='cache_unverified'),
                       dict(status='ok', rights_issue_present=True)):
            with self.subTest(change=change):
                history.update(change)
                self.write_json(folder / 'actions' / '600000.json', history)
                current = runner.prepare_stock(job)
                self.assertIsNone(current['feature_error'])
                self.assertNotEqual(current['action_sha256'], previous['action_sha256'])
                self.assertNotEqual(current['input_sha256'], previous['input_sha256'])
                self.assertEqual(current['action_status'], history['status'])
                self.assertEqual(current['rights_issue_present'], history['rights_issue_present'])
                self.assertEqual(current['action_count'], 1)
                expected_cash, expected_mult = np.zeros(len(calendar)), np.ones(len(calendar))
                expected_cash[260], expected_mult[260] = .25, 1.05
                with np.load(folder / 'features' / '600000.npz') as data:
                    np.testing.assert_array_equal(data['raw'], ohlcv(np.full(len(calendar), 10.)))
                    np.testing.assert_array_equal(data['event_cash'], expected_cash)
                    np.testing.assert_array_equal(data['event_mult'], expected_mult)
                    if history['status'] != 'ok' or history['rights_issue_present']:
                        self.assertFalse(data['eligible'].any())
                    else:
                        self.assertTrue(data['eligible'].any())
                self.assertEqual(runner.read_json(folder / 'features' / '600000.json'), current)
                self.assertEqual(runner.prepare_stock(job), current)
                self.assertEqual((folder / 'source.sqlite').read_bytes(), original_source)
                previous = current


class TestRunIdentity(SnapshotCase):
    def resume(self, folder, manifest, **changes):
        request = dict(source=manifest.get('source_path', self.root / 'absent-original.sqlite'),
                       start=manifest['start'], end=manifest['requested_end'],
                       limit=manifest.get('universe_limit', 0), workers=1, fetch_actions=False)
        request.update(changes)
        with redirect_stdout(io.StringIO()):
            return runner.run(folder, **request)

    def completed_fixture(self, **changes):
        folder = self.fixture()
        manifest = runner.read_json(folder / 'manifest.json')
        manifest.update(source_path=str(self.root / 'original.sqlite'), universe_limit=0,
                        requested_end='2024-12-31', feature_start='2015-01-01',
                        action_coverage_start='2015-01-01',
                        source_sha256=hashlib.sha256((folder / 'source.sqlite').read_bytes()).hexdigest())
        manifest.update(changes)
        self.write_json(folder / 'manifest.json', manifest)
        self.resume(folder, manifest)
        return folder, manifest

    @staticmethod
    def artifact_bytes(folder):
        return {str(path.relative_to(folder)): path.read_bytes()
                for path in folder.rglob('*') if path.is_file()}

    def assert_resume_rejected(self, folder, manifest, message, **changes):
        before = self.artifact_bytes(folder)
        self.assertIn('summary.json', before)
        with patch.object(runner, 'connect_ro', side_effect=AssertionError('DB before validation')), \
                patch.object(runner, 'prepare_stock', side_effect=AssertionError('features before validation')), \
                patch('action_fallback.resolve_action_snapshot',
                      side_effect=AssertionError('actions before validation')), \
                self.assertRaisesRegex(ValueError, message):
            self.resume(folder, manifest, **changes)
        # Preserve the actual prior summary/completion marker and ALL artifacts,
        # including manifest, trials, ledgers and cached features, byte for byte.
        self.assertEqual(self.artifact_bytes(folder), before)

    def test_create_records_resolved_source_and_requested_limit_not_universe_size(self):
        fixture = self.fixture()
        source = fixture / 'unused' / '..' / 'source.sqlite'
        for limit in (1, 10):
            with self.subTest(limit=limit):
                folder = self.root / f'limited-{limit}'
                manifest = runner.create_run(folder, source, str(sessions()[0]), '2024-12-31', limit)
                self.assertEqual(manifest['source_path'], str(source.resolve()))
                self.assertEqual(manifest['universe_limit'], limit)
                self.assertEqual(len(manifest['universe']), 1)
                self.assertEqual(runner.read_json(folder / 'manifest.json')['universe_limit'], limit)

    def test_new_run_rejects_before_coverage_without_creating_or_opening_anything(self):
        folder = self.root / 'unsupported'
        for entry in (runner.create_run, runner.run):
            with self.subTest(entry=entry.__name__), \
                    patch.object(runner, 'connect_ro', side_effect=AssertionError('unexpected DB')), \
                    self.assertRaisesRegex(ValueError, '2015-01-01'):
                entry(folder, self.root / 'absent.sqlite', '2014-12-31', '2024-01-31')
            self.assertFalse(folder.exists())

    def test_main_uses_same_date_limit_validation_before_run(self):
        for extra, message in ((['--start', '2014-12-31'], '2015-01-01'),
                               (['--start', 'invalid'], 'isoformat'),
                               (['--start', '2024-02-01', '--end', '2024-02-01'], 'dates'),
                               (['--limit', '-1'], 'limit'), (['--workers', '0'], 'workers')):
            with self.subTest(extra=extra), patch.object(runner, 'run') as run, \
                    patch('sys.argv', ['pareto_backtest.py', '--output', str(self.root / 'cli')] + extra), \
                    redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit) as raised:
                runner.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn(message, errors.getvalue())
            run.assert_not_called()
        self.assertFalse((self.root / 'cli').exists())

    def test_main_accepts_coverage_boundary(self):
        with patch('sys.argv', ['pareto_backtest.py', '--output', str(self.root / 'cli'),
                                '--start', '2015-01-01', '--end', '2015-02-01']), \
                patch.object(runner, 'run') as run:
            runner.main()
        self.assertEqual(run.call_args.args[2:4], ('2015-01-01', '2015-02-01'))

    def test_coverage_boundary_keeps_short_history_ineligible_not_a_feature_error(self):
        calendar = sessions('2015-01-01', 18)
        prior = np.arange('2014-01-01', '2015-01-01', dtype='datetime64[D]')
        prior = prior[np.is_busday(prior)].astype('U10')
        folder = self.fixture(calendar, [dict(code='600000', prepared=False,
                                               dates=np.concatenate((prior, calendar)))])
        runner.create_run(folder, folder / 'source.sqlite', str(calendar[0]), str(calendar[-1]))
        result = self.run_fixture(folder)
        meta = runner.read_json(folder / 'features' / '600000.json')
        self.assertEqual(meta['bars'], len(calendar))
        self.assertEqual(meta['first_date'], '2015-01-01')
        self.assertIsNone(meta['feature_error'])
        with np.load(folder / 'features' / '600000.npz') as data:
            np.testing.assert_array_equal(data['dates'], calendar)
            self.assertFalse(data['eligible'].any())
        self.assertEqual(result['totals']['feature_errors'], 0)
        self.assertEqual(result['totals']['final_cash'], PRINCIPAL)
        self.assertEqual(result['totals']['build_count'], 0)

    def test_resume_rejects_changed_start_or_requested_end_even_with_same_calendar(self):
        folder, manifest = self.completed_fixture()
        for changes, field in ((dict(start='2014-12-31'), 'start'),
                               (dict(start='2024-01-03'), 'start'),
                               (dict(start='20240102'), 'start'),
                               (dict(end=manifest['effective_end']), 'requested_end'),
                               (dict(end='2024-12-30'), 'requested_end')):
            with self.subTest(changes=changes):
                self.assert_resume_rejected(folder, manifest, field, fetch_actions=True, **changes)

    def test_resume_rejects_changed_limit_even_when_resulting_universe_is_identical(self):
        for recorded, changed in ((0, 1), (1, 0), (10, 1)):
            with self.subTest(recorded=recorded, changed=changed):
                folder, manifest = self.completed_fixture(universe_limit=recorded)
                self.assert_resume_rejected(folder, manifest, 'universe_limit', limit=changed)

    def test_resume_rejects_different_source_path_even_with_identical_source_bytes(self):
        folder, manifest = self.completed_fixture()
        other = self.root / 'other.sqlite'
        shutil.copyfile(folder / 'source.sqlite', other)
        self.assert_resume_rejected(folder, manifest, 'source_path', source=other)

    def test_each_resume_hashes_frozen_bytes_not_size_or_mtime_and_preserves_summary(self):
        folder, manifest = self.completed_fixture()
        snapshot = folder / 'source.sqlite'
        before = snapshot.stat()
        with closing(sqlite3.connect(snapshot)) as db, db:
            db.execute('UPDATE klines SET close=close+0.01')
        os.utime(snapshot, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertEqual(snapshot.stat().st_size, before.st_size)
        self.assertEqual(snapshot.stat().st_mtime_ns, before.st_mtime_ns)
        self.assertNotEqual(hashlib.sha256(snapshot.read_bytes()).hexdigest(), manifest['source_sha256'])
        self.assert_resume_rejected(folder, manifest, 'source.sqlite SHA-256')

    def test_resume_with_missing_frozen_backup_preserves_summary(self):
        folder, manifest = self.completed_fixture()
        (folder / 'source.sqlite').unlink()
        self.assert_resume_rejected(folder, manifest, 'cannot verify frozen source.sqlite SHA-256')

    def test_present_but_empty_hash_is_not_treated_as_legacy(self):
        folder, manifest = self.completed_fixture()
        for value in (None, ''):
            with self.subTest(value=value):
                manifest['source_sha256'] = value
                self.write_json(folder / 'manifest.json', manifest)
                self.assert_resume_rejected(folder, manifest, 'source.sqlite SHA-256')

    def test_changed_original_contents_at_same_resolved_path_are_not_read_or_modified(self):
        folder, manifest = self.completed_fixture()
        source = Path(manifest['source_path'])
        shutil.copyfile(folder / 'source.sqlite', source)
        with closing(sqlite3.connect(source)) as db, db:
            db.execute('DELETE FROM klines')
        source_bytes = source.read_bytes()
        before = self.artifact_bytes(folder)
        real_open, real_connect = Path.open, runner.connect_ro

        def snapshot_open(path, *args, **kwargs):
            if path.resolve() == source:
                raise AssertionError('resume must not open/hash the current external DB')
            return real_open(path, *args, **kwargs)

        def snapshot_connect(path):
            self.assertEqual(Path(path).resolve(), folder / 'source.sqlite')
            return real_connect(path)

        with patch.object(Path, 'open', new=snapshot_open), \
                patch.object(runner, 'connect_ro', side_effect=snapshot_connect), \
                patch.object(runner.hashlib, 'file_digest', wraps=hashlib.file_digest) as digest:
            for _ in range(2):
                result = self.resume(folder, manifest, source=source.parent / 'unused' / '..' / source.name)
            self.assertEqual(digest.call_count, 2)
        self.assertEqual(result, runner.read_json(folder / 'summary.json'))
        self.assertEqual(self.artifact_bytes(folder), before)
        self.assertEqual(source.read_bytes(), source_bytes)

    def test_copied_run_does_not_require_external_source_to_exist(self):
        folder, manifest = self.completed_fixture()
        self.assertFalse(Path(manifest['source_path']).exists())
        copied = self.root / 'copied-run'
        shutil.copytree(folder, copied)
        before = self.artifact_bytes(folder)
        self.resume(copied, manifest)
        self.assertEqual(self.artifact_bytes(copied), before)
        self.assertEqual(self.artifact_bytes(folder), before)

    def test_legacy_missing_fields_default_limit_zero_without_backfill_or_hash(self):
        folder = self.fixture()
        original = self.run_fixture(folder)
        manifest = runner.read_json(folder / 'manifest.json')
        before = self.artifact_bytes(folder)
        self.assert_resume_rejected(folder, manifest, 'universe_limit', limit=1)
        with patch.object(runner.hashlib, 'file_digest', side_effect=AssertionError('legacy has no hash')):
            result = self.resume(folder, manifest, source=self.root / 'nonexistent.sqlite')
        self.assertEqual(result, original)
        self.assertEqual(self.artifact_bytes(folder), before)

    def test_legacy_with_hash_still_checks_backup_without_requiring_source_path(self):
        folder, manifest = self.completed_fixture()
        for key in ('source_path', 'universe_limit', 'feature_start', 'action_coverage_start'):
            manifest.pop(key)
        self.write_json(folder / 'manifest.json', manifest)
        self.resume(folder, manifest, source=self.root / 'nonexistent.sqlite')
        with closing(sqlite3.connect(folder / 'source.sqlite')) as db, db:
            db.execute('UPDATE klines SET volume=volume+1')
        self.assert_resume_rejected(folder, manifest, 'source.sqlite SHA-256')


class TestRunnerRegressions(SnapshotCase):
    def test_nan_bar_at_training_cutoff_must_carry_previous_valuation(self):
        """Skipped invalid rows carry the held account's prior NAV, not principal."""
        calendar = sessions('2021-12-20', 35)
        cutoff = int(np.searchsorted(calendar, runner.TRAIN_END, side='right')) - 1
        raw = ohlcv(np.full(len(calendar), 10.))
        raw[cutoff, 1] = np.nan
        folder = self.fixture(calendar, [dict(code='600000', raw=raw)])
        summary, curve, ledger = self.replay(folder)
        self.assertTrue(fills(ledger))
        self.assertLess(curve[cutoff - 1], PRINCIPAL)  # A real held account paid fees.
        self.assertEqual(summary['train_nav'], curve[cutoff - 1])
        self.assertEqual(curve[cutoff], curve[cutoff - 1])
        self.assertTrue(summary['feature_error'])

    def test_nan_cached_feature_input_must_be_flagged_even_if_metadata_says_ok(self):
        """Invalid cached quotes must override apparently clean feature metadata."""
        raw = ohlcv(np.full(len(sessions()), 10.))
        raw[:, 1] = np.nan
        folder = self.fixture(stocks=[dict(code='600000', raw=raw)])
        summary, curve, ledger = self.replay(folder)
        self.assertEqual(fills(ledger), [])
        self.assertEqual(summary['final_cash'], PRINCIPAL)
        np.testing.assert_array_equal(curve, PRINCIPAL)
        self.assertTrue(summary['feature_error'], 'invalid cached raw must not be reported clean')

    def test_holdout_starting_after_validation_uses_principal_not_negative_index(self):
        """A calendar entirely after validation starts holdout with all principal."""
        folder = self.fixture()
        effective_end = runner.read_json(folder / 'manifest.json')['effective_end']
        # Cached trials isolate holdout indexing from candidate computation.
        for vol in runner.CANDIDATE_VOL_TARGETS:
            self.write_json(folder / 'trials' / f'vol_{vol:.2f}.json', dict(
                vol_target=vol, train_pnl=0., validation_pnl=0.,
                validation_end_nav=PRINCIPAL, scope_end=effective_end))
        result = self.run_fixture(folder)
        self.assertLess(result['totals']['pnl'], 0.)  # Real fees/slippage, not a zero-trade case.
        self.assertEqual(result['holdout_start_equity'], PRINCIPAL)
        self.assertEqual(result['holdout_2024_onward_pnl'], result['totals']['pnl'])

    def test_selected_validation_phase_matches_actual_holdout_starting_equity(self):
        """Trials and the selected replay share a continuous validation boundary.

        This is phase/account continuity, NOT a claim that validation selects
        the winner. The separate training-only selection test covers that.
        """
        calendar = sessions('2023-12-11', 35)
        folder = self.fixture(calendar)
        result = self.run_fixture(folder)
        selected = runner.read_json(folder / 'selection.json')['selected']
        with (folder / 'equity.csv').open(encoding='utf-8') as stream:
            # NumPy reads the actual selected run, rather than constructing a
            # second Account which could hide a reset at the phase boundary.
            curve = np.genfromtxt(stream, delimiter=',', names=True,
                                  dtype=None, encoding='utf-8')
        cutoff = int(np.searchsorted(calendar, runner.VALIDATION_END, side='right')) - 1
        boundary_nav = float(curve['equity'][cutoff])
        self.assertAlmostEqual(result['holdout_2024_onward_pnl'],
                               result['totals']['final_equity'] - boundary_nav)
        self.assertAlmostEqual(selected['validation_end_nav'], boundary_nav)
        self.assertAlmostEqual(result['holdout_start_equity'], boundary_nav)
        self.assertEqual(selected['scope_end'], result['effective_end'])
        ledger = runner.read_json(folder / 'ledgers' / '600000.json')
        boundary_orders = [row for row in fills(ledger) if row['day'] <= calendar[cutoff]]
        self.assertTrue(boundary_orders)
        self.assertGreater(boundary_orders[-1]['shares'], 0.)
        self.assertFalse(any(row['event'] == 'exit' for row in boundary_orders))


if __name__ == '__main__':
    unittest.main()