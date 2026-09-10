"""Offline fixtures and adversarial tests: never read the production cache/run."""

from collections import Counter
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import socket
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import pareto_report as report


CALENDAR = ['2021-12-29', '2021-12-30', '2021-12-31', '2022-04-28',
            '2022-04-29', '2022-05-02', '2023-08-25', '2023-08-28',
            '2023-08-29', '2023-12-29', '2024-01-02', '2024-01-03',
            '2024-01-04', '2024-01-05', '2024-01-08']
CONFIG = dict(initial_capital=1_000_000., commission_rate=.0003,
              minimum_commission=5., slippage_bps=5., max_drawdown=.15, cooldown_sessions=20)


def trade(d, signal, kind, quantity, cash, shares, raw=10., target=.2, reason='pareto_risk_target'):
    """Fixture-only arithmetic, not a call to the production account."""
    buy = kind in ('build', 'add')
    price = raw * (1.0005 if buy else .9995)
    gross = quantity * price
    commission = max(5., gross * .0003)
    stamp = 0. if buy else gross * (.0005 if d >= '2023-08-28' else .001)
    transfer = gross * (.00001 if d >= '2022-04-29' else .00002)
    fee = commission + stamp + transfer
    return dict(day=d, signal_day=signal, event=kind, reason=reason, quantity=quantity,
                price=price, raw_price=raw, gross=gross, fee=fee,
                cash=cash - gross - fee if buy else cash + gross - fee,
                shares=shares + quantity if buy else shares - quantity,
                target_weight=target, side='buy' if buy else 'sell', commission=commission,
                stamp_tax=stamp, transfer_fee=transfer)


def action(d, cash, shares, cps=.5, multiplier=1.1):
    return dict(day=d, signal_day=None, event='action', reason='ex_date_cash_and_share_availability_approximation',
                quantity=shares * multiplier - shares, price=None, gross=shares * cps, fee=0.,
                cash=cash + shares * cps, shares=shares * multiplier, target_weight=None,
                eligible_shares=shares, cash_per_share=cps, share_multiplier=multiplier)


def unfilled(d, signal, cash, shares, block='below_lot', quantity=10.):
    return dict(day=d, signal_day=signal, event='unfilled', reason='pareto_risk_target',
                quantity=quantity, price=10., gross=0., fee=0., cash=cash,
                shares=shares, target_weight=.1, block_reason=block)


def account_for(ledger, code='000001', price=10.):
    final = ledger[-1] if ledger else dict(cash=1_000_000., shares=0.)
    cash, shares = final['cash'], final['shares']
    counts = Counter(e['event'] for e in ledger)
    result = dict(code=code, initial_capital=1_000_000., cash=cash, final_cash=cash, shares=shares,
                  marked_nav=cash + shares * price, pnl=cash + shares * price - 1_000_000.,
                  residual_value=shares * price, last_price=price, liquidation_complete=shares == 0,
                  fees=sum(e['fee'] for e in ledger),
                  cash_dividends=sum(e['gross'] for e in ledger if e['event'] == 'action'))
    result.update({kind + '_count': counts[kind] for kind in (*report.FILLS, 'unfilled')})
    return result


def sample_ledger():
    buy = trade(CALENDAR[1], CALENDAR[0], 'build', 100, 1_000_000., 0.)
    ent = action(CALENDAR[2], buy['cash'], buy['shares'])
    sell = trade(CALENDAR[3], CALENDAR[2], 'exit', ent['shares'], ent['cash'], ent['shares'], target=0.)
    zero = action(CALENDAR[4], sell['cash'], 0., cps=.2, multiplier=1.)
    buy2 = trade(CALENDAR[6], CALENDAR[5], 'build', 100, zero['cash'], 0.)
    small = unfilled(CALENDAR[7], CALENDAR[6], buy2['cash'], buy2['shares'])
    return [buy, ent, sell, zero, buy2, small]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def fixture(folder):
    """Complete tiny frozen run; real NPZs, two accounts, stale mark and residual."""
    folder.mkdir()
    codes = ['000001', '000002']
    risk = dict(annual_vol_target=.15, es_budget=.02, risk_window=63, es_confidence=.95,
                tracking_penalty=5., max_weight=1.)
    snapshot = folder / 'source.sqlite'
    snapshot.write_bytes(b'opaque frozen snapshot bytes: the exporter never connects to SQLite')
    manifest = dict(version='pareto_daily_atoms_risk_v1', start=CALENDAR[0], requested_end='2024-01-10',
                    effective_end=CALENDAR[-1], calendar=CALENDAR, universe=[dict(code=c, name=c) for c in codes],
                    source_sha256=report.digest(snapshot.read_bytes()), risk_defaults=risk,
                    account_defaults=CONFIG, train_end='2021-12-31', validation_end='2023-12-31',
                    candidate_vol_targets=[.1, .15, .2], limitations=['synthetic fixture'],
                    action_snapshot_dir=str(folder / 'unavailable_snapshot'))
    save(folder / 'manifest.json', manifest)
    ledgers = [sample_ledger(), []]
    accounts, curves, hashes = [], [], []
    eligible_counts = np.zeros(len(CALENDAR), dtype=int)
    fronts = np.full((len(CALENDAR), 2), 5, dtype=np.int8)
    for col, (code, ledger) in enumerate(zip(codes, ledgers)):
        # Second stock starts late and ends early; full principal still retained.
        dates = np.array(CALENDAR if col == 0 else CALENDAR[4:-2], dtype='U10')
        raw = np.tile([10., 11., 9., 10., 1000.], (len(dates), 1)).astype(np.float64)
        events = [dict(ex_date=e['day'], cash_per_share=e['cash_per_share'], share_multiplier=e['share_multiplier'])
                  for e in ledger if e['event'] == 'action']
        history = dict(code=code, status='ok', rights_issue_present=False, events=events,
                       metadata=dict(source='eastmoney_datacenter', evidence=[]))
        encoded = report.canonical(history)
        input_hash = report.digest(dates.tobytes() + raw.tobytes() + encoded)
        hashes.append(input_hash)
        meta = dict(code=code, name=code, feature_version=manifest['version'], bars=len(dates),
                    first_date=str(dates[0]), last_date=str(dates[-1]), late_start=col == 1, stale=col == 1,
                    feature_error=None, action_status='ok', rights_issue_present=False,
                    action_count=len(events), unexplained_gap_count=0,
                    input_sha256=input_hash, action_sha256=report.digest(encoded))
        save(folder / 'features' / f'{code}.json', meta)
        save(folder / 'resolved_actions' / f'{code}.json', history)
        save(folder / 'actions' / f'{code}.json', history)
        event_cash, event_mult = np.zeros(len(dates)), np.ones(len(dates))
        for event in events:
            i = int(np.searchsorted(dates, event['ex_date']))
            event_cash[i], event_mult[i] = event['cash_per_share'], event['share_multiplier']
        eligible = np.ones(len(dates), dtype=bool)
        np.savez_compressed(folder / 'features' / f'{code}.npz', dates=dates, raw=raw,
                            eligible=eligible, gates=eligible, event_cash=event_cash, event_mult=event_mult)
        idx = np.searchsorted(CALENDAR, dates)
        fronts[idx, col] = 1
        eligible_counts[idx] += 1
        account = account_for(ledger, code)
        account.update({k: v for k, v in meta.items() if k in (
            'name', 'first_date', 'last_date', 'late_start', 'stale', 'feature_error',
            'action_status', 'action_count', 'rights_issue_present', 'unexplained_gap_count')})
        account.update(end=CALENDAR[-1], max_drawdown_observed=.01,
                       fractional_share_approximation=True, average_exposure=.1, years={})
        curve = []
        for d in CALENDAR:
            earlier = [e for e in ledger if e['day'] <= d]
            state = earlier[-1] if earlier else dict(cash=1_000_000., shares=0.)
            nav = state['cash'] + state['shares'] * 10.
            curve.append(nav)
            if d in dates:
                account['years'][d[:4]] = dict(date=d, nav=nav, cash=state['cash'], shares=state['shares'])
        account['train_nav'], account['validation_nav'] = curve[2], curve[9]
        accounts.append(account)
        curves.append(curve)
        save(folder / 'accounts' / f'{code}.json', account)
        save(folder / 'ledgers' / f'{code}.json', ledger)
    np.save(folder / 'fronts.npy', fronts)
    save(folder / 'generation.json', dict(version=manifest['version'], hashes=hashes))
    save(folder / 'front_stats.json', [dict(date=d, eligible=int(eligible_counts[i]), first_front=int(eligible_counts[i]))
                                      for i, d in enumerate(CALENDAR)])
    report.write_csv(folder / 'accounts.csv', [{k: v for k, v in a.items() if k != 'years'} for a in accounts])
    curve = np.sum(curves, axis=0)
    report.write_csv(folder / 'equity.csv', [dict(date=d, equity=float(nav)) for d, nav in zip(CALENDAR, curve)])
    totals = dict(account_count=2, initial_capital=2_000_000., final_cash=sum(a['cash'] for a in accounts),
                  residual_marked_value=sum(a['residual_value'] for a in accounts), final_equity=float(curve[-1]),
                  pnl=float(curve[-1]) - 2_000_000., cash_minus_all_principal=sum(a['cash'] for a in accounts)-2_000_000.,
                  fully_liquidated_count=1, profitable_accounts=sum(a['pnl'] > 0 for a in accounts),
                  losing_accounts=sum(a['pnl'] < 0 for a in accounts), fees=sum(a['fees'] for a in accounts),
                  cash_dividends=50., action_fetch_failures=0, rights_issue_accounts=0, feature_errors=0,
                  unexplained_gap_accounts=0, stale_accounts=1, late_start_accounts=1)
    totals.update({kind + '_count': sum(a[kind + '_count'] for a in accounts) for kind in report.FILLS})
    trials = []
    for i, vol in enumerate(manifest['candidate_vol_targets']):
        trial = dict(vol_target=vol, train_pnl=float(curve[2])-2_000_000.-i,
                     validation_pnl=float(curve[9]-curve[2]), validation_end_nav=float(curve[9])-i,
                     scope_end=CALENDAR[-1])
        trials.append(trial)
        save(folder / 'trials' / f'vol_{vol:.2f}.json', trial)
    selection = dict(trials=trials, selected=trials[0], criterion='maximum training currency P/L; ties lower risk')
    save(folder / 'selection.json', selection)
    annual, previous = [], 2_000_000.
    for year in sorted({d[:4] for d in CALENDAR}):
        index = max(i for i, d in enumerate(CALENDAR) if d.startswith(year))
        nav = float(curve[index])
        annual.append(dict(year=year, date=CALENDAR[index], final_equity=nav, pnl=nav-previous,
                           return_pct=(nav/previous-1)*100))
        previous = nav
    summary = {k: manifest[k] for k in ('version', 'start', 'requested_end', 'effective_end', 'limitations')}
    summary.update(selected_risk=dict(risk, annual_vol_target=.1), totals=totals, annual_results=annual,
                   trials=trials, holdout_start_equity=float(curve[9]), holdout_2024_onward_pnl=float(curve[-1]-curve[9]),
                   verdict='unverified_research_not_principal_protected')
    save(folder / 'summary.json', summary)
    return manifest


class OfflineTest(unittest.TestCase):
    def setUp(self):
        self.network = patch.object(socket, 'socket', side_effect=AssertionError('network forbidden'))
        self.database = patch.object(sqlite3, 'connect', side_effect=AssertionError('SQLite forbidden'))
        self.network.start()
        self.database.start()
        self.addCleanup(self.network.stop)
        self.addCleanup(self.database.stop)


class LedgerTests(OfflineTest):
    def verify(self, ledger, account=None, **kwargs):
        return report.replay_ledger(account or account_for(ledger), ledger, CALENDAR, CONFIG, **kwargs)

    def test_independent_replay_dividends_fees_and_below_lot(self):
        ledger = sample_ledger()
        with patch.dict('sys.modules', {'capital_account': None, 'pareto_backtest': None}):
            result = self.verify(ledger)
        self.assertEqual(result['counts']['action'], 2)
        self.assertEqual(result['blocks']['below_lot'], 1)
        self.assertEqual(result['cash_dividends'], 50.)
        self.assertEqual(result['shares'], 100.)
        self.assertAlmostEqual(result['cash'], ledger[-1]['cash'])

    def test_zero_holdings_action_requires_no_price(self):
        event = action(CALENDAR[0], 1_000_000., 0.)
        result = self.verify([event], actions={CALENDAR[0]: (.5, 1.1)})
        self.assertEqual(result['cash'], 1_000_000.)
        event['price'] = 10.
        with self.assertRaisesRegex(report.VerificationError, 'action has order fields'):
            self.verify([event])

    def test_t_plus_one_can_sell_old_shares_after_new_buy(self):
        a = trade(CALENDAR[1], CALENDAR[0], 'build', 100, 1_000_000., 0.)
        b = trade(CALENDAR[2], CALENDAR[1], 'add', 100, a['cash'], a['shares'])
        c = trade(CALENDAR[2], CALENDAR[1], 'reduce', 100, b['cash'], b['shares'], target=0.)
        result = self.verify([a, b, c])
        self.assertEqual(result['shares'], 100.)
        bad = trade(CALENDAR[2], CALENDAR[1], 'exit', 200, b['cash'], b['shares'], target=0.)
        with self.assertRaisesRegex(report.VerificationError, r'T\+1'):
            self.verify([a, b, bad])

    def test_same_day_all_new_shares_cannot_sell(self):
        a = trade(CALENDAR[1], CALENDAR[0], 'build', 100, 1_000_000., 0.)
        b = trade(CALENDAR[1], CALENDAR[0], 'exit', 100, a['cash'], a['shares'], target=0.)
        with self.assertRaisesRegex(report.VerificationError, r'T\+1'):
            self.verify([a, b])

    def test_action_before_orders_and_duplicate(self):
        a = trade(CALENDAR[1], CALENDAR[0], 'build', 100, 1_000_000., 0.)
        b = action(CALENDAR[1], a['cash'], a['shares'])
        with self.assertRaisesRegex(report.VerificationError, 'action duplicate/after order'):
            self.verify([a, b])
        zero = action(CALENDAR[0], 1_000_000., 0.)
        with self.assertRaisesRegex(report.VerificationError, 'action duplicate/after order'):
            self.verify([zero, zero])

    def test_nonzero_target_signal_next_market_day(self):
        a = trade(CALENDAR[2], CALENDAR[0], 'build', 100, 1_000_000., 0.)
        with self.assertRaisesRegex(report.VerificationError, 'not next market session'):
            self.verify([a])
        a['signal_day'] = a['day']
        with self.assertRaisesRegex(report.VerificationError, 'signal not earlier'):
            self.verify([a])

    def test_zero_exit_may_wait_after_suspension(self):
        a = trade(CALENDAR[1], CALENDAR[0], 'build', 100, 1_000_000., 0.)
        b = trade(CALENDAR[4], CALENDAR[1], 'exit', 100, a['cash'], a['shares'], target=0.)
        self.assertEqual(self.verify([a, b])['shares'], 0.)

    def test_final_five_sessions_no_buys(self):
        a = trade(CALENDAR[-5], CALENDAR[-6], 'build', 100, 1_000_000., 0.)
        with self.assertRaisesRegex(report.VerificationError, 'final five'):
            self.verify([a])

    def test_scheduled_exit_and_terminal_close_dates(self):
        a = trade(CALENDAR[1], CALENDAR[0], 'build', 100, 1_000_000., 0.)
        b = trade(CALENDAR[-1], None, 'exit', 100, a['cash'], a['shares'],
                  target=0., reason='deadline_close')
        self.verify([a, b])
        b['day'] = CALENDAR[-2]
        with self.assertRaisesRegex(report.VerificationError, 'premature final close'):
            self.verify([a, b])

    def test_fee_date_boundaries_and_minimum_commission(self):
        for sell_day, signal in [(CALENDAR[3], CALENDAR[2]), (CALENDAR[4], CALENDAR[3]),
                                 (CALENDAR[6], CALENDAR[5]), (CALENDAR[7], CALENDAR[6])]:
            with self.subTest(day=sell_day):
                a = trade(CALENDAR[1], CALENDAR[0], 'build', 100, 1_000_000., 0.)
                b = trade(sell_day, signal, 'exit', 100, a['cash'], a['shares'], target=0.)
                self.verify([a, b])
                b['stamp_tax'] += .03
                with self.assertRaisesRegex(report.VerificationError, 'stamp_tax'):
                    self.verify([a, b])

    def test_proportional_commission_large_order(self):
        a = trade(CALENDAR[1], CALENDAR[0], 'build', 80000, 1_000_000., 0.)
        self.assertGreater(a['commission'], 5.)
        self.verify([a])

    def test_fractional_entitlement_and_exit(self):
        a = trade(CALENDAR[1], CALENDAR[0], 'build', 100, 1_000_000., 0.)
        b = action(CALENDAR[2], a['cash'], a['shares'], multiplier=1.00123)
        c = trade(CALENDAR[3], CALENDAR[2], 'exit', b['shares'], b['cash'], b['shares'], target=0.)
        self.assertEqual(self.verify([a, b, c])['shares'], 0.)

    def test_currency_and_share_absolute_tolerances(self):
        ledger = sample_ledger()
        ledger[0]['cash'] += .019
        ledger[0]['shares'] += 5e-8
        self.verify(ledger)
        ledger[0]['cash'] += .002
        with self.assertRaisesRegex(report.VerificationError, 'cash'):
            self.verify(ledger)
        ledger = sample_ledger()
        ledger[0]['shares'] += 2e-7
        with self.assertRaisesRegex(report.VerificationError, 'shares'):
            self.verify(ledger)

    def test_missing_and_nonfinite_and_unknown_event_fail(self):
        for key, value in [('gross', None), ('fee', float('nan')), ('event', 'ignored')]:
            ledger = sample_ledger()
            ledger[0][key] = value
            with self.subTest(key=key), self.assertRaises(report.VerificationError):
                self.verify(ledger)
        ledger = sample_ledger()
        del ledger[0]['cash']
        with self.assertRaisesRegex(report.VerificationError, 'missing fields'):
            self.verify(ledger)

    def test_missing_even_zero_holding_action_fails(self):
        a = action(CALENDAR[0], 1_000_000., 0.)
        with self.assertRaisesRegex(report.VerificationError, 'missing/extra action'):
            self.verify([], actions={a['day']: (.5, 1.1)})

    def test_do_not_double_count_dividend_fee_or_slippage(self):
        ledger = sample_ledger()
        for key in ('cash_dividends', 'fees', 'marked_nav', 'residual_value', 'pnl'):
            a = account_for(ledger)
            a[key] += 1.
            with self.subTest(key=key), self.assertRaises(report.VerificationError):
                self.verify(ledger, a)
        ledger[0]['price'] = ledger[0]['raw_price']
        with self.assertRaisesRegex(report.VerificationError, 'slippage price'):
            self.verify(ledger)

    def test_unfilled_never_changes_balances_or_fees(self):
        ledger = sample_ledger()
        for key in ('cash', 'shares', 'fee', 'gross'):
            corrupt = deepcopy(ledger)
            corrupt[-1][key] += 1.
            with self.subTest(key=key), self.assertRaises(report.VerificationError):
                self.verify(corrupt)

    def test_frozen_raw_price_and_lot_size(self):
        a = trade(CALENDAR[1], CALENDAR[0], 'build', 100, 1_000_000., 0.)
        with self.assertRaisesRegex(report.VerificationError, 'raw bar price'):
            self.verify([a], bars={a['day']: (11., 10.)})
        odd = trade(CALENDAR[1], CALENDAR[0], 'build', 101, 1_000_000., 0.)
        with self.assertRaisesRegex(report.VerificationError, 'lot size'):
            self.verify([odd])


class ExportTests(OfflineTest):
    def setUp(self):
        super().setUp()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.run = self.base / 'run'
        self.output = self.base / 'report'
        fixture(self.run)

    def load(self, relative):
        return report.decode((self.run / relative).read_bytes())

    def export(self):
        with patch('sys.stdout', new_callable=io.StringIO):
            return report.export_report(self.run, self.output)

    def assert_failed(self, message):
        with self.assertRaisesRegex(report.VerificationError, message):
            self.export()
        self.assertFalse(self.output.exists())

    def test_complete_export_reproducible_and_no_large_copies(self):
        before = {p: report.digest(p.read_bytes()) for p in self.run.rglob('*') if p.is_file()}
        result = self.export()
        self.assertEqual(result['verified_accounts'], 2)
        self.assertEqual(result['verified_ledgers'], 2)
        self.assertEqual(result['ledger_events'], 6)
        self.assertEqual(result['no_trade_accounts'], 1)
        self.assertFalse(result['performance_certified'])
        self.assertFalse(result['unit_tests']['executed_by_export'])
        self.assertEqual(before, {p: report.digest(p.read_bytes()) for p in before})
        names = {p.name for p in self.output.iterdir()}
        self.assertEqual(names, {'README.md', 'verification.json', 'audit.csv', 'accounts.csv',
                                 'equity.csv', 'manifest.json', 'summary.json', 'front_stats.json', 'selection.json'})
        text = (self.output / 'README.md').read_text(encoding='utf-8')
        for expected in ('未保本', '非纯价差', '不是独立盲测', '2/2', '50.00'):
            self.assertIn(expected, text)
        for p in self.output.iterdir():
            self.assertNotIn(str(self.base), p.read_text(encoding='utf-8-sig'))
        first_output = self.output
        self.output = self.base / 'repeat'
        self.export()
        for p in first_output.iterdir():
            self.assertEqual(p.read_bytes(), (self.output / p.name).read_bytes())
        with (self.output / 'audit.csv').open(encoding='utf-8-sig') as stream:
            audits = list(csv.DictReader(stream))
        self.assertEqual(audits[0]['ledger_sha256'], before[self.run / 'ledgers/000001.json'])
        self.assertEqual(audits[0]['input_sha256'], self.load('features/000001.json')['input_sha256'])
        self.assertEqual(audits[1]['ledger_events'], '0')

    def test_missing_ledger_never_exports_success(self):
        (self.run / 'ledgers/000002.json').unlink()
        self.assert_failed('missing/extra population')

    def test_extra_account_never_silently_excluded(self):
        save(self.run / 'accounts/000003.json', {})
        self.assert_failed('missing/extra population')

    def test_corruption_reports_exact_successful_count(self):
        ledger = self.load('ledgers/000001.json')
        ledger[0]['cash'] += 1.
        save(self.run / 'ledgers/000001.json', ledger)
        self.assert_failed('000001: verification failed after 0/2 successful ledgers')

    def test_snapshot_hash_and_wal_guard(self):
        (self.run / 'source.sqlite-wal').write_bytes(b'not empty')
        self.assert_failed('nonempty SQLite WAL')
        (self.run / 'source.sqlite-wal').unlink()
        (self.run / 'source.sqlite').write_bytes(b'changed')
        self.assert_failed('source snapshot hash mismatch')

    def test_input_and_generation_hashes_checked(self):
        generation = self.load('generation.json')
        generation['hashes'][0] = '0' * 64
        save(self.run / 'generation.json', generation)
        self.assert_failed('original input/generation hash mismatch')

    def test_missing_completion_marker(self):
        (self.run / 'summary.json').unlink()
        self.assert_failed('missing input')

    def test_daily_curve_middle_corruption_detected(self):
        path = self.run / 'equity.csv'
        with path.open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        rows[5]['equity'] = float(rows[5]['equity']) + .03
        report.write_csv(path, rows)
        self.assert_failed('whole daily equity curve mismatch')

    def test_annual_selected_cutoff_and_total_corruptions(self):
        original = self.load('summary.json')
        for section, key in [('totals', 'fees'), ('totals', 'initial_capital'),
                             ('annual_results', 'pnl'), ('annual_results', 'date'),
                             ('root', 'holdout_start_equity')]:
            corrupt = deepcopy(original)
            row = corrupt if section == 'root' else corrupt[section]
            row = row[0] if isinstance(row, list) else row
            row[key] = '2021-12-30' if key == 'date' else row[key] + 1.
            save(self.run / 'summary.json', corrupt)
            with self.subTest(section=section, key=key):
                self.assert_failed('totals|annual|holdout')
        save(self.run / 'summary.json', original)

    def test_wrong_selection_and_trial_file(self):
        selection = self.load('selection.json')
        selection['selected'] = selection['trials'][1]
        save(self.run / 'selection.json', selection)
        self.assert_failed('not training-only maximum')
        selection['selected'] = selection['trials'][0]
        save(self.run / 'selection.json', selection)
        trial = self.load('trials/vol_0.10.json')
        trial['validation_pnl'] += 1.
        save(self.run / 'trials/vol_0.10.json', trial)
        self.assert_failed('trial files/selection/summary')

    def test_account_year_and_accounts_csv_reconcile(self):
        account = self.load('accounts/000001.json')
        account['years']['2021']['cash'] += .03
        save(self.run / 'accounts/000001.json', account)
        self.assert_failed('annual 2021 cash')
        account['years']['2021']['cash'] -= .03
        save(self.run / 'accounts/000001.json', account)
        path = self.run / 'accounts.csv'
        with path.open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        rows[0]['cash'] = float(rows[0]['cash']) + 1.
        report.write_csv(path, rows)
        self.assert_failed('accounts.csv cash')

    def test_front_counts_ratio_not_mean_of_daily_ratios(self):
        fronts = np.load(self.run / 'fronts.npy')
        fronts[0, 0] = 2
        np.save(self.run / 'fronts.npy', fronts)
        stats = self.load('front_stats.json')
        stats[0]['first_front'] = 0
        save(self.run / 'front_stats.json', stats)
        result = self.export()
        f = result['front_population']
        self.assertEqual(f['ratio'], sum(r['first_front'] for r in stats)/sum(r['eligible'] for r in stats))
        self.assertNotAlmostEqual(f['ratio'], sum(r['first_front']/r['eligible'] for r in stats)/len(stats))

    def test_available_market_evidence_and_original_hash(self):
        folder = self.base / 'snapshot'
        market = dict(datasets=dict(dividends=dict(status='ok', count=1, expected_pages=1,
                                                   pages=[dict(response_sha256='a'*64, page=1)])))
        save(folder / 'market_manifest.json', market)
        save(folder / 'request.json', dict(start='2021-01-01'))
        manifest = self.load('manifest.json')
        manifest['action_snapshot_dir'] = str(folder)
        save(self.run / 'manifest.json', manifest)
        result = self.export()
        self.assertTrue(result['action_snapshot']['available'])
        self.assertEqual(result['action_snapshot']['market_manifest_sha256'],
                         report.digest((folder / 'market_manifest.json').read_bytes()))
        self.assertTrue((self.output / 'evidence/market_manifest.json').is_file())

    def test_evidence_reference_hash_mismatch_raises(self):
        folder = self.base / 'snapshot'
        save(folder / 'market_manifest.json', dict(datasets={}))
        save(folder / 'manifests/a.json', {})
        with self.assertRaisesRegex(report.VerificationError, 'evidence hash mismatch'):
            report.snapshot_evidence(folder, report.Inputs(),
                                     [dict(snapshot_manifest='manifests/a.json', snapshot_manifest_sha256='0'*64)])

    def test_input_change_detection_and_duplicate_json(self):
        inputs = report.Inputs()
        inputs.read(self.run / 'summary.json')
        save(self.run / 'summary.json', {})
        with self.assertRaisesRegex(report.VerificationError, 'input changed'):
            inputs.unchanged()
        with self.assertRaisesRegex(report.VerificationError, 'duplicate JSON'):
            report.decode('{"cash":1,"cash":2}')
        with self.assertRaisesRegex(report.VerificationError, 'nonfinite'):
            report.decode('{"cash":NaN}')

    def test_no_overwrite_or_input_overlap(self):
        with self.assertRaisesRegex(report.VerificationError, 'overlap'):
            report.export_report(self.run, self.run / 'report')
        self.export()
        with self.assertRaisesRegex(report.VerificationError, 'do not overwrite'):
            self.export()

    def test_cli_failure_returns_nonzero_and_no_false_marker(self):
        (self.run / 'ledgers/000002.json').unlink()
        with patch('sys.stderr', new_callable=io.StringIO) as stream:
            code = report.main(['--run', str(self.run), '--output', str(self.output)])
        self.assertEqual(code, 1)
        self.assertIn('VERIFICATION FAILED', stream.getvalue())
        self.assertFalse((self.output / 'verification.json').exists())

    def test_redaction_posix_windows_unc_and_embedded_paths(self):
        values = ['/home/me/data/run', 'C:\\Users\\me\\run', '\\\\host\\share\\run',
                  'error opening /home/me/private/run', 'https://example.com/api/path']
        result = report.redact_paths(values)
        self.assertTrue(all('<LOCAL_PATH>' in x for x in result[:4]))
        self.assertEqual(result[-1], values[-1])


if __name__ == '__main__':
    unittest.main()