#!/usr/bin/env python3
"""Offline, independent ledger audit and compact report of a completed Pareto run.

No Account/runner imports, database connections, HTTP, strategy reruns or writes
to the input run. A successful audit certifies arithmetic consistency with the
frozen inputs, NOT actual fills, data completeness or investment performance.
"""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
from datetime import date
import hashlib
import io
import json
import math
from pathlib import Path, PureWindowsPath
import re
import sys

import numpy as np


MONEY_TOL = .02
SHARE_TOL = 1e-7
NUMBER_TOL = 1e-9
INITIAL = 1_000_000.
FILLS = ('build', 'add', 'reduce', 'exit')
VERSION = 'pareto_report_v1'
ROOT = Path(__file__).resolve().parent


class VerificationError(ValueError):
    """Missing evidence or a failed invariant; never convert this to success."""


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def number(value, label):
    require(not isinstance(value, bool) and isinstance(value, (int, float, np.number)),
            f'{label}: numeric value required')
    value = float(value)
    require(math.isfinite(value), f'{label}: nonfinite value')
    return value


def close(actual, expected, label, tolerance=MONEY_TOL):
    actual, expected = number(actual, label), number(expected, label)
    require(abs(actual - expected) <= tolerance,
            f'{label}: actual={actual!r}, expected={expected!r}, tolerance={tolerance}')
    return abs(actual - expected)


def day(value):
    require(isinstance(value, str), f'invalid date: {value!r}')
    try:
        require(date.fromisoformat(value).isoformat() == value, f'non-ISO date: {value}')
    except ValueError as exc:
        raise VerificationError(f'invalid date: {value!r}') from exc
    return value


def fields(value, names, label):
    require(isinstance(value, dict), f'{label}: object required')
    missing = set(names.split()) - value.keys()
    require(not missing, f'{label}: missing fields {sorted(missing)}')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    # Exactly the original runner's input/action hash encoding, including ASCII escapes.
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'duplicate JSON key: {key}')
        result[key] = value
    return result


def decode(data):
    def invalid(value):
        raise VerificationError(f'nonfinite JSON constant: {value}')
    return json.loads(data, object_pairs_hook=_pairs, parse_constant=invalid)


class Inputs:
    """Hash actual bytes, and reject files changed during this export."""

    def __init__(self):
        self.fingerprints = {}
        self.hashes = {}

    def stamp(self, path):
        path = Path(path)
        require(path.is_file(), f'missing input: {path}')
        stat = path.stat()
        stamp = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
        if path in self.fingerprints:
            require(self.fingerprints[path] == stamp, f'input changed: {path}')
        self.fingerprints[path] = stamp
        return path

    def read(self, path):
        path = self.stamp(path)
        data = path.read_bytes()
        self.stamp(path)
        hashed = digest(data)
        if path in self.hashes:
            require(self.hashes[path] == hashed, f'input bytes changed: {path}')
        self.hashes[path] = hashed
        return data

    def json(self, path):
        return decode(self.read(path))

    def large_hash(self, path):
        path = self.stamp(path)
        with path.open('rb') as stream:
            hashed = hashlib.file_digest(stream, 'sha256').hexdigest()
        self.stamp(path)
        self.hashes[path] = hashed
        return hashed

    def unchanged(self):
        for path in list(self.fingerprints):
            self.stamp(path)


def redact_paths(value):
    """Remove local absolute paths, including paths embedded in diagnostic text."""
    if isinstance(value, dict):
        return {k: redact_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_paths(v) for v in value]
    if isinstance(value, str):
        if value.startswith(('https://', 'http://')):
            return value
        if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
            return '<LOCAL_PATH>/' + value.replace('\\', '/').rstrip('/').split('/')[-1]
        return re.sub(r"(?:[A-Za-z]:[\\/]|\\\\|/(?:home|Users|tmp|mnt|var)/)[^\s\"'<>]+",
                      '<LOCAL_PATH>', value)
    return value


def replay_ledger(account, ledger, calendar, config, bars=None, actions=None):
    """Recompute cash/shares, never instantiate or call the production Account.

    bars maps date to (open, close). actions maps date to (cash/share, multiplier).
    Empty-holding actions are required too. Unfilled/below_lot records have no
    balance effect but their timestamps, zero fees/gross and balances are checked.
    """
    fields(account, 'code initial_capital cash final_cash shares marked_nav pnl fees '
           'cash_dividends residual_value last_price liquidation_complete '
           'build_count add_count reduce_count exit_count unfilled_count', 'account')
    fields(config, 'initial_capital commission_rate minimum_commission slippage_bps', 'config')
    close(config['initial_capital'], INITIAL, 'config initial', 0)
    close(account['initial_capital'], INITIAL, 'account initial', 0)
    require(isinstance(ledger, list), 'ledger: array required')
    indices = {d: i for i, d in enumerate(calendar)}
    require(len(indices) == len(calendar) and calendar == sorted(calendar) and calendar,
            'calendar: empty/duplicate/unsorted')
    for d in calendar:
        day(d)
    commission_rate = number(config['commission_rate'], 'commission_rate')
    minimum = number(config['minimum_commission'], 'minimum_commission')
    slip = number(config['slippage_bps'], 'slippage_bps') / 10000
    require(commission_rate >= 0 and minimum >= 0 and 0 <= slip < 1, 'invalid costs')
    cash, shares, new_today, fees, dividends = INITIAL, 0., 0., 0., 0.
    current = None
    order_seen = action_seen = False
    counts, blocks = Counter(), Counter()
    states = {}
    annual_fees, annual_dividends = defaultdict(float), defaultdict(float)
    max_cash_error = max_share_error = 0.
    for n, event in enumerate(ledger):
        label = f"{account['code']} ledger[{n}]"
        fields(event, 'day signal_day event reason quantity price gross fee cash shares target_weight', label)
        d, kind = day(event['day']), event['event']
        require(d in indices and (current is None or d >= current), f'{label}: event chronology/calendar')
        if d != current:
            new_today, order_seen, action_seen = 0., False, False
            current = d
        require(kind in (*FILLS, 'action', 'unfilled'), f'{label}: unknown event {kind!r}')
        require(bars is None or d in bars, f'{label}: no valid frozen execution bar')
        quantity = number(event['quantity'], label + ' quantity')
        counts[kind] += 1
        if kind == 'action':
            fields(event, 'cash_per_share share_multiplier eligible_shares', label)
            require(not order_seen and not action_seen, f'{label}: action duplicate/after order')
            require(event['signal_day'] is None and event['price'] is None
                    and event['target_weight'] is None, f'{label}: action has order fields')
            action_seen = True
            cps = number(event['cash_per_share'], label + ' cash_per_share')
            multiplier = number(event['share_multiplier'], label + ' multiplier')
            require(cps >= 0 and multiplier > 0, f'{label}: invalid action entitlement')
            if actions is not None:
                require(d in actions, f'{label}: unexpected action')
                close(cps, actions[d][0], label + ' frozen dividend/share', NUMBER_TOL)
                close(multiplier, actions[d][1], label + ' frozen multiplier', NUMBER_TOL)
            close(event['eligible_shares'], shares, label + ' pre-action shares', SHARE_TOL)
            credit = shares * cps
            close(quantity, shares * (multiplier - 1), label + ' action delta', SHARE_TOL)
            close(event['gross'], credit, label + ' action gross')
            close(event['fee'], 0., label + ' action fee', NUMBER_TOL)
            cash += credit
            shares *= multiplier
            dividends += credit
            annual_dividends[d[:4]] += credit
        else:
            order_seen = True
            require(quantity >= 0, f'{label}: negative order quantity')
            target = number(event['target_weight'], label + ' target')
            require(0 <= target <= 1, f'{label}: target outside [0,1]')
            reason, signal = event['reason'], event['signal_day']
            if reason in ('scheduled_deadline', 'deadline_close'):
                require(signal is None and target == 0, f'{label}: deadline order contract')
                require(indices[d] >= max(0, len(calendar) - 5), f'{label}: premature deadline')
                require(reason != 'deadline_close' or d == calendar[-1], f'{label}: premature final close')
            else:
                require(reason in ('pareto_risk_target', 'risk_exit'), f'{label}: unknown order reason')
                require(signal in indices and indices[signal] < indices[d], f'{label}: signal not earlier')
                require(target == 0 or indices[d] == indices[signal] + 1,
                        f'{label}: nonzero target not next market session')
                require(reason != 'risk_exit' or target == 0, f'{label}: risk exit nonzero target')
            if kind == 'unfilled':
                fields(event, 'block_reason', label)
                block = event['block_reason']
                require(block in ('block_buy', 'block_sell', 'below_lot', 'not_tradable',
                                  'insufficient_cash', 'insufficient_cash_for_fees', 't_plus_one'),
                        f'{label}: unknown unfilled reason')
                blocks[block] += 1
                close(event['gross'], 0., label + ' unfilled gross', NUMBER_TOL)
                close(event['fee'], 0., label + ' unfilled fee', NUMBER_TOL)
                if block == 'below_lot':
                    require(0 < quantity < 100, f'{label}: below_lot quantity')
                if event['price'] is None:
                    require(block == 'not_tradable', f'{label}: missing order price')
                else:
                    require(number(event['price'], label + ' price') > 0, f'{label}: nonpositive price')
                    if bars is not None:
                        raw = bars[d][reason == 'deadline_close']
                        close(event['price'], raw, label + ' unfilled raw price', NUMBER_TOL)
            else:
                fields(event, 'side raw_price commission stamp_tax transfer_fee', label)
                buy = kind in ('build', 'add')
                require(event['side'] == ('buy' if buy else 'sell'), f'{label}: side/event mismatch')
                require(quantity > 0, f'{label}: zero fill')
                require(not buy or indices[d] < len(calendar) - 5, f'{label}: buy in final five sessions')
                raw = number(event['raw_price'], label + ' raw_price')
                require(raw > 0, f'{label}: nonpositive raw price')
                if bars is not None:
                    close(raw, bars[d][reason == 'deadline_close'], label + ' raw bar price', NUMBER_TOL)
                execution = raw * (1 + slip if buy else 1 - slip)
                close(event['price'], execution, label + ' slippage price', NUMBER_TOL)
                gross = quantity * execution
                commission = max(minimum, gross * commission_rate)
                stamp = 0. if buy else gross * (.0005 if d >= '2023-08-28' else .001)
                transfer = gross * (.00001 if d >= '2022-04-29' else .00002)
                fee = commission + stamp + transfer
                for key, expected in [('gross', gross), ('commission', commission),
                                      ('stamp_tax', stamp), ('transfer_fee', transfer), ('fee', fee)]:
                    close(event[key], expected, label + ' ' + key)
                if buy or target != 0:
                    close(quantity, round(quantity / 100) * 100, label + ' lot size', SHARE_TOL)
                before = shares
                if buy:
                    cash -= gross + fee
                    shares += quantity
                    new_today += quantity
                    expected_kind = 'build' if before == 0 else 'add'
                else:
                    available = max(0., before - new_today)
                    require(quantity <= available + SHARE_TOL, f'{label}: T+1/oversell')
                    cash += gross - fee
                    shares -= quantity
                    if abs(quantity - available) <= SHARE_TOL:
                        shares = new_today
                    expected_kind = 'exit' if shares == 0 else 'reduce'
                require(kind == expected_kind, f'{label}: wrong build/add/reduce/exit classification')
                fees += fee
                annual_fees[d[:4]] += fee
        require(cash >= -MONEY_TOL and shares >= -SHARE_TOL, f'{label}: negative balance')
        max_cash_error = max(max_cash_error, close(event['cash'], cash, label + ' cash'))
        max_share_error = max(max_share_error, close(event['shares'], shares, label + ' shares', SHARE_TOL))
        states[d] = (cash, shares)
    if actions is not None:
        actual_days = {e['day'] for e in ledger if e['event'] == 'action'}
        require(actual_days == set(actions), f"{account['code']}: missing/extra action records")
    residual = shares * number(account['last_price'], 'last_price')
    require(account['last_price'] > 0, 'nonpositive final mark')
    for key, expected in [('cash', cash), ('final_cash', cash), ('fees', fees),
                          ('cash_dividends', dividends), ('residual_value', residual),
                          ('marked_nav', cash + residual), ('pnl', cash + residual - INITIAL)]:
        close(account[key], expected, account['code'] + ' summary ' + key)
    close(account['shares'], shares, account['code'] + ' summary shares', SHARE_TOL)
    require(account['liquidation_complete'] is (shares == 0), 'liquidation flag mismatch')
    for kind in (*FILLS, 'unfilled'):
        require(account[kind + '_count'] == counts[kind], f'{account["code"]}: {kind} count mismatch')
    return dict(cash=cash, shares=shares, fees=fees, cash_dividends=dividends,
                residual_value=residual, marked_nav=cash + residual, pnl=cash + residual - INITIAL,
                counts=counts, blocks=blocks, states=states, annual_fees=annual_fees,
                annual_dividends=annual_dividends, max_cash_error=max_cash_error,
                max_share_error=max_share_error, ledger_events=len(ledger))


def stock_inputs(run, code, inputs, manifest, generation_hash):
    meta = inputs.json(run / 'features' / f'{code}.json')
    history = inputs.json(run / 'resolved_actions' / f'{code}.json')
    fields(meta, 'code input_sha256 action_sha256 feature_version bars first_date last_date '
           'action_status rights_issue_present action_count', code + ' features')
    fields(history, 'code status events rights_issue_present metadata', code + ' actions')
    require(meta['code'] == history['code'] == code, code + ': input code mismatch')
    require(meta['feature_version'] == manifest['version'], code + ': feature version mismatch')
    encoded = canonical(history)
    require(digest(encoded) == meta['action_sha256'], code + ': action canonical hash mismatch')
    data_bytes = inputs.read(run / 'features' / f'{code}.npz')
    with np.load(io.BytesIO(data_bytes), allow_pickle=False) as loaded:
        needed = ('dates', 'raw', 'eligible', 'gates', 'event_cash', 'event_mult')
        require(set(needed) <= set(loaded.files), code + ': missing feature arrays')
        data = {k: loaded[k] for k in needed}
    dates, raw = data['dates'], data['raw']
    require(dates.ndim == 1 and dates.dtype == np.dtype('U10') and raw.dtype == np.float64
            and raw.shape == (len(dates), 5), code + ': raw feature shape/dtype')
    require(all(data[k].shape == dates.shape for k in needed[2:]), code + ': feature lengths')
    require(len(dates) == meta['bars'] and (len(dates) < 2 or np.all(dates[1:] > dates[:-1])),
            code + ': feature dates/bars')
    require(digest(dates.tobytes() + raw.tobytes() + encoded) == meta['input_sha256']
            == generation_hash, code + ': original input/generation hash mismatch')
    require(meta['first_date'] == (str(dates[0]) if len(dates) else None)
            and meta['last_date'] == (str(dates[-1]) if len(dates) else None), code + ': feature endpoints')
    require(meta['action_status'] == history['status']
            and meta['rights_issue_present'] == history['rights_issue_present']
            and meta['action_count'] == len(history['events']), code + ': action metadata mismatch')
    # Independently align sequential pre-share cash and share multipliers.
    expected_cash, expected_mult = np.zeros(len(dates)), np.ones(len(dates))
    for event in sorted(history['events'], key=lambda e: e['ex_date']):
        fields(event, 'ex_date cash_per_share share_multiplier', code + ' source action')
        ex_date = day(event['ex_date'])
        cps, mult = number(event['cash_per_share'], 'source cash/share'), number(event['share_multiplier'], 'source multiplier')
        require(cps >= 0 and mult > 0, code + ': invalid source action')
        i = int(np.searchsorted(dates, ex_date))
        if 0 < i < len(dates):
            expected_cash[i] += expected_mult[i] * cps
            expected_mult[i] *= mult
    require(np.allclose(expected_cash, data['event_cash'], rtol=0, atol=NUMBER_TOL)
            and np.allclose(expected_mult, data['event_mult'], rtol=0, atol=NUMBER_TOL),
            code + ': aligned action arrays mismatch')
    return meta, history, data


def verify_stock(run, code, inputs, manifest, generation_hash, front_column):
    account = inputs.json(run / 'accounts' / f'{code}.json')
    require(account['code'] == code, code + ': account code mismatch')
    ledger = inputs.json(run / 'ledgers' / f'{code}.json')
    meta, history, data = stock_inputs(run, code, inputs, manifest, generation_hash)
    for key in ('name', 'first_date', 'last_date', 'late_start', 'stale', 'action_status',
                'action_count', 'rights_issue_present', 'unexplained_gap_count'):
        require(account[key] == meta[key], code + ': summary/meta ' + key)
    require(account['end'] == manifest['effective_end'], code + ': account cutoff')
    calendar = manifest['calendar']
    dates, raw = data['dates'], data['raw']
    ci = np.searchsorted(calendar, dates)
    in_range = (dates >= calendar[0]) & (dates <= calendar[-1])
    require(all(calendar[int(i)] == str(d) for i, d in zip(ci[in_range], dates[in_range])),
            code + ': feature date absent from market calendar')
    valid = in_range & np.isfinite(raw).all(axis=1) & (raw[:, :4] > 0).all(axis=1)
    valid_indices = np.flatnonzero(valid)
    bars = {str(dates[i]): (float(raw[i, 0]), float(raw[i, 3])) for i in valid_indices}
    actions = {str(dates[i]): (float(data['event_cash'][i]), float(data['event_mult'][i]))
               for i in valid_indices if data['event_cash'][i] or data['event_mult'][i] != 1}
    result = replay_ledger(account, ledger, calendar, manifest['account_defaults'], bars, actions)
    eligible = np.zeros(len(calendar), dtype=bool)
    eligible[ci[in_range]] = (data['eligible'] & data['gates'])[in_range]
    require(not np.any((front_column == 1) & ~eligible), code + ': F1 without eligibility')
    if history['status'] != 'ok' or history['rights_issue_present']:
        require(not any(result['counts'][k] for k in FILLS) and not eligible.any(),
                code + ': quarantined account traded/eligible')
        close(result['cash'], INITIAL, code + ': quarantined principal')
    # Forward carry only valuations and balances, never trading signals.
    state_days = sorted(result['states'])
    positions = np.searchsorted(state_days, calendar, side='right')
    states = np.array([(INITIAL, 0.)] + [result['states'][d] for d in state_days])
    cash, shares = states[positions, 0], states[positions, 1]
    marks = np.ones(len(calendar))
    mark_indices = np.zeros(len(calendar), dtype=int)
    marks[ci[valid]] = raw[valid, 3]
    mark_indices[ci[valid]] = ci[valid]
    marks = marks[np.maximum.accumulate(mark_indices)]
    curve = cash + shares * marks
    last_price = float(raw[valid_indices[-1], 3]) if len(valid_indices) else 1.
    close(account['last_price'], last_price, code + ': final price', NUMBER_TOL)
    close(curve[-1], account['marked_nav'], code + ': curve endpoint')
    snapshots = account['years']
    expected_years = {str(d)[:4] for d in dates[valid]}
    require(set(snapshots) == expected_years, code + ': missing/extra annual account snapshots')
    for year, snap in snapshots.items():
        i = [int(i) for i in valid_indices if str(dates[i]).startswith(year)][-1]
        index = int(ci[i])
        require(snap['date'] == str(dates[i]), code + ': annual date')
        for key, expected in [('cash', cash[index]), ('shares', shares[index]), ('nav', curve[index])]:
            close(snap[key], expected, code + ': annual ' + year + ' ' + key,
                  SHARE_TOL if key == 'shares' else MONEY_TOL)
    for name, cutoff in [('train_nav', manifest['train_end']), ('validation_nav', manifest['validation_end'])]:
        i = bisect_right(calendar, cutoff) - 1
        close(account[name], curve[i] if i >= 0 else INITIAL, code + ': ' + name)
    metadata = history['metadata']
    source = metadata.get('cash_events_source') or metadata.get('source')
    require(isinstance(source, str) and source, code + ': missing action source attribution')
    cross = metadata.get('rights_crosscheck', {})
    evidence = metadata.get('evidence', []) + cross.get('evidence', [])
    original = run / 'actions' / f'{code}.json'
    original_hash = digest(inputs.read(original))
    primary = metadata.get('primary_snapshot_evidence')
    if primary is not None:
        require(original_hash == primary['sha256'], code + ': primary evidence hash mismatch')
    audit = dict(code=code, verified=True, ledger_events=result['ledger_events'],
                 account_sha256=inputs.hashes[run / 'accounts' / f'{code}.json'],
                 ledger_sha256=inputs.hashes[run / 'ledgers' / f'{code}.json'],
                 input_sha256=meta['input_sha256'],
                 feature_sha256=inputs.hashes[run / 'features' / f'{code}.npz'],
                 feature_metadata_sha256=inputs.hashes[run / 'features' / f'{code}.json'],
                 action_sha256=meta['action_sha256'],
                 resolved_actions_sha256=inputs.hashes[run / 'resolved_actions' / f'{code}.json'],
                 original_actions_sha256=original_hash, action_source=source,
                 action_status=history['status'], source_action_count=len(history['events']),
                 ledger_action_count=result['counts']['action'], action_evidence_count=len(evidence),
                 action_evidence_sha256=digest(canonical(evidence)),
                 action_snapshot_manifest_sha256=metadata.get('snapshot_manifest_sha256')
                 or cross.get('snapshot_manifest_sha256'),
                 response_sha256=history.get('response_sha256'),
                 replay_cash=result['cash'], replay_shares=result['shares'], replay_fees=result['fees'],
                 replay_cash_dividends=result['cash_dividends'],
                 max_event_cash_error=result['max_cash_error'], max_event_share_error=result['max_share_error'])
    return account, result, curve, eligible, audit, metadata


def reconcile_accounts_csv(rows, accounts):
    require(len(rows) == len(accounts), 'accounts.csv population mismatch')
    indexed = {r['code']: r for r in rows}
    require(len(indexed) == len(rows) and set(indexed) == {a['code'] for a in accounts},
            'accounts.csv duplicate/missing/extra code')
    for account in accounts:
        row = indexed[account['code']]
        for key, expected in account.items():
            if key == 'years':
                continue
            require(key in row, 'accounts.csv missing column: ' + key)
            actual = row[key]
            label = account['code'] + ' accounts.csv ' + key
            if expected is None:
                require(actual == '', label)
            elif isinstance(expected, bool):
                require(actual == str(expected), label)
            elif isinstance(expected, (int, float)):
                close(float(actual), expected, label, SHARE_TOL if key == 'shares' else MONEY_TOL)
            else:
                require(actual == str(expected), label)


def reconcile_population(manifest, summary, selection, trial_files, accounts, curve, equity_rows, front_stats,
                         eligible_counts, front_counts, annual_fees, annual_dividends):
    calendar, totals = manifest['calendar'], summary['totals']
    for key in ('version', 'start', 'requested_end', 'effective_end'):
        require(summary[key] == manifest[key], 'summary/manifest ' + key)
    require([r['date'] for r in equity_rows] == calendar, 'equity.csv calendar mismatch')
    original_curve = np.array([float(r['equity']) for r in equity_rows])
    require(np.isfinite(original_curve).all(), 'equity.csv nonfinite')
    differences = np.abs(original_curve - curve)
    require(np.all(differences <= MONEY_TOL), f'whole daily equity curve mismatch: max={differences.max()}')
    require([r['date'] for r in front_stats] == calendar, 'front_stats calendar mismatch')
    for i, row in enumerate(front_stats):
        require(row['eligible'] == int(eligible_counts[i]) and row['first_front'] == int(front_counts[i]),
                'front_stats counts mismatch: ' + calendar[i])
    expected = dict(account_count=len(accounts), initial_capital=len(accounts) * INITIAL,
                    final_cash=math.fsum(a['cash'] for a in accounts),
                    residual_marked_value=math.fsum(a['residual_value'] for a in accounts),
                    fees=math.fsum(a['fees'] for a in accounts),
                    cash_dividends=math.fsum(a['cash_dividends'] for a in accounts),
                    fully_liquidated_count=sum(a['liquidation_complete'] for a in accounts),
                    profitable_accounts=sum(a['pnl'] > 0 for a in accounts),
                    losing_accounts=sum(a['pnl'] < 0 for a in accounts),
                    action_fetch_failures=sum(a['action_status'] != 'ok' for a in accounts),
                    rights_issue_accounts=sum(a['rights_issue_present'] for a in accounts),
                    feature_errors=sum(bool(a['feature_error']) for a in accounts),
                    unexplained_gap_accounts=sum(a['unexplained_gap_count'] > 0 for a in accounts),
                    stale_accounts=sum(a['stale'] for a in accounts),
                    late_start_accounts=sum(a['late_start'] for a in accounts))
    expected['final_equity'] = expected['final_cash'] + expected['residual_marked_value']
    expected['pnl'] = expected['final_equity'] - expected['initial_capital']
    expected['cash_minus_all_principal'] = expected['final_cash'] - expected['initial_capital']
    expected.update({kind + '_count': sum(a[kind + '_count'] for a in accounts) for kind in FILLS})
    require(set(totals) == set(expected), 'unknown/missing totals fields')
    for key, value in expected.items():
        close(totals[key], value, 'totals ' + key, 0 if isinstance(value, int) else MONEY_TOL)
    close(math.fsum(a['pnl'] for a in accounts), totals['pnl'], 'sum account pnl')
    close(curve[-1], totals['final_equity'], 'last daily curve/totals')
    years = sorted({d[:4] for d in calendar})
    require([r['year'] for r in summary['annual_results']] == years, 'annual year coverage')
    previous, annual = expected['initial_capital'], []
    for row in summary['annual_results']:
        year = row['year']
        index = max(i for i, d in enumerate(calendar) if d.startswith(year))
        require(row['date'] == calendar[index], 'annual cutoff date ' + year)
        close(row['final_equity'], curve[index], 'annual equity ' + year)
        close(row['pnl'], curve[index] - previous, 'annual pnl ' + year)
        close(row['return_pct'], (curve[index] / previous - 1) * 100, 'annual return ' + year, NUMBER_TOL)
        annual.append(dict(row, cash_dividends=annual_dividends[year], fees=annual_fees[year],
                           price_pnl_proxy=row['pnl'] - annual_dividends[year]))
        previous = float(curve[index])
    close(math.fsum(r['pnl'] for r in annual), totals['pnl'], 'sum annual pnl')
    close(math.fsum(r['cash_dividends'] for r in annual), totals['cash_dividends'], 'annual dividends')
    close(math.fsum(r['fees'] for r in annual), totals['fees'], 'annual fees')
    trials = selection['trials']
    require(trials == summary['trials'] == trial_files, 'trial files/selection/summary mismatch')
    require([t['vol_target'] for t in trials] == manifest['candidate_vol_targets'], 'trial candidates mismatch')
    for trial in trials:
        require(trial['scope_end'] == manifest['effective_end'], 'trial scope cutoff')
        close(trial['validation_end_nav'], expected['initial_capital'] + trial['train_pnl'] + trial['validation_pnl'],
              'trial validation arithmetic')
    selected = max(trials, key=lambda t: (t['train_pnl'], -t['vol_target']))
    require(selection['selected'] == selected, 'selection is not training-only maximum, lower-risk tie break')
    require(selection['criterion'] == 'maximum training currency P/L; ties lower risk', 'unknown selection criterion')
    risk = dict(manifest['risk_defaults'], annual_vol_target=selected['vol_target'])
    require(summary['selected_risk'] == risk, 'selected risk/default risk mismatch')
    train_index = bisect_right(calendar, manifest['train_end']) - 1
    validation_index = bisect_right(calendar, manifest['validation_end']) - 1
    train_nav = float(curve[train_index]) if train_index >= 0 else expected['initial_capital']
    validation_nav = float(curve[validation_index]) if validation_index >= 0 else expected['initial_capital']
    close(math.fsum(a['train_nav'] for a in accounts), train_nav, 'summed training NAV')
    close(math.fsum(a['validation_nav'] for a in accounts), validation_nav, 'summed validation NAV')
    close(selected['train_pnl'], train_nav - expected['initial_capital'], 'selected training pnl')
    close(selected['validation_pnl'], validation_nav - train_nav, 'selected validation pnl')
    close(selected['validation_end_nav'], validation_nav, 'selected validation endpoint')
    close(summary['holdout_start_equity'], validation_nav, 'holdout start equity')
    close(summary['holdout_2024_onward_pnl'], expected['final_equity'] - validation_nav, 'holdout pnl')
    return annual, float(differences.max())


def snapshot_evidence(folder, inputs, metadata_rows):
    """Verify available frozen evidence. Never fetch missing external evidence."""
    if folder is None or not folder.is_dir():
        return dict(available=False, note='External action snapshot unavailable; per-stock frozen hashes retained.'), {}
    market_path = folder / 'market_manifest.json'
    market = inputs.json(market_path)
    files = {'market_manifest.json': market}
    info = dict(available=True, market_manifest_sha256=inputs.hashes[market_path], datasets={}, files={})
    for name, data in market['datasets'].items():
        fields(data, 'status pages count expected_pages', 'market evidence ' + name)
        require(data['status'] == 'ok' and len(data['pages']) == data['expected_pages'],
                'incomplete market evidence: ' + name)
        info['datasets'][name] = dict(status=data['status'], count=data['count'], pages=len(data['pages']),
                                     page_chain_sha256=digest(canonical(data['pages'])))
    references = {}
    for metadata in metadata_rows:
        cross = metadata.get('rights_crosscheck', {})
        for entry in (metadata, cross):
            if entry.get('snapshot_manifest'):
                references[entry['snapshot_manifest']] = entry['snapshot_manifest_sha256']
        primary = metadata.get('primary_snapshot_evidence')
        if primary:
            references[primary['path']] = primary['sha256']
    for relative, expected in sorted(references.items()):
        path = (folder / relative).resolve()
        require(path.is_relative_to(folder.resolve()), 'evidence path escapes snapshot')
        require(digest(inputs.read(path)) == expected, 'snapshot evidence hash mismatch: ' + relative)
    info['verified_referenced_files'] = len(references)
    info['referenced_files_sha256'] = digest(canonical(references))
    for path in sorted(folder.glob('*.json')):
        data = inputs.json(path)
        info['files'][path.name] = dict(sha256=inputs.hashes[path], bytes=path.stat().st_size)
        if path.name in ('request.json', 'validation_report.json'):
            files[path.name] = data
    return info, files


def readme(summary, manifest, verification, accounts):
    """All numbers/tables come from this audit, not hard-coded narrative results."""
    t, v, f = summary['totals'], verification, verification['front_population']
    money = lambda n: f'{n:,.2f}'
    residuals = [a for a in accounts if a['shares'] != 0]
    lines = [
        '# Pareto34 历史运行独立核验报告', '',
        '**未保本，用户的保本盈利目标未完成。核验通过仅表示冻结证据下的账务与汇总一致，不是绩效认证。**', '',
        f"- 策略版本：{summary['version']}；34 个三值原子分别比较，同日 F1 全保留但不等于满仓。",
        f"- 请求区间：{manifest['start']} ～ {manifest['requested_end']}；实际市场日首尾：{manifest['calendar'][0]} ～ {manifest['calendar'][-1]}，共 {len(manifest['calendar'])} 个市场日。",
        f"- 全部 {t['account_count']} 股各自独立初始 1,000,000.00 元，本金 {money(t['initial_capital'])} 元（{t['initial_capital']/1e8:.2f} 亿元）；不筛掉失败、隔离、晚起始、陈旧或无交易账户。",
        f"- 已成功核验 {v['verified_accounts']}/{v['expected_accounts']} 个账户、{v['verified_ledgers']} 份完整流水、{v['ledger_events']:,} 条事件；没有抽样或跳过账户。",
        f"- 本轮冻结所选波动目标 {summary['selected_risk']['annual_vol_target']:.0%}；模型默认 {manifest['risk_defaults']['annual_vol_target']:.0%}，两者不同。", '',
        '## 全量资金结果（人民币元）', '',
        '| 项目 | 金额 / 数量 |', '|---|---:|',
        f"| 全部本金 | {money(t['initial_capital'])} |",
        f"| 最终现金 | {money(t['final_cash'])} |",
        f"| 未清仓残值 | {money(t['residual_marked_value'])} |",
        f"| 最终权益＝现金＋残值 | {money(t['final_equity'])} |",
        f"| 总盈亏＝权益－全部本金 | {money(t['pnl'])}（{t['pnl']/1e8:.6f} 亿元） |",
        f"| 区间总收益率（非年化） | {100*t['pnl']/t['initial_capital']:.8f}% |",
        f"| 现金－全部本金 | {money(t['cash_minus_all_principal'])} |",
        f"| 显式费用（已扣） | {money(t['fees'])}（{t['fees']/1e8:.6f} 亿元） |",
        f"| 现金分红（已入账） | {money(t['cash_dividends'])} |",
        f"| 价差代理＝盈亏－现金分红 | {money(t['pnl']-t['cash_dividends'])} |", '',
        '现金分红按元/权息前持有股计算，已包含在权益与盈亏中，不能再加一次；费用已扣，不能重复减。滑点已经体现在成交价，不计入显式费用，也不再次扣除。',
        '**价差代理并非纯股价差**：它仍含交易费用、送转/零碎股与现金/股数口径近似，未把持股数量变动归一为可比每股价格收益。', '',
        '## 年度结果（连续账户，不重置本金）', '',
        '| 年份 | 实际截止 | 期末权益/元 | 当年盈亏/元 | 当年收益率 | 现金分红/元（已含） | 费用/元（已扣） | 价差代理/元（非纯价差） |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in summary['annual_results']:
        lines.append(f"| {row['year']} | {row['date']} | {money(row['final_equity'])} | {money(row['pnl'])} | {row['return_pct']:.8f}% | {money(row['cash_dividends'])} | {money(row['fees'])} | {money(row['price_pnl_proxy'])} |")
    lines += ['', '首尾年份不是完整自然年；年度分红/费用来自当年流水，年度盈亏之和与全区间总盈亏核对。表格金额展示到分，JSON/CSV 保留原始浮点精度。', '',
              '## F1、交易与数据质量', '',
              f"- F1 人次合计 {f['first_front_sum']:,}，同日 eligible 人次合计 {f['eligible_sum']:,}；人口加权比率＝两者总和之比 **{f['ratio']:.10f}（{100*f['ratio']:.8f}%）**，不是日比率的简单均值。",
              f"- 日均 F1 {f['first_front_sum']/len(manifest['calendar']):.6f}，日均 eligible {f['eligible_sum']/len(manifest['calendar']):.6f}。",
              f"- 建仓 {t['build_count']:,}，加仓 {t['add_count']:,}，减仓 {t['reduce_count']:,}，退出 {t['exit_count']:,}；高频 F1 切换与订单成本是本轮主要问题之一。",
              f"- 未成交 {v['event_counts'].get('unfilled',0):,} 条，其中 below_lot {v['unfilled_reasons'].get('below_lot',0)} 条；未成交不改变现金/股份，但每条均核验，不删除记录。",
              f"- 盈利 {t['profitable_accounts']}，亏损 {t['losing_accounts']}，持平 {t['account_count']-t['profitable_accounts']-t['losing_accounts']}；无成交账户 {v['no_trade_accounts']}；已清仓 {t['fully_liquidated_count']}，残留 {len(residuals)}。",
              f"- 权息失败 {t['action_fetch_failures']}，含配股 {t['rights_issue_accounts']}，特征错误 {t['feature_errors']}，未解释跳空 {t['unexplained_gap_accounts']}，陈旧 {t['stale_accounts']}，晚起始 {t['late_start_accounts']}。这些类别可以重叠，不能相加作排除数。", '',
              '## 全部残留持仓（不能冒充现金）', '',
              '| 代码 | 名称 | 最后行情日 | 剩余股数 | 估值价/元 | 残值/元 | 现金/元 | 权益/元 |',
              '|---|---|---|---:|---:|---:|---:|---:|']
    for a in residuals:
        name = a['name'].replace('|', '\\|').replace('\n', ' ')
        lines.append(f"| {a['code']} | {name} | {a['last_date']} | {a['shares']:.8f} | {a['last_price']:.6f} | {money(a['residual_value'])} | {money(a['cash'])} | {money(a['marked_nav'])} |")
    lines += ['', '## 训练、验证及已看过的后段', '',
              f"训练截止 {manifest['train_end']}，验证截止 {manifest['validation_end']}；选择规则仅取训练货币盈亏最大，并列取较低波动目标。", '',
              '| 波动目标 | 训练盈亏/元 | 验证段盈亏/元 | 验证期末权益/元 | 选择 |',
              '|---|---:|---:|---:|---|']
    for trial in summary['trials']:
        lines.append(f"| {trial['vol_target']:.0%} | {money(trial['train_pnl'])} | {money(trial['validation_pnl'])} | {money(trial['validation_end_nav'])} | {'是' if trial['vol_target']==summary['selected_risk']['annual_vol_target'] else '否'} |")
    lines += ['', f"2024 年起设计上的留出段起始权益 {money(summary['holdout_start_equity'])} 元，盈亏 {money(summary['holdout_2024_onward_pnl'])} 元。本轮已多次查看后段结果，**不是独立盲测**。未入选候选仅核对冻结 trial 文件及其内部算术，没有重新运行其全部账户。", '',
              '## 独立核验范围与限制', '',
              '- 不导入或调用生产 Account/回测器。从每股 100 万元、零股份开始，逐笔重算买卖、税费、权息、现金与股份，并核对每条记账余额、累计费用/分红及最终账户。',
              '- 交易量与成交价重算 gross；按日期重算佣金、印花税和过户费；以冻结开盘价和滑点核对成交价。收盘截止单仅允许实际最终市场日。',
              '- 校验事件先后、权息在当日订单之前、信号早于成交、非零目标为下一市场日、最后五市场日禁买；零目标退出允许跨缺失行情日等待。T+1 逐日追踪此前可卖股份与当日新买股份，不错误禁止卖出旧仓。',
              '- 权息记录可以对应零持仓，没有执行价格；分红只按权息前股数入账。独立对齐冻结权息与特征事件，检查应有的零持仓权息也未遗漏。',
              f'- 货币绝对容差 {MONEY_TOL:.2f} 元（全量总账仍为此容差，不乘账户数）；股份容差 {SHARE_TOL:g} 股；价格/比例容差 {NUMBER_TOL:g}。缺字段、文件、未知事件或不一致立即报错，无静默通过。',
              f"- 冻结原价逐日重建全部净值；与原始总曲线 {len(manifest['calendar'])} 个点、每股年度快照、训练/验证截止、年度合计和总账核对。总曲线最大绝对差 {v['max_daily_equity_error']:.10f} 元。",
              '- 原始输入哈希按原算法重新计算，NPZ/流水/账户/权息文件按原字节 SHA-256 记录；只验证冻结材料一致性，未重新计算34个原子或证明F1非支配算法正确。未证明每个未成交阻断原因、风险优化最优性或真实市场可成交性。',
              '- 当前缓存股票池有幸存者偏差；事后隔离不是历史时点可投资池。不保证供应商权息完整，不把 EM 配股交叉核验说成双源分红一致。',
              '- 日线涨跌停代理不是官方逐笔成交；没有参与率限制。权息按除权日现金到账/送转可用、无差别化红利税、零碎股处理及股份口径均是近似；配股未建模，含配股及权息失败账户保留现金。',
              '- 15% 回撤退出、20 市场日冷静期并非保本或最大回撤保证；闲置现金零利息，残值可能无法变现。不提供实时推荐。', '',
              '## 数据血缘与复现', '',
              f"- 原始 source.sqlite 快照 SHA-256：{manifest['source_sha256']}；只读取文件字节核验哈希，不打开 SQLite 连接，不刷新缓存。",
              f"- 权息来源（账户数）：{json.dumps(v['action_source_counts'], ensure_ascii=False, sort_keys=True)}。来源事件合计 {v['source_action_events']}，实际入账权息记录 {v['event_counts'].get('action',0)}（包含零持仓）。",
              f"- 市场冻结证据可用：{'是' if v['action_snapshot']['available'] else '否'}；各页计数与哈希链见 [verification.json](verification.json)，市场清单原字节哈希：{v['action_snapshot'].get('market_manifest_sha256', '不可用')}。",
              '- [audit.csv](audit.csv) 逐股保留 ledger/input/feature/权息原始哈希、来源、事件计数及独立余额；[accounts.csv](accounts.csv) 保留全部账户。',
              '- [summary.json](summary.json)、[manifest.json](manifest.json)、[equity.csv](equity.csv)、[front_stats.json](front_stats.json)、[selection.json](selection.json) 为精简可跟踪产物。绝对本机路径替换为 <LOCAL_PATH>，原文件哈希仍对应未改动源字节。',
              '- 不复制大体积价格、特征、逐笔流水或 source.sqlite。复现核验必须保留原运行目录及可用权息快照；此精简包本身不足以重新演算逐笔账本。',
              '- [verification.json](verification.json) 是本次验证证据，标记为 tests/validation_not_performance_certificate；不声称导出时运行了单元测试。测试命令与源码哈希另存其中。',
              '- 在仓库根目录使用项目虚拟环境：python -B pareto_report.py --run output/pareto_mainboard_20260910 --output reports/pareto_20260910；通用 CLI 也接受其他冻结目录。',
              '- 新增测试：python -B -m unittest unittests.test_pareto_report -v。本中文报告由同一导出源码自动生成；冻结输入与源码不变时，重复导出字节一致。', '']
    return '\n'.join(lines)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(redact_paths(value), ensure_ascii=False, separators=(',', ':'),
                               allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    require(bool(rows), 'cannot export empty CSV')
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(redact_paths(rows))


def export_report(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    require(not output.is_relative_to(run) and not run.is_relative_to(output), 'output must not overlap input run')
    require(not output.exists() or not any(output.iterdir()), 'output must be new or empty; do not overwrite a completed report')
    inputs = Inputs()
    manifest = inputs.json(run / 'manifest.json')
    summary = inputs.json(run / 'summary.json')  # Completion marker is mandatory.
    selection = inputs.json(run / 'selection.json')
    generation = inputs.json(run / 'generation.json')
    front_stats = inputs.json(run / 'front_stats.json')
    fields(manifest, 'version start requested_end effective_end calendar universe account_defaults '
           'risk_defaults candidate_vol_targets train_end validation_end source_sha256', 'manifest')
    require(manifest['version'] == 'pareto_daily_atoms_risk_v1', 'unsupported strategy version')
    calendar = manifest['calendar']
    require(calendar and calendar == sorted(set(calendar)), 'invalid calendar')
    for d in (*calendar, manifest['start'], manifest['requested_end'], manifest['effective_end'],
              manifest['train_end'], manifest['validation_end']):
        day(d)
    require(manifest['start'] <= calendar[0] <= calendar[-1] == manifest['effective_end']
            <= manifest['requested_end'], 'manifest endpoints')
    codes = [row['code'] for row in manifest['universe']]
    require(codes and codes == sorted(set(codes)) and all(re.fullmatch(r'\d{6}', c) for c in codes), 'invalid universe')
    for folder, suffix in [('accounts', '.json'), ('ledgers', '.json'), ('features', '.json'),
                           ('features', '.npz'), ('resolved_actions', '.json'), ('actions', '.json')]:
        actual = {p.stem for p in (run / folder).glob('*' + suffix)}
        require(actual == set(codes), f'{folder}{suffix}: missing/extra population files')
    require(generation['version'] == manifest['version'] and len(generation['hashes']) == len(codes),
            'generation version/population')
    wal = run / 'source.sqlite-wal'
    require(not wal.exists() or wal.stat().st_size == 0, 'nonempty SQLite WAL: byte hash is not a complete snapshot')
    source_hash = inputs.large_hash(run / 'source.sqlite')
    require(source_hash == manifest['source_sha256'], 'source snapshot hash mismatch')
    equity_rows = list(csv.DictReader(io.StringIO(inputs.read(run / 'equity.csv').decode('utf-8-sig'))))
    csv_accounts = list(csv.DictReader(io.StringIO(inputs.read(run / 'accounts.csv').decode('utf-8-sig'))))
    trial_files = [inputs.json(run / 'trials' / f'vol_{vol:.2f}.json') for vol in manifest['candidate_vol_targets']]
    require({p.name for p in (run / 'trials').glob('*.json')} ==
            {f'vol_{vol:.2f}.json' for vol in manifest['candidate_vol_targets']}, 'extra/missing trial files')
    inputs.large_hash(run / 'fronts.npy')
    fronts = np.load(run / 'fronts.npy', mmap_mode='r', allow_pickle=False)
    require(fronts.shape == (len(calendar), len(codes)) and np.isin(fronts, (1, 2, 5)).all(), 'fronts shape/labels')
    curve = np.zeros(len(calendar))
    eligible_counts = np.zeros(len(calendar), dtype=np.int64)
    front_counts = np.count_nonzero(fronts == 1, axis=1)
    accounts, audits, metadata_rows = [], [], []
    counts, blocks, sources = Counter(), Counter(), Counter()
    yearly_fees, yearly_dividends = defaultdict(float), defaultdict(float)
    events = no_trade = 0
    for index, code in enumerate(codes):
        try:
            account, replay, stock_curve, eligible, audit, metadata = verify_stock(
                run, code, inputs, manifest, generation['hashes'][index], fronts[:, index])
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise VerificationError(f'{code}: verification failed after {index}/{len(codes)} successful ledgers: {exc}') from exc
        accounts.append(account)
        audits.append(audit)
        metadata_rows.append(metadata)
        curve += stock_curve
        eligible_counts += eligible
        counts.update(replay['counts'])
        blocks.update(replay['blocks'])
        sources[audit['action_source']] += 1
        events += replay['ledger_events']
        no_trade += not any(replay['counts'][k] for k in FILLS)
        for year, value in replay['annual_fees'].items():
            yearly_fees[year] += value
        for year, value in replay['annual_dividends'].items():
            yearly_dividends[year] += value
        if (index + 1) % 250 == 0:
            print(f'Independent ledger audit: {index + 1}/{len(codes)}', flush=True)
    reconcile_accounts_csv(csv_accounts, accounts)
    annual, curve_error = reconcile_population(manifest, summary, selection, trial_files, accounts, curve,
                                               equity_rows, front_stats, eligible_counts, front_counts,
                                               yearly_fees, yearly_dividends)
    action_dir = manifest.get('action_snapshot_dir')
    action_dir = Path(action_dir) if action_dir else None
    if action_dir is not None and not action_dir.is_absolute():
        action_dir = ROOT / action_dir
    evidence, evidence_files = snapshot_evidence(action_dir, inputs, metadata_rows)
    first_sum, eligible_sum = int(front_counts.sum()), int(eligible_counts.sum())
    require(eligible_sum > 0, 'no eligible population for F1 ratio')
    report_summary = dict(summary, annual_results=annual,
                          price_pnl_proxy=summary['totals']['pnl'] - summary['totals']['cash_dividends'])
    verification = dict(schema_version=VERSION, status='passed',
                        classification='tests/validation_not_performance_certificate',
                        performance_certified=False, principal_protected=False, user_goal_met=False,
                        expected_accounts=len(codes), verified_accounts=len(accounts), verified_ledgers=len(audits),
                        ledger_events=events, skipped_accounts=0, no_trade_accounts=no_trade,
                        event_counts=dict(counts), unfilled_reasons=dict(blocks),
                        currency_absolute_tolerance=MONEY_TOL, shares_absolute_tolerance=SHARE_TOL,
                        price_ratio_absolute_tolerance=NUMBER_TOL, max_daily_equity_error=curve_error,
                        max_event_cash_error=max(a['max_event_cash_error'] for a in audits),
                        max_event_share_error=max(a['max_event_share_error'] for a in audits),
                        daily_equity_points_verified=len(calendar),
                        front_population=dict(first_front_sum=first_sum, eligible_sum=eligible_sum,
                                              ratio=first_sum / eligible_sum),
                        action_source_counts=dict(sources), source_action_events=sum(a['source_action_count'] for a in audits),
                        action_snapshot=evidence,
                        checks=['all account/ledger events independently replayed', 'T+1 prior vs new shares',
                                'gross/fees/slippage/raw price and causal time', 'aligned actions including zero holdings',
                                'all daily aggregate equity and annual account snapshots',
                                'population totals, annual sum, cutoffs and selected trial',
                                'original input/generation/action hashes and file SHA-256',
                                'front/eligible counts, not front dominance recomputation'],
                        unit_tests=dict(executed_by_export=False,
                                        command='python -B -m unittest unittests.test_pareto_report -v'),
                        exporter_sha256=digest(Path(__file__).read_bytes()),
                        test_source_sha256=digest((ROOT / 'unittests' / 'test_pareto_report.py').read_bytes()))
    report_manifest = dict(manifest, report_schema=VERSION,
                           path_redaction_note='Absolute local paths replaced by <LOCAL_PATH>/basename; original-byte hashes unchanged.',
                           source_files={str(p.relative_to(run)): dict(sha256=h, bytes=inputs.fingerprints[p][0])
                                         for p, h in sorted(inputs.hashes.items())
                                         if p.parent in (run, run / 'trials')},
                           omitted_large_inputs=['source.sqlite', 'features/', 'ledgers/', 'fronts.npy', 'accounts/'],
                           per_stock_hashes='audit.csv')
    inputs.unchanged()
    # No output exists until EVERY input ledger and aggregate invariant passes.
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [('summary.json', report_summary), ('manifest.json', report_manifest),
                        ('front_stats.json', front_stats), ('selection.json', selection)]:
        write_json(output / name, value)
    write_csv(output / 'accounts.csv', [{k: v for k, v in a.items() if k != 'years'} for a in accounts])
    write_csv(output / 'audit.csv', audits)
    write_csv(output / 'equity.csv', equity_rows)
    for name, value in evidence_files.items():
        write_json(output / 'evidence' / name, value)
    (output / 'README.md').write_text(readme(report_summary, report_manifest, verification, accounts), encoding='utf-8')
    verification['report_files_sha256'] = {p.relative_to(output).as_posix(): digest(p.read_bytes())
                                           for p in sorted(output.rglob('*')) if p.is_file()}
    # This is the REPORT completion marker, written last. No self-referential hash.
    write_json(output / 'verification.json', verification)
    print(json.dumps(dict(status='passed', verified_accounts=len(accounts), verified_ledgers=len(audits),
                          ledger_events=events, pnl=summary['totals']['pnl'],
                          front_population_ratio=first_sum / eligible_sum), ensure_ascii=False), flush=True)
    return verification


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        export_report(args.run, args.output)
    except (VerificationError, KeyError, TypeError, ValueError, OSError) as exc:
        print(f'VERIFICATION FAILED: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())