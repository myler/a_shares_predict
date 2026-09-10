#!/usr/bin/env python3
"""Snapshot-based daily Pareto / independent-account research replay.

No production DB writes. Source is backed up through a read-only transaction;
network collection writes only the new run's action snapshots. Never treat
missing actions, stale quotes or failed liquidation as verified cash proceeds.

New runs start no earlier than 2015-01-01, the feature/action coverage boundary;
insufficient warm-up remains the existing eligible mask's responsibility.
Resume requires the exact frozen start/requested end/limit and, when recorded,
the resolved source path. The frozen source.sqlite is the only quote input:
each resume verifies its byte SHA-256 when recorded, never the current source
DB. Changed production contents at the same path are allowed; the external
path need not still exist (e.g. copied runs). Legacy manifests without a limit
mean zero; absent source-path/hash fields are not inferred or backfilled.
Validation precedes removing summary.json, the completion marker. These are
runner checks only, not per-request Web hashes or external-path existence checks
for readers of copied reports.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

import numpy as np
import pandas as pd

from capital_account import Account, AccountConfig
from pareto_strategy import (OBJECTIVE_NAMES, STRATEGY_VERSION, RiskConfig,
                             daily_front_layers, make_objectives,
                             optimal_rebalance_weight, risk_statistics)

ROOT = Path(__file__).resolve().parent
MAINBOARD = ('600', '601', '603', '605', '000', '001', '002', '003')
CANDIDATE_VOL_TARGETS = (.10, .15, .20)
TRAIN_END = '2021-12-31'
VALIDATION_END = '2023-12-31'
FEATURE_START = '2015-01-01'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def connect_ro(path):
    connection = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    connection.execute('PRAGMA query_only=ON')
    return connection


def aligned_actions(events, dates):
    """Map actions during a missing/holiday bar to first subsequent bar.

    This preserves pre-event holdings but assumes no intervening trades. A
    separate date-availability warning remains in every result.
    """
    result = {}
    for event in sorted(events, key=lambda x: x['ex_date']):
        i = int(np.searchsorted(dates, event['ex_date']))
        if not 0 < i < len(dates):
            continue
        d, m = result.get(i, (0., 1.))
        result[i] = (d + m * event['cash_per_share'], m * event['share_multiplier'])
    return result


def causal_adjusted_prices(raw, actions):
    """Forward-only ex-rights continuity transform, never adjust past signals.

    At ex-date the theoretical reference is (previous raw close - cash)/shares.
    Price levels after that event are rescaled, not earlier observations. This
    is for indicators only; executions and the cash/share ledger remain raw.
    """
    result = raw.copy()
    scale = 1.
    share_scale = 1.
    for i in range(len(raw)):
        if i in actions:
            dividend, multiplier = actions[i]
            reference = (raw[i - 1, 3] - dividend) / multiplier
            if reference <= 0 or multiplier <= 0:
                raise ValueError('invalid ex-rights reference')
            scale *= raw[i - 1, 3] / reference
            share_scale *= multiplier
        result[i, :4] *= scale
        result[i, 4] /= share_scale
    return result


def prepare_stock(job):
    folder, code, name, start, end = job
    folder = Path(folder)
    dest = folder / 'features' / (code + '.npz')
    meta_path = folder / 'features' / (code + '.json')
    resolved = folder / 'resolved_actions' / (code + '.json')
    history = read_json(resolved if resolved.exists() else folder / 'actions' / (code + '.json'))
    action_hash = hashlib.sha256(json.dumps(history, sort_keys=True).encode()).hexdigest()
    if dest.exists() and meta_path.exists():
        existing = read_json(meta_path)
        if (existing.get('feature_version') == STRATEGY_VERSION
                and existing.get('action_sha256', action_hash) == action_hash):
            return existing
    with connect_ro(folder / 'source.sqlite') as db:
        rows = db.execute('SELECT date,open,high,low,close,volume FROM klines '
                          'WHERE code=? AND date>=? AND date<=? ORDER BY date',
                          (code, FEATURE_START, end)).fetchall()
    meta = dict(code=code, name=name, bars=len(rows), action_status=history['status'],
                rights_issue_present=history.get('rights_issue_present', False),
                action_count=len(history.get('events', [])), feature_version=STRATEGY_VERSION,
                action_sha256=action_hash)
    dates = np.array([r[0] for r in rows], dtype='U10')
    raw = np.array([r[1:] for r in rows], dtype=np.float64).reshape(-1, 5)
    meta.update(first_date=str(dates[0]) if len(dates) else None,
                last_date=str(dates[-1]) if len(dates) else None,
                late_start=not len(dates) or dates[0] > start,
                stale=not len(dates) or dates[-1] < end)
    actions = aligned_actions(history.get('events', []), dates)
    event_cash = np.zeros(len(dates)); event_mult = np.ones(len(dates))
    for i, (cash, mult) in actions.items():
        event_cash[i], event_mult[i] = cash, mult
    try:
        adjusted = causal_adjusted_prices(raw, actions)
        objectives = make_objectives(*adjusted.T)
        risk = risk_statistics(adjusted[:, 3])
        # Discontinuities NOT explained by recorded actions invalidate trading
        # thereafter. Do not turn an unknown split into a profitable/crash signal.
        reference = raw[:-1, 3].copy()
        for i, (cash, mult) in actions.items():
            reference[i - 1] = (reference[i - 1] - cash) / mult
        jump = np.zeros(len(raw), bool)
        if len(raw) > 1:
            jump[1:] = np.abs(raw[1:, 0] / reference - 1) > .25
        quarantined = np.maximum.accumulate(jump) if len(jump) else jump
        objectives['eligible'] &= ~quarantined
        # Retrospective data-quality isolation, NOT a point-in-time universe:
        # undated rights history cannot be priced safely. Keep these accounts
        # and their full initial cash, but never simulate unknown entitlements.
        if history['status'] != 'ok' or meta['rights_issue_present']:
            objectives['eligible'][:] = False
        meta['unexplained_gap_count'] = int(jump.sum())
        meta['feature_error'] = None
    except (ValueError, TypeError, FloatingPointError) as error:
        objectives = dict(votes=np.zeros((len(rows), 34), np.int8),
                          eligible=np.zeros(len(rows), bool), gate_pass=np.zeros(len(rows), bool))
        risk = dict(annual_vol=np.full(len(rows), np.nan), daily_es=np.full(len(rows), np.nan))
        meta['unexplained_gap_count'] = 0
        meta['feature_error'] = str(error)
    meta['input_sha256'] = hashlib.sha256(dates.tobytes() + raw.tobytes() +
                                        json.dumps(history, sort_keys=True).encode()).hexdigest()
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, dates=dates, raw=raw, votes=objectives['votes'],
                        eligible=objectives['eligible'], gates=objectives['gate_pass'],
                        annual_vol=risk['annual_vol'], daily_es=risk['daily_es'],
                        event_cash=event_cash, event_mult=event_mult)
    write_json(meta_path, meta)
    return meta


def compute_fronts(folder, codes, calendar):
    """Compare only contemporaneous eligible, gate-passing vectors each day."""
    folder = Path(folder)
    shape = (len(calendar), len(codes))
    votes = np.zeros(shape + (34,), dtype=np.int8)
    eligible = np.zeros(shape, bool)
    for col, code in enumerate(codes):
        with np.load(folder / 'features' / (code + '.npz')) as data:
            idx = np.searchsorted(calendar, data['dates'])
            valid = (idx < len(calendar)) & (data['dates'] >= calendar[0])
            votes[idx[valid], col] = data['votes'][valid]
            eligible[idx[valid], col] = (data['eligible'] & data['gates'])[valid]
    layers = np.full(shape, 5, dtype=np.int8)
    stats = []
    for i, day in enumerate(calendar):
        selected = np.flatnonzero(eligible[i])
        if len(selected):
            layers[i, selected] = daily_front_layers(votes[i, selected], max_layers=1)
        stats.append(dict(date=str(day), eligible=int(len(selected)),
                          first_front=int(np.count_nonzero(layers[i] == 1))))
        if (i + 1) % 250 == 0:
            print(f'Pareto daily fronts {i+1}/{len(calendar)}', flush=True)
    np.save(folder / 'fronts.npy', layers)
    write_json(folder / 'front_stats.json', stats)
    return stats


def execution_blocks(raw, i, action_cash, action_mult):
    """Conservative opening-limit proxy, without using that day's high/low.

    Historical ST flags and official limit references are unavailable: block
    either side at adverse >=4.8% opening gaps (even for normal 10% stocks).
    volume is end-of-day liquidity availability, not a signal/timing feature.
    """
    tradable = bool(np.isfinite(raw[i]).all() and raw[i, 0] > 0 and raw[i, 4] > 0)
    if i == 0:
        return tradable, False, False
    reference = (raw[i - 1, 3] - action_cash[i]) / action_mult[i]
    gap = raw[i, 0] / reference - 1 if reference > 0 else 0
    return tradable, bool(gap >= .048), bool(gap <= -.048)


def replay_stock(folder, code, col, risk_config, end=None, record_ledger=False):
    folder = Path(folder)
    manifest = read_json(folder / 'manifest.json')
    start = manifest['start']
    end = end or manifest['effective_end']
    calendar = np.array(manifest['calendar'], dtype='U10')
    layer_column = np.load(folder / 'fronts.npy', mmap_mode='r')[:, col]
    data = dict(np.load(folder / 'features' / (code + '.npz')))
    dates, raw = data['dates'], data['raw']
    meta = read_json(folder / 'features' / (code + '.json'))
    account = Account(code, record_ledger=record_ledger)
    nav_curve = np.full(len(calendar), account.cash)
    exposure_curve = np.zeros(len(calendar))
    pending = None
    last_price = 1.
    risk_exit = False
    snapshots = {}
    present = set()
    quality_allowed = meta['action_status'] == 'ok' and not meta['rights_issue_present']
    deadline_start = max(0, int(np.searchsorted(calendar, end, side='right')) - 5)
    valid_rows = np.flatnonzero((dates >= start) & (dates <= end))
    for i in valid_rows:
        day = str(dates[i]); ci = int(np.searchsorted(calendar, day))
        if ci >= len(calendar) or calendar[ci] != day:
            continue
        o, h, l, close, volume = raw[i]
        if not np.isfinite(raw[i]).all() or min(o, h, l, close) <= 0:
            meta['feature_error'] = meta['feature_error'] or 'invalid cached raw bar skipped'
            continue
        present.add(ci)
        if data['event_cash'][i] or data['event_mult'][i] != 1:
            account.apply_action(day, data['event_cash'][i], data['event_mult'][i])
        tradable, block_buy, block_sell = execution_blocks(
            raw, i, data['event_cash'], data['event_mult'])
        # A scheduled deadline is known in advance. No new shares during its
        # final five market sessions, avoiding an impossible same-day T+1 exit.
        if ci >= deadline_start:
            account.rebalance(day, o, 0., tradable=tradable, block_sell=block_sell,
                              reason='scheduled_deadline', signal_day=None)
        elif pending is not None:
            target, signal_day, signal_ci, reason = pending
            # Do not execute stale BUY orders after a suspension. Exits persist.
            if ci == signal_ci + 1 or target == 0:
                account.rebalance(day, o, target, tradable=tradable,
                                  block_buy=block_buy or account.risk_blocked(ci),
                                  block_sell=block_sell, reason=reason, signal_day=signal_day)
        pending = None
        risk_exit = account.observe_close(day, ci, close)
        nav = account.nav(close)
        current = account.shares * close / nav if nav else 0.
        vol, es = float(data['annual_vol'][i]), float(data['daily_es'][i])
        allowed = (quality_allowed and layer_column[ci] == 1 and data['eligible'][i] and data['gates'][i]
                   and not risk_exit and not account.risk_blocked(ci))
        desired = (min(risk_config.max_weight, risk_config.annual_vol_target / max(vol, 1e-8),
                       risk_config.es_budget / max(es, 1e-8))
                   if allowed and np.isfinite(vol) and np.isfinite(es) else 0.)
        if ci < deadline_start:
            # Proportional cost proxy for the optimizer; exact minimum fees and
            # share/lot rounding are applied by the execution account.
            buy_cost = account.config.commission_rate + .00002 + account.config.slippage_bps / 10000
            sell_cost = buy_cost + (.0005 if day >= '2023-08-28' else .001)
            target = optimal_rebalance_weight(current, desired, vol if np.isfinite(vol) else 0.,
                                              buy_cost, sell_cost, risk_config, risk_cap=desired)
            pending = (target, day, ci, 'risk_exit' if risk_exit else 'pareto_risk_target')
        # Only at actual common deadline: terminal close order cannot use a
        # stale stock's last date as though known to be its final trade day.
        if day == end and account.shares:
            previous = raw[i-1, 3] if i else close
            ref = (previous-data['event_cash'][i])/data['event_mult'][i]
            limit_down = close <= ref * (1-.048)
            account.rebalance(day, close, 0., tradable=tradable, block_sell=bool(limit_down),
                              reason='deadline_close')
            account.observe_close(day, ci, close)
        nav_curve[ci] = account.nav(close)
        exposure_curve[ci] = account.shares * close / max(account.nav(close), 1e-8)
        last_price = close
        snapshots[day[:4]] = dict(date=day, nav=account.nav(close), cash=account.cash,
                                  shares=account.shares)
    # Forward fill valuations across missing bars, never forward-fill signals.
    for ci in range(1, len(calendar)):
        if ci not in present:
            nav_curve[ci] = nav_curve[ci-1]
            exposure_curve[ci] = exposure_curve[ci-1]
    summary = account.summary(last_price)
    summary.update(name=meta['name'], last_price=last_price, end=end,
                   first_date=meta['first_date'], last_date=meta['last_date'],
                   feature_error=meta['feature_error'], late_start=meta['late_start'],
                   stale=meta['stale'], rights_issue_present=meta['rights_issue_present'],
                   action_status=meta['action_status'], action_count=meta['action_count'],
                   unexplained_gap_count=meta['unexplained_gap_count'],
                   final_cash=account.cash, residual_value=account.shares*last_price,
                   liquidation_complete=account.shares == 0., years=snapshots,
                   average_exposure=float(exposure_curve[calendar <= end].mean())
                   if np.any(calendar <= end) else 0.)
    for key, cutoff in [('train_nav', TRAIN_END), ('validation_nav', VALIDATION_END)]:
        idx = min(len(calendar)-1, int(np.searchsorted(calendar, cutoff, side='right'))-1)
        summary[key] = float(nav_curve[idx]) if idx >= 0 else account.config.initial_capital
    return summary, nav_curve, account.ledger


def replay_job(job):
    folder, code, col, vol_target, end, ledger = job
    config = RiskConfig(annual_vol_target=vol_target)
    return replay_stock(folder, code, col, config, end, ledger)


def replay_population(folder, codes, vol_target, workers, end=None, ledgers=False):
    folder = Path(folder)
    summaries = []
    calendar = read_json(folder / 'manifest.json')['calendar']
    total_curve = np.zeros(len(calendar))
    jobs = [(str(folder), c, i, vol_target, end, ledgers) for i, c in enumerate(codes)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(replay_job, job): job[1] for job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            summary, curve, ledger = future.result()
            summaries.append(summary); total_curve += curve
            if ledgers:
                write_json(folder / 'accounts' / (summary['code'] + '.json'), summary)
                write_json(folder / 'ledgers' / (summary['code'] + '.json'), ledger)
            if done % 250 == 0:
                print(f'Accounts vol={vol_target:.0%}: {done}/{len(codes)}', flush=True)
    summaries.sort(key=lambda s: s['code'])
    return summaries, total_curve


def aggregate(summaries):
    initial = sum(s['initial_capital'] for s in summaries)
    cash = sum(s['final_cash'] for s in summaries)
    residual = sum(s['residual_value'] for s in summaries)
    return dict(account_count=len(summaries), initial_capital=initial, final_cash=cash,
                residual_marked_value=residual, final_equity=cash+residual,
                pnl=cash+residual-initial, cash_minus_all_principal=cash-initial,
                fully_liquidated_count=sum(s['liquidation_complete'] for s in summaries),
                profitable_accounts=sum(s['pnl'] > 0 for s in summaries),
                losing_accounts=sum(s['pnl'] < 0 for s in summaries),
                fees=sum(s['fees'] for s in summaries),
                cash_dividends=sum(s['cash_dividends'] for s in summaries),
                action_fetch_failures=sum(s['action_status'] != 'ok' for s in summaries),
                rights_issue_accounts=sum(s['rights_issue_present'] for s in summaries),
                feature_errors=sum(bool(s['feature_error']) for s in summaries),
                unexplained_gap_accounts=sum(s['unexplained_gap_count'] > 0 for s in summaries),
                stale_accounts=sum(s['stale'] for s in summaries),
                late_start_accounts=sum(s['late_start'] for s in summaries),
                build_count=sum(s['build_count'] for s in summaries),
                add_count=sum(s['add_count'] for s in summaries),
                reduce_count=sum(s['reduce_count'] for s in summaries),
                exit_count=sum(s['exit_count'] for s in summaries))


def validate_new_run(start, requested_end, limit):
    """Reject unsupported requests before making a folder or opening a DB."""
    date.fromisoformat(start); date.fromisoformat(requested_end)
    if start < FEATURE_START:
        raise ValueError(f'start must be >= {FEATURE_START} (feature/action coverage)')
    if start >= requested_end or limit < 0:
        raise ValueError('invalid dates or limit')


def validate_resume(folder, manifest, source, start, requested_end, limit):
    """Validate identity and frozen bytes without accessing the external DB."""
    for key, actual, expected in (
            ('start', start, manifest['start']),
            ('requested_end', requested_end, manifest['requested_end']),
            ('universe_limit', limit, manifest.get('universe_limit', 0))):
        if actual != expected:
            raise ValueError(f'refuse resume: {key} differs from manifest '
                             f'({actual!r} != {expected!r}); use a new run directory')
    if 'source_path' in manifest and str(Path(source).resolve()) != manifest['source_path']:
        raise ValueError('refuse resume: source_path differs from manifest; use a new run directory')
    if 'source_sha256' in manifest:
        try:
            with (Path(folder) / 'source.sqlite').open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        except OSError as error:
            raise ValueError('refuse resume: cannot verify frozen source.sqlite SHA-256') from error
        if digest != manifest['source_sha256']:
            raise ValueError('refuse resume: frozen source.sqlite SHA-256 differs from manifest')


def create_run(folder, source, start, requested_end, limit=0):
    validate_new_run(start, requested_end, limit)
    source = Path(source).resolve()
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    snapshot = folder / 'source.sqlite'
    if not snapshot.exists():
        with connect_ro(source) as origin, sqlite3.connect(snapshot) as dest:
            origin.backup(dest)
    with snapshot.open('rb') as stream:
        source_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    with connect_ro(snapshot) as db:
        listing = db.execute('SELECT k.code,COALESCE(s.name,k.code),MIN(k.date),MAX(k.date) '
                             'FROM klines k LEFT JOIN stocks s ON k.code=s.code '
                             'GROUP BY k.code ORDER BY k.code').fetchall()
        listing = [r for r in listing if r[0].startswith(MAINBOARD)]
        if limit:
            listing = listing[:limit]
        effective_end = min(requested_end, db.execute('SELECT MAX(date) FROM klines').fetchone()[0])
        calendar = [r[0] for r in db.execute('SELECT DISTINCT date FROM klines '
                                             'WHERE date>=? AND date<=? ORDER BY date',
                                             (start, effective_end))]
    if not calendar or not listing:
        raise ValueError('empty calendar or mainboard universe')
    manifest = dict(version=STRATEGY_VERSION, start=start, requested_end=requested_end,
                    effective_end=effective_end, calendar=calendar,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    universe=[dict(code=r[0], name=r[1], first_date=r[2], last_date=r[3]) for r in listing],
                    objective_names=OBJECTIVE_NAMES, risk_defaults=asdict(RiskConfig()),
                    account_defaults=asdict(AccountConfig()),
                    candidate_vol_targets=CANDIDATE_VOL_TARGETS,
                    train_end=TRAIN_END, validation_end=VALIDATION_END,
                    source_path=str(source), source_sha256=source_sha256,
                    universe_limit=limit, feature_start=FEATURE_START,
                    action_coverage_start=FEATURE_START,
                    limitations=['current cached universe, survivorship bias',
                                 'not all stocks have ten-year history',
                                 'Sina/Eastmoney ex-date cash/share availability approximation; no dividend tax',
                                 'rights issues not modeled: all-history flagged accounts held in cash',
                                 'failed/unverified actions and malformed histories held in cash',
                                 'retrospective quality screening, not a point-in-time investable universe',
                                 'opening 4.8% gap conservative limit proxy',
                                 'daily OHLC cannot certify fills or official limit status',
                                 'unexplained >25% opening gaps quarantine later signals',
                                 'zero interest on idle cash; no principal guarantee'])
    write_json(folder / 'manifest.json', manifest)
    return manifest


def run(folder, source, start, end, workers=8, limit=0, fetch_actions=True, action_snapshot=None):
    folder = Path(folder)
    resuming = (folder/'manifest.json').exists()
    manifest = (read_json(folder/'manifest.json') if resuming
                else create_run(folder, source, start, end, limit))
    if manifest['version'] != STRATEGY_VERSION:
        raise ValueError('refuse resuming a different strategy version')
    if resuming:
        validate_resume(folder, manifest, source, start, end, limit)
    # summary.json is the completion marker: preserve it on validation failure.
    (folder/'summary.json').unlink(missing_ok=True)
    codes = [r['code'] for r in manifest['universe']]
    if fetch_actions:
        from action_fallback import resolve_action_snapshot
        action_folder = Path(action_snapshot) if action_snapshot else folder/'market_actions'
        histories = resolve_action_snapshot(codes, action_folder, start=FEATURE_START,
                                            end=manifest['requested_end'], primary_dir=folder/'actions')
        for code, history in histories.items():
            write_json(folder/'resolved_actions'/(code+'.json'), history)
        manifest['action_snapshot_dir'] = str(action_folder.resolve())
        manifest['action_sources'] = 'public Eastmoney batch; existing Sina dividends retained, rights crosschecked'
        write_json(folder/'manifest.json', manifest)
        print(f"Resolved corporate actions: {sum(h['status']=='ok' for h in histories.values())}/{len(codes)}", flush=True)
    else:
        with connect_ro(folder / 'source.sqlite') as db:
            for code in codes:
                path = folder/'actions'/(code+'.json')
                if path.exists():
                    continue
                rows = db.execute('SELECT ex_date,dividend_per_share,bonus_share,transfer_share '
                                  'FROM dividends WHERE code=? ORDER BY ex_date', (code,)).fetchall()
                write_json(path, dict(code=code, status='cache_unverified', rights_issue_present=False,
                                     events=[dict(ex_date=r[0],cash_per_share=r[1] or 0,
                                                  share_multiplier=1+((r[2] or 0)+(r[3] or 0))/10) for r in rows]))
    jobs = [(str(folder), row['code'], row['name'], manifest['start'], manifest['effective_end'])
            for row in manifest['universe']]
    feature_hashes = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(prepare_stock, jobs), 1):
            feature_hashes.append(result.get('input_sha256', result['code']))
            if i % 100 == 0:
                print(f'Objective histories {i}/{len(codes)}', flush=True)
    generation = dict(version=STRATEGY_VERSION, hashes=feature_hashes)
    generation_path = folder/'generation.json'
    if generation_path.exists() and read_json(generation_path) != generation:
        (folder/'fronts.npy').unlink(missing_ok=True)
        for path in (folder/'trials').glob('*.json'):
            path.unlink()
    write_json(generation_path, generation)
    if not (folder/'fronts.npy').exists():
        compute_fronts(folder, codes, np.array(manifest['calendar'], dtype='U10'))
    trials = []
    # A fixed three-point risk-sensitivity grid, not unbounded profit searching.
    # Selection only reads training NAV. Validation data do not select winner.
    for vol in CANDIDATE_VOL_TARGETS:
        trial_path = folder/'trials'/f'vol_{vol:.2f}.json'
        cached = read_json(trial_path) if trial_path.exists() else None
        if cached and cached['scope_end'] == manifest['effective_end']:
            trial = cached
        else:
            # One continuous account and the SAME scheduled final deadline as
            # the selected replay: no artificial liquidation at phase borders.
            # Selection below reads training P/L only, never later outcomes.
            summaries, _ = replay_population(folder, codes, vol, workers)
            principal = len(codes)*AccountConfig().initial_capital
            train_nav = sum(s['train_nav'] for s in summaries)
            validation_nav = sum(s['validation_nav'] for s in summaries)
            trial = dict(vol_target=vol, train_pnl=train_nav-principal,
                         validation_pnl=validation_nav-train_nav, validation_end_nav=validation_nav,
                         scope_end=manifest['effective_end'])
            write_json(trial_path, trial)
        trials.append(trial)
    selected = max(trials, key=lambda t: (t['train_pnl'], -t['vol_target']))
    write_json(folder/'selection.json', dict(trials=trials, selected=selected,
                                           criterion='maximum training currency P/L; ties lower risk'))
    summaries, curve = replay_population(folder, codes, selected['vol_target'], workers, ledgers=True)
    totals = aggregate(summaries)
    years = []
    previous = totals['initial_capital']
    calendar = np.array(manifest['calendar'])
    for year in sorted(set(d[:4] for d in calendar)):
        last = np.flatnonzero(np.char.startswith(calendar, year))[-1]
        value = float(curve[last])
        years.append(dict(year=year, date=str(calendar[last]), final_equity=value,
                          pnl=value-previous, return_pct=(value/previous-1)*100))
        previous = value
    validation_index = int(np.searchsorted(calendar, VALIDATION_END, side='right'))-1
    holdout_start_equity = (float(curve[validation_index]) if validation_index >= 0
                            else totals['initial_capital'])
    holdout_pnl = totals['final_equity']-holdout_start_equity
    result = dict(version=STRATEGY_VERSION, start=manifest['start'],
                  requested_end=manifest['requested_end'], effective_end=manifest['effective_end'],
                  selected_risk=asdict(RiskConfig(annual_vol_target=selected['vol_target'])),
                  totals=totals, annual_results=years, holdout_2024_onward_pnl=holdout_pnl,
                  holdout_start_equity=holdout_start_equity,
                  trials=trials, limitations=manifest['limitations'],
                  verdict='unverified_research_not_principal_protected')
    pd.DataFrame([{k:v for k,v in s.items() if k!='years'} for s in summaries]).to_csv(
        folder/'accounts.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(dict(date=calendar, equity=curve)).to_csv(folder/'equity.csv', index=False)
    write_json(folder/'summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source', default=str(ROOT/'stock_cache.db'))
    parser.add_argument('--start', default='2016-09-10')
    parser.add_argument('--end', default='2026-09-10')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0, help='smoke test only; frozen in manifest')
    parser.add_argument('--cached-actions', action='store_true')
    parser.add_argument('--action-snapshot', help='reuse a frozen Eastmoney batch directory (same coverage dates)')
    args = parser.parse_args()
    try:
        validate_new_run(args.start, args.end, args.limit)
    except ValueError as error:
        parser.error(str(error))
    if not 1 <= args.workers <= 16:
        parser.error('invalid workers (expected 1..16)')
    run(args.output, args.source, args.start, args.end, args.workers, args.limit,
        fetch_actions=not args.cached_actions, action_snapshot=args.action_snapshot)


if __name__ == '__main__':
    main()