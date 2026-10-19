"""Independent current cross-sections; never overwrite historical backtests."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
from threading import Event
from uuid import uuid4

import numpy as np

import action_fallback
import live_data
import pareto_web
from live_data import SHANGHAI, _day
from pareto_backtest import aligned_actions, causal_adjusted_prices, write_json
from pareto_strategy import STRATEGY_VERSION, make_objectives


SNAPSHOT_ROOT = Path(os.environ.get('FUSION_SNAPSHOT_DIR',
                     Path(__file__).resolve().parent / 'output' / 'fusion_current'))


def _peer(code, source, folder, history, now, stopped, evidence=None):
    if stopped.is_set():
        return dict(code=code, price_status='error', error='stopped after access denial')
    dest = folder / 'prices' / code
    dest.mkdir(parents=True)
    checked_at = now.isoformat()
    try:
        old = (json.loads((evidence / 'old_target.json').read_bytes()) if evidence else
               live_data._read_target(source, code))
        live_data._json(dest / 'old_target.json', old)
        latest = max((row['date'] for row in old['klines']), default=None)
        failures = []
        for provider, url, unit in live_data._price_sources(code, now.date().isoformat()):
            if stopped.is_set():
                raise ValueError('stopped after access denial')
            try:
                if evidence:
                    proof = json.loads((evidence / 'attempts' / f'price_{provider}.json').read_bytes())
                    if proof.get('http_status') != 200 or 'error' in proof:
                        raise ValueError('saved request unsuccessful')
                    raw = (evidence / proof['body_file']).read_bytes()
                    if live_data._sha(raw) != proof['response_sha256'] or proof['source_url'] != url:
                        raise ValueError('saved price evidence mismatch')
                    checked_at = proof['checked_at']
                    live_data._write(dest / proof['body_file'], raw)
                    live_data._json(dest / 'attempts' / f'price_{provider}.json', proof)
                else:
                    raw = live_data._request_bytes(url, dest, 'price_' + provider, checked_at)
                bars, name = live_data._parse_price(provider, raw, code)
                source_last = max(live_data._day(row['day']) for row in bars)
                if source_last < live_data.START:
                    if latest and source_last < latest:
                        raise ValueError('historical source older than cached latest')
                    result = dict(code=code, price_status='ok', date=source_last, bars=0,
                                  provider=provider, checked_at=checked_at, eligible=False, gate=False,
                                  votes=[0] * 34, action_status=history['status'],
                                  rights_issue_present=history['rights_issue_present'], feature_error=None,
                                  history_only=True)
                    live_data._json(dest / 'feature.json', result)
                    return result
                bars, filtered = live_data._validate_prices(bars, now)
                if latest and bars[-1]['day'] < latest:
                    raise ValueError('provider latest date is older than cached MAX(date)')
                bars, volume = live_data._normalize_volume(bars, old['klines'], unit)
            except live_data._HTTPFailure as error:
                failures.append(dict(provider=provider, error=live_data._safe_error(error)))
                if error.status in live_data.DENIED:
                    stopped.set()
                    raise ValueError('access denied; collection stopped') from None
                continue
            except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
                failures.append(dict(provider=provider, error=live_data._safe_error(error)))
                continue
            live_data._json(dest / 'accepted_price.json', dict(provider=provider,
                response_sha256=live_data._sha(raw), rows=bars, volume_normalization=volume,
                latest_date=bars[-1]['day'], intraday_bar_filtered=filtered, failures=failures))
            break
        else:
            live_data._json(dest / 'failures.json', failures)
            raise ValueError('all price sources failed')
        merged = {row['date']: dict(day=row['date'], **{key: row[key] for key in live_data.FIELDS})
                  for row in old['klines'] if row['date'] >= live_data.START}
        merged.update({row['day']: row for row in bars})
        rows, unused = live_data._validate_prices(list(merged.values()), now)
        live_data._json(dest / 'rows.json', rows)
        dates = np.array([row['day'] for row in rows], dtype='U10')
        raw = np.array([[row[key] for key in live_data.FIELDS] for row in rows])
        result = dict(code=code, date=str(dates[-1]), price_status='ok', provider=provider,
                      checked_at=checked_at, bars=len(rows), eligible=False, gate=False,
                      action_status=history['status'], rights_issue_present=history['rights_issue_present'],
                      feature_error=None, votes=[0] * 34)
        if history['status'] == 'ok' and history['rights_issue_present'] is False:
            try:
                meta = history['metadata']
                if (meta['coverage_start'] > dates[0] or meta['coverage_end'] < dates[-1]
                        or meta['pagination_complete'] is not True):
                    raise ValueError('action coverage does not include prices')
                events = aligned_actions(history['events'], dates)
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    adjusted = causal_adjusted_prices(raw, events)
                objectives = make_objectives(*adjusted.T)
                reference = raw[:-1, 3].copy()
                for position, (cash, multiplier) in events.items():
                    reference[position - 1] = (reference[position - 1] - cash) / multiplier
                jumps = bool(np.any(reference <= 0) or np.any(np.abs(raw[1:, 0] / reference - 1) > .25))
                result.update(votes=objectives['votes'][-1].tolist(),
                              eligible=bool(objectives['eligible'][-1]) and not jumps,
                              gate=bool(objectives['gate_pass'][-1]))
            except (ValueError, TypeError, KeyError, FloatingPointError, OverflowError) as error:
                result['feature_error'] = type(error).__name__
        live_data._json(dest / 'feature.json', result)
        return result
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        result = dict(code=code, price_status='error', error=live_data._safe_error(error))
        live_data._json(dest / 'failure.json', result)
        return result


def build_snapshot(*, root=None, source=None, workers=4, now=None, resume=None):
    root = Path(root or SNAPSHOT_ROOT).resolve()
    source = Path(source or live_data.DB_PATH).resolve()
    now = now or datetime.now(SHANGHAI)
    if now.utcoffset() is None or not 1 <= workers <= 8:
        raise ValueError('aware clock and 1..8 workers required')
    now = now.astimezone(SHANGHAI)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report, manifest, summary, risk, config = pareto_web._load_run()
        universe = sorted(manifest['universe'], key=lambda stock: stock['code'])
        codes = [stock['code'] for stock in universe]
        if len(codes) < 2 or len(codes) != len(set(codes)):
            raise ValueError('a complete comparison universe is required')
        generation = now.strftime('%Y%m%dT%H%M%S_') + uuid4().hex
        previous = None
        if resume:
            if not isinstance(resume, str) or re.fullmatch(r'[0-9T_]+[a-f0-9]{32}', resume) is None:
                raise ValueError('invalid resume generation')
            previous_folder = root / resume
            previous_raw = (previous_folder / 'snapshot.json').read_bytes()
            previous = json.loads(previous_raw)
            pointer = json.loads((root / f'{_day(previous["date"])}.json').read_bytes())
            if pointer['generation'] != resume or pointer['sha256'] != live_data._sha(previous_raw):
                raise ValueError('resume requires latest verified generation')
            previous_request = json.loads((previous_folder / 'request.json').read_bytes())
            if (previous['feature_version'] != STRATEGY_VERSION or previous['complete'] is not True
                    or previous_request['universe'] != universe
                    or datetime.fromisoformat(previous['checked_at']).astimezone(SHANGHAI).date() != now.date()
                    or Path(previous_request['source']) != source):
                raise ValueError('resume requires same source, universe and Shanghai collection date')
            if any(json.loads(path.read_bytes()).get('http_status') in live_data.DENIED
                   for path in (previous_folder / 'prices').glob('*/attempts/*.json')):
                raise ValueError('cannot resume after endpoint access denial')
        folder = root / generation
        folder.mkdir()
        live_data._json(folder / 'request.json', dict(checked_at=now.isoformat(), source=str(source),
            historical_report=str(report), historical_manifest_sha256=live_data._sha(live_data._bytes(manifest)),
            universe=universe, source_read_only=True, previous=resume))
        state = dict(generation=generation, status='collecting_actions', checked_at=now.isoformat(),
                     total=len(codes), completed=0)
        write_json(root / 'status.json', state)
        print(f'Collecting current actions for {len(codes)} stocks', flush=True)
        action_folder = previous_folder / 'actions' if previous else folder / 'actions'
        if previous and not action_folder.exists():
            action_folder = Path(previous['action_folder'])
        if previous:
            action_manifest = json.loads((action_folder / 'market_manifest.json').read_bytes())
            if (action_manifest['status'] != 'ok'
                    or action_manifest['request'] != action_fallback._spec(live_data.START, now.date().isoformat())):
                raise ValueError('action snapshot window mismatch')
            action_index = action_fallback._index(action_folder, action_manifest)
            actions = {code: action_fallback._history(code, action_manifest, action_index, scan_all_rights=True)
                       for code in codes}
        else:
            actions = action_fallback.resolve_action_snapshot(codes, action_folder,
                        start=live_data.START, end=now.date().isoformat())
        if not (action_folder / 'market_manifest.json').is_file():
            state.update(status='failed', error='action snapshot incomplete; no price requests or publication')
            write_json(root / 'status.json', state)
            raise ValueError(state['error'])
        stopped = Event()
        stocks = []
        prior = {row['code']: row for row in previous['stocks']} if previous else {}
        def collect(code):
            if code in prior and prior[code]['price_status'] == 'ok':
                return prior[code]
            if previous:
                evidence = previous_folder / 'prices' / code
                proof_path = evidence / 'attempts' / 'price_sina.json'
                if proof_path.exists():
                    proof = json.loads(proof_path.read_bytes())
                    if proof.get('http_status') == 200 and 'error' not in proof:
                        return _peer(code, source, folder, actions[code], now, stopped, evidence=evidence)
            return _peer(code, source, folder, actions[code], now, stopped)
        state['status'] = 'collecting_prices'
        write_json(root / 'status.json', state)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = pool.map(collect, codes)
            for row in results:
                stocks.append(row)
                state['completed'] = len(stocks)
                if len(stocks) % 50 == 0 or len(stocks) == len(codes):
                    write_json(root / 'status.json', state)
                    print(f'Prices/features: {len(stocks)}/{len(codes)}', flush=True)
        if stopped.is_set():
            state.update(status='failed', error='access denied; stopped without publication')
            write_json(root / 'status.json', state)
            raise ValueError(state['error'])
        dates = [row['date'] for row in stocks if row['price_status'] == 'ok']
        if not dates:
            state.update(status='failed', error='no validated prices; no publication')
            write_json(root / 'status.json', state)
            raise ValueError(state['error'])
        day = max(dates)
        snapshot = dict(schema_version=1, feature_version=STRATEGY_VERSION, date=day,
                        checked_at=previous['checked_at'] if previous else now.isoformat(),
                        completed_at=datetime.now(SHANGHAI).isoformat(),
                        complete=True, actions_complete=True, stocks=stocks,
                        action_folder=str(action_folder), previous=resume)
        raw = live_data._bytes(snapshot)
        live_data._write(folder / 'snapshot.json', raw)
        write_json(root / f'{day}.json', dict(generation=generation, sha256=live_data._sha(raw)))
        failures = sum(row['price_status'] != 'ok' for row in stocks)
        state.update(status='complete', date=day, price_failures=failures,
                     same_day=sum(row.get('date') == day for row in stocks))
        write_json(root / 'status.json', state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
        return state


def compare_current(code, day, vector, universe=None):
    _day(day)
    pointer = SNAPSHOT_ROOT / f'{day}.json'
    if not pointer.exists():
        return None
    index = json.loads(pointer.read_bytes())
    generation = index['generation']
    if not isinstance(generation, str) or re.fullmatch(r'[0-9T_]+[a-f0-9]{32}', generation) is None:
        raise ValueError('invalid snapshot generation')
    raw = (SNAPSHOT_ROOT / generation / 'snapshot.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != index['sha256']:
        raise ValueError('current snapshot hash mismatch')
    snapshot = json.loads(raw)
    checked = datetime.fromisoformat(snapshot['checked_at'])
    if (snapshot['schema_version'] != 1 or snapshot['feature_version'] != STRATEGY_VERSION
            or snapshot['complete'] is not True or snapshot['date'] != day
            or checked.utcoffset() is None or day > checked.astimezone(SHANGHAI).date().isoformat()
            or snapshot['actions_complete'] is not True):
        raise ValueError('incomplete or incompatible current snapshot')
    clock = checked.astimezone(SHANGHAI)
    if day == clock.date().isoformat() and clock.time() < time(15):
        raise ValueError('snapshot includes uncompleted daily bars')
    rows = snapshot['stocks']
    codes = [row['code'] for row in rows]
    if (not codes or codes != sorted(set(codes))
            or any(re.fullmatch(r'[0-9]{6}', peer) is None for peer in codes)
            or universe is not None and codes != sorted(universe)):
        raise ValueError('current snapshot universe mismatch')
    vector = np.asarray(vector)
    if vector.shape != (34,) or vector.dtype.kind not in 'iuf' or not np.isin(vector, [-1, 0, 1]).all():
        raise ValueError('invalid current vector')
    compared, unknown, excluded, dominators = 0, 0, 0, []
    for row in rows:
        if row['code'] == code:
            continue
        if row['price_status'] == 'error':
            unknown += 1
            continue
        if row['price_status'] != 'ok' or _day(row['date']) > day:
            raise ValueError('invalid peer price coverage')
        if row['date'] != day:
            excluded += 1
            continue
        if (type(row['eligible']) is not bool or type(row['gate']) is not bool
                or type(row['bars']) is not int or row['bars'] < 1):
            raise ValueError('invalid peer eligibility')
        if (row['action_status'] != 'ok' or row['rights_issue_present'] is not False
                or row['feature_error'] or not row['eligible'] or not row['gate']):
            excluded += 1
            continue
        other = np.asarray(row['votes'])
        if (row['bars'] < 251 or other.shape != (34,) or other.dtype.kind not in 'iuf'
            or not np.isin(other, [-1, 0, 1]).all()):
            raise ValueError('invalid peer vector')
        compared += 1
        if np.all(other >= vector) and np.any(other > vector):
            dominators.append(row['code'])
    status = 'dominated' if dominators else ('front1' if compared and not unknown else 'unknown')
    return dict(status=status, compared=compared, expected=len(codes) - int(code in codes),
                dominators=dominators, date=day, in_universe=code in codes, source='current',
                checked_at=snapshot['checked_at'], generation=generation,
                completed_at=snapshot.get('completed_at'),
                unavailable=unknown, excluded=excluded,
                coverage=f'独立当前截面：{day}，固定股票池{len(codes)}只；'
                         f'行情采集失败{unknown}只，非同日或未通过资格/门禁/权息{excluded}只；'
                         '不是盘中全市场实时行情，不沿用历史回测前沿')


def main():
    parser = argparse.ArgumentParser(description='Collect a separate current Pareto cross-section; source DB is read-only')
    parser.add_argument('--output', type=Path, default=SNAPSHOT_ROOT)
    parser.add_argument('--source', type=Path, default=Path(live_data.DB_PATH))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', help='Same-day completed generation; preserve successes, retry failed price requests only')
    args = parser.parse_args()
    try:
        build_snapshot(root=args.output, source=args.source, workers=args.workers, resume=args.resume)
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f'Current snapshot unavailable: {error}', flush=True)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()