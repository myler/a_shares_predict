"""Fresh single-stock daily data, independent of legacy fetch/cache and backtests.

Public API: refresh_stock(code, *, now=None, db_path=None, snapshot_root=None).
The DB MUST already exist with db.py's klines and stocks schema. No initializer,
DELETE, legacy dividends write, whole-DB backup or frozen-backtest access occurs.
Every call uses a new evidence directory and real HTTP (unless mocked by tests).
All price sources failing raises RefreshError, NEVER returns cached prices.

Return contract (JSON-serializable):
  rows: complete committed target history >=2015-01-01, ascending dictionaries
        with day/open/high/low/close/volume; numeric values are floats, raw prices.
  name: cached/provider name, or None (no extra name request).
  checked_at: request clock, timezone-aware ISO; NOT a quote timestamp.
  provider: sina | tencent | eastmoney, the successful PRICE provider only.
  latest_date, date: same actual latest completed daily-bar date, NOT checked_at.
  actions: action-history dict with status/events/rights_issue_present/metadata,
           code/retrieved_at/source_url/response_sha256 and optional error.
  refresh_id, evidence_dir: unique ID and absolute evidence directory.
  warnings: strings; callers must display age/quality caveats, not invent freshness.
An action error does not undo committed prices. Callers MUST block live advice
when actions.status != 'ok', rights_issue_present is not False, or
actions.metadata.live_recommendation_blocked is true. This is a data gate, not a
recommendation engine or Web integration; passing it does not establish safety.

Source adapter interface:
  _price_sources(code, asof) -> ordered (provider, HTTPS URL, native volume unit).
  _parse_price(provider, raw_bytes, code) -> (unvalidated bar dicts, name | None).
  _request_bytes(...) -> exact body bytes, one attempt, frozen before parsing.
Sina daily JSON uses named OHLC fields, volume=shares. Tencent uses ONLY `day`
(never qfqday/hfqday): date/open/close/high/low/volume, volume=lots of 100 shares.
EM klt=101/fqt=0: f51 date, f52 open, f53 close, f54 high, f55 low, f56 lots.
No history: normalize known native units to shares. With history: require >=3
positive overlapping historical volume pairs, each within 1% of ONE of
cached/provider ratios 1, 100, .01. Exclude cached latest day from calibration
because same-day corrections may change volume; zero/zero pairs give no proof.
Unknown/mixed/insufficient units reject that source, rather than splice volumes.
The factor and calibration days are frozen; existing historical units are not
relabeled as shares. Earlier history outside the response is preserved verbatim.

Aware `now` is converted to Asia/Shanghai; naive times are rejected. Before
15:00, today's bar is removed; future dates, nonfinite/invalid OHLCV and conflicting
duplicate dates reject the source. Provider latest must be >= actual cached MAX,
not stocks.last_kline_date. Same-date online verification/correction is allowed,
but is not called a newer quote. No exchange holiday/suspension calendar assumed.

Actions: fresh per-code EM datacenter queries on EVERY successful price refresh:
inclusive ex-date 2015-01-01..Shanghai asof, ALL undated implemented dividends,
ALL rights lifecycles (no date filter). pageSize=500, max256 pages/query, checked
counts and nonoverlapping pages. Explicit count=0 permits data=None; a missing
result/count or API failure otherwise never means empty. One EM provider-schema
exception: on page 1 only, integer code=9201, success=false, explicit result=null
and exact message="返回数据为空" denote a source-empty result, normalized to
count=0/pages=0/rows=[]. This is not a transport failure; retain the raw response
and normal page proof without another retrieval. Other errors/mismatches and
later-page empty sentinels reject. Empty queries (including rights=False) mean
only no source-reported records, not proof of historical absence.
Uses action_fallback's event
parsers, never its cached market snapshots or Sina requests. Endpoint access
denial stops remaining action requests; no retry/backoff/access workaround.
Coverage is checked against quote date; cash pay dates, bonus availability and
dividend tax remain unverified holding-account approximations, disclosed in meta.

Evidence: target-only old rows, raw bodies + SHA256, per-attempt metadata,
normalized accepted prices, actions and result under output/live_refresh/<id>.
Writes use exclusive new files. Read old DB via mode=ro; after network validation
BEGIN IMMEDIATE rechecks target for concurrent changes, upserts only that code,
sets stocks.last_kline_date using SELECT MAX, commits, then reads committed rows.
RefreshError exposes refresh_id/evidence_dir/price_committed. A post-commit I/O
failure cannot undo a completed commit. Public errors omit exception strings,
response contents and remote URLs; exact raw bodies remain local evidence only.
"""

from contextlib import closing
from datetime import date as Date, datetime, time
import hashlib
from http.client import HTTPException, IncompleteRead
import json
import math
from pathlib import Path
import re
import sqlite3
from urllib.error import HTTPError
from urllib.parse import urlencode
import urllib.request
from uuid import uuid4
from zoneinfo import ZoneInfo

from action_fallback import _dividend, _rights, _sorted_unique
from db import DB_PATH
from fetcher import _parse_tencent_day_klines


START = '2015-01-01'
SHANGHAI = ZoneInfo('Asia/Shanghai')
SNAPSHOT_ROOT = Path(__file__).resolve().parent / 'output' / 'live_refresh'
ACTION_ENDPOINT = 'https://datacenter-web.eastmoney.com/api/data/v1/get'
PAGE_SIZE = 500
MAX_PAGES = 256
DENIED = {401, 403, 429, 456}
FIELDS = ('open', 'high', 'low', 'close', 'volume')


class RefreshError(RuntimeError):
    """No usable refresh; inspect local evidence, never substitute stale cache."""

    def __init__(self, message, *, refresh_id=None, evidence_dir=None,
                 price_committed=False):
        super().__init__(message)
        self.refresh_id = refresh_id
        self.evidence_dir = str(evidence_dir) if evidence_dir else None
        self.price_committed = price_committed


class _Invalid(ValueError):
    """Internal safe, constant validation messages (no provider values)."""


class _HTTPFailure(Exception):
    def __init__(self, kind, status=None):
        self.kind, self.status = kind, status
        super().__init__(f'HTTP {status}' if status is not None else kind)


def _safe_error(exc):
    if isinstance(exc, (_Invalid, _HTTPFailure)):
        return str(exc)
    return type(exc).__name__  # Never echo URL, API message or exception args.


def _bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)
            + '\n').encode('utf-8')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)


def _json(path, value):
    _write(path, _bytes(value))


def _request_bytes(url, folder, label, checked_at):
    """Exactly one public HTTP attempt; persistence failures propagate."""
    proof = dict(source_url=url, checked_at=checked_at, response_sha256=None,
                 http_status=None, label=label)
    raw, failure = None, None
    request = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json',
        'Referer': 'https://' + urllib.parse.urlsplit(url).netloc + '/',
    })
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            proof['http_status'] = response.status
            raw = response.read()
            if response.status != 200:
                failure = _HTTPFailure('HTTPError', response.status)
    except HTTPError as exc:
        proof['http_status'] = exc.code
        failure = _HTTPFailure('HTTPError', exc.code)
        try:
            raw = exc.read()
        except (OSError, HTTPException) as read_error:
            if isinstance(read_error, IncompleteRead):
                raw = read_error.partial
        finally:
            exc.close()
    except (OSError, HTTPException) as exc:
        failure = _HTTPFailure(type(exc).__name__)
        if isinstance(exc, IncompleteRead):
            raw = exc.partial
    if raw is not None:
        proof['response_sha256'] = _sha(raw)
        proof['body_file'] = f'raw/{label}.{_sha(raw)}.body'
        _write(folder / proof['body_file'], raw)
    if failure:
        proof['error'] = str(failure)
    _json(folder / 'attempts' / f'{label}.json', proof)
    if failure:
        raise failure
    return raw


def _price_sources(code, asof):
    symbol = ('sh' if code.startswith('6') else 'sz') + code
    return (
        ('sina', 'https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/'
         'CN_MarketData.getKLineData?' + urlencode(dict(
             symbol=symbol, scale=240, ma='no', datalen=2500)), 'shares'),
        ('tencent', 'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?' +
         urlencode(dict(param=f'{symbol},day,,,2500')), 'lots'),
        ('eastmoney', 'https://push2his.eastmoney.com/api/qt/stock/kline/get?' +
         urlencode(dict(secid=f'{1 if code.startswith("6") else 0}.{code}',
                        fields1='f1,f2,f3,f4,f5,f6',
                        fields2='f51,f52,f53,f54,f55,f56', klt=101, fqt=0,
                        beg='20150101', end=asof.replace('-', ''), lmt=10000)), 'lots'),
    )


def _name(value):
    return value.strip() if isinstance(value, str) and value.strip() else None


def _parse_price(provider, raw, code):
    if provider not in ('sina', 'tencent', 'eastmoney'):
        raise _Invalid('unknown price provider')
    payload = json.loads(raw)
    symbol = ('sh' if code.startswith('6') else 'sz') + code
    if provider == 'sina':
        if not isinstance(payload, list) or not payload:
            raise _Invalid('empty or malformed Sina daily data')
        return payload, None
    if provider == 'tencent':
        if not isinstance(payload, dict) or payload.get('code', 0) != 0:
            raise _Invalid('unsuccessful Tencent response')
        rows = _parse_tencent_day_klines(payload, symbol)
        stock = payload['data'][symbol]
        qt = stock.get('qt', {})
        quote = qt.get(symbol) if isinstance(qt, dict) else None
        name = _name(quote[1]) if isinstance(quote, list) and len(quote) > 1 else None
        return rows, name
    if not isinstance(payload, dict) or payload.get('rc', 0) != 0:
        raise _Invalid('unsuccessful Eastmoney response')
    data = payload.get('data')
    if not isinstance(data, dict) or data.get('code', code) != code:
        raise _Invalid('missing or wrong-code Eastmoney data')
    lines = data.get('klines')
    if not isinstance(lines, list) or not lines:
        raise _Invalid('empty Eastmoney daily data')
    rows = []
    for line in lines:
        if not isinstance(line, str) or len(line.split(',')) < 6:
            raise _Invalid('malformed Eastmoney daily row')
        day, open_, close, high, low, volume = line.split(',')[:6]
        rows.append(dict(day=day, open=open_, high=high, low=low,
                         close=close, volume=volume))
    return rows, _name(data.get('name'))


def _day(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', value):
        raise _Invalid('invalid daily date')
    try:
        Date.fromisoformat(value)
    except ValueError:
        raise _Invalid('invalid daily date') from None
    return value


def _number(value):
    if isinstance(value, bool):
        raise _Invalid('invalid OHLCV number')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise _Invalid('invalid OHLCV number') from None
    if not math.isfinite(result):
        raise _Invalid('nonfinite OHLCV number')
    return result


def _validate_prices(rows, now):
    today = now.date().isoformat()
    by_day, filtered = {}, False
    for row in rows:
        if not isinstance(row, dict):
            raise _Invalid('daily row must be an object')
        day = _day(row.get('day'))
        if day > today:
            raise _Invalid('future daily bar')
        values = {key: _number(row.get(key)) for key in FIELDS}
        if (any(values[key] <= 0 for key in FIELDS[:-1]) or values['volume'] < 0
                or values['high'] < max(values['open'], values['low'], values['close'])
                or values['low'] > min(values['open'], values['high'], values['close'])):
            raise _Invalid('invalid OHLC range or volume')
        normalized = dict(day=day, **values)
        if day in by_day and by_day[day] != normalized:
            raise _Invalid('conflicting duplicate daily bars')
        by_day[day] = normalized
    result = []
    for day, row in sorted(by_day.items()):
        if day == today and now.time() < time(15):
            filtered = True
        elif day >= START:
            result.append(row)
    if not result:
        raise _Invalid('no completed daily bars in requested history')
    return result, filtered


def _normalize_volume(rows, old, native_unit):
    if native_unit not in ('shares', 'lots'):
        raise _Invalid('unknown provider volume unit')
    if not old:
        factor = 1. if native_unit == 'shares' else 100.
        proof = dict(factor=factor, native_unit=native_unit, output_unit='shares',
                     basis='known provider units; no cached history', overlap_days=[])
    else:
        cached = {row['date']: row for row in old}
        latest = max(cached)
        ratios, days = [], []
        for row in rows:
            day = row['day']
            if day not in cached or day == latest:
                continue
            prior, incoming = _number(cached[day]['volume']), row['volume']
            if prior == incoming == 0:
                continue
            if prior <= 0 or incoming <= 0:
                raise _Invalid('inconsistent overlapping volume')
            ratios.append(prior / incoming)
            days.append(day)
        if len(ratios) < 3:
            raise _Invalid('insufficient overlap to establish volume units')
        candidates = [factor for factor in (1., 100., .01)
                      if all(math.isclose(ratio, factor, rel_tol=.01)
                             for ratio in ratios)]
        if len(candidates) != 1:
            raise _Invalid('ambiguous or mixed overlapping volume units')
        factor = candidates[0]
        proof = dict(factor=factor, native_unit=native_unit, output_unit='cached_units',
                     basis='stable cached/provider overlap ratios', overlap_days=days,
                     ratios=ratios, excluded_latest_day=latest, relative_tolerance=.01)
    normalized = [dict(row, volume=_number(row['volume'] * factor)) for row in rows]
    return normalized, proof


def _target(conn, code):
    rows = [dict(row) for row in conn.execute(
        'SELECT code,date,open,high,low,close,volume FROM klines WHERE code=? ORDER BY date',
        (code,))]
    stock = conn.execute(
        'SELECT code,name,last_kline_date,last_div_date FROM stocks WHERE code=?',
        (code,)).fetchone()
    return dict(klines=rows, stock=dict(stock) if stock is not None else None)


def _read_target(path, code):
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA query_only=ON')
        conn.execute('BEGIN')
        return _target(conn, code)


def _commit_prices(path, code, old, rows, name):
    with closing(sqlite3.connect(path.as_uri() + '?mode=rw', uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        with conn:
            conn.execute('BEGIN IMMEDIATE')
            if _target(conn, code) != old:
                raise _Invalid('target changed concurrently; start a new refresh')
            conn.executemany(
                'INSERT INTO klines(code,date,open,high,low,close,volume) VALUES(?,?,?,?,?,?,?) '
                'ON CONFLICT(code,date) DO UPDATE SET open=excluded.open,high=excluded.high,'
                'low=excluded.low,close=excluded.close,volume=excluded.volume',
                [(code, row['day'], *(row[key] for key in FIELDS)) for row in rows])
            conn.execute(
                'INSERT INTO stocks(code,name,last_kline_date) '
                'VALUES(?,?,(SELECT MAX(date) FROM klines WHERE code=?)) '
                'ON CONFLICT(code) DO UPDATE SET name=COALESCE(excluded.name,stocks.name),'
                'last_kline_date=excluded.last_kline_date', (code, name, code))
            expected = _target(conn, code)
        return expected  # Transaction has successfully COMMITTED here.


def _action_queries(code, asof):
    code_filter = f'(SECURITY_CODE="{code}")'
    return (
        ('dividends', 'RPT_SHAREBONUS_DET', code_filter +
         f"(EX_DIVIDEND_DATE>='{START}')(EX_DIVIDEND_DATE<='{asof}')",
         'EX_DIVIDEND_DATE,SECURITY_CODE,REPORT_DATE'),
        ('undated', 'RPT_SHAREBONUS_DET', code_filter +
         '(EX_DIVIDEND_DATE="NULL")(ASSIGN_PROGRESS="实施分配")',
         'NOTICE_DATE,SECURITY_CODE,REPORT_DATE'),
        ('rights', 'RPT_IPO_ALLOTMENT', code_filter,
         'EX_DIVIDEND_DATE,SECURITY_CODE,FINANCE_CODE'),
    )


def _action_page(raw, number, code):
    payload = json.loads(raw)
    if (number == 1 and isinstance(payload, dict)
            and type(payload.get('code')) is int and payload['code'] == 9201
            and payload.get('success') is False
            and 'result' in payload and payload['result'] is None
            and payload.get('message') == '返回数据为空'):
        return 0, 0, []  # Exact EM source-empty sentinel, never a mid-page fallback.
    if (not isinstance(payload, dict) or payload.get('success') is not True
            or type(payload.get('code')) is not int or payload['code'] != 0):
        raise _Invalid('unsuccessful action API response')
    result = payload.get('result')
    if not isinstance(result, dict):
        raise _Invalid('missing action result; empty history not established')
    count, pages, rows = result.get('count'), result.get('pages'), result.get('data')
    if type(count) is not int or count < 0 or type(pages) is not int:
        raise _Invalid('missing or invalid action pagination counts')
    if pages not in ({0, 1} if count == 0 else {math.ceil(count / PAGE_SIZE)}):
        raise _Invalid('inconsistent action pagination counts')
    if pages > MAX_PAGES or not 1 <= number <= max(1, pages):
        raise _Invalid('action page limit exceeded')
    if count == 0 and rows is None:
        rows = []  # Only an explicit zero proves a valid empty code query.
    expected = min(PAGE_SIZE, max(0, count - (number - 1) * PAGE_SIZE))
    if not isinstance(rows, list) or len(rows) != expected:
        raise _Invalid('truncated or malformed action page')
    if any(not isinstance(row, dict) or row.get('SECURITY_CODE') != code for row in rows):
        raise _Invalid('wrong-code or malformed action row')
    return count, pages, rows


def _action_dataset(code, query, folder, checked_at):
    """Keep normal raw-body proofs even for the first-page source-empty sentinel."""
    name, report, filter_, sort = query
    result, seen, first, proofs = [], set(), None, []
    number = 1
    while True:
        url = ACTION_ENDPOINT + '?' + urlencode(dict(
            reportName=report, columns='ALL', filter=filter_, pageSize=PAGE_SIZE,
            pageNumber=number, sortColumns=sort, sortTypes='1,1,1', source='WEB', client='WEB'))
        label = f'actions_{name}_{number:04d}'
        raw = _request_bytes(url, folder, label, checked_at)
        count, pages, rows = _action_page(raw, number, code)
        if first is None:
            first = (count, pages)
        elif first != (count, pages):
            raise _Invalid('action pagination changed during request')
        hashes = [_sha(_bytes(row)) for row in rows]
        if seen.intersection(hashes) or len(set(hashes)) != len(hashes):
            raise _Invalid('duplicate action pagination rows')
        seen.update(hashes)
        result.extend(rows)
        proofs.append(dict(source_url=url, response_sha256=_sha(raw),
                           attempt=f'attempts/{label}.json', page=number))
        if number >= max(1, pages):
            break
        number += 1
    return result, dict(status='ok', count=first[0], pages=proofs)


def _fetch_actions(code, asof, latest, folder, checked_at):
    meta = dict(schema_version=1, source='eastmoney_datacenter', coverage_start=START,
                coverage_end=asof, quote_date=latest, coverage_matches_quote=False,
                history_scope='inclusive ex-date window + ALL undated implemented + ALL rights lifecycles',
                pagination_complete=False, datasets={}, rights_history_status='unknown',
                rights_events=[], live_recommendation_blocked=True,
                supplier_history_completeness_guaranteed=False,
                corporate_actions_reconciled=False, historical_point_in_time=False,
                snapshot_transactional=False, cash_payment_date_verified=False,
                bonus_listing_date_verified=False, cash_tax_applied=False,
                rights_issue_simulated=False, cash_basis='pre_tax_yuan_per_pre_event_share',
                notes=[
                    '现金分红为税前元/权息前股；除权日不等于现金到账日。',
                    '持仓核算若在除权日计现金、立即使用送转股、忽略红利税均为近似。',
                    '配股未模拟；未知/存在配股或权息错误必须阻止实时建议。',
                    '完整分页不保证供应商历史完整；未逐笔核对价格、权息与持仓。',
                    '空结果仅表示本次供应商查询无记录，不证明历史上不存在权息/配股。',
                ])
    result = dict(code=code, status='error', events=[], rights_issue_present=None,
                  metadata=meta, retrieved_at=checked_at, source_url=None,
                  response_sha256=None)
    datasets, errors, denied = {}, [], False
    for query in _action_queries(code, asof):
        name = query[0]
        if denied:
            meta['datasets'][name] = dict(status='error', error='not requested after endpoint access denial')
            errors.append(f'{name}: not requested after endpoint access denial')
            continue
        try:
            rows, proof = _action_dataset(code, query, folder, checked_at)
            datasets[name] = rows
            meta['datasets'][name] = proof
            if name == 'dividends':
                result.update(source_url=proof['pages'][0]['source_url'],
                              response_sha256=proof['pages'][0]['response_sha256'])
        except (_HTTPFailure, ValueError, TypeError, KeyError, OverflowError) as exc:
            error = _safe_error(exc)
            meta['datasets'][name] = dict(status='error', error=error)
            errors.append(f'{name}: {error}')
            denied = isinstance(exc, _HTTPFailure) and exc.status in DENIED
    meta['pagination_complete'] = len(datasets) == 3
    if 'rights' in datasets:
        try:
            rights = [event for row in datasets['rights']
                      if (event := _rights(row, START, asof)) is not None]
            meta['rights_events'] = rights
            result['rights_issue_present'] = bool(rights)
            meta['rights_history_status'] = 'reported_issues' if rights else 'no_reported_issues_in_window'
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            errors.append(f'rights parse: {_safe_error(exc)}')
    events = []
    if all(name in datasets for name in ('dividends', 'undated')):
        try:
            for name in ('dividends', 'undated'):
                for row in datasets[name]:
                    event = _dividend(row, START, asof)
                    if event is not None:
                        events.append(event)
            events = _sorted_unique(events)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            errors.append(f'dividend parse: {_safe_error(exc)}')
    if not START <= latest <= asof:
        errors.append('action coverage does not include quote date')
    if result['rights_issue_present'] is None and not errors:
        errors.append('rights history remains unknown')
    if errors:
        result['error'] = '; '.join(errors)
    else:
        result.update(status='ok', events=events)
        meta['coverage_matches_quote'] = True
        meta['live_recommendation_blocked'] = result['rights_issue_present'] is not False
    _json(folder / 'actions.json', result)
    return result


def refresh_stock(code, *, now=None, db_path=None, snapshot_root=None) -> dict:
    """Fetch -> validate/freeze -> targeted transaction -> full rows -> actions.

    ValueError: invalid caller code/clock. RefreshError: unavailable DB/schema,
    all sources invalid/unavailable, concurrent target edit, or persistence error.
    Action-only failures are returned with status='error' and the blocking gate.
    """
    if not isinstance(code, str) or not re.fullmatch(r'[0-9]{6}', code.strip()):
        raise ValueError('code must be a six-digit string')
    code = code.strip()
    now = datetime.now(SHANGHAI) if now is None else now
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('now must be a timezone-aware datetime')
    now = now.astimezone(SHANGHAI)
    checked_at, asof = now.isoformat(), now.date().isoformat()
    if asof < START:
        raise ValueError('now must be on or after 2015-01-01')
    path = Path(DB_PATH if db_path is None else db_path).resolve()
    root = Path(SNAPSHOT_ROOT if snapshot_root is None else snapshot_root).resolve()
    refresh_id = f'{now.strftime("%Y%m%dT%H%M%S%f")}_{code}_{uuid4().hex}'
    folder, committed = root / refresh_id, False
    try:
        folder.mkdir(parents=True, exist_ok=False)
        old = _read_target(path, code)
        old_raw = _bytes(old)
        _write(folder / 'old_target.json', old_raw)
        _json(folder / 'request.json', dict(code=code, checked_at=checked_at, asof=asof,
              start=START, refresh_id=refresh_id, old_target_sha256=_sha(old_raw)))
        cached_latest = max((row['date'] for row in old['klines']), default=None)
        warnings, failures = [], []
        for provider, url, unit in _price_sources(code, asof):
            try:
                raw = _request_bytes(url, folder, f'price_{provider}', checked_at)
                bars, provider_name = _parse_price(provider, raw, code)
                bars, filtered = _validate_prices(bars, now)
                if cached_latest and bars[-1]['day'] < cached_latest:
                    raise _Invalid('provider latest date is older than cached MAX(date)')
                bars, volume = _normalize_volume(bars, old['klines'], unit)
            except (_HTTPFailure, ValueError, TypeError, KeyError, OverflowError) as exc:
                error = _safe_error(exc)
                failures.append(dict(provider=provider, error=error))
                _json(folder / f'rejected_{provider}.json', failures[-1])
                continue
            _json(folder / 'accepted_price.json', dict(provider=provider,
                  response_sha256=_sha(raw), rows=bars, volume_normalization=volume,
                  latest_date=bars[-1]['day'], intraday_bar_filtered=filtered))
            if filtered:
                warnings.append('已过滤上海时间15:00前的当日临时日线，不将盘中价格当作收盘价。')
            break
        else:
            raise _Invalid('all price sources failed; stale fallback forbidden')
        name = _name((old['stock'] or {}).get('name')) or provider_name
        expected = _commit_prices(path, code, old, bars, name)
        committed = True
        loaded = _read_target(path, code)  # Only after successful commit, full history.
        if loaded != expected:
            raise _Invalid('target changed after commit; start a new refresh')
        rows = [dict(day=row['date'], **{key: row[key] for key in FIELDS})
                for row in loaded['klines'] if row['date'] >= START]
        latest = rows[-1]['day']
        warnings.extend(f'{failure["provider"]}: {failure["error"]}' for failure in failures)
        if latest < asof:
            warnings.append(f'本次已联网核验；最新完整日线为{latest}，并非核验日{asof}行情；未验证交易日历/停牌。')
        if cached_latest == latest:
            warnings.append('供应商最新日期与缓存相同：本次为在线核验/同日修正，不是新增交易日。')
        actions = _fetch_actions(code, asof, latest, folder, checked_at)
        if actions['metadata']['live_recommendation_blocked']:
            warnings.append('权息核验失败或存在未建模配股，禁止生成实时买卖建议。')
        warnings.extend(actions['metadata']['notes'])
        result = dict(rows=rows, name=name, checked_at=checked_at, provider=provider,
                      latest_date=latest, date=latest, actions=actions, refresh_id=refresh_id,
                      evidence_dir=str(folder), warnings=warnings)
        _json(folder / 'result.json', result)
        return result
    except (OSError, sqlite3.Error, _Invalid, ValueError, TypeError, KeyError) as exc:
        error = _safe_error(exc)
        try:
            if folder.is_dir():
                _json(folder / 'failure.json', dict(error=error, price_committed=committed))
        except OSError:
            pass
        raise RefreshError(error, refresh_id=refresh_id, evidence_dir=folder,
                           price_committed=committed) from None