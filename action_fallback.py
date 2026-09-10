"""Public Eastmoney corporate-action fallback; stdlib only, no DB or Sina I/O.

download_market_snapshot(folder, start, end) freezes whole-market pages (500)
once. collect_action_snapshot(codes, folder, start=..., end=..., primary_dir=...)
matches codes in memory and returns the fetch_action_history result contract.
Valid primary Sina snapshots are returned unchanged. Fallback results are NEW,
content-addressed files under results/, never replacements of primary files.
fetch_action_history(code, folder) is an OFFLINE reader of a complete snapshot.
resolve_action_snapshot has the same batch contract, but cross-checks even
successful Sina snapshots against frozen EM rights. It returns new dictionaries
for a runner to write separately under resolved/<code>.json; never write those
over the original primary snapshots. One market index is built per invocation.

Coverage is inclusive EX_DIVIDEND_DATE, not report year or announcement year
(a December announcement can have a January ex-date). An additional undated
implemented-dividend query prevents silently dropping material unknown dates.
Rights use the entire public RPT_IPO_ALLOTMENT list, then overlap the requested
window with ex/record/subscription/listing dates. Reported rights have details,
are flagged, and are NOT simulated. Missing rights coverage is None + error,
never False. No source list proves historical absence/completeness.

Amounts are PRE-TAX RMB per ten pre-event shares. Profiles cross-check nulls
and units; no dividend tax, cash pay date or bonus listing date is invented.
Pagination is checked, but the remote endpoint is not a transactional or PIT
historical database. A new data version requires a fresh directory. Sequential
HTTP only, timeout 15, NO retries or access-control workarounds. Storage errors
propagate. Raw HTTP/error bytes and URLs/hashes survive parser failures.
"""

from copy import deepcopy
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from urllib.error import HTTPError
from urllib.parse import urlencode
import urllib.request
from uuid import uuid4

from action_snapshot import (_compact, _day, _now, _number, _read_valid_snapshot,
                             _result, _sorted_unique, _validate_code)


ENDPOINT = 'https://datacenter-web.eastmoney.com/api/data/v1/get'
PAGE_SIZE = 500
MAX_PAGES = 256
DIVIDENDS = 'RPT_SHAREBONUS_DET'
RIGHTS = 'RPT_IPO_ALLOTMENT'


class _SourceError(ValueError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _write_new(path, raw):
    """Atomic exclusive publication: even concurrent writers cannot overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f'immutable snapshot mismatch: {path}')
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != raw:
                raise ValueError(f'concurrent snapshot mismatch: {path}')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _spec(start, end):
    start = _day(start, 'start')
    end = _day(end or date.today().isoformat(), 'end')
    if start > end:
        raise ValueError('start must not exceed end')
    return dict(schema_version=1, start=start, end=end, page_size=PAGE_SIZE,
                endpoint=ENDPOINT, dividends_report=DIVIDENDS, rights_report=RIGHTS)


def _queries(spec):
    window = (f"(EX_DIVIDEND_DATE>='{spec['start']}')"
              f"(EX_DIVIDEND_DATE<='{spec['end']}')")
    return {
        'dividends': (DIVIDENDS, window, 'EX_DIVIDEND_DATE,SECURITY_CODE,REPORT_DATE'),
        'undated': (DIVIDENDS, '(EX_DIVIDEND_DATE="NULL")(ASSIGN_PROGRESS="实施分配")',
                    'NOTICE_DATE,SECURITY_CODE,REPORT_DATE'),
        'rights': (RIGHTS, '', 'EX_DIVIDEND_DATE,SECURITY_CODE,FINANCE_CODE'),
    }


def _url(query, page):
    report, filter_, sort = query
    return ENDPOINT + '?' + urlencode(dict(
        reportName=report, columns='ALL', filter=filter_, pageSize=PAGE_SIZE,
        pageNumber=page, sortColumns=sort, sortTypes='1,1,1', source='WEB', client='WEB'))


def _page(raw, page):
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get('success') is not True or value.get('code') != 0:
        raise ValueError(f'API unsuccessful: {str(value)[:200]}')
    result = value.get('result')
    if not isinstance(result, dict):
        raise ValueError('missing result; not proof of empty history')
    count, pages, rows = result.get('count'), result.get('pages'), result.get('data')
    if type(count) is not int or count < 0 or type(pages) is not int:
        raise ValueError('invalid pagination counts')
    expected_pages = math.ceil(count / PAGE_SIZE)
    if pages not in ({0, 1} if count == 0 else {expected_pages}) or pages > MAX_PAGES:
        raise ValueError('inconsistent or excessive page count')
    expected_rows = min(PAGE_SIZE, max(0, count - (page - 1) * PAGE_SIZE))
    if (not isinstance(rows, list) or len(rows) != expected_rows
            or page > max(1, pages) or not all(isinstance(row, dict) for row in rows)):
        raise ValueError('truncated or invalid page')
    for row in rows:
        _validate_code(row.get('SECURITY_CODE'))
    return dict(count=count, pages=pages, rows=rows, version=value.get('version'))


def _read_evidence(folder, evidence):
    digest = evidence['response_sha256']
    if not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None:
        raise ValueError('invalid raw response hash')
    raw = (folder / 'raw' / f'{digest}.json').read_bytes()
    if _sha(raw) != digest:
        raise ValueError('raw response hash mismatch')
    return raw


def _get_page(folder, name, query, page):
    url = _url(query, page)
    cache = folder / 'pages' / f'{name}_{page:04d}.json'
    if cache.exists():
        evidence = json.loads(cache.read_bytes())
        if evidence.get('source_url') != url:
            raise ValueError('cached request mismatch; use a new snapshot directory')
        return _page(_read_evidence(folder, evidence), page), evidence
    evidence = dict(source_url=url, page=page, retrieved_at=_now(), response_sha256=None)
    raw = None
    try:
        with urllib.request.urlopen(urllib.request.Request(
                url, headers={'User-Agent': 'action-snapshot/1.0', 'Accept': 'application/json'}),
                timeout=15) as response:
            raw = response.read()
            evidence['http_status'] = response.status
    except HTTPError as exc:
        raw = exc.read()
        evidence.update(http_status=exc.code, error=f'{type(exc).__name__}: {exc}')
    except (OSError, TimeoutError) as exc:
        evidence['error'] = f'{type(exc).__name__}: {exc}'
    evidence['retrieved_at'] = _now()
    if raw is not None:
        evidence['response_sha256'] = _sha(raw)
        _write_new(folder / 'raw' / f'{_sha(raw)}.json', raw)
    # Every actual HTTP attempt is append-only, including denied/non-JSON bodies.
    _write_new(folder / 'attempts' / f'{uuid4().hex}.json', _json_bytes(evidence))
    if 'error' in evidence:
        raise _SourceError(evidence['error'], evidence)
    try:
        parsed = _page(raw, page)
    except (ValueError, TypeError, KeyError) as exc:
        raise _SourceError(str(exc), evidence) from exc
    _write_new(cache, _json_bytes(evidence))
    return parsed, evidence


def _dataset(folder, name, query):
    result = dict(status='error', pages=[], count=None, error=None)
    seen = set()
    try:
        first, evidence = _get_page(folder, name, query, 1)
        result.update(count=first['count'], expected_pages=first['pages'])
        for page in range(1, max(1, first['pages']) + 1):
            parsed, proof = (first, evidence) if page == 1 else _get_page(folder, name, query, page)
            result['pages'].append(proof)
            if (parsed['count'], parsed['pages']) != (first['count'], first['pages']):
                raise ValueError('pagination changed during download')
            # Overlapping exact rows on different pages imply unstable paging.
            hashes = {_sha(_json_bytes(row)) for row in parsed['rows']}
            if seen & hashes:
                raise ValueError('duplicate rows across pages; pagination not trustworthy')
            seen.update(hashes)
        result.update(status='ok', error=None)
    except (ValueError, KeyError, TypeError) as exc:
        result['error'] = str(exc)
        if isinstance(exc, _SourceError):
            result['failure_evidence'] = exc.evidence
    return result


def download_market_snapshot(output_dir, start='2015-01-01', end=None):
    """Freeze/resume one data version. Failed attempts persist; no HTTP retry.

    Returns a manifest even on source failure. Only an entirely downloaded
    version gets market_manifest.json; failed versions live in manifests/.
    Reusing a complete version never refreshes it. No production DB is opened.
    """
    folder = Path(output_dir)
    spec = _spec(start, end)
    _write_new(folder / 'request.json', _json_bytes(spec))
    completed = folder / 'market_manifest.json'
    if completed.exists():
        manifest = json.loads(completed.read_bytes())
        if manifest['request'] != spec:
            raise ValueError('snapshot window mismatch')
        _index(folder, manifest)  # Check all raw hashes before calling it reusable.
        return manifest
    manifest = dict(request=spec, retrieved_at=_now(), datasets={})
    denied = False
    for name, query in _queries(spec).items():
        if denied:
            dataset = dict(status='error', pages=[], count=None,
                           error='not requested after public endpoint access denial')
        else:
            dataset = _dataset(folder, name, query)
        manifest['datasets'][name] = dataset
        denied = denied or dataset.get('failure_evidence', {}).get('http_status') in (401, 403, 429, 456)
    manifest['retrieved_at'] = _now()
    manifest['status'] = ('ok' if all(d['status'] == 'ok' for d in manifest['datasets'].values())
                          else 'error')
    raw = _json_bytes(manifest)
    _write_new(folder / 'manifests' / f'{_sha(raw)}.json', raw)
    if manifest['status'] == 'ok':
        _write_new(completed, raw)
    return manifest


def _index(folder, manifest):
    index = {name: {} for name in _queries(manifest['request'])}
    for name, dataset in manifest['datasets'].items():
        if dataset['status'] != 'ok':
            continue  # Partial history can never become a valid per-code result.
        total = 0
        expected_pages = max(1, math.ceil(dataset['count'] / PAGE_SIZE))
        if len(dataset['pages']) != expected_pages:
            raise ValueError('manifest missing pages')
        for page, proof in enumerate(dataset['pages'], 1):
            if proof['source_url'] != _url(_queries(manifest['request'])[name], page):
                raise ValueError('manifest source URL mismatch')
            parsed = _page(_read_evidence(folder, proof), page)
            if parsed['count'] != dataset['count']:
                raise ValueError('manifest count mismatch')
            total += len(parsed['rows'])
            for offset, row in enumerate(parsed['rows']):
                index[name].setdefault(row['SECURITY_CODE'], []).append((row, dict(
                    source_url=proof['source_url'], response_sha256=proof['response_sha256'],
                    page=page, row_index=offset, dataset=name)))
        if total != dataset['count']:
            raise ValueError('incomplete manifest rows')
    return index


def _date(value, name, optional=False):
    if optional and (value is None or value in ('', '--')):
        return None
    if isinstance(value, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2} 00:00:00', value):
        value = value[:10]
    return _day(value, name)


def _amount(value, name):
    if isinstance(value, bool) or value is None:
        raise ValueError(f'invalid {name}: {value!r}')
    return _number(str(value), name)


def _dividend_amounts(row):
    """Null means absent ONLY if the explicit ten-share profile proves it."""
    profile = _compact(row.get('IMPL_PLAN_PROFILE') or '')
    main = re.split(r'[（(]', profile, maxsplit=1)[0]
    token = r'(送|转增|转|派)([0-9]+(?:\.[0-9]+)?)(?:股|元)?'
    parsed = {'bonus': 0., 'transfer': 0., 'cash': 0.}
    tokens = re.findall(token, main[2:]) if main.startswith('10') else []
    known = bool(tokens) and re.fullmatch(f'(?:{token})+', main[2:]) is not None
    tolerances = {}
    for kind, text in tokens:
        key = {'送': 'bonus', '转': 'transfer', '转增': 'transfer', '派': 'cash'}[kind]
        if key in tolerances:
            raise ValueError('ambiguous repeated profile component')
        parsed[key] = _amount(text, 'profile amount')
        decimals = len(text.partition('.')[2])
        tolerances[key] = .5 * 10 ** -max(2, decimals) + 1e-10
    values = {}
    for key, field in (('bonus', 'BONUS_RATIO'), ('transfer', 'IT_RATIO'),
                       ('cash', 'PRETAX_BONUS_RMB')):
        value = row[field]  # Absent schema fields are errors, not null components.
        if value is None and known and parsed[key] == 0:
            value = 0.
        values[key] = _amount(value, field)
        if known and abs(values[key] - parsed[key]) > tolerances.get(key, 1e-10):
            raise ValueError(f'{field} conflicts with ten-share profile')
    combined = row['BONUS_IT_RATIO']
    if combined is not None and not math.isclose(
            _amount(combined, 'BONUS_IT_RATIO'), values['bonus'] + values['transfer'],
            rel_tol=1e-9, abs_tol=1e-8):
        raise ValueError('BONUS_IT_RATIO conflicts with bonus + transfer')
    multiplier = 1 + (values['bonus'] + values['transfer']) / 10
    if not math.isfinite(multiplier):
        raise ValueError('share_multiplier must be finite')
    return values['cash'] / 10, multiplier


def _dividend(row, start, end):
    status = _compact(row.get('ASSIGN_PROGRESS') or '')
    if status in ('预案', '董事会预案', '股东大会预案', '股东大会通过', '不分配', '取消分配', '取消'):
        return None
    if status != '实施分配':
        raise ValueError(f'unknown implementation status: {status!r}')
    cash, multiplier = _dividend_amounts(row)
    announce = _date(row['NOTICE_DATE'], 'announce_date')
    record = _date(row['EQUITY_RECORD_DATE'], 'record_date', optional=True)
    ex = _date(row['EX_DIVIDEND_DATE'], 'ex_date', optional=True)
    if ex is None:
        if cash == 0 and multiplier == 1:
            return None
        # Even a very old undated material row is not silently declared outside
        # the window: no actual ex-date is available to prove that assertion.
        raise ValueError('implemented nonzero dividend has no actual ex_date')
    if not start <= ex <= end:
        return None
    if record and record > ex:
        raise ValueError('record_date after ex_date')
    return dict(ex_date=ex, cash_per_share=cash, share_multiplier=multiplier,
                record_date=record, announce_date=announce, pay_date=None)


def _rights(row, start, end):
    fields = dict(ex_date='EX_DIVIDEND_DATE', record_date='EQUITY_RECORD_DATE',
                  subscription_start='PAY_START_DATE', subscription_end='PAY_END_DATE',
                  listing_date='LISTING_DATE', announce_date='FIRST_NOTICE_DATE')
    dates = {key: _date(row[field], key, optional=True) for key, field in fields.items()}
    known = [value for value in dates.values() if value]
    if not known:
        raise ValueError('rights row has no verifiable dates')
    # The entire lifecycle may overlap the window, even with an earlier ex-date.
    if max(known) < start or min(known) > end:
        return None
    if not dates['ex_date']:
        raise ValueError('potential rights issue without actual ex_date')
    ratio = _amount(row['PLACING_RATIO'], 'rights per 10')
    price = _amount(row['ISSUE_PRICE'], 'rights price')
    issued = _amount(row['ISSUE_NUM'], 'actual issued shares')
    if ratio <= 0 or price <= 0 or issued <= 0:
        raise ValueError('rights implementation not established by actual issuance')
    return dict(dates, rights_per_10=ratio, rights_per_share=ratio / 10,
                subscription_price=price, issued_shares=issued,
                finance_code=str(row['FINANCE_CODE']), status='reported_issued',
                implementation_basis='positive ISSUE_NUM and actual EX_DIVIDEND_DATE',
                simulated=False)


def _history(code, manifest, index, *, scan_all_rights=False):
    result = _result(code)
    spec = manifest['request']
    datasets = manifest['datasets']
    result.update(rights_issue_present=None, source_url=_url(_queries(spec)['dividends'], 1))
    pages = datasets['dividends']['pages']
    if pages:
        result['response_sha256'] = pages[0]['response_sha256']
    meta = result['metadata']
    meta.update(source='eastmoney_datacenter', coverage_start=spec['start'], coverage_end=spec['end'],
                history_scope='inclusive ex-date window + ALL undated implemented rows; ALL rights lifecycles',
                snapshot_manifest=f'manifests/{_sha(_json_bytes(manifest))}.json',
                snapshot_manifest_sha256=_sha(_json_bytes(manifest)),
                pagination_complete=manifest['status'] == 'ok',
                snapshot_transactional=False, historical_point_in_time=False,
                response_sha256_basis='first dividend HTTP page; complete page chain in manifest',
                cash_tax_applied=False, bonus_listing_date_verified=False,
                rights_history_status='unknown', rights_events=[], evidence=[])
    meta['notes'] += [
        '东方财富 PRETAX_BONUS_RMB/送转比例均按每10股；已除以10。现金为税前，不模拟差别化红利税。',
        '未提供现金到账日和送转股上市日；配股 PAY_START/END 是认购缴款期，不是现金分红到账日。',
        'False 仅表示完整下载的供应商配股列表在窗口内无记录，不保证真实历史无配股。',
        '分页计数一致不等于供应商历史完整或事务快照；未匹配报价/大跳空，runner 仍须隔离未知跳空。',
    ]
    errors = [f'{name}: {ds["error"]}' for name, ds in datasets.items() if ds['status'] != 'ok']
    rights_errors = []
    if datasets['rights']['status'] == 'ok':
        for row, proof in index['rights'].get(code, []):
            meta['evidence'].append(proof)
            try:
                event = _rights(row, spec['start'], spec['end'])
                if event is not None:
                    meta['rights_events'].append(dict(event, evidence=proof))
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                rights_errors.append(f'rights parse: {exc}')
                if not scan_all_rights:
                    break  # Preserve legacy collect's first-error behavior.
        if not rights_errors:
            result['rights_issue_present'] = bool(meta['rights_events'])
            meta['rights_history_status'] = ('reported_issues' if meta['rights_events']
                                             else 'no_reported_issues_in_window')
        else:
            errors.extend(rights_errors)
            # Keep known positive evidence, but never claim absence on failure.
            result['rights_issue_present'] = True if meta['rights_events'] else None
    else:
        rights_errors.append(f'rights: {datasets["rights"]["error"]}')
    if scan_all_rights:
        meta.update(rights_source_status=datasets['rights']['status'],
                    rights_error='; '.join(rights_errors) or None)
    events = []
    if all(datasets[name]['status'] == 'ok' for name in ('dividends', 'undated')):
        try:
            for name in ('dividends', 'undated'):
                for row, proof in index[name].get(code, []):
                    meta['evidence'].append(proof)
                    event = _dividend(row, spec['start'], spec['end'])
                    if event is not None:
                        events.append(event)
            events = _sorted_unique(events)
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            errors.append(f'dividend parse: {exc}')
    result['retrieved_at'] = manifest['retrieved_at']
    if errors:
        result['error'] = '; '.join(errors)
    else:
        result.update(status='ok', events=events)
    return result


def fetch_action_history(code, snapshot_dir):
    """Offline single-code reader. Download the whole market once, not per code."""
    code = _validate_code(code)
    folder = Path(snapshot_dir)
    manifest = json.loads((folder / 'market_manifest.json').read_bytes())
    return _history(code, manifest, _index(folder, manifest))


def collect_action_snapshot(codes, output_dir, *, start='2015-01-01', end=None, primary_dir=None):
    """Batch API for runner integration; never writes to the primary directory.

    Returns all codes in input order. Fallback bytes go to results/<code>.<hash>
    and an append-only result_manifest.<hash>.json indexes this invocation.
    Primary failures are attached as evidence, never overwritten. No automatic
    invocation is added to the Sina collector, runner, Web or database layer.
    """
    return _collect_action_snapshot(codes, output_dir, start=start, end=end,
                                    primary_dir=primary_dir, resolve_primary=False)


def resolve_action_snapshot(codes, output_dir, *, start='2015-01-01', end=None, primary_dir=None):
    """Collect-compatible results with a market-wide rights check for EVERY code.

    Reuse frozen EM pages without HTTP and build their index exactly once. If no
    completed snapshot exists, freeze/resume one as collect_action_snapshot does.
    Successful Sina cash/events and provenance stay primary; EM dividend errors
    cannot replace them. EM rights unknown => error (None unless either source
    reports True), independently of EM dividend status. False is not proof of
    absence. No dividend agreement or historical completeness is asserted.

    Results/evidence are archived append-only as in collect; the caller writes
    the returned mapping to its independent resolved/<code>.json directory.
    No primary snapshot, runner, database or Sina endpoint is modified/called.
    """
    return _collect_action_snapshot(codes, output_dir, start=start, end=end,
                                    primary_dir=primary_dir, resolve_primary=True)


def _resolve_primary(primary, fallback):
    """Merge into a NEW result; never mutate either source or its metadata."""
    result = deepcopy(primary)
    meta = result.get('metadata')
    if not isinstance(meta, dict):
        meta = {}
        result['metadata'] = meta
    em = fallback['metadata']
    rights_status = em['rights_history_status']
    unknown = rights_status == 'unknown'
    known_true = (primary['rights_issue_present'] is True
                  or fallback['rights_issue_present'] is True)
    resolved = True if known_true else None if unknown else False
    result['rights_issue_present'] = resolved
    crosscheck = dict(
        source='eastmoney_datacenter', rights_status=rights_status,
        source_status=em['rights_source_status'], error=em['rights_error'],
        primary_rights_issue_present=primary['rights_issue_present'],
        eastmoney_rights_issue_present=fallback['rights_issue_present'],
        resolved_rights_issue_present=resolved,
        coverage_start=em['coverage_start'], coverage_end=em['coverage_end'],
        snapshot_manifest=em['snapshot_manifest'],
        snapshot_manifest_sha256=em['snapshot_manifest_sha256'],
        retrieved_at=fallback['retrieved_at'],
        rights_events=deepcopy(em['rights_events']),
        evidence=[deepcopy(proof) for proof in em['evidence'] if proof['dataset'] == 'rights'],
        absence_guaranteed=False)
    meta.update(rights_crosscheck=crosscheck, corporate_actions_reconciled=False,
                cash_events_source='sina_primary',
                cash_events_crosscheck=dict(status='single_source_not_reconciled',
                                           eastmoney_status=fallback['status'],
                                           eastmoney_error=fallback.get('error')))
    if unknown:
        result.update(status='error', error='EM rights crosscheck unknown; see rights_crosscheck evidence')
    return result


def _collect_action_snapshot(codes, output_dir, *, start, end, primary_dir, resolve_primary):
    if isinstance(codes, (str, bytes)):
        raise ValueError('codes must be an iterable of six-digit strings')
    codes = list(dict.fromkeys(_validate_code(code) for code in codes))
    spec = _spec(start, end)
    if not codes:
        return {}
    folder = Path(output_dir)
    primary = Path(primary_dir) if primary_dir is not None else None
    if primary is not None and (folder.resolve() == primary.resolve()
                                or folder.resolve() in primary.resolve().parents):
        raise ValueError('fallback output must not contain the primary directory')
    results, paths = {}, {}
    for code in codes:
        existing = _read_valid_snapshot(primary / f'{code}.json', code) if primary else None
        if existing is not None:
            results[code] = existing
            paths[code] = dict(source='primary_unchanged', path=str(primary / f'{code}.json'))
    pending = codes if resolve_primary else [code for code in codes if code not in results]
    if pending:
        completed = folder / 'market_manifest.json'
        if resolve_primary and completed.exists():
            # download_market_snapshot validates by indexing, so do not call it
            # here and then index again. All raw hashes are checked below once.
            manifest = json.loads(completed.read_bytes())
            if manifest['request'] != spec:
                raise ValueError('snapshot window mismatch')
        else:
            manifest = download_market_snapshot(folder, start, end)
        index = _index(folder, manifest)
        for code in pending:
            result = _history(code, manifest, index, scan_all_rights=resolve_primary)
            existing = results.get(code)
            if resolve_primary and existing is not None:
                result = _resolve_primary(existing, result)
            if primary is not None and (primary / f'{code}.json').exists():
                raw = (primary / f'{code}.json').read_bytes()
                proof_path = f'primary_evidence/{code}.{_sha(raw)}.json'
                _write_new(folder / proof_path, raw)
                result['metadata']['primary_snapshot_evidence'] = dict(
                    path=proof_path, sha256=_sha(raw), original_path=str(primary / f'{code}.json'))
                try:
                    original = json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    original = {}
                if (existing is None and isinstance(original, dict)
                    and 'conflicting' in str(original.get('error', '')).lower()):
                    # A second supplier is evidence, not resolution of an actual
                    # first-supplier conflict. Leave it for explicit reconciliation.
                    result['metadata']['primary_conflict_unresolved'] = True
                    result['metadata']['fallback_candidate_events'] = result['events']
                    result.update(status='error', events=[], error=(result.get('error', '') +
                                  '; unresolved primary conflict: ' + str(original['error'])).lstrip('; '))
            raw = _json_bytes(result)
            path = f'results/{code}.{_sha(raw)}.json'
            _write_new(folder / path, raw)
            results[code] = result
            paths[code] = dict(source='resolved_primary' if existing is not None else 'fallback',
                               path=path, sha256=_sha(raw))
        raw = _json_bytes(paths)
        _write_new(folder / f'result_manifest.{_sha(raw)}.json', raw)
    return {code: results[code] for code in codes}