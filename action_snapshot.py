"""Independent Sina corporate-action snapshots; stdlib only, no database access.

Runner entry points: fetch_action_history(code), collect_action_snapshot(codes,
output_dir, workers=4). Supply a fresh run directory for a new data version; an
existing directory resumes successful snapshots rather than refreshing them.
Do not share a run directory between independent writer processes. Storage
errors propagate: returning success must not imply an unsaved snapshot exists.

Only implemented dividends/bonus shares are parsed. Cash is PRE-TAX yuan per
pre-event share, NOT a verified payment on ex-date. pay_date is always None;
actual cash payment, bonus listing, taxation and price/action reconciliation
belong to the runner. status='ok' means a recognized table parsed successfully,
NOT that the supplier's history is complete or all corporate actions matched.
Implemented rights issues are flagged, never simulated: the runner must mark
such results unreliable. Error results must not be treated as no dividends.

At most TWO total HTTP attempts per code (one retry), timeout=10 each, no sleep.
Malformed responses are not retried. download_action_response returns exact
response-body bytes for callers needing them; normal results contain only their
SHA-256, not large HTML bodies. A hash identifies bytes, not provider accuracy.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import hashlib
import html
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
from urllib.error import HTTPError
import urllib.request


SOURCE_URL = ('https://vip.stock.finance.sina.com.cn/corp/go.php/'
              'vISSUE_ShareBonus/stockid/{code}.phtml')
MAX_ATTEMPTS = 2
_SAVE_LOCK = threading.Lock()
_DATE = re.compile(r'[0-9]{4}-[0-9]{2}-[0-9]{2}\Z')
_NUMBER = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)'
                     r'(?:[eE][+-]?[0-9]+)?\Z')
_MISSING_DATE = {'', '--', '-', '—'}


def _validate_code(code):
    if not isinstance(code, str) or re.fullmatch(r'[0-9]{6}', code.strip()) is None:
        raise ValueError('code must be a six-digit string')
    return code.strip()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _result(code):
    return {
        'code': code, 'status': 'error', 'events': [],
        'rights_issue_present': False, 'source_url': SOURCE_URL.format(code=code),
        'retrieved_at': _now(), 'response_sha256': None,
        'metadata': {
            'schema_version': 1,
            'supplier_history_completeness_guaranteed': False,
            'corporate_actions_reconciled': False,
            'cash_payment_date_verified': False,
            'rights_issue_simulated': False,
            'cash_basis': 'pre_tax_yuan_per_pre_event_share',
            'notes': [
                '不担保供应商历史完整性；尚未与行情、持仓或其他来源匹配权息。',
                'pay_date 未知；除权日不代表现金到账日或送转股实际上市日。',
                '配股仅标记不模拟；runner 应将含配股或采集错误的结果标为不可靠。',
                '哈希对应响应体原始字节，不证明数据正确；无响应时哈希为 null。',
                'retrieved_at 为本次获取尝试结束时间；失败不表示获取到数据。',
            ],
        },
    }


def download_action_response(code):
    """One HTTP attempt returning bytes, without decoding, persistence or retry."""
    code = _validate_code(code)
    request = urllib.request.Request(
        SOURCE_URL.format(code=code),
        headers={'User-Agent': 'Mozilla/5.0',
                 'Referer': 'https://finance.sina.com.cn'},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.read()


def _clean(text):
    return ' '.join(html.unescape(text).split())


def _compact(text):
    return ''.join(text.split())


@dataclass
class _Table:
    table_id: str
    rows: list = field(default_factory=list)
    caption: list = field(default_factory=list)
    headers: list = field(default_factory=list)
    row: object = None
    cell: object = None
    cell_is_header: bool = False
    in_caption: bool = False
    closed: bool = False

    def end_cell(self):
        if self.cell is not None:
            if self.row is None:
                self.row = []
            text = _clean(''.join(self.cell))
            self.row.append(text)
            if self.cell_is_header:
                self.headers.append(text)
            self.cell = None
            self.cell_is_header = False

    def end_row(self):
        self.end_cell()
        if self.row is not None:
            self.rows.append(self.row)
            self.row = None


class _Tables(HTMLParser):
    """Keep nested tables separate; retain empty cells and inline nested text.

    Starting a new row/cell closes an omitted end tag, as common Sina HTML does.
    Parent layout tables never inherit a nested action table's rows.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = []
        self.stack = []
        self.ignored_tag = None

    def handle_starttag(self, tag, attrs):
        if self.ignored_tag:
            return
        if tag in ('script', 'style'):
            self.ignored_tag = tag
        elif tag == 'table':
            table = _Table((dict(attrs).get('id') or '').strip().lower())
            self.tables.append(table)
            self.stack.append(table)
        elif self.stack:
            table = self.stack[-1]
            if tag == 'tr':
                table.end_row()
                table.row = []
            elif tag in ('td', 'th'):
                table.end_cell()
                table.cell = []
                table.cell_is_header = tag == 'th'
            elif tag == 'caption':
                table.in_caption = True
            elif tag == 'br' and table.cell is not None:
                table.cell.append(' ')

    def handle_endtag(self, tag):
        if self.ignored_tag:
            if tag == self.ignored_tag:
                self.ignored_tag = None
            return
        if not self.stack:
            return
        table = self.stack[-1]
        if tag == 'table':
            table.end_row()
            table.closed = True
            self.stack.pop()
        elif tag == 'tr':
            table.end_row()
        elif tag in ('td', 'th'):
            table.end_cell()
        elif tag == 'caption':
            table.in_caption = False

    def handle_data(self, data):
        if self.stack and not self.ignored_tag:
            table = self.stack[-1]
            if table.cell is not None:
                table.cell.append(data)
            elif table.in_caption:
                table.caption.append(data)


def _parse_tables(raw):
    if not isinstance(raw, bytes) or not raw:
        raise ValueError('empty or non-byte response')
    # Sina is normally GBK; also support UTF-8 fixtures/mirrors and entities.
    # Strict decoding avoids silently accepting replacement characters in data.
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        text = raw.decode('gb18030')
    parser = _Tables()
    parser.feed(text)
    parser.close()
    return parser.tables


def _headings(table):
    # TH headings may span several rows. Legacy TD headings are also supported,
    # but unrelated prose elsewhere on the page must not select a dividend table.
    rows = [''.join(row) for row in table.rows
            if any(_compact(cell) in ('公告日期', '除权除息日', '方案进度')
                   for cell in row)]
    return _compact(''.join(table.caption + table.headers + rows))


def _dividend_table(tables):
    primary = [table for table in tables if table.table_id == 'sharebonus_1']
    if primary:
        candidates = primary
    else:
        candidates = []
        for table in tables:
            title = _headings(table)
            if table.table_id == 'sharebonus_2' or '配股' in title:
                continue
            if '派息' in title or ('送股' in title and '除权' in title):
                candidates.append(table)
    if len(candidates) != 1:
        raise ValueError('missing dividend table' if not candidates
                         else 'ambiguous dividend tables')
    if not candidates[0].closed:
        raise ValueError('incomplete dividend table')
    return candidates[0]


def _day(value, name, optional=False):
    if optional and value in _MISSING_DATE:
        return None
    if not isinstance(value, str) or not _DATE.fullmatch(value):
        raise ValueError(f'invalid {name}: {value!r}')
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise ValueError(f'invalid {name}: {value!r}') from exc


def _has_rights(tables):
    for table in tables:
        if table.table_id != 'sharebonus_2' and '配股' not in _headings(table):
            continue
        for row in table.rows:
            if not any(_compact(cell) == '实施' for cell in row):
                continue
            for cell in row:
                try:
                    _day(cell, 'rights date')
                except ValueError:
                    continue
                return True
    return False


def _number(text, name):
    if not _NUMBER.fullmatch(text):
        raise ValueError(f'invalid {name}: {text!r}')
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return value


def _sorted_unique(events):
    by_day = {}
    for event in events:
        day = event['ex_date']
        if day in by_day and by_day[day] != event:
            # Never add together different announcements for the same ex-date.
            raise ValueError(f'conflicting events on ex_date {day}')
        by_day[day] = event
    return [by_day[day] for day in sorted(by_day)]


def _events(table, diagnostics=None):
    events = []
    saw_data = saw_empty = False
    for row in table.rows:
        if not row:
            continue
        nonempty = [_compact(cell) for cell in row if _compact(cell)]
        if nonempty == ['暂时没有数据']:
            saw_empty = True
            continue
        dated = re.match(r'^[0-9]{4}[-/年]', row[0]) is not None
        implemented = any(_compact(cell) == '实施' for cell in row)
        if not dated and not implemented:
            continue
        saw_data = True
        if len(row) < 5 or _compact(row[4]) in ('', '--'):
            raise ValueError('dividend row missing status/columns')
        if implemented and _compact(row[4]) != '实施':
            raise ValueError('implemented dividend row has shifted status/columns')
        if _compact(row[4]) != '实施':
            continue  # Proposals/cancelled plans are not events; no numeric coercion.
        if len(row) < 7:
            raise ValueError('implemented dividend row missing dates/columns')
        bonus = _number(row[1], 'bonus per 10 shares')
        transfer = _number(row[2], 'transfer per 10 shares')
        cash = _number(row[3], 'cash per 10 shares')
        # Some implemented rows describe an explicitly zero distribution. Only
        # those may lack an ex-date; missing/invalid amounts are never zeros.
        if bonus == transfer == cash == 0 and row[5] in _MISSING_DATE:
            _day(row[0], 'announce_date')
            _day(row[6], 'record_date', optional=True)
            if diagnostics is not None:
                name = 'ignored_zero_undated_rows'
                diagnostics[name] = diagnostics.get(name, 0) + 1
            continue
        multiplier = 1 + (bonus + transfer) / 10
        if not math.isfinite(multiplier):
            raise ValueError('share_multiplier must be finite')
        events.append({
            'ex_date': _day(row[5], 'ex_date'),
            'cash_per_share': cash / 10,
            'share_multiplier': multiplier,
            'record_date': _day(row[6], 'record_date', optional=True),
            'announce_date': _day(row[0], 'announce_date'),
            'pay_date': None,  # Sina column 7 is BONUS LISTING, not cash payment.
        })
    if not saw_data and not saw_empty:
        raise ValueError('dividend table has no recognized data or explicit no-data row')
    if saw_data and saw_empty:
        raise ValueError('dividend table mixes data and no-data marker')
    return _sorted_unique(events)


def fetch_action_history(code) -> dict:
    """Return JSON-safe data/error with provenance; invalid code raises ValueError.

    No partial events escape on parse/conflict errors. A received malformed body
    still has its byte hash, while a transport failure without a body has None.
    """
    code = _validate_code(code)
    result = _result(code)
    for attempt in range(MAX_ATTEMPTS):
        try:
            raw = download_action_response(code)
        except Exception as exc:
            result['retrieved_at'] = _now()
            result['error'] = f'download attempt {attempt + 1}/{MAX_ATTEMPTS}: {type(exc).__name__}: {exc}'
            if isinstance(exc, HTTPError) and exc.code in (401, 403, 429, 456):
                result['metadata']['http_status'] = exc.code
                result['metadata']['access_denial_not_retried'] = True
                break
            continue
        result['retrieved_at'] = _now()
        if isinstance(raw, bytes):
            result['response_sha256'] = hashlib.sha256(raw).hexdigest()
        try:
            tables = _parse_tables(raw)
            result['rights_issue_present'] = _has_rights(tables)
            result['events'] = _events(_dividend_table(tables), result['metadata'])
        except (ValueError, OverflowError) as exc:
            result['error'] = f'parse: {exc}'
            return result
        result['status'] = 'ok'
        result.pop('error', None)
        return result
    return result


def _read_valid_snapshot(path, code):
    """Only reuse complete, internally valid successful snapshots for this code."""
    try:
        with path.open(encoding='utf-8') as stream:
            value = json.load(stream)
        if not isinstance(value, dict) or value.get('code') != code or value.get('status') != 'ok':
            return None
        if value.get('source_url') != SOURCE_URL.format(code=code):
            return None
        if type(value.get('rights_issue_present')) is not bool:
            return None
        if not isinstance(value.get('response_sha256'), str) or not re.fullmatch(
                r'[0-9a-f]{64}', value['response_sha256']):
            return None
        timestamp = datetime.fromisoformat(value['retrieved_at'].replace('Z', '+00:00'))
        if timestamp.utcoffset() is None:
            return None
        events = value['events']
        if not isinstance(events, list):
            return None
        for event in events:
            if not isinstance(event, dict):
                return None
            for name in ('ex_date', 'announce_date', 'record_date'):
                if name == 'record_date' and event[name] is None:
                    continue
                _day(event[name], name)
            if event['pay_date'] is not None:
                return None
            for name, minimum in (('cash_per_share', 0), ('share_multiplier', 1)):
                number = event[name]
                if type(number) not in (int, float) or not math.isfinite(number) or number < minimum:
                    return None
        if events != _sorted_unique(events):
            return None
        json.dumps(value, allow_nan=False)
        return value
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
        return None


def _save_snapshot(path, result):
    # Serialize writers within this process, including concurrent collector calls.
    with _SAVE_LOCK:
        existing = _read_valid_snapshot(path, result['code'])
        if existing is not None:
            return existing
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode='w', encoding='utf-8', dir=path.parent,
                    prefix=f'.{result["code"]}.', suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return result


def collect_action_snapshot(codes, output_dir, workers=4) -> dict:
    """Save <code>.json atomically in the caller's run folder, including failures.

    Successful valid files are reused byte-for-byte. Failed/invalid files may be
    replaced on resume. Codes are deduplicated before dispatch, workers capped at
    eight, and progress is printed for every 100 completed codes (resumes count).
    Returns all results keyed by normalized code, in input order. Does not touch
    stockcache or import any production data-access module.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer')
    if isinstance(codes, (str, bytes)):
        raise ValueError('codes must be an iterable of six-digit strings')
    codes = list(dict.fromkeys(_validate_code(code) for code in codes))
    if not codes:
        return {}
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    def collect_one(code):
        path = directory / f'{code}.json'
        existing = _read_valid_snapshot(path, code)
        if existing is not None:
            return existing
        try:
            result = fetch_action_history(code)
        except Exception as exc:
            result = _result(code)
            result['error'] = f'collector: {type(exc).__name__}: {exc}'
        return _save_snapshot(path, result)

    results = {}
    with ThreadPoolExecutor(max_workers=min(workers, 8)) as executor:
        futures = {executor.submit(collect_one, code): code for code in codes}
        for completed, future in enumerate(as_completed(futures), 1):
            results[futures[future]] = future.result()
            if completed % 100 == 0:
                print(f'action snapshots: {completed}/{len(codes)}', flush=True)
    return {code: results[code] for code in codes}