"""Offline live-refresh contracts: temporary full schemas, mock HTTP, socket guard."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from http.client import IncompleteRead
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, unquote, urlsplit

import live_data as live


SCHEMA = '''
CREATE TABLE stocks(code TEXT PRIMARY KEY, name TEXT,
                    last_kline_date TEXT, last_div_date TEXT);
CREATE TABLE klines(code TEXT, date TEXT, open REAL, high REAL, low REAL,
                    close REAL, volume REAL, PRIMARY KEY(code,date));
CREATE TABLE dividends(code TEXT, ex_date TEXT, announce_date TEXT,
                       dividend_10 REAL, dividend_per_share REAL,
                       bonus_share REAL, transfer_share REAL, status TEXT,
                       PRIMARY KEY(code,ex_date));
CREATE TABLE unrelated(value TEXT);
INSERT INTO unrelated VALUES('must survive');
INSERT INTO stocks VALUES('000001','缓存名称',NULL,'2020-01-01');
INSERT INTO stocks VALUES('600000','其他股票','2026-09-09','2019-01-01');
INSERT INTO dividends VALUES('000001','2020-01-01','2019-12-20',1,.1,0,0,'实施');
'''
NOW = datetime(2026, 9, 10, 16, 0, tzinfo=live.SHANGHAI)
DAYS = ['2026-09-04', '2026-09-07', '2026-09-08', '2026-09-09']
SOURCE_EMPTY = ('{"version":null,"result":null,"success":false,'
                '"message":"返回数据为空","code":9201}').encode('utf-8')


def bar(day, volume=1000, close=10.5, **changes):
    row = dict(day=day, open=10., high=12., low=9., close=close, volume=volume)
    row.update(changes)
    return row


def dividend(**changes):
    row = dict(SECURITY_CODE='000001', BONUS_IT_RATIO=2, BONUS_RATIO=None, IT_RATIO=2,
               PRETAX_BONUS_RMB=1.74, IMPL_PLAN_PROFILE='10转2.00派1.74元(含税)',
               EX_DIVIDEND_DATE='2015-04-13 00:00:00',
               EQUITY_RECORD_DATE='2015-04-10 00:00:00',
               NOTICE_DATE='2015-04-07 00:00:00', REPORT_DATE='2014-12-31 00:00:00',
               ASSIGN_PROGRESS='实施分配')
    row.update(changes)
    return row


def rights(**changes):
    row = dict(SECURITY_CODE='000001', FINANCE_CODE='43263', PLACING_RATIO=3,
               ISSUE_PRICE=21.16, ISSUE_NUM=85251672,
               EX_DIVIDEND_DATE='2023-12-08 00:00:00',
               EQUITY_RECORD_DATE='2023-11-29 00:00:00',
               PAY_START_DATE='2023-11-30 00:00:00', PAY_END_DATE='2023-12-06 00:00:00',
               LISTING_DATE='2023-12-25 00:00:00', FIRST_NOTICE_DATE='2023-11-27 00:00:00')
    row.update(changes)
    return row


def encode(value):
    return json.dumps(value, ensure_ascii=False).encode('utf-8')


def action_page(rows, count=None, pages=None):
    count = (0 if rows is None else len(rows)) if count is None else count
    pages = max(1, (count + 499) // 500) if pages is None else pages
    return dict(success=True, code=0, result=dict(count=count, pages=pages, data=rows))


class Response(io.BytesIO):
    status = 200


class LiveDataTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.db = self.root / 'stock.sqlite'
        self.snapshots = self.root / 'snapshots'
        self.connect = sqlite3.connect
        with self.connect(self.db) as conn:
            conn.executescript(SCHEMA)
            for code in ('000001', '600000'):
                rows = [bar('2014-12-31'), bar('2015-01-05')]
                rows += [bar(day, volume=1000 * (i + 1)) for i, day in enumerate(DAYS)]
                conn.executemany('INSERT INTO klines VALUES(?,?,?,?,?,?,?)',
                                 [(code, row['day'], *(row[key] for key in live.FIELDS))
                                  for row in rows])
        self.bars = [bar(day, volume=1000 * (i + 1)) for i, day in enumerate(DAYS)]
        self.bars.append(bar('2026-09-10', volume=5000))
        self.prices = {}  # Per-provider exceptions/raw-byte/JSON overrides.
        self.actions = dict(dividends=[dividend()], undated=[], rights=[])
        self.action_overrides = {}
        self.requests, self.connections, self.statements = [], [], []
        self.guard('socket.socket.connect', side_effect=AssertionError('real network forbidden'))
        self.guard('socket.create_connection', side_effect=AssertionError('real network forbidden'))
        self.guard('db.get_db', side_effect=AssertionError('legacy DB initializer forbidden'))
        self.guard('fetcher.save_klines', side_effect=AssertionError('legacy overwrite forbidden'))
        self.guard('fetcher.get_name', side_effect=AssertionError('extra name HTTP forbidden'))
        self.guard('sqlite3.connect', side_effect=self.guarded_connect)
        self.urlopen = self.guard('live_data.urllib.request.urlopen', side_effect=self.respond)

    def guard(self, target, **kwargs):
        patcher = patch(target, **kwargs)
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def guarded_connect(self, database, *args, **kwargs):
        parsed = urlsplit(str(database))
        path = Path(unquote(parsed.path)) if parsed.scheme == 'file' else Path(database)
        self.assertTrue(path.resolve().is_relative_to(self.root), 'production DB forbidden')
        self.connections.append((str(database), kwargs))
        conn = self.connect(database, *args, **kwargs)
        conn.set_trace_callback(self.statements.append)
        return conn

    def contents(self):
        with self.connect(self.db) as conn:
            return {table: conn.execute(f'SELECT * FROM {table} ORDER BY 1,2').fetchall()
                    for table in ('klines', 'stocks', 'dividends')}

    def execute(self, statement, parameters=()):
        with self.connect(self.db) as conn:
            return conn.execute(statement, parameters).fetchall()

    def price_payload(self, provider):
        if provider == 'sina':
            return self.bars
        lines = [[row['day'], row['open'], row['close'], row['high'],
                  row['low'], row['volume'] / 100] for row in self.bars]
        if provider == 'tencent':
            return dict(code=0, data={'sz000001': dict(day=lines,
                        qt={'sz000001': ['51', '供应商名称', '000001']})})
        return dict(rc=0, data=dict(code='000001', name='供应商名称',
                                   klines=[','.join(map(str, line)) for line in lines]))

    def respond(self, request, timeout):
        self.assertEqual(timeout, 15)
        self.assertEqual(urlsplit(request.full_url).scheme, 'https')
        query = parse_qs(urlsplit(request.full_url).query)
        if 'reportName' in query:
            name = ('rights' if query['reportName'] == ['RPT_IPO_ALLOTMENT'] else
                    'undated' if 'NULL' in query['filter'][0] else 'dividends')
            number = int(query['pageNumber'][0])
            self.assertEqual(query['pageSize'], ['500'])
            self.assertIn('(SECURITY_CODE="000001")', query['filter'][0])
            if name == 'rights':
                self.assertEqual(query['filter'], ['(SECURITY_CODE="000001")'])
            elif name == 'undated':
                self.assertNotIn('>=', query['filter'][0])
                self.assertIn('(ASSIGN_PROGRESS="实施分配")', query['filter'][0])
            else:
                self.assertIn("(EX_DIVIDEND_DATE>='2015-01-01')", query['filter'][0])
            self.requests.append((name, number, request.full_url))
            rows = self.actions[name]
            payload = self.action_overrides.get((name, number), action_page(
                rows[(number - 1) * 500:number * 500], len(rows)))
        else:
            host = urlsplit(request.full_url).hostname
            provider = ('sina' if 'sina' in host else 'tencent' if 'gtimg' in host else 'eastmoney')
            self.requests.append((provider, 1, request.full_url))
            if provider == 'tencent':
                self.assertEqual(query['param'], ['sz000001,day,,,2500'])
            elif provider == 'eastmoney':
                self.assertEqual((query['fqt'], query['klt']), (['0'], ['101']))
                self.assertEqual(query['fields2'], ['f51,f52,f53,f54,f55,f56'])
            payload = self.prices.get(provider, self.price_payload(provider))
        if isinstance(payload, BaseException):
            raise payload
        return Response(payload if isinstance(payload, bytes) else encode(payload))

    def refresh(self, **kwargs):
        return live.refresh_stock('000001', now=kwargs.pop('now', NOW),
                                  db_path=kwargs.pop('db_path', self.db),
                                  snapshot_root=self.snapshots, **kwargs)

    def evidence(self, result, name):
        return json.loads((Path(result['evidence_dir']) / name).read_bytes())

    def fail_all(self, value):
        self.prices = {provider: value for provider in ('sina', 'tencent', 'eastmoney')}

    def test_full_committed_rows_preserve_long_history_other_codes_and_tables(self):
        before = self.contents()
        result = self.refresh()
        after = self.contents()
        self.assertEqual(set(result), {'rows', 'name', 'checked_at', 'provider', 'latest_date',
                         'date', 'actions', 'refresh_id', 'evidence_dir', 'warnings'})
        self.assertEqual([row['day'] for row in result['rows']], ['2015-01-05'] + DAYS + ['2026-09-10'])
        self.assertEqual(set(result['rows'][0]), {'day', *live.FIELDS})
        self.assertTrue(all(isinstance(row[key], float) for row in result['rows'] for key in live.FIELDS))
        for table in ('klines', 'stocks'):
            self.assertEqual([row for row in before[table] if row[0] == '600000'],
                             [row for row in after[table] if row[0] == '600000'])
        self.assertEqual(after['dividends'], before['dividends'])
        self.assertEqual(after['klines'][0], before['klines'][0])  # pre2015 retained
        self.assertEqual(self.execute('SELECT * FROM unrelated'), [('must survive',)])
        self.assertEqual(self.execute('SELECT * FROM stocks WHERE code="000001"'),
                         [('000001', '缓存名称', '2026-09-10', '2020-01-01')])
        self.assertEqual(result['name'], '缓存名称')
        self.assertEqual(result['date'], result['latest_date'])
        self.assertEqual(result['provider'], 'sina')
        self.assertEqual(result, self.evidence(result, 'result.json'))

    def test_old_target_snapshot_hash_and_exact_provider_bodies(self):
        result = self.refresh()
        folder = Path(result['evidence_dir'])
        old = self.evidence(result, 'old_target.json')
        self.assertEqual({row['code'] for row in old['klines']}, {'000001'})
        self.assertEqual(len(old['klines']), 6)
        request = self.evidence(result, 'request.json')
        self.assertEqual(request['old_target_sha256'], hashlib.sha256(
            (folder / 'old_target.json').read_bytes()).hexdigest())
        for proof_path in (folder / 'attempts').glob('*.json'):
            proof = json.loads(proof_path.read_bytes())
            raw = (folder / proof['body_file']).read_bytes()
            self.assertEqual(proof['response_sha256'], hashlib.sha256(raw).hexdigest())
            if proof['label'] == 'price_sina':
                self.assertEqual(raw, encode(self.bars))
        self.assertFalse(list(folder.rglob('*.db')))
        self.assertFalse(list(folder.rglob('*.sqlite')))
        self.assertEqual(len(list(folder.glob('attempts/*.json'))), 4)

    def test_same_day_correction_and_volume_correction_are_upserted(self):
        self.bars.pop()
        self.bars[-1].update(close=11.5, volume=7000.)
        result = self.refresh()
        self.assertEqual(result['rows'][-1]['close'], 11.5)
        self.assertEqual(result['rows'][-1]['volume'], 7000.)
        self.assertEqual(result['latest_date'], '2026-09-09')
        self.assertTrue(any('不是新增交易日' in warning for warning in result['warnings']))
        self.assertEqual(self.execute('SELECT COUNT(*) FROM klines WHERE code="000001"')[0][0], 6)

    def test_sql_readonly_before_network_no_initialization_and_commit_before_load(self):
        original = self.respond

        def inspect(request, timeout):
            if not self.requests:
                self.assertTrue(all('mode=ro' in item[0] for item in self.connections))
                self.assertFalse(any(sql.startswith(('INSERT', 'UPDATE', 'DELETE', 'CREATE'))
                                     for sql in self.statements))
            return original(request, timeout)

        self.urlopen.side_effect = inspect
        self.refresh()
        self.assertEqual([parse_qs(urlsplit(item[0]).query)['mode'][0] for item in self.connections],
                         ['ro', 'rw', 'ro'])
        self.assertIn('BEGIN IMMEDIATE', self.statements)
        commit = self.statements.index('COMMIT')
        self.assertIn('PRAGMA query_only=ON', self.statements[commit + 1:])
        self.assertTrue(any('SELECT MAX(date)' in sql for sql in self.statements))
        self.assertFalse(any(sql.startswith(('DELETE', 'CREATE', 'ALTER', 'DROP')) for sql in self.statements))

    def test_all_sources_failed_no_stale_fallback_no_writable_db_or_row_changes(self):
        before, raw_before = self.contents(), self.db.read_bytes()
        self.fail_all(TimeoutError('secret_token=do_not_print'))
        with self.assertRaises(live.RefreshError) as raised:
            self.refresh()
        exc = raised.exception
        self.assertEqual(self.contents(), before)
        self.assertEqual(self.db.read_bytes(), raw_before)
        self.assertFalse(exc.price_committed)
        self.assertNotIn('secret_token', str(exc))
        self.assertEqual([name for name, _, _ in self.requests], ['sina', 'tencent', 'eastmoney'])
        self.assertTrue(all('mode=ro' in uri for uri, _ in self.connections))
        self.assertTrue(Path(exc.evidence_dir, 'failure.json').is_file())

    def test_denied_price_sources_are_not_retried_and_error_body_is_frozen(self):
        for provider, status in zip(('sina', 'tencent', 'eastmoney'), (456, 403, 429)):
            self.prices[provider] = HTTPError('https://example.invalid/?token=secret', status,
                                             'secret-message', {}, io.BytesIO(b'raw denied'))
        with self.assertRaises(live.RefreshError) as raised:
            self.refresh()
        self.assertEqual(self.urlopen.call_count, 3)
        folder = Path(raised.exception.evidence_dir)
        for provider in ('sina', 'tencent', 'eastmoney'):
            proof = json.loads((folder / 'attempts' / f'price_{provider}.json').read_bytes())
            self.assertEqual((folder / proof['body_file']).read_bytes(), b'raw denied')
            self.assertNotIn('secret', json.dumps(proof))

    def test_tencent_correct_fields_https_unadjusted_and_volume_lots_to_cache(self):
        self.prices['sina'] = TimeoutError('offline')
        result = self.refresh()
        self.assertEqual(result['provider'], 'tencent')
        self.assertEqual(result['rows'][-1], self.bars[-1])
        volume = self.evidence(result, 'accepted_price.json')['volume_normalization']
        self.assertEqual((volume['factor'], volume['output_unit']), (100., 'cached_units'))
        self.assertEqual(volume['overlap_days'], DAYS[:3])

    def test_eastmoney_correct_fields_and_new_name_from_selected_provider(self):
        self.execute('UPDATE stocks SET name=NULL WHERE code="000001"')
        self.prices.update(sina=b'[]', tencent=b'[]')
        result = self.refresh()
        self.assertEqual(result['provider'], 'eastmoney')
        self.assertEqual(result['name'], '供应商名称')
        self.assertEqual(result['rows'][-1], self.bars[-1])
        self.assertEqual(self.execute('SELECT name FROM stocks WHERE code="000001"'), [('供应商名称',)])

    def test_missing_name_is_none_not_fabricated_or_extra_http(self):
        self.execute('UPDATE stocks SET name=NULL WHERE code="000001"')
        result = self.refresh()
        self.assertIsNone(result['name'])
        self.assertEqual(self.urlopen.call_count, 4)

    def test_existing_volume_cache_in_lots_normalizes_sina_shares_by_point01(self):
        self.execute('UPDATE klines SET volume=volume/100 WHERE code="000001"')
        result = self.refresh()
        self.assertEqual(result['rows'][-1]['volume'], 50.)
        self.assertEqual(self.evidence(result, 'accepted_price.json')['volume_normalization']['factor'], .01)

    def test_no_history_uses_known_shares_standard_for_each_provider(self):
        for provider in ('sina', 'tencent', 'eastmoney'):
            with self.subTest(provider=provider):
                self.execute('DELETE FROM klines WHERE code="000001"')
                self.execute('DELETE FROM stocks WHERE code="000001"')
                self.prices = {earlier: b'[]' for earlier in ('sina', 'tencent', 'eastmoney')
                               if earlier != provider}
                result = self.refresh()
                self.assertEqual(result['provider'], provider)
                self.assertEqual(result['rows'][-1]['volume'], 5000.)
                meta = self.evidence(result, 'accepted_price.json')['volume_normalization']
                self.assertEqual(meta['output_unit'], 'shares')

    def test_ambiguous_or_mixed_volume_units_reject_all_without_writes(self):
        self.bars[0]['volume'] *= 10
        before = self.contents()
        with self.assertRaisesRegex(live.RefreshError, 'all price sources failed'):
            self.refresh()
        self.assertEqual(self.contents(), before)
        self.assertEqual(self.urlopen.call_count, 3)

    def test_insufficient_and_zero_overlap_never_guess_cache_units(self):
        for rows in ([self.bars[-1]], self.bars[-3:],
                     [dict(row, volume=0) for row in self.bars]):
            with self.subTest(rows=rows):
                self.bars = deepcopy(rows)
                with self.assertRaises(live.RefreshError):
                    self.refresh()

    def test_zero_zero_overlap_not_counted_as_unit_evidence(self):
        self.execute('UPDATE klines SET volume=0 WHERE code="000001"')
        self.bars = [dict(row, volume=0) for row in self.bars]
        with self.assertRaises(live.RefreshError):
            self.refresh()

    def test_volume_ambiguous_first_source_can_fall_back_to_consistent_source(self):
        wrong = deepcopy(self.bars)
        wrong[0]['volume'] *= 10
        self.prices['sina'] = wrong
        result = self.refresh()
        self.assertEqual(result['provider'], 'tencent')

    def test_before_15_filters_today_but_at_15_accepts_it(self):
        result = self.refresh(now=NOW.replace(hour=14, minute=59, second=59))
        self.assertEqual(result['latest_date'], '2026-09-09')
        self.assertNotIn('2026-09-10', [row['day'] for row in result['rows']])
        self.assertTrue(any('临时日线' in warning for warning in result['warnings']))
        self.assertEqual(result['actions']['metadata']['coverage_end'], '2026-09-10')
        result = self.refresh(now=NOW.replace(hour=15))
        self.assertEqual(result['latest_date'], '2026-09-10')

    def test_only_intraday_bar_is_not_a_completed_refresh(self):
        self.execute('DELETE FROM klines WHERE code="000001"')
        self.bars = [self.bars[-1]]
        with self.assertRaises(live.RefreshError):
            self.refresh(now=NOW.replace(hour=14))

    def test_aware_utc_converted_to_shanghai_and_date_is_not_checked_at(self):
        self.bars.pop()
        result = self.refresh(now=datetime(2026, 9, 10, 6, tzinfo=timezone.utc))
        self.assertEqual(result['checked_at'], '2026-09-10T14:00:00+08:00')
        self.assertEqual(result['date'], '2026-09-09')
        self.assertTrue(any('本次已联网核验' in warning for warning in result['warnings']))

    def test_default_clock_and_paths_can_be_injected_without_touching_production(self):
        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 9, 10, 16, tzinfo=tz)

        with patch.object(live, 'datetime', Clock), patch.object(live, 'DB_PATH', self.db), \
                patch.object(live, 'SNAPSHOT_ROOT', self.snapshots):
            result = live.refresh_stock('000001')
        self.assertEqual(result['checked_at'], NOW.isoformat())
        self.assertEqual(Path(result['evidence_dir']).parent, self.snapshots)

    def test_older_provider_date_cannot_masquerade_as_refresh_from_cached_max(self):
        self.prices['sina'] = self.bars[:3]
        result = self.refresh()
        self.assertEqual(result['provider'], 'tencent')
        rejection = self.evidence(result, 'rejected_sina.json')
        self.assertIn('older than cached MAX', rejection['error'])

    def test_all_older_quotes_keep_database_unchanged_even_if_stock_lastdate_null(self):
        self.bars = self.bars[:3]
        before = self.contents()
        with self.assertRaises(live.RefreshError):
            self.refresh()
        self.assertEqual(self.contents(), before)

    def test_invalid_bar_future_nan_ohlc_negative_volume_and_duplicate_fall_back(self):
        variants = [dict(day='2026-09-11'), dict(day='2026-02-30'),
                    dict(day='20260910'), dict(open='NaN'), dict(close=float('inf')),
                    dict(open=None), dict(close=True), dict(high=8), dict(low=13),
                    dict(open=0), dict(volume=-1), dict(volume='NaN')]
        for changes in variants:
            with self.subTest(changes=changes):
                wrong = deepcopy(self.bars)
                wrong[-1].update(changes)
                self.prices['sina'] = wrong
                result = self.refresh()
                self.assertEqual(result['provider'], 'tencent')
        self.prices['sina'] = self.bars + [dict(self.bars[-1], close=11)]
        self.assertEqual(self.refresh()['provider'], 'tencent')

    def test_malformed_intraday_bar_is_rejected_before_filtering(self):
        self.prices['sina'] = self.bars[:-1] + [dict(self.bars[-1], close='NaN')]
        result = self.refresh(now=NOW.replace(hour=14))
        self.assertEqual(result['provider'], 'tencent')

    def test_exact_duplicate_bars_deduplicate_and_unsorted_rows_are_sorted(self):
        self.prices['sina'] = list(reversed(self.bars)) + [self.bars[-1]]
        result = self.refresh()
        self.assertEqual(result['provider'], 'sina')
        self.assertEqual(len(result['rows']), 6)

    def test_newer_name_or_price_from_rejected_source_does_not_leak(self):
        self.execute('UPDATE stocks SET name=NULL WHERE code="000001"')
        self.prices['sina'] = b'[]'
        payload = self.price_payload('tencent')
        payload['data']['sz000001']['day'][-1][2] = -10
        payload['data']['sz000001']['qt']['sz000001'][1] = '拒绝的名称'
        self.prices['tencent'] = payload
        self.assertEqual(self.refresh()['name'], '供应商名称')

    def test_missing_db_and_schema_do_not_create_or_initialize_database(self):
        missing = self.root / 'missing.sqlite'
        with self.assertRaises(live.RefreshError):
            self.refresh(db_path=missing)
        self.assertFalse(missing.exists())
        empty = self.root / 'empty.sqlite'
        with self.connect(empty) as conn:
            conn.execute('CREATE TABLE sentinel(value TEXT)')
        before = empty.read_bytes()
        with self.assertRaises(live.RefreshError):
            self.refresh(db_path=empty)
        self.assertEqual(empty.read_bytes(), before)
        self.urlopen.assert_not_called()

    def test_invalid_inputs_reject_before_database_or_network(self):
        for code in (123456, '../123', '', '00001', '１２３４５６'):
            with self.subTest(code=code), self.assertRaises(ValueError):
                live.refresh_stock(code, db_path=self.db, snapshot_root=self.snapshots)
        for now in (datetime(2026, 9, 10), '2026-09-10', NOW.replace(year=2014)):
            with self.subTest(now=now), self.assertRaises(ValueError):
                self.refresh(now=now)
        self.assertFalse(self.connections)
        self.urlopen.assert_not_called()

    def test_transaction_error_rolls_back_all_target_rows_and_stock_metadata(self):
        self.execute('CREATE TRIGGER fail_stock BEFORE UPDATE ON stocks '
                     'BEGIN SELECT RAISE(ABORT,"fixture failure secret"); END')
        before = self.contents()
        with self.assertRaises(live.RefreshError) as raised:
            self.refresh()
        self.assertEqual(before, self.contents())
        self.assertFalse(raised.exception.price_committed)
        self.assertNotIn('secret', str(raised.exception))
        self.assertEqual([name for name, _, _ in self.requests], ['sina'])

    def test_concurrent_target_change_aborts_without_overwriting_other_writer(self):
        original = self.respond

        def race(request, timeout):
            response = original(request, timeout)
            if self.requests[-1][0] == 'sina':
                self.execute('UPDATE klines SET close=11 WHERE code="000001" AND date="2026-09-09"')
            return response

        self.urlopen.side_effect = race
        with self.assertRaisesRegex(live.RefreshError, 'concurrently'):
            self.refresh()
        self.assertEqual(self.execute('SELECT close FROM klines WHERE code="000001" '
                                     'AND date="2026-09-09"'), [(11.,)])
        self.assertEqual(self.execute('SELECT COUNT(*) FROM klines WHERE date="2026-09-10"'), [(0,)])

    def test_postcommit_load_returns_actual_database_rows_not_network_rolling_window(self):
        self.execute('CREATE TRIGGER adjust_close AFTER INSERT ON klines '
                     'WHEN NEW.code="000001" AND NEW.date="2026-09-10" '
                     'BEGIN UPDATE klines SET close=11 WHERE code=NEW.code AND date=NEW.date; END')
        result = self.refresh()
        self.assertEqual(result['rows'][-1]['close'], 11.)
        self.assertEqual(self.bars[-1]['close'], 10.5)
        self.assertEqual(result['rows'][0]['day'], '2015-01-05')

    def test_storage_failure_before_commit_does_not_write_price(self):
        before = self.contents()
        with patch.object(live, '_write', side_effect=OSError('secret disk path')):
            with self.assertRaises(live.RefreshError) as raised:
                self.refresh()
        self.assertFalse(raised.exception.price_committed)
        self.assertEqual(self.contents(), before)
        self.urlopen.assert_not_called()

    def test_postcommit_storage_error_discloses_commit_not_false_rollback(self):
        original = live._json

        def fail_result(path, value):
            if path.name == 'result.json':
                raise OSError('fixture disk failure')
            return original(path, value)

        with patch.object(live, '_json', side_effect=fail_result):
            with self.assertRaises(live.RefreshError) as raised:
                self.refresh()
        self.assertTrue(raised.exception.price_committed)
        self.assertEqual(self.execute('SELECT last_kline_date FROM stocks WHERE code="000001"'),
                         [('2026-09-10',)])

    def test_actions_contract_parser_units_coverage_and_holding_approximations(self):
        result = self.refresh()
        actions = result['actions']
        self.assertEqual(actions['status'], 'ok')
        self.assertFalse(actions['rights_issue_present'])
        self.assertEqual(actions['events'][0]['cash_per_share'], .174)
        self.assertEqual(actions['events'][0]['share_multiplier'], 1.2)
        self.assertIsNone(actions['events'][0]['pay_date'])
        meta = actions['metadata']
        self.assertTrue(meta['pagination_complete'])
        self.assertTrue(meta['coverage_matches_quote'])
        self.assertEqual(meta['coverage_start'], '2015-01-01')
        self.assertGreaterEqual(meta['coverage_end'], result['latest_date'])
        for key in ('live_recommendation_blocked', 'cash_payment_date_verified',
                    'bonus_listing_date_verified', 'cash_tax_applied',
                    'supplier_history_completeness_guaranteed', 'corporate_actions_reconciled'):
            self.assertFalse(meta[key])
        self.assertTrue(any('持仓' in note for note in meta['notes']))
        self.assertEqual(actions, self.evidence(result, 'actions.json'))
        filters = [parse_qs(urlsplit(url).query)['filter'][0] for name, _, url in self.requests
                   if name == 'dividends']
        self.assertIn("(EX_DIVIDEND_DATE<='2026-09-10')", filters[0])

    def test_each_refresh_fetches_new_prices_and_actions_unique_immutable_evidence(self):
        first = self.refresh()
        folder = Path(first['evidence_dir'])
        original = {str(path): path.read_bytes() for path in folder.rglob('*') if path.is_file()}
        self.actions['dividends'] = []
        second = self.refresh()
        self.assertNotEqual(first['refresh_id'], second['refresh_id'])
        self.assertNotEqual(first['evidence_dir'], second['evidence_dir'])
        self.assertEqual(self.urlopen.call_count, 8)
        self.assertEqual(second['actions']['events'], [])
        self.assertEqual(original, {str(path): path.read_bytes() for path in folder.rglob('*') if path.is_file()})

    def test_explicit_count_zero_data_none_is_legitimate_empty_history(self):
        for name in self.actions:
            self.action_overrides[(name, 1)] = action_page(None, 0, 0)
        result = self.refresh()['actions']
        self.assertEqual((result['status'], result['events'], result['rights_issue_present']),
                         ('ok', [], False))
        self.assertIs(result['rights_issue_present'], False)
        self.assertFalse(result['metadata']['supplier_history_completeness_guaranteed'])
        self.assertTrue(any('不证明历史上不存在权息/配股' in note
                            for note in result['metadata']['notes']))

    def test_exact_source_empty_sentinel_normalizes_only_first_page(self):
        self.assertEqual(live._action_page(SOURCE_EMPTY, 1, '000001'), (0, 0, []))
        for number in (0, 2):
            with self.subTest(number=number), self.assertRaises(live._Invalid):
                live._action_page(SOURCE_EMPTY, number, '000001')
        self.urlopen.assert_not_called()

    def test_all_source_empty_queries_keep_normal_proofs_raw_and_absence_caveat(self):
        for name in self.actions:
            self.action_overrides[(name, 1)] = SOURCE_EMPTY
        result = self.refresh()
        actions = result['actions']
        self.assertEqual((actions['status'], actions['events']), ('ok', []))
        self.assertIs(actions['rights_issue_present'], False)
        meta = actions['metadata']
        self.assertEqual(meta['rights_history_status'], 'no_reported_issues_in_window')
        self.assertEqual(meta['rights_events'], [])
        self.assertTrue(meta['pagination_complete'])
        self.assertTrue(meta['coverage_matches_quote'])
        self.assertFalse(meta['live_recommendation_blocked'])
        self.assertFalse(meta['supplier_history_completeness_guaranteed'])
        self.assertFalse(meta['corporate_actions_reconciled'])
        self.assertTrue(any('不证明历史上不存在权息/配股' in note for note in meta['notes']))
        self.assertTrue(any('不证明历史上不存在权息/配股' in note for note in result['warnings']))
        self.assertEqual([(name, number) for name, number, _ in self.requests],
                         [('sina', 1), ('dividends', 1), ('undated', 1), ('rights', 1)])
        self.assertEqual(self.urlopen.call_count, 4)
        folder = Path(result['evidence_dir'])
        self.assertEqual(len(list(folder.glob('attempts/*.json'))), 4)
        digest = hashlib.sha256(SOURCE_EMPTY).hexdigest()
        for name, _, url in self.requests[1:]:
            attempt = f'attempts/actions_{name}_0001.json'
            self.assertEqual(meta['datasets'][name], dict(status='ok', count=0, pages=[
                dict(source_url=url, response_sha256=digest, attempt=attempt, page=1)]))
            proof = self.evidence(result, attempt)
            self.assertEqual(proof['http_status'], 200)
            self.assertNotIn('error', proof)
            self.assertEqual(proof['response_sha256'], digest)
            self.assertEqual((folder / proof['body_file']).read_bytes(), SOURCE_EMPTY)
        self.assertEqual(actions, self.evidence(result, 'actions.json'))

    def test_empty_undated_query_does_not_hide_reported_rights_or_dividends(self):
        self.action_overrides[('undated', 1)] = SOURCE_EMPTY
        self.actions['rights'] = [rights()]
        actions = self.refresh()['actions']
        self.assertEqual(actions['status'], 'ok')
        self.assertEqual(len(actions['events']), 1)
        self.assertIs(actions['rights_issue_present'], True)
        self.assertEqual(len(actions['metadata']['rights_events']), 1)
        self.assertTrue(actions['metadata']['live_recommendation_blocked'])
        self.assertEqual(actions['metadata']['datasets']['undated']['count'], 0)
        self.assertEqual(actions['metadata']['datasets']['rights']['count'], 1)
        self.assertEqual(self.urlopen.call_count, 4)

    def test_source_empty_mismatches_and_other_api_errors_remain_unknown(self):
        sentinel = json.loads(SOURCE_EMPTY)
        malformed = [dict(sentinel, **{key: value}) for key, values in (
            ('code', (True, False, '9201', 9201.0, 9202, 0, None)),
            ('success', (True, 0, None, 'false')),
            ('result', ({}, [], dict(count=0, pages=0, data=None))),
            ('message', ('返回数据为空 ', ' 返回数据为空', '其他错误', '', None)),
        ) for value in values]
        malformed += [{key: value for key, value in sentinel.items() if key != missing}
                      for missing in ('code', 'success', 'result', 'message')]
        for payload in malformed:
            with self.subTest(payload=payload):
                self.requests.clear()
                self.action_overrides[('rights', 1)] = payload
                result = self.refresh()
                actions = result['actions']
                self.assertEqual(actions['status'], 'error')
                self.assertIsNone(actions['rights_issue_present'])
                self.assertTrue(actions['metadata']['live_recommendation_blocked'])
                self.assertFalse(actions['metadata']['pagination_complete'])
                self.assertIn('unsuccessful action API response', actions['error'])
                self.assertEqual(result['latest_date'], '2026-09-10')
                self.assertEqual([(name, number) for name, number, _ in self.requests],
                                 [('sina', 1), ('dividends', 1), ('undated', 1), ('rights', 1)])

    def test_missing_result_count_or_failed_response_is_never_empty_history(self):
        malformed = [dict(success=True, code=0, result=None),
                     dict(success=True, code=0, result=dict(data=None, pages=0)),
                     dict(success=False, code=0, result=dict(data=None, count=0, pages=0)),
                     dict(success=True, code=0, result=dict(data=None, count=1, pages=1)),
                     dict(success=True, code=0, result=dict(data=[], count=True, pages=1)),
                     b'not JSON token=private']
        for payload in malformed:
            with self.subTest(payload=payload):
                self.action_overrides[('rights', 1)] = payload
                result = self.refresh()
                self.assertEqual(result['actions']['status'], 'error')
                self.assertIsNone(result['actions']['rights_issue_present'])
                self.assertTrue(result['actions']['metadata']['live_recommendation_blocked'])
                self.assertEqual(result['latest_date'], '2026-09-10')
                self.assertNotIn('private', result['actions']['error'])

    def test_rights_http_error_is_unknown_blocks_advice_but_prices_commit(self):
        self.action_overrides[('rights', 1)] = HTTPError(
            'https://invalid/?token=private', 403, 'secret', {}, io.BytesIO(b'rights denied'))
        result = self.refresh()
        self.assertEqual(result['actions']['status'], 'error')
        self.assertIsNone(result['actions']['rights_issue_present'])
        self.assertIn('HTTP 403', result['actions']['error'])
        self.assertNotIn('secret', result['actions']['error'])
        self.assertEqual(self.execute('SELECT last_kline_date FROM stocks WHERE code="000001"'),
                         [('2026-09-10',)])
        self.assertTrue(any('禁止生成' in warning for warning in result['warnings']))
        self.assertEqual(self.urlopen.call_count, 4)

    def test_reported_rights_are_not_cash_dividends_and_block_live_recommendation(self):
        self.actions['rights'] = [rights()]
        result = self.refresh()['actions']
        self.assertEqual(result['status'], 'ok')
        self.assertTrue(result['rights_issue_present'])
        self.assertTrue(result['metadata']['live_recommendation_blocked'])
        event = result['metadata']['rights_events'][0]
        self.assertEqual(event['rights_per_share'], .3)
        self.assertFalse(event['simulated'])
        self.assertEqual(len(result['events']), 1)

    def test_rights_parse_errors_are_unknown_not_false(self):
        for change in (dict(ISSUE_NUM=0), dict(EX_DIVIDEND_DATE=None),
                       dict(ISSUE_PRICE='NaN'), dict(PLACING_RATIO=-1)):
            with self.subTest(change=change):
                self.actions['rights'] = [rights(**change)]
                result = self.refresh()['actions']
                self.assertEqual(result['status'], 'error')
                self.assertIsNone(result['rights_issue_present'])

    def test_undated_material_dividend_even_old_is_error_not_silently_dropped(self):
        self.actions['undated'] = [dividend(EX_DIVIDEND_DATE=None, NOTICE_DATE='2000-01-01')]
        result = self.refresh()['actions']
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['events'], [])
        self.assertTrue(result['metadata']['live_recommendation_blocked'])

    def test_conflicting_dividends_and_profile_errors_are_fail_closed(self):
        variants = [[dividend(), dividend(NOTICE_DATE='2015-04-08')],
                    [dividend(PRETAX_BONUS_RMB=.174)], [dividend(ASSIGN_PROGRESS='unknown')]]
        for rows in variants:
            with self.subTest(rows=rows):
                self.actions['dividends'] = rows
                actions = self.refresh()['actions']
                self.assertEqual((actions['status'], actions['events']), ('error', []))

    def test_action_denial_stops_remaining_queries_no_retry_after_price_fallback(self):
        self.prices['sina'] = HTTPError('https://invalid', 456, 'denied', {}, io.BytesIO(b'sina'))
        self.action_overrides[('dividends', 1)] = HTTPError(
            'https://invalid', 429, 'limited', {}, io.BytesIO(b'em'))
        result = self.refresh()
        self.assertEqual(result['provider'], 'tencent')
        self.assertEqual([name for name, _, _ in self.requests], ['sina', 'tencent', 'dividends'])
        self.assertEqual(result['actions']['status'], 'error')
        self.assertIsNone(result['actions']['rights_issue_present'])

    def test_paginate_all_three_code_queries_at_500_with_complete_counts(self):
        proposals = [dividend(ASSIGN_PROGRESS='预案', REPORT_DATE=f'fixture-{i}') for i in range(501)]
        zero_undated = [dividend(EX_DIVIDEND_DATE=None, PRETAX_BONUS_RMB=0,
                        BONUS_RATIO=0, IT_RATIO=0, BONUS_IT_RATIO=0,
                        IMPL_PLAN_PROFILE='10派0', REPORT_DATE=f'fixture-{i}') for i in range(501)]
        old_rights = [rights(FINANCE_CODE=str(i), EX_DIVIDEND_DATE='2010-01-02',
                     EQUITY_RECORD_DATE='2010-01-01', PAY_START_DATE='2010-01-01',
                     PAY_END_DATE='2010-01-02', LISTING_DATE='2010-01-03',
                     FIRST_NOTICE_DATE='2010-01-01') for i in range(501)]
        self.actions = dict(dividends=proposals, undated=zero_undated, rights=old_rights)
        result = self.refresh()['actions']
        self.assertEqual(result['status'], 'ok', result.get('error'))
        self.assertEqual(result['events'], [])
        self.assertFalse(result['rights_issue_present'])
        self.assertEqual([(name, number) for name, number, _ in self.requests],
                         [('sina', 1), ('dividends', 1), ('dividends', 2),
                          ('undated', 1), ('undated', 2), ('rights', 1), ('rights', 2)])
        self.assertTrue(all(len(ds['pages']) == 2 and ds['count'] == 501
                            for ds in result['metadata']['datasets'].values()))

    def test_excessive_pages_wrong_code_and_truncated_pages_are_errors(self):
        bad = [action_page([], 500 * 257, 257), action_page([], 1, 1),
               action_page([dividend(SECURITY_CODE='600000')]), action_page([], 0, 2)]
        for payload in bad:
            with self.subTest(payload=payload):
                self.action_overrides[('dividends', 1)] = payload
                result = self.refresh()['actions']
                self.assertEqual(result['status'], 'error')
                self.assertFalse(result['metadata']['pagination_complete'])

    def test_mid_page_failure_duplicate_or_count_change_never_returns_partial_actions(self):
        rows = [dividend(ASSIGN_PROGRESS='预案', REPORT_DATE=str(i)) for i in range(501)]
        self.actions['dividends'] = rows
        malformed = (
            TimeoutError('private query'), action_page([rows[0]], 501, 2),
            action_page([rows[-1], rows[-2]], 502, 2), SOURCE_EMPTY,
            action_page(None, 0, 0), action_page([], 0, 1),
        )
        for payload in malformed:
            with self.subTest(payload=payload):
                self.action_overrides[('dividends', 2)] = payload
                result = self.refresh()['actions']
                self.assertEqual((result['status'], result['events']), ('error', []))
                self.assertTrue(result['metadata']['live_recommendation_blocked'])
                self.assertFalse(result['metadata']['pagination_complete'])
                self.assertNotIn('private', result['error'])

    def test_insufficient_action_window_is_blocked_even_with_complete_pages(self):
        folder = self.snapshots / 'coverage-test'
        folder.mkdir(parents=True)
        actions = live._fetch_actions('000001', '2026-09-09', '2026-09-10', folder, NOW.isoformat())
        self.assertEqual(actions['status'], 'error')
        self.assertTrue(actions['metadata']['live_recommendation_blocked'])
        self.assertFalse(actions['metadata']['coverage_matches_quote'])

    def test_incomplete_http_body_falls_back_once_and_freezes_partial_bytes(self):
        self.prices['sina'] = IncompleteRead(b'partial response', 100)
        result = self.refresh()
        self.assertEqual(result['provider'], 'tencent')
        proof = self.evidence(result, 'attempts/price_sina.json')
        self.assertEqual(proof['error'], 'IncompleteRead')
        self.assertEqual(Path(result['evidence_dir'], proof['body_file']).read_bytes(), b'partial response')
        self.assertEqual([name for name, _, _ in self.requests].count('sina'), 1)

    def test_incomplete_action_body_is_returned_as_error_after_price_commit(self):
        self.action_overrides[('rights', 1)] = IncompleteRead(b'partial rights', 100)
        result = self.refresh()
        self.assertEqual(result['latest_date'], '2026-09-10')
        self.assertEqual(result['actions']['status'], 'error')
        self.assertIsNone(result['actions']['rights_issue_present'])
        self.assertIn('IncompleteRead', result['actions']['error'])

    def test_all_invalid_responses_leave_db_unchanged_and_retain_bodies(self):
        self.fail_all(b'not JSON token=private')
        before = self.contents()
        with self.assertRaises(live.RefreshError) as raised:
            self.refresh()
        self.assertEqual(self.contents(), before)
        folder = Path(raised.exception.evidence_dir)
        self.assertEqual(len(list(folder.glob('raw/*.body'))), 3)
        for path in folder.glob('rejected_*.json'):
            self.assertNotIn('private', path.read_text())

    def test_tencent_adjusted_only_and_eastmoney_wrong_code_rejected(self):
        self.prices['sina'] = b'[]'
        tencent = self.price_payload('tencent')
        stock = tencent['data']['sz000001']
        stock['qfqday'] = stock.pop('day')
        self.prices['tencent'] = tencent
        eastmoney = self.price_payload('eastmoney')
        eastmoney['data']['code'] = '600000'
        self.prices['eastmoney'] = eastmoney
        before = self.contents()
        with self.assertRaises(live.RefreshError):
            self.refresh()
        self.assertEqual(self.contents(), before)

    def test_tencent_name_from_successful_response_only(self):
        self.execute('UPDATE stocks SET name=NULL WHERE code="000001"')
        self.prices['sina'] = b'[]'
        self.assertEqual(self.refresh()['name'], '供应商名称')

    def test_unknown_units_and_provider_are_never_assumed(self):
        with self.assertRaisesRegex(ValueError, 'unknown provider volume unit'):
            live._normalize_volume(self.bars, [], 'unknown')
        with self.assertRaisesRegex(ValueError, 'unknown price provider'):
            live._parse_price('unknown', encode(self.price_payload('eastmoney')), '000001')

    def test_action_403_429_456_each_stops_remaining_endpoint_requests(self):
        for status in (403, 429, 456):
            with self.subTest(status=status):
                self.requests.clear()
                self.action_overrides[('dividends', 1)] = HTTPError(
                    'https://invalid', status, 'denied', {}, io.BytesIO(b'no'))
                result = self.refresh()['actions']
                self.assertEqual([name for name, _, _ in self.requests], ['sina', 'dividends'])
                self.assertEqual(result['status'], 'error')
                self.assertIsNone(result['rights_issue_present'])


if __name__ == '__main__':
    unittest.main()