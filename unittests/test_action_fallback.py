"""Offline public-source contracts: synthetic pages, no DB, temporary files only."""

from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import action_fallback as fallback
import action_snapshot as sina


def dividend(code='000001', ex='2015-04-13 00:00:00', **changes):
    # Field values/profile match the verified 2015 public Ping An example.
    row = dict(SECURITY_CODE=code, BONUS_IT_RATIO=2, BONUS_RATIO=None, IT_RATIO=2,
               PRETAX_BONUS_RMB=1.74, IMPL_PLAN_PROFILE='10转2.00派1.74元(含税,扣税后1.566元)',
               EX_DIVIDEND_DATE=ex, EQUITY_RECORD_DATE='2015-04-10 00:00:00',
               NOTICE_DATE='2015-04-07 00:00:00', PLAN_NOTICE_DATE='2015-03-13 00:00:00',
               REPORT_DATE='2014-12-31 00:00:00', ASSIGN_PROGRESS='实施分配')
    row.update(changes)
    return row


def rights(code='000049', **changes):
    row = dict(SECURITY_CODE=code, FINANCE_CODE='43263', PLACING_RATIO=3,
               ISSUE_PRICE=21.16, ISSUE_NUM=85251672, EX_DIVIDEND_DATE='2023-12-08 00:00:00',
               EQUITY_RECORD_DATE='2023-11-29 00:00:00', PAY_START_DATE='2023-11-30 00:00:00',
               PAY_END_DATE='2023-12-06 00:00:00', LISTING_DATE='2023-12-25 00:00:00',
               FIRST_NOTICE_DATE='2023-11-27 00:00:00')
    row.update(changes)
    return row


def page(rows, count=None, pages=None):
    count = len(rows) if count is None else count
    pages = (max(1, (count + fallback.PAGE_SIZE - 1) // fallback.PAGE_SIZE)
             if pages is None else pages)
    return fallback._json_bytes(dict(success=True, code=0, version='fixture',
                                     result=dict(count=count, pages=pages, data=rows)))


class Response(io.BytesIO):
    status = 200


class FallbackTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name) / 'fallback'
        self.datasets = dict(dividends=[dividend()], undated=[], rights=[])
        self.urls = []
        patcher = patch('action_fallback.urllib.request.urlopen', side_effect=self.respond)
        self.urlopen = patcher.start()
        self.addCleanup(patcher.stop)

    def respond(self, request, timeout):
        self.assertEqual(timeout, 15)
        self.assertTrue(request.full_url.startswith(fallback.ENDPOINT))
        self.urls.append(request.full_url)
        query = parse_qs(urlsplit(request.full_url).query)
        name = ('rights' if query['reportName'] == [fallback.RIGHTS] else
                'undated' if 'NULL' in query['filter'][0] else 'dividends')
        number, size = int(query['pageNumber'][0]), int(query['pageSize'][0])
        rows = self.datasets[name]
        return Response(page(rows[(number - 1) * size:number * size], len(rows)))

    def collect(self, codes=('000001',), **kwargs):
        return fallback.collect_action_snapshot(codes, self.folder, start='2015-01-01',
                                                 end='2026-09-10', **kwargs)

    def resolve(self, codes=('000001',), **kwargs):
        return fallback.resolve_action_snapshot(codes, self.folder, start='2015-01-01',
                                                 end='2026-09-10', **kwargs)

    def freeze(self):
        manifest = fallback.download_market_snapshot(self.folder, '2015-01-01', '2026-09-10')
        self.urlopen.reset_mock()
        self.urlopen.side_effect = AssertionError('frozen resolver must not fetch')
        return manifest

    def primary_snapshot(self, code='000001', present=False):
        primary = self.folder.parent / 'sina'
        primary.mkdir(exist_ok=True)
        value = sina._result(code)
        # Deliberately different from EM: resolution must not replace these.
        value.update(status='ok', response_sha256='a' * 64, rights_issue_present=present,
                     events=[dict(ex_date='2015-04-13', cash_per_share=9.87,
                                  share_multiplier=1.5, record_date='2015-04-10',
                                  announce_date='2015-04-07', pay_date=None)])
        value['metadata'].update(custom={'keep': ['original']}, evidence=[{'primary': 'proof'}])
        (primary / f'{code}.json').write_bytes(fallback._json_bytes(value))
        self.assertEqual(sina._read_valid_snapshot(primary / f'{code}.json', code), value)
        return primary, value

    def test_units_dates_metadata_original_bytes_and_offline_reader(self):
        result = self.collect()['000001']
        self.assertEqual(result['status'], 'ok', result.get('error'))
        self.assertEqual(result['events'], [dict(ex_date='2015-04-13', cash_per_share=.174,
                         share_multiplier=1.2, record_date='2015-04-10',
                         announce_date='2015-04-07', pay_date=None)])
        self.assertFalse(result['rights_issue_present'])
        meta = result['metadata']
        self.assertTrue(meta['pagination_complete'])
        for flag in ('supplier_history_completeness_guaranteed', 'corporate_actions_reconciled',
                     'cash_payment_date_verified', 'rights_issue_simulated', 'cash_tax_applied'):
            self.assertFalse(meta[flag])
        self.assertEqual(meta['coverage_start'], '2015-01-01')
        self.assertEqual(meta['rights_history_status'], 'no_reported_issues_in_window')
        raw = (self.folder / 'raw' / f'{result["response_sha256"]}.json').read_bytes()
        self.assertEqual(raw, page([dividend()]))
        manifest = (self.folder / meta['snapshot_manifest']).read_bytes()
        self.assertEqual(fallback._sha(manifest), meta['snapshot_manifest_sha256'])
        self.assertEqual(self.urlopen.call_count, 3)
        self.assertIn("(EX_DIVIDEND_DATE>='2015-01-01')", parse_qs(urlsplit(self.urls[0]).query)['filter'][0])
        self.assertNotIn('REPORT_DATE>=', self.urls[0])
        self.urlopen.side_effect = AssertionError('offline reader must not fetch')
        self.assertEqual(fallback.fetch_action_history('000001', self.folder), result)

    def test_pagination_matches_codes_not_report_year_and_resumes_without_network(self):
        self.datasets['dividends'] = [dividend(code=f'{i:06d}') for i in range(1, 6)]
        with patch.object(fallback, 'PAGE_SIZE', 2):
            results = self.collect(['000005', '000003', '000001', '000005', '999999'])
            self.assertEqual(self.urlopen.call_count, 5)  # 3 dividend + 1 undated + 1 rights
            self.assertEqual(list(results), ['000005', '000003', '000001', '999999'])
            self.assertEqual(len(results['000005']['events']), 1)
            self.assertEqual(results['999999']['events'], [])
            before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.folder.rglob('*') if p.is_file()}
            self.urlopen.side_effect = AssertionError('no refetch')
            self.assertEqual(self.collect(list(results)), results)
            self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns)
                                    for p in self.folder.rglob('*') if p.is_file()})

    def test_rights_details_are_not_a_free_bonus_or_dividend_pay_date(self):
        self.datasets['rights'] = [rights('000001')]
        result = self.collect()['000001']
        self.assertEqual(result['status'], 'ok')
        self.assertTrue(result['rights_issue_present'])
        event = result['metadata']['rights_events'][0]
        self.assertEqual((event['rights_per_10'], event['rights_per_share']), (3, .3))
        self.assertEqual(event['subscription_price'], 21.16)
        self.assertEqual(event['subscription_start'], '2023-11-30')
        self.assertEqual(event['ex_date'], '2023-12-08')
        self.assertEqual(event['listing_date'], '2023-12-25')
        self.assertFalse(event['simulated'])
        self.assertIsNone(result['events'][0]['pay_date'])
        self.assertEqual(len(result['events']), 1)

    def test_rights_unavailable_is_unknown_error_with_raw_denial_not_false(self):
        real = self.respond

        def denied(request, timeout):
            if fallback.RIGHTS in request.full_url:
                raise HTTPError(request.full_url, 403, 'denied', {}, io.BytesIO(b'public access denied'))
            return real(request, timeout)

        self.urlopen.side_effect = denied
        result = self.collect()['000001']
        self.assertEqual((result['status'], result['events']), ('error', []))
        self.assertIsNone(result['rights_issue_present'])
        self.assertEqual(result['metadata']['rights_history_status'], 'unknown')
        self.assertIn('403', result['error'])
        self.assertEqual(self.urlopen.call_count, 3)
        self.assertTrue((self.folder / 'raw' / f'{fallback._sha(b"public access denied")}.json').exists())
        self.assertFalse((self.folder / 'market_manifest.json').exists())

    def test_missing_rights_implementation_or_dates_is_not_no_rights(self):
        for changes in ({'ISSUE_NUM': None}, {'ISSUE_NUM': 0}, {'EX_DIVIDEND_DATE': None},
                        {'PLACING_RATIO': -1}, {'ISSUE_PRICE': float('inf')}):
            with self.subTest(changes=changes):
                with self.assertRaises((ValueError, TypeError)):
                    fallback._rights(rights(**changes), '2015-01-01', '2026-09-10')
        self.assertIsNone(fallback._rights(rights(), '2024-01-01', '2026-09-10'))
        # Ex-date precedes start but pending rights shares/listing still overlap.
        self.assertIsNotNone(fallback._rights(rights(), '2023-12-15', '2026-09-10'))

    def test_null_components_require_explicit_profile_and_totals_must_match(self):
        row = dividend(PRETAX_BONUS_RMB=None, IMPL_PLAN_PROFILE='10转2.00')
        self.assertEqual(fallback._dividend_amounts(row), (0, 1.2))
        row = dividend(BONUS_RATIO=3, IT_RATIO=2, BONUS_IT_RATIO=5,
                       PRETAX_BONUS_RMB=12.5, IMPL_PLAN_PROFILE='10送3.00转2.00派12.50元(含税)')
        self.assertEqual(fallback._dividend_amounts(row), (1.25, 1.5))
        for changes in ({'PRETAX_BONUS_RMB': None}, {'PRETAX_BONUS_RMB': .174},
                        {'BONUS_IT_RATIO': 20}, {'BONUS_RATIO': -1}, {'IT_RATIO': True},
                        {'PRETAX_BONUS_RMB': '--'}, {'IMPL_PLAN_PROFILE': None},
                        {'IMPL_PLAN_PROFILE': '每1股派1.74元'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                fallback._dividend_amounts(dividend(**changes))

    def test_undated_nonzero_never_disappears_and_zero_requires_explicit_amounts(self):
        self.datasets['undated'] = [dividend(EX_DIVIDEND_DATE=None)]
        result = self.collect()['000001']
        self.assertEqual((result['status'], result['events']), ('error', []))
        self.assertIn('no actual ex_date', result['error'])
        zero = dividend(EX_DIVIDEND_DATE=None, BONUS_RATIO=0, IT_RATIO=0,
                        BONUS_IT_RATIO=0, PRETAX_BONUS_RMB=0, IMPL_PLAN_PROFILE='不分配')
        self.assertIsNone(fallback._dividend(zero, '2015-01-01', '2026-09-10'))
        zero['PRETAX_BONUS_RMB'] = None
        with self.assertRaises(ValueError):
            fallback._dividend(zero, '2015-01-01', '2026-09-10')

    def test_proposals_unknown_status_conflicts_and_bad_dates(self):
        for status in ('预案', '股东大会预案', '取消分配'):
            self.assertIsNone(fallback._dividend(dividend(ASSIGN_PROGRESS=status,
                              PRETAX_BONUS_RMB='--'), '2015-01-01', '2026-09-10'))
        for changes in ({'ASSIGN_PROGRESS': '未知'}, {'EX_DIVIDEND_DATE': '2015-02-30'},
                        {'EQUITY_RECORD_DATE': '2015-04-14'}, {'NOTICE_DATE': None}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                fallback._dividend(dividend(**changes), '2015-01-01', '2026-09-10')
        self.datasets['dividends'] += [dividend(NOTICE_DATE='2015-04-08 00:00:00')]
        result = self.collect()['000001']
        self.assertEqual((result['status'], result['events']), ('error', []))
        self.assertIn('conflicting', result['error'])

    def test_truncated_api_failures_and_malformed_responses_are_not_empty_history(self):
        bad = [b'<html>blocked</html>', b'{}', b'{"success":false,"message":"bad"}',
               b'{"success":true,"code":0,"result":null}', page([], 1),
               page([], 0, 12), page([dividend()], 1, 9999)]
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                fallback._page(raw, 1)
        self.assertEqual(fallback._page(page([]), 1)['count'], 0)

    def test_mid_pagination_failure_no_partial_events_and_explicit_resume(self):
        self.datasets['dividends'] = [dividend(code=f'{i:06d}') for i in range(1, 4)]
        original = self.respond

        def fail_second(request, timeout):
            query = parse_qs(urlsplit(request.full_url).query)
            if query['pageNumber'] == ['2']:
                raise TimeoutError('offline second page')
            return original(request, timeout)

        with patch.object(fallback, 'PAGE_SIZE', 2):
            self.urlopen.side_effect = fail_second
            result = self.collect()['000001']
            self.assertEqual((result['status'], result['events']), ('error', []))
            self.assertEqual(self.urlopen.call_count, 4)
            self.urlopen.side_effect = self.respond
            result = self.collect()['000001']
            self.assertEqual(result['status'], 'ok')
            self.assertEqual(self.urlopen.call_count, 5)  # Only the missing page is fetched.

    def test_primary_success_never_overwritten_failed_bytes_preserved_separately(self):
        primary = self.folder.parent / 'sina'
        primary.mkdir()
        valid = sina._result('000001')
        valid.update(status='ok', response_sha256='a' * 64)
        good = primary / '000001.json'
        good.write_bytes(fallback._json_bytes(valid))
        failed = primary / '000002.json'
        bad_bytes = b'{"code":"000002","status":"error","error":"HTTP 456"}'
        failed.write_bytes(bad_bytes)
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in primary.iterdir()}
        results = self.collect(['000001', '000002'], primary_dir=primary)
        self.assertEqual(results['000001'], valid)
        self.assertEqual(results['000002']['status'], 'ok')
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in primary.iterdir()})
        proof = results['000002']['metadata']['primary_snapshot_evidence']
        self.assertEqual((self.folder / proof['path']).read_bytes(), bad_bytes)
        self.assertEqual(list((self.folder / 'results').glob('000001.*')), [])

    def test_all_primary_success_skips_network_and_creation(self):
        primary = self.folder.parent / 'sina'
        primary.mkdir()
        valid = sina._result('000001')
        valid.update(status='ok', response_sha256='a' * 64)
        (primary / '000001.json').write_bytes(fallback._json_bytes(valid))
        self.assertEqual(self.collect(primary_dir=primary), {'000001': valid})
        self.urlopen.assert_not_called()
        self.assertFalse(self.folder.exists())

    def test_immutable_window_hash_corruption_storage_errors_and_input_validation(self):
        self.collect()
        with self.assertRaises(ValueError):
            fallback.download_market_snapshot(self.folder, '2016-01-01', '2026-09-10')
        manifest = json.loads((self.folder / 'market_manifest.json').read_bytes())
        digest = manifest['datasets']['dividends']['pages'][0]['response_sha256']
        (self.folder / 'raw' / f'{digest}.json').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.collect()
        for codes in ('000001', ['../escape'], ['12345']):
            with self.assertRaises(ValueError):
                self.collect(codes)
        with self.assertRaises(ValueError):
            self.collect(primary_dir=self.folder)
        with self.assertRaises(ValueError):
            fallback._spec('2026-09-10', '2015-01-01')
        with patch('action_fallback.os.link', side_effect=OSError('disk failed')):
            with self.assertRaises(OSError):
                fallback._write_new(self.folder / 'new.json', b'{}')
        self.assertFalse((self.folder / 'new.json').exists())

    def test_access_denial_stops_remaining_endpoint_requests_and_keeps_evidence(self):
        self.urlopen.side_effect = lambda request, timeout: (_ for _ in ()).throw(
            HTTPError(request.full_url, 429, 'limited', {}, io.BytesIO(b'limit')))
        result = self.collect()['000001']
        self.assertEqual(self.urlopen.call_count, 1)
        self.assertIsNone(result['rights_issue_present'])
        self.assertEqual(result['status'], 'error')
        manifest = json.loads((self.folder / result['metadata']['snapshot_manifest']).read_bytes())
        proof = manifest['datasets']['dividends']['failure_evidence']
        self.assertEqual(proof['response_sha256'], fallback._sha(b'limit'))
        self.assertEqual(proof['http_status'], 429)

    def test_primary_conflicts_remain_unresolved_even_when_fallback_parses(self):
        primary = self.folder.parent / 'sina'
        primary.mkdir()
        (primary / '000001.json').write_bytes(fallback._json_bytes(dict(
            code='000001', status='error', error='parse: conflicting events on ex_date 2015-04-13')))
        result = self.collect(primary_dir=primary)['000001']
        self.assertEqual((result['status'], result['events']), ('error', []))
        self.assertTrue(result['metadata']['primary_conflict_unresolved'])
        self.assertEqual(len(result['metadata']['fallback_candidate_events']), 1)

    def test_cross_page_duplicates_or_changed_counts_fail_closed(self):
        for changed_count in (False, True):
            with tempfile.TemporaryDirectory() as folder, patch.object(fallback, 'PAGE_SIZE', 1):
                calls = [page([dividend()], 2),
                         page([dividend()], 3 if changed_count else 2)]
                self.urlopen.side_effect = lambda *a, **kw: Response(calls.pop(0))
                result = fallback._dataset(Path(folder), 'dividends',
                            fallback._queries(fallback._spec('2015-01-01', '2026-09-10'))['dividends'])
                self.assertEqual(result['status'], 'error')
                self.assertIn('pagination', result['error'])

    def test_overflow_cannot_escape_as_infinite_multiplier(self):
        with self.assertRaises(ValueError):
            fallback._dividend_amounts(dividend(BONUS_RATIO=1e308, IT_RATIO=1e308,
                BONUS_IT_RATIO=None, PRETAX_BONUS_RMB=0, IMPL_PLAN_PROFILE=''))

    def test_resolve_primary_preserved_new_metadata_and_runner_serializable(self):
        primary, original = self.primary_snapshot()
        self.freeze()
        path = primary / '000001.json'
        before = path.read_bytes(), path.stat().st_mtime_ns
        result = self.resolve(primary_dir=primary)['000001']
        self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)
        for field in ('events', 'source_url', 'response_sha256', 'retrieved_at'):
            self.assertEqual(result[field], original[field])
        self.assertEqual(result['status'], 'ok')
        self.assertIs(result['rights_issue_present'], False)
        self.assertEqual(result['metadata']['custom'], original['metadata']['custom'])
        self.assertEqual(result['metadata']['evidence'], original['metadata']['evidence'])
        self.assertFalse(result['metadata']['corporate_actions_reconciled'])
        self.assertEqual(result['metadata']['cash_events_source'], 'sina_primary')
        self.assertEqual(result['metadata']['cash_events_crosscheck']['status'],
                         'single_source_not_reconciled')
        proof = result['metadata']['primary_snapshot_evidence']
        self.assertEqual((self.folder / proof['path']).read_bytes(), before[0])
        crosscheck = result['metadata']['rights_crosscheck']
        self.assertEqual(crosscheck['rights_status'], 'no_reported_issues_in_window')
        self.assertFalse(crosscheck['absence_guaranteed'])
        raw = (self.folder / crosscheck['snapshot_manifest']).read_bytes()
        self.assertEqual(fallback._sha(raw), crosscheck['snapshot_manifest_sha256'])
        resolved = primary.parent / 'resolved' / '000001.json'
        resolved.parent.mkdir()
        resolved.write_bytes(fallback._json_bytes(result))  # Runner owns this publication.
        self.assertEqual(json.loads(resolved.read_bytes()), result)
        self.assertEqual(path.read_bytes(), before[0])
        self.assertEqual(self.resolve(primary_dir=primary)['000001'], result)
        self.urlopen.assert_not_called()

    def test_resolve_em_rights_override_false_despite_dividend_conflict(self):
        self.datasets['rights'] = [rights('000001')]
        self.datasets['dividends'] = [dividend(PRETAX_BONUS_RMB=999)]
        primary, original = self.primary_snapshot()
        self.freeze()
        result = self.resolve(primary_dir=primary)['000001']
        self.assertEqual(result['status'], 'ok', result.get('error'))
        self.assertIs(result['rights_issue_present'], True)
        self.assertEqual(result['events'], original['events'])
        meta = result['metadata']
        check = meta['rights_crosscheck']
        self.assertEqual(check['rights_status'], 'reported_issues')
        self.assertIs(check['primary_rights_issue_present'], False)
        self.assertIs(check['eastmoney_rights_issue_present'], True)
        self.assertIsNone(check['error'])
        self.assertEqual(check['rights_events'][0]['rights_per_share'], .3)
        self.assertEqual(check['evidence'][0]['dataset'], 'rights')
        self.assertEqual(json.loads(fallback._read_evidence(self.folder, check['evidence'][0]))
                         ['result']['data'], self.datasets['rights'])
        self.assertEqual(meta['cash_events_crosscheck']['eastmoney_status'], 'error')
        self.assertIn('conflicts', meta['cash_events_crosscheck']['eastmoney_error'])
        self.assertEqual(sina._read_valid_snapshot(primary / '000001.json', '000001'), original)
        self.urlopen.assert_not_called()

    def test_resolve_sina_true_not_overridden_by_em_false(self):
        primary, original = self.primary_snapshot(present=True)
        self.freeze()
        result = self.resolve(primary_dir=primary)['000001']
        self.assertEqual(result['status'], 'ok')
        self.assertIs(result['rights_issue_present'], True)
        check = result['metadata']['rights_crosscheck']
        self.assertIs(check['eastmoney_rights_issue_present'], False)
        self.assertIs(check['primary_rights_issue_present'], True)
        self.assertEqual(result['events'], original['events'])

    def test_resolve_unknown_rights_rejects_primary_false_and_preserves_cash(self):
        self.datasets['rights'] = [rights('000001', EX_DIVIDEND_DATE=None)]
        primary, original = self.primary_snapshot()
        self.freeze()
        result = self.resolve(primary_dir=primary)['000001']
        self.assertEqual(result['status'], 'error')
        self.assertIsNone(result['rights_issue_present'])
        self.assertEqual(result['events'], original['events'])
        self.assertIn('rights crosscheck unknown', result['error'])
        check = result['metadata']['rights_crosscheck']
        self.assertEqual((check['source_status'], check['rights_status']), ('ok', 'unknown'))
        self.assertIn('without actual ex_date', check['error'])
        self.assertEqual(len(check['evidence']), 1)

    def test_resolve_unknown_rights_keeps_sina_true_but_still_errors(self):
        self.datasets['rights'] = [rights('000001', ISSUE_NUM=None)]
        primary, original = self.primary_snapshot(present=True)
        self.freeze()
        result = self.resolve(primary_dir=primary)['000001']
        self.assertEqual(result['status'], 'error')
        self.assertIs(result['rights_issue_present'], True)
        self.assertIsNone(result['metadata']['rights_crosscheck']['eastmoney_rights_issue_present'])
        self.assertEqual(result['events'], original['events'])

    def test_resolve_checks_all_rights_rows_known_em_true_survives_unknown(self):
        self.datasets['rights'] = [rights('000001', ISSUE_NUM=None), rights('000001')]
        primary, original = self.primary_snapshot()
        self.freeze()
        result = self.resolve(primary_dir=primary)['000001']
        self.assertEqual(result['status'], 'error')
        self.assertIs(result['rights_issue_present'], True)
        self.assertEqual(result['events'], original['events'])
        check = result['metadata']['rights_crosscheck']
        self.assertEqual(check['rights_status'], 'unknown')
        self.assertIs(check['eastmoney_rights_issue_present'], True)
        self.assertEqual(len(check['evidence']), 2)
        self.assertEqual(len(check['rights_events']), 1)
        # Legacy collect still stops at the first error, unchanged.
        legacy = self.collect()['000001']
        self.assertIsNone(legacy['rights_issue_present'])
        self.assertNotIn('rights_source_status', legacy['metadata'])

    def test_resolve_dividend_source_failure_is_not_rights_unknown(self):
        primary, original = self.primary_snapshot()
        real = self.respond

        def fail_dividends(request, timeout):
            if fallback.RIGHTS not in request.full_url:
                raise TimeoutError('dividends unavailable')
            return real(request, timeout)

        self.urlopen.side_effect = fail_dividends
        result = self.resolve(primary_dir=primary)['000001']
        self.assertEqual(result['status'], 'ok', result.get('error'))
        self.assertIs(result['rights_issue_present'], False)
        self.assertEqual(result['events'], original['events'])
        check = result['metadata']['rights_crosscheck']
        self.assertEqual(check['source_status'], 'ok')
        self.assertEqual(check['rights_status'], 'no_reported_issues_in_window')
        self.assertEqual(result['metadata']['cash_events_crosscheck']['eastmoney_status'], 'error')

    def test_resolve_rights_source_failure_rejects_primary_false_with_evidence(self):
        primary, original = self.primary_snapshot()
        real = self.respond

        def fail_rights(request, timeout):
            if fallback.RIGHTS in request.full_url:
                raise HTTPError(request.full_url, 403, 'denied', {}, io.BytesIO(b'rights denied'))
            return real(request, timeout)

        self.urlopen.side_effect = fail_rights
        result = self.resolve(primary_dir=primary)['000001']
        self.assertEqual(result['status'], 'error')
        self.assertIsNone(result['rights_issue_present'])
        self.assertEqual(result['events'], original['events'])
        check = result['metadata']['rights_crosscheck']
        self.assertEqual((check['source_status'], check['rights_status']), ('error', 'unknown'))
        self.assertIn('403', check['error'])
        manifest = json.loads((self.folder / check['snapshot_manifest']).read_bytes())
        proof = manifest['datasets']['rights']['failure_evidence']
        self.assertEqual(fallback._read_evidence(self.folder, proof), b'rights denied')

    def test_resolve_one_index_and_one_read_per_page_for_mixed_batch(self):
        codes = [f'{i:06d}' for i in range(1, 13)]
        self.datasets['rights'] = [rights(code) for code in codes]
        for code in codes[:8]:
            primary, _ = self.primary_snapshot(code)
        self.freeze()
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in primary.iterdir()}
        with patch.object(fallback, '_index', wraps=fallback._index) as index, \
                patch.object(fallback, '_read_evidence', wraps=fallback._read_evidence) as read, \
                patch.object(fallback, 'fetch_action_history', side_effect=AssertionError('per-code I/O')), \
                patch.object(fallback, 'download_market_snapshot', side_effect=AssertionError('refreeze')):
            results = self.resolve(iter(codes + codes[:3]), primary_dir=primary)
        self.assertEqual(index.call_count, 1)
        self.assertEqual(read.call_count, 3)  # One dividend, undated and rights page, not 12 * 3.
        self.assertEqual(list(results), codes)
        self.assertTrue(all(r['status'] == 'ok' and r['rights_issue_present'] is True
                            for r in results.values()))
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in primary.iterdir()})
        self.urlopen.assert_not_called()

    def test_resolve_failed_primary_conflict_and_missing_primary_match_collect(self):
        primary = self.folder.parent / 'sina'
        primary.mkdir()
        raw = fallback._json_bytes(dict(code='000001', status='error', error='conflicting events'))
        (primary / '000001.json').write_bytes(raw)
        self.freeze()
        legacy = self.collect(['000001', '000002'], primary_dir=primary)
        results = self.resolve(['000001', '000002'], primary_dir=primary)
        for code in results:
            for field in ('status', 'events', 'rights_issue_present', 'error', 'source_url'):
                self.assertEqual(results[code].get(field), legacy[code].get(field))
        self.assertTrue(results['000001']['metadata']['primary_conflict_unresolved'])
        self.assertEqual((primary / '000001.json').read_bytes(), raw)
        self.assertFalse((primary / '000002.json').exists())

    def test_resolve_metadata_deep_copy_even_if_original_metadata_is_not_a_dict(self):
        _, original = self.primary_snapshot()
        candidate = self.resolve()['000001']
        before = deepcopy((original, candidate))
        result = fallback._resolve_primary(original, candidate)
        result['events'][0]['cash_per_share'] = 100
        result['metadata']['custom']['keep'].append('changed')
        result['metadata']['rights_crosscheck']['evidence'].append({'changed': True})
        self.assertEqual((original, candidate), before)
        original['metadata'] = None
        self.assertIsInstance(fallback._resolve_primary(original, candidate)['metadata'], dict)
        self.assertIsNone(original['metadata'])

    def test_resolve_input_window_and_frozen_hash_validation(self):
        primary, _ = self.primary_snapshot()
        for codes in ('000001', ['../escape'], ['12345']):
            with self.assertRaises(ValueError):
                self.resolve(codes, primary_dir=primary)
        self.assertEqual(self.resolve([], primary_dir=primary), {})
        self.urlopen.assert_not_called()
        with self.assertRaises(ValueError):
            self.resolve(primary_dir=self.folder)
        manifest = self.freeze()
        with self.assertRaisesRegex(ValueError, 'window mismatch'):
            fallback.resolve_action_snapshot(['000001'], self.folder, start='2016-01-01',
                                             end='2026-09-10', primary_dir=primary)
        digest = manifest['datasets']['rights']['pages'][0]['response_sha256']
        (self.folder / 'raw' / f'{digest}.json').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.resolve(primary_dir=primary)
        self.urlopen.assert_not_called()


if __name__ == '__main__':
    unittest.main()