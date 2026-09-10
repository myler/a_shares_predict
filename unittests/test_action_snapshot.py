"""Offline Sina HTML contracts. All network is mocked; files use temp dirs only."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import action_snapshot as snapshot


HEADERS = ('公告日期', '送股(股)', '转增(股)', '派息(税前)(元)', '方案进度',
           '除权除息日', '股权登记日', '红股上市日', '查看详细')


def dividend_row(announce='2026-06-01', bonus='0', transfer='0', cash='2.5',
                 status='实施', ex='2026-06-10', record='2026-06-09'):
    # Exact Sina positions, including listing date which is NOT pay_date.
    cells = (announce, bonus, transfer, cash, status, ex, record,
             '2026-06-15', '<a href="detail">查看</a>')
    return '<tr>' + ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>'


def dividend_table(rows='', table_id='sharebonus_1'):
    identity = f' id="{table_id}"' if table_id else ''
    header = ''.join(f'<th>{label}</th>' for label in HEADERS)
    return (f'<table{identity}><thead><tr><th colspan="9">分红</th></tr>'
            f'<tr>{header}</tr></thead><tbody>{rows}</tbody></table>')


def no_data_table():
    return dividend_table('<tr><td colspan="9">&nbsp;暂时没有数据&#160;</td></tr>')


def rights_table(status='实施', day='2026-05-01', table_id='sharebonus_2'):
    return (f'<table id="{table_id}"><tr><th>公告日期</th><th>配股方案</th>'
            '<th>配股价格</th><th>方案进度</th><th>除权日</th></tr>'
            f'<tr><td>{day}</td><td>10配3</td><td>8</td><td>{status}</td>'
            f'<td>{day}</td></tr></table>')


class OfflineTest(unittest.TestCase):
    def setUp(self):
        patcher = patch('action_snapshot.urllib.request.urlopen',
                        side_effect=AssertionError('unexpected network access'))
        self.urlopen = patcher.start()
        self.addCleanup(patcher.stop)

    def fetch(self, markup, code='000001', encoding='gbk'):
        raw = markup.encode(encoding)
        self.urlopen.side_effect = lambda *args, **kwargs: io.BytesIO(raw)
        return snapshot.fetch_action_history(code)

    def valid_result(self, code='000001'):
        return self.fetch(dividend_table(dividend_row()), code)


class TestFetchActionHistory(OfflineTest):
    def test_exact_sina_positions_units_headers_and_byte_provenance(self):
        markup = dividend_table(dividend_row(bonus='2', transfer='3', cash='12.5'))
        result = self.fetch(markup)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['events'], [{
            'ex_date': '2026-06-10', 'cash_per_share': 1.25,
            'share_multiplier': 1.5, 'record_date': '2026-06-09',
            'announce_date': '2026-06-01', 'pay_date': None,
        }])
        self.assertFalse(result['rights_issue_present'])
        self.assertEqual(result['response_sha256'], hashlib.sha256(markup.encode('gbk')).hexdigest())
        self.assertEqual(result['source_url'], snapshot.SOURCE_URL.format(code='000001'))
        self.assertIsNotNone(datetime.fromisoformat(result['retrieved_at']).utcoffset())
        self.assertNotIn('raw_html', result)
        self.assertNotIn('error', result)
        request = self.urlopen.call_args.args[0]
        self.assertEqual(request.full_url, result['source_url'])
        headers = {key.lower(): value for key, value in request.header_items()}
        self.assertEqual(headers['user-agent'], 'Mozilla/5.0')
        self.assertEqual(headers['referer'], 'https://finance.sina.com.cn')
        self.assertEqual(self.urlopen.call_args.kwargs, {'timeout': 10})
        self.assertEqual(self.urlopen.call_count, 1)
        metadata = result['metadata']
        for flag in ('supplier_history_completeness_guaranteed',
                     'corporate_actions_reconciled', 'cash_payment_date_verified',
                     'rights_issue_simulated'):
            self.assertIs(metadata[flag], False)
        json.dumps(result, allow_nan=False)

    def test_nested_tags_tables_entities_nbsp_and_omitted_cell_end_tags(self):
        row = dividend_row(
            announce='<a><span>2026-06-01</span></a>',
            bonus='&nbsp;<b>2</b>&#160;', transfer='\u00a03\u00a0',
            cash='&#49;&#50;.5', status='<span>实<b>施</b></span>')
        row = row.replace('</td>', '').replace('</tr>', '')
        inner = dividend_table(row).replace('id="sharebonus_1"', "id='sharebonus_1'")
        markup = '<table><tr><td><div>' + inner + '</div></td></tr></table>'
        result = self.fetch(markup, encoding='utf-8')
        self.assertEqual(result['status'], 'ok', result.get('error'))
        self.assertEqual(len(result['events']), 1)
        self.assertEqual(result['events'][0]['cash_per_share'], 1.25)
        self.assertEqual(result['events'][0]['share_multiplier'], 1.5)

    def test_zeros_are_valid_and_unknown_record_date_is_not_announcement(self):
        result = self.fetch(dividend_table(dividend_row(cash='0', record='&nbsp;--')))
        self.assertEqual(result['status'], 'ok')
        event = result['events'][0]
        self.assertEqual((event['cash_per_share'], event['share_multiplier']), (0., 1.))
        self.assertIsNone(event['record_date'])
        self.assertEqual(event['announce_date'], '2026-06-01')
        self.assertIsNone(event['pay_date'])

    def test_no_data_is_ok_but_missing_empty_and_truncated_table_are_errors(self):
        result = self.fetch(no_data_table())
        self.assertEqual((result['status'], result['events']), ('ok', []))
        for markup in ('<html>暂时没有数据</html>', '<table><tr><td>广告</td></tr></table>',
                       rights_table(), dividend_table(), no_data_table().replace('</table>', '')):
            with self.subTest(markup=markup):
                result = self.fetch(markup)
                self.assertEqual((result['status'], result['events']), ('error', []))
                self.assertTrue(result['error'])
                self.assertEqual(result['response_sha256'], hashlib.sha256(markup.encode('gbk')).hexdigest())

    def test_implemented_only_not_proposal_cancelled_or_nonimplemented(self):
        rows = ''.join(dividend_row(status=status, cash='--', ex='--')
                       for status in ('预案', '股东大会预案', '取消', '未实施'))
        result = self.fetch(dividend_table(rows))
        self.assertEqual((result['status'], result['events']), ('ok', []))
        result = self.fetch(dividend_table(rows + dividend_row()))
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(len(result['events']), 1)

    def test_id_priority_header_fallback_and_ambiguous_fallback(self):
        fallback = dividend_table(dividend_row(cash='90'), table_id='other')
        result = self.fetch(fallback + no_data_table())
        self.assertEqual((result['status'], result['events']), ('ok', []))
        result = self.fetch(fallback)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['events'][0]['cash_per_share'], 9.)
        result = self.fetch(fallback + dividend_table(dividend_row(), table_id='another'))
        self.assertEqual(result['status'], 'error')
        result = self.fetch(fallback + dividend_table())
        self.assertEqual(result['status'], 'error')  # Do not bypass broken primary.
        result = self.fetch(no_data_table() + no_data_table())
        self.assertEqual(result['status'], 'error')

    def test_fallback_multirow_th_legacy_td_and_unrelated_prose(self):
        multirow = ('<table><tr><th rowspan="2">公告日期</th><th colspan="3">分红方案</th>'
                    '<th>除权除息日</th></tr><tr><th>送股(股)</th><th>转增(股)</th>'
                    '<th>派息(元)</th></tr>' + dividend_row() + '</table>')
        legacy = (dividend_table(dividend_row(), table_id='')
                  .replace('<th>', '<td>').replace('<th ', '<td ')
                  .replace('</th>', '</td>'))
        for markup in (multirow, legacy):
            with self.subTest(markup=markup):
                result = self.fetch(markup)
                self.assertEqual(result['status'], 'ok', result.get('error'))
                self.assertEqual(len(result['events']), 1)
        unrelated = ('<p>派息 送股 除权除息日</p><table><tr><td>派息说明</td></tr>'
                     + dividend_row() + '</table>')
        self.assertEqual(self.fetch(unrelated)['status'], 'error')

    def test_rights_are_flagged_not_simulated_and_survive_dividend_errors(self):
        for status, day, expected in (('实施', '2026-05-01', True),
                                       ('预案', '2026-05-01', False),
                                       ('未实施', '2026-05-01', False),
                                       ('实施', '--', False),
                                       ('实施', '2026-02-30', False)):
            with self.subTest(status=status, day=day):
                result = self.fetch(no_data_table() + rights_table(status, day))
                self.assertEqual(result['status'], 'ok')
                self.assertEqual(result['rights_issue_present'], expected)
                self.assertEqual(result['events'], [])
        result = self.fetch(rights_table())
        self.assertEqual(result['status'], 'error')
        self.assertTrue(result['rights_issue_present'])
        result = self.fetch(no_data_table() + rights_table(table_id='rights'))
        self.assertTrue(result['rights_issue_present'])

    def test_sort_deduplicate_and_reject_same_ex_date_conflicts(self):
        older = dividend_row(announce='1999-06-01', ex='1999-06-10', record='1999-06-09')
        current = dividend_row()
        result = self.fetch(dividend_table(current + older + current))
        self.assertEqual(result['status'], 'ok')
        self.assertEqual([event['ex_date'] for event in result['events']], ['1999-06-10', '2026-06-10'])
        for changes in ({'cash': '3'}, {'bonus': '1'}, {'transfer': '2'},
                        {'record': '2026-06-08'}, {'announce': '2026-06-02'}):
            with self.subTest(changes=changes):
                result = self.fetch(dividend_table(older + current + dividend_row(**changes)))
                self.assertEqual((result['status'], result['events']), ('error', []))
                self.assertIn('conflicting', result['error'])

    def test_untrusted_numeric_never_coerced_to_zero_or_skipped(self):
        for field in ('cash', 'bonus', 'transfer'):
            for bad in ('--', '', '&nbsp;', 'nan', 'NaN', 'inf', '-Infinity',
                        '-0.01', '1e309', '1_0', '1,000', '1元', '1&nbsp;2'):
                with self.subTest(field=field, bad=bad):
                    result = self.fetch(dividend_table(dividend_row(**{field: bad})))
                    self.assertEqual((result['status'], result['events']), ('error', []))
                    json.dumps(result, allow_nan=False)
        result = self.fetch(dividend_table(dividend_row(bonus='1e308', transfer='1e308')))
        self.assertEqual(result['status'], 'error')

    def test_bad_dates_short_rows_and_mixed_empty_marker_fail_closed(self):
        for field in ('announce', 'ex', 'record'):
            for bad in ('2026-02-30', '2026-13-01', 'invalid'):
                with self.subTest(field=field, bad=bad):
                    result = self.fetch(dividend_table(dividend_row(**{field: bad})))
                    self.assertEqual((result['status'], result['events']), ('error', []))
        for row in ('<tr><td>2026-06-01</td><td>0</td></tr>',
                    dividend_row(ex='--'), dividend_row(status=''),
                    dividend_row().replace('<td>0</td>', '', 1),
                    dividend_row() + '<tr><td>暂时没有数据</td></tr>'):
            with self.subTest(row=row):
                self.assertEqual(self.fetch(dividend_table(row))['status'], 'error')

    def test_transport_retry_is_bounded_and_does_not_sleep(self):
        raw = no_data_table().encode('gbk')
        self.urlopen.side_effect = [URLError('temporary'), io.BytesIO(raw)]
        with patch('time.sleep', side_effect=AssertionError('must not sleep')):
            result = snapshot.fetch_action_history('000001')
        self.assertEqual(result['status'], 'ok')
        self.assertNotIn('error', result)
        self.assertEqual(self.urlopen.call_count, 2)
        self.urlopen.reset_mock()
        self.urlopen.side_effect = TimeoutError('offline')
        result = snapshot.fetch_action_history('000001')
        self.assertEqual(self.urlopen.call_count, 2)
        self.assertEqual((result['status'], result['events']), ('error', []))
        self.assertIsNone(result['response_sha256'])
        self.assertIn('2/2', result['error'])

    def test_malformed_response_not_retried_and_raw_download_available(self):
        for raw in (b'', b'\x81', b'<html>blocked</html>'):
            with self.subTest(raw=raw):
                self.urlopen.reset_mock()
                self.urlopen.side_effect = lambda *args, **kwargs: io.BytesIO(raw)
                result = snapshot.fetch_action_history('000001')
                self.assertEqual(result['status'], 'error')
                self.assertEqual(result['response_sha256'], hashlib.sha256(raw).hexdigest())
                self.assertEqual(self.urlopen.call_count, 1)
        raw = no_data_table().encode('gbk')
        self.urlopen.side_effect = lambda *args, **kwargs: io.BytesIO(raw)
        self.assertEqual(snapshot.download_action_response('000001'), raw)

    def test_code_validation_precedes_network(self):
        for code in ('../000001', '00001', '1234567', '１２３４５６', 1, None, True):
            with self.subTest(code=code), self.assertRaises(ValueError):
                snapshot.fetch_action_history(code)
        self.urlopen.assert_not_called()
        self.assertEqual(self.fetch(no_data_table(), code=' 000001 ')['code'], '000001')

    def test_only_explicit_zero_implemented_row_can_have_missing_ex_date(self):
        for missing in ('--', '', '—'):
            result = self.fetch(dividend_table(dividend_row(cash='0', ex=missing, record='--')))
            self.assertEqual((result['status'], result['events']), ('ok', []))
            self.assertEqual(result['metadata']['ignored_zero_undated_rows'], 1)
        for change in ({'cash': '0.01'}, {'bonus': '1'}, {'transfer': '1'},
                       {'cash': '--'}, {'bonus': '--'}, {'transfer': '--'},
                       {'announce': '2026-02-30'}, {'record': '2026-02-30'}):
            args = dict(cash='0', ex='--', record='--')
            args.update(change)
            with self.subTest(change=change):
                self.assertEqual(self.fetch(dividend_table(dividend_row(**args)))['status'], 'error')
        # A zero row cannot make real same-ex-date contradictions disappear.
        rows = dividend_row(cash='0', ex='--') + dividend_row() + dividend_row(cash='5')
        result = self.fetch(dividend_table(rows))
        self.assertEqual((result['status'], result['events']), ('error', []))
        self.assertIn('conflicting', result['error'])

    def test_access_denials_are_not_retried_or_redirected_to_another_source(self):
        for status in (401, 403, 429, 456):
            with self.subTest(status=status):
                self.urlopen.reset_mock()
                self.urlopen.side_effect = HTTPError('https://sina.invalid', status, 'denied', {}, None)
                result = snapshot.fetch_action_history('000001')
                self.assertEqual((result['status'], result['events']), ('error', []))
                self.assertEqual(self.urlopen.call_count, 1)
                self.assertEqual(result['metadata']['http_status'], status)
                self.assertTrue(result['metadata']['access_denial_not_retried'])


class TestCollectActionSnapshot(OfflineTest):
    def test_successful_resume_preserves_bytes_and_deduplicates_codes(self):
        expected = self.valid_result()
        with tempfile.TemporaryDirectory() as directory:
            with patch('action_snapshot.fetch_action_history', return_value=expected) as fetch:
                first = snapshot.collect_action_snapshot(['000001', ' 000001 '], directory)
            fetch.assert_called_once_with('000001')
            path = Path(directory) / '000001.json'
            before = path.read_bytes(), path.stat().st_mtime_ns
            with patch('action_snapshot.fetch_action_history', side_effect=AssertionError('must resume')) as fetch:
                second = snapshot.collect_action_snapshot(['000001'], directory)
            fetch.assert_not_called()
            self.assertEqual(first, second)
            self.assertEqual(second['000001'], expected)
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_failures_saved_then_retried_once_per_code_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            self.urlopen.side_effect = URLError('offline')
            results = snapshot.collect_action_snapshot(['000001', '000002'], directory)
            self.assertEqual(self.urlopen.call_count, 4)
            for code, result in results.items():
                self.assertEqual(result['status'], 'error')
                self.assertEqual(json.loads((Path(directory) / f'{code}.json').read_text(encoding='utf-8')), result)
            raw = no_data_table().encode('gbk')
            self.urlopen.side_effect = lambda *args, **kwargs: io.BytesIO(raw)
            results = snapshot.collect_action_snapshot(['000001', '000002'], directory)
            self.assertTrue(all(result['status'] == 'ok' for result in results.values()))
            self.assertEqual(self.urlopen.call_count, 6)

    def test_corrupt_mismatched_or_invalid_ok_cache_is_refetched(self):
        valid = self.valid_result()
        invalids = ['{not json', json.dumps({'code': '000001', 'status': 'ok'})]
        for field, value in (('code', '000002'), ('events', 'not a list'),
                             ('response_sha256', 'bad'), ('source_url', 'wrong'),
                             ('rights_issue_present', 1), ('retrieved_at', 'yesterday')):
            changed = deepcopy(valid)
            changed[field] = value
            invalids.append(json.dumps(changed))
        for field, value in (('cash_per_share', float('nan')), ('share_multiplier', .5),
                             ('cash_per_share', True), ('pay_date', '2026-06-10')):
            changed = deepcopy(valid)
            changed['events'][0][field] = value
            invalids.append(json.dumps(changed))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '000001.json'
            for text in invalids:
                with self.subTest(text=text):
                    path.write_text(text, encoding='utf-8')
                    with patch('action_snapshot.fetch_action_history', return_value=valid) as fetch:
                        results = snapshot.collect_action_snapshot(['000001'], directory)
                    fetch.assert_called_once_with('000001')
                    self.assertEqual(results['000001'], valid)
                    self.assertEqual(json.loads(path.read_text(encoding='utf-8')), valid)

    def test_atomic_replace_and_late_valid_snapshot_are_preserved(self):
        valid = self.valid_result()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '000001.json'
            real_replace = snapshot.os.replace

            def check_replace(source, destination):
                self.assertEqual(Path(source).parent, Path(directory))
                self.assertEqual(Path(destination), path)
                self.assertEqual(json.loads(Path(source).read_text(encoding='utf-8')), valid)
                return real_replace(source, destination)

            with patch('action_snapshot.fetch_action_history', return_value=valid), \
                    patch('action_snapshot.os.replace', side_effect=check_replace) as replace:
                snapshot.collect_action_snapshot(['000001'], directory)
            replace.assert_called_once()
            before = path.read_bytes()
            failed = deepcopy(valid)
            failed.update(status='error', error='later failure', events=[])
            # Another collector may finish between the first cache check and save.
            self.assertEqual(snapshot._save_snapshot(path, failed), valid)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_storage_failure_propagates_and_temporary_file_is_cleaned(self):
        valid = self.valid_result()
        with tempfile.TemporaryDirectory() as directory:
            with patch('action_snapshot.fetch_action_history', return_value=valid), \
                    patch('action_snapshot.os.replace', side_effect=OSError('disk error')):
                with self.assertRaises(OSError):
                    snapshot.collect_action_snapshot(['000001'], directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_worker_cap_all_results_input_order_and_progress_every_100(self):
        valid = self.valid_result()
        codes = [f'{number:06d}' for number in range(205)]

        def fetch(code):
            result = deepcopy(valid)
            result.update(code=code, source_url=snapshot.SOURCE_URL.format(code=code))
            return result

        with tempfile.TemporaryDirectory() as directory:
            with patch('action_snapshot.fetch_action_history', side_effect=fetch), \
                    patch('action_snapshot.ThreadPoolExecutor', wraps=ThreadPoolExecutor) as pool, \
                    patch('builtins.print') as progress:
                results = snapshot.collect_action_snapshot(codes, directory, workers=99)
            pool.assert_called_once_with(max_workers=8)
            self.assertEqual(list(results), codes)
            self.assertEqual(len(list(Path(directory).glob('*.json'))), 205)
            self.assertEqual([call.args[0] for call in progress.call_args_list],
                             ['action snapshots: 100/205', 'action snapshots: 200/205'])

    def test_unexpected_worker_error_is_saved_without_retry_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('action_snapshot.fetch_action_history', side_effect=RuntimeError('broken')) as fetch:
                result = snapshot.collect_action_snapshot(['000001'], directory)['000001']
            fetch.assert_called_once_with('000001')
            self.assertEqual(result['status'], 'error')
            self.assertIn('broken', result['error'])
            self.assertTrue((Path(directory) / '000001.json').exists())

    def test_empty_inputs_and_validation_do_not_fetch_or_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'not_created'
            self.assertEqual(snapshot.collect_action_snapshot([], output), {})
            for workers in (0, -1, True, 1.5):
                with self.subTest(workers=workers), self.assertRaises(ValueError):
                    snapshot.collect_action_snapshot(['000001'], output, workers=workers)
            for codes in ('000001', ['000001', '../escape']):
                with self.subTest(codes=codes), self.assertRaises(ValueError):
                    snapshot.collect_action_snapshot(codes, output)
            self.assertFalse(output.exists())
            self.urlopen.assert_not_called()


if __name__ == '__main__':
    unittest.main()