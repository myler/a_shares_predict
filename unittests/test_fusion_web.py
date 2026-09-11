"""Only this suite: synthetic in-memory analysis, no real HTTP or database.

The one chart smoke test renders a single synthetic daily bar in memory. No
analysis refresh, report loading, backtest, or other agent's suite is executed.
"""

import base64
from copy import deepcopy
from datetime import date, timedelta
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

import fusion_web as web
from fusion_chart import render_chart as real_render_chart
from live_data import RefreshError
from pareto_strategy import OBJECTIVE_NAMES, RiskConfig


PNG = ('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8'
       '/x8AAwMCAO+a9mQAAAAASUVORK5CYII=')


class Node:
    def __init__(self, tag='', attrs=()):
        self.tag, self.attrs, self.children = tag, dict(attrs), []

    @property
    def text(self):
        return ''.join(c.text if isinstance(c, Node) else c for c in self.children)

    def find(self, tag=None, css=None):
        result = []
        for child in self.children:
            if isinstance(child, Node):
                if ((tag is None or child.tag == tag)
                        and (css is None or css in child.attrs.get('class', '').split())):
                    result.append(child)
                result.extend(child.find(tag, css))
        return result


class Document(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.root = Node()
        self.stack = [self.root]
        self.feed(html)
        self.close()

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in ('img', 'br', 'hr', 'input', 'meta', 'link'):
            self.stack.append(node)

    def handle_endtag(self, tag):
        if len(self.stack) <= 1 or self.stack[-1].tag != tag:
            raise AssertionError('unbalanced HTML: ' + tag)
        self.stack.pop()

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def fixture(n=300):
    days = (np.datetime64('2026-09-09') - np.arange(n - 1, -1, -1)).astype('U10')
    close = 20 + np.arange(n) * .01
    raw = np.column_stack((close, close + .5, close - .5, close, np.full(n, 1000.)))
    votes = np.tile(np.resize(np.array([-1, 0, 1], dtype=np.int8), 34), (n, 1))
    actions = dict(status='ok', rights_issue_present=False, retrieved_at='2026-09-10T10:00:01+08:00',
                   metadata=dict(live_recommendation_blocked=False, notes=['现金基数以各次权息前股计。']),
                   events=[dict(ex_date='2025-09-10', cash_per_share=8., share_multiplier=1.),
                           dict(ex_date='2025-09-11', cash_per_share=.2, share_multiplier=1.5),
                           dict(ex_date='2026-09-10', cash_per_share=.3, share_multiplier=1.)])
    # Actual stock initial capital intentionally differs from the model's 1m.
    account = dict(code='000001', name='历史股票名', initial_capital=200_000.,
                   final_cash=180_000., residual_value=5_000., marked_nav=185_000.,
                   pnl=-15_000., fees=321., cash_dividends=123., shares=500.,
                   end='2026-09-08', last_date='2026-09-07', build_count=6,
                   add_count=6, reduce_count=6, exit_count=6, unfilled_count=2, action_count=1,
                   years={'2023': dict(date='2023-12-29', nav=210_000., cash=210_000., shares=0),
                          '2026': dict(date='2026-09-08', nav=185_000., cash=180_000., shares=500)})
    ledger = []
    for i in range(24):
        day = date(2026, 8, 1) + timedelta(days=i)
        ledger.append(dict(day=day.isoformat(), signal_day=(day - timedelta(days=1)).isoformat(),
                           event=('build', 'add', 'reduce', 'exit')[i % 4],
                           quantity=100, price=20.01, fee=5., cash=180_000., shares=500,
                           reason=f'fill-{i:03d}'))
    ledger += [dict(day='2026-08-25', event='unfilled', reason='not-a-fill'),
               dict(day='2026-08-26', event='action', reason='dividend-not-a-fill',
                    cash_per_share=.2, share_multiplier=1.5)]
    history = dict(manifest=dict(start='2019-01-02', requested_end='2026-09-10',
                                 effective_end='2026-09-08', limitations=['历史冻结数据不是今日行情。']),
                   summary=dict(totals=dict(pnl=999_999_999.), verdict='模拟亏损，无盈利保证',
                                limitations=['历史后段已看过，不是盲测。']),
                   account=account, ledger=ledger, error=None,
                   risk=RiskConfig(annual_vol_target=.10), folder='/never/read/history')
    return dict(code='000001', name='模拟股票', holding=False,
                fresh=dict(checked_at='2026-09-10T10:00:00+08:00', latest_date='2026-09-09',
                           date='2026-09-09', provider='sina', actions=actions,
                           warnings=['来源最新完整日线早于核验日。'],
                           evidence_dir='/never/expose/live', refresh_id='mock-refresh'),
                dates=days, raw=raw, adjusted=raw.copy(), votes=votes,
                eligible=n >= 251, gate=True, verified_actions=True, rsi=56.2, obv_flow=.12,
                annual_vol=.20, daily_es=.03, conditional_cap=.50,
                comparison=dict(status='unknown', compared=0, expected=3330,
                                coverage='无同日可比截面', dominators=[], date='2026-09-09'),
                action='暂不下买卖结论', symbol='⚠️', css='warn', reason='缺少同日可比截面，不能确认F1',
                history=history, warnings=['个人账户约束未提供。'])


class TestFusionWeb(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.temp = Path(temp.name)
        # Patches block even accidental fallback to production helpers.
        for target in ('socket.create_connection', 'socket.socket.connect',
                       'socket.socket.connect_ex', 'socket.getaddrinfo', 'sqlite3.connect',
                       'live_data.refresh_stock', 'pareto_web._load_run', 'pareto_web._json'):
            guard = patch(target, side_effect=AssertionError('no real network/DB/report reads'))
            guard.start()
            self.addCleanup(guard.stop)
        for target, value in (('live_data.SNAPSHOT_ROOT', self.temp / 'refresh'),
                              ('pareto_web.RUN_DIR', self.temp / 'report')):
            guard = patch(target, value)
            guard.start()
            self.addCleanup(guard.stop)
        self.a = fixture()
        analyze = patch.object(web, 'analyze_stock', return_value=self.a)
        chart = patch.object(web, 'render_chart', return_value=PNG)
        self.analyze, self.chart = analyze.start(), chart.start()
        self.addCleanup(analyze.stop)
        self.addCleanup(chart.stop)

    def render(self, **kwargs):
        html = web.render_analysis('000001', **kwargs)
        parsed = Document(html)
        self.assertEqual(len(parsed.stack), 1)
        return html, parsed.root

    def test_layout_conclusion_summary_indicators_history_then_last_image(self):
        html, dom = self.render(calc_dividend=True)
        positions = [html.index(value) for value in (
            'class="conclusion warn"', 'class="top-row"', 'class="strategy indicators"',
            'class="historical-detail"', 'class="dividend-history"', 'class="strategy methodology"', '<img ')]
        self.assertEqual(positions, sorted(positions))
        result = dom.find(css='result')[0]
        self.assertEqual(result.children[0].attrs['class'], 'conclusion warn')
        self.assertEqual(result.children[-1].tag, 'img')
        self.assertEqual(len(dom.find(css='top-row')[0].find('table')), 2)
        self.assertEqual(dom.find(css='conclusion')[0].text,
                         '⚠️ 暂不下买卖结论 · 数据日期 2026-09-09')
        self.assertIn(web._LEGEND, dom.text)
        self.assertEqual(dom.find('img')[0].attrs['src'], 'data:image/png;base64,' + PNG)
        self.analyze.assert_called_once_with('000001', holding=False)
        self.chart.assert_called_once()

    def test_all_34_coordinates_once_in_correct_five_groups(self):
        html, dom = self.render()
        indicators = dom.find(css='indicators')[0]
        cards = [node for node in indicators.find() if 'data-group' in node.attrs]
        self.assertEqual([card.attrs['data-group'] for card in cards], ['M', 'F', 'P', 'Q', 'A'])
        self.assertEqual([len(card.find('tbody')[0].find('tr')) for card in cards], [6, 5, 2, 3, 18])
        names = [row.find('td')[0].text for card in cards for row in card.find('tbody')[0].find('tr')]
        self.assertEqual(names, list(OBJECTIVE_NAMES))
        self.assertEqual(len(set(names)), 34)
        for name in OBJECTIVE_NAMES:
            self.assertEqual(html.count('<td>' + name + '</td>'), 1)
            self.assertRegex(web.OBJECTIVE_LABELS[name], '[\u4e00-\u9fff]')
        for css, label in (('up', '↑ +1 偏强'), ('flat', '→ 0 无方向'), ('down', '↓ −1 偏弱')):
            nodes = dom.find(css='vote-' + css)
            self.assertTrue(nodes)
            self.assertIn('color:', nodes[0].attrs['style'])
            self.assertEqual(nodes[0].text, label)
        for forbidden in ('S*', 'Sstar', '融合评分', '共识度', '权重', '一致性聚合'):
            self.assertNotIn(forbidden, html)

    def test_holding_arguments_forwarded_and_only_current_action_changes(self):
        original_account = deepcopy(self.a['history']['account'])

        def analyzed(code, holding=False):
            return dict(self.a, holding=holding, action='持有候选' if holding else '建仓候选',
                        css='buy', symbol='✅', comparison=dict(self.a['comparison'], status='front1'))

        self.analyze.side_effect = analyzed
        _, flat = self.render(holding=False)
        _, held = self.render(holding=True)
        self.assertIn('建仓候选', flat.find(css='conclusion')[0].text)
        self.assertIn('持有候选', held.find(css='conclusion')[0].text)
        self.assertEqual(self.analyze.call_count, 2)
        self.assertEqual(self.analyze.call_args_list[0].kwargs, {'holding': False})
        self.assertEqual(self.analyze.call_args_list[1].kwargs, {'holding': True})
        self.assertEqual(flat.find(css='historical-summary')[0].text, held.find(css='historical-summary')[0].text)
        self.assertEqual(original_account, self.a['history']['account'])

    def test_dividend_toggle_does_not_change_any_history_amount_or_chart(self):
        _, off = self.render(holding=True)
        _, on = self.render(holding=True, calc_dividend=True)
        self.assertFalse(off.find(css='dividend-history'))
        self.assertTrue(on.find(css='dividend-history'))
        for cls in ('historical-summary', 'historical-detail'):
            self.assertEqual(off.find(css=cls)[0].text, on.find(css=cls)[0].text)
        self.assertIn('-15,000.00 元', off.text)
        self.assertIn('-7.50%', off.text)
        self.assertIn('123.00 元', off.text)
        self.assertIn('现金分红已入账，不重复加减', off.text)
        self.assertEqual(self.chart.call_args_list[0].kwargs, self.chart.call_args_list[1].kwargs)

    def test_dividend_can_be_displayed_without_current_holding(self):
        _, dom = self.render(holding=False, calc_dividend=True)
        self.assertTrue(dom.find(css='dividend-history'))
        self.analyze.assert_called_once_with('000001', holding=False)
        self.assertEqual(self.chart.call_count, 1)

    def test_dividend_events_and_twelve_calendar_months_not_yield(self):
        _, dom = self.render(calc_dividend=True)
        block = dom.find(css='dividend-history')[0]
        self.assertIn('0.5000 元/各次权息前股', block.text)
        self.assertIn('(2025-09-10, 2026-09-10]', block.text)
        self.assertIn('2025-09-10', block.text)  # Still present in full history.
        self.assertIn('1.5000', block.text)
        self.assertIn('税前现金（元/权息前股）', block.text)
        self.assertIn('送转前后每股基数未归一化', block.text)
        self.assertIn('不计算股息率', block.text)
        self.assertNotIn('%', block.text)
        self.assertNotIn('2026-09-08', block.text)

    def test_empty_verified_actions_is_zero_but_error_is_not_zero(self):
        self.a['fresh']['actions']['events'] = []
        _, dom = self.render(calc_dividend=True)
        self.assertIn('0.0000 元/各次权息前股', dom.text)
        self.a['fresh']['actions']['status'] = 'error'
        self.a['fresh']['actions']['metadata']['live_recommendation_blocked'] = True
        self.a['verified_actions'] = False
        _, dom = self.render(calc_dividend=True)
        self.assertIn('空列表不等于零分红', dom.text)
        self.assertNotIn('0.0000 元/各次权息前股', dom.text)

    def test_actual_initial_final_cash_residual_pnl_fees_and_counts(self):
        _, dom = self.render()
        history = dom.find(css='historical-summary')[0]
        for value in ('200,000.00 元', '180,000.00 元', '5,000.00 元', '185,000.00 元',
                      '-15,000.00 元', '321.00 元', '123.00 元', '-7.50%', '6 / 6 / 6 / 6', '2 / 1'):
            self.assertIn(value, history.text)
        self.assertNotIn('999,999,999', dom.text)
        self.assertNotIn('1,000,000.00', dom.text)
        self.assertIn('期末权益＝现金＋残值', history.text)
        self.assertIn('残值不等于已兑现现金', history.text)

    def test_live_check_quote_and_backtest_dates_are_distinct(self):
        _, dom = self.render()
        current = dom.find(css='current-analysis')[0].text
        history = dom.find(css='historical-summary')[0].text
        self.assertIn('2026-09-10T10:00:00+08:00', current)
        self.assertIn('2026-09-09', current)
        self.assertIn('provider', current)
        self.assertIn('sina', current)
        self.assertNotIn('2019-01-02', current)
        self.assertIn('2019-01-02', history)
        self.assertIn('回测有效截止2026-09-08', history)
        self.assertIn('回测请求截止2026-09-10', history)
        self.assertIn('2026-09-07', history)  # Distinct residual valuation day.
        self.assertNotIn('2026-09-09', history)
        self.assertEqual(self.chart.call_args.kwargs['history']['end'], '2026-09-08')

    def test_chart_receives_original_arrays_and_full_frozen_ledger(self):
        self.render()
        self.assertTrue(self.chart.call_args.kwargs['action_continuous'])
        for actual, key in zip(self.chart.call_args.args, ('dates', 'raw', 'adjusted', 'votes')):
            self.assertIs(actual, self.a[key])
        self.assertIs(self.chart.call_args.kwargs['history']['ledger'], self.a['history']['ledger'])
        self.assertEqual(len(self.chart.call_args.kwargs['history']['ledger']), 26)

    def test_unknown_f1_stays_warn_even_with_all_positive_votes(self):
        self.a['votes'][:] = 1
        html, dom = self.render()
        self.assertIn('conclusion warn', html)
        self.assertNotIn('conclusion buy', html)
        self.assertIn('F1 未知：不能确认，不据此买入', dom.text)
        self.assertIn('条件风险上限（非执行目标）50.00%', dom.text)
        self.assertNotIn('建仓候选', dom.text)

    def test_inconsistent_buy_unknown_f1_fails_closed(self):
        self.a.update(css='buy', action='建仓候选', symbol='✅')
        html, _ = self.render()
        self.assertIn('conclusion warn', html)
        self.assertNotIn('建仓候选', html)
        self.chart.assert_not_called()

    def test_action_failure_preserves_diagnostic_chart_and_warning(self):
        self.a['fresh']['actions'].update(status='error', rights_issue_present=None)
        self.a['fresh']['actions']['metadata']['live_recommendation_blocked'] = True
        self.a.update(verified_actions=False, reason='权息核验失败', conditional_cap=0.)
        html, dom = self.render()
        self.assertIn('conclusion warn', html)
        self.assertIn('权息未通过，MACD 实际输入为未连续化原价', dom.text)
        self.assertEqual(len(dom.find(css='indicators')[0].find(css='vote')), 34)
        self.assertEqual(len(dom.find('img')), 1)
        self.chart.assert_called_once()
        self.assertFalse(self.chart.call_args.kwargs['action_continuous'])

    def test_warmup_zeros_and_nan_diagnostics_are_not_neutral_recommendation(self):
        self.a = fixture(1)
        self.a.update(rsi=float('nan'), annual_vol=float('nan'), daily_es=float('nan'), conditional_cap=0.)
        self.a['votes'][:] = 0
        self.analyze.return_value = self.a
        _, dom = self.render()
        for text in ('原子预热未完成：仅 1 根日线', '完整资格至少需要 251 根',
                     '预热占位 0 不能当作中性推荐', '—（数据不足）', 'RSI 相对强弱', '日历史 ES'):
            self.assertIn(text, dom.text)
        self.assertEqual(len(dom.find(css='vote-flat')), 34)
        self.assertNotIn('nan', dom.text)
        self.chart.assert_called_once()

    def test_annual_details_uses_account_nav_not_population_returns(self):
        _, dom = self.render()
        years = dom.find(css='historical-years')[0]
        self.assertEqual(years.tag, 'details')
        self.assertNotIn('open', years.attrs)
        self.assertIn('10,000.00 元', years.text)
        self.assertIn('5.00%', years.text)
        self.assertIn('-25,000.00 元', years.text)
        self.assertIn('-11.90%', years.text)
        self.assertIn('年度缺口不插补', years.text)
        self.assertNotIn('999,999,999', years.text)

    def test_recent_twenty_fills_not_last_twenty_mixed_events(self):
        _, dom = self.render()
        detail = dom.find(css='historical-detail')[0]
        self.assertIn('共 24 次，展示最近 20 次', detail.text)
        self.assertNotIn('fill-003', detail.text)
        self.assertIn('fill-004', detail.text)
        self.assertIn('fill-023', detail.text)
        self.assertNotIn('not-a-fill', detail.text)
        self.assertNotIn('dividend-not-a-fill', detail.text)
        self.assertIn('不配对交易、不推算胜率', detail.text)
        self.assertNotRegex(detail.text, r'胜率[：:]?\s*\d')
        self.assertIn('2026-08-04', detail.text)  # First visible fill's signal day.

    def test_no_history_is_friendly_and_never_substitutes_population(self):
        self.a['history'].update(account=None, ledger=[], error='没有该股账户')
        _, dom = self.render(calc_dividend=True)
        self.assertIn('暂无该股可用的完整历史账户', dom.text)
        self.assertNotIn('本期总收益率', dom.text)
        self.assertFalse(dom.find(css='historical-detail'))
        self.assertIsNone(self.chart.call_args.kwargs['history'])
        self.assertTrue(dom.find(css='dividend-history'))
        self.a['history'] = None
        _, dom = self.render()
        self.assertTrue(dom.find('img'))
        self.assertIsNone(self.chart.call_args.kwargs['history'])

    def test_no_fills_and_no_annual_records_are_not_fabricated(self):
        self.a['history']['ledger'] = []
        self.a['history']['account']['years'] = {}
        _, dom = self.render()
        self.assertIn('历史账户无实际成交', dom.text)
        self.assertIn('暂无年度账户记录', dom.text)
        self.assertEqual(self.chart.call_args.kwargs['history']['ledger'], [])

    def test_invalid_history_funds_or_future_fills_do_not_hide_current_analysis(self):
        original = deepcopy(self.a['history'])
        for kind in ('money', 'end', 'future', 'quantity'):
            with self.subTest(kind=kind):
                self.a['history'] = deepcopy(original)
                if kind == 'money':
                    self.a['history']['account']['pnl'] += 1
                elif kind == 'end':
                    self.a['history']['account']['end'] = '2026-09-09'
                elif kind == 'future':
                    self.a['history']['ledger'][-1]['day'] = '2026-09-11'
                else:
                    self.a['history']['ledger'][0]['quantity'] = 0
                _, dom = self.render()
                self.assertIn('暂不展示历史收益及成交标记', dom.text)
                self.assertFalse(dom.find(css='historical-detail'))
                self.assertTrue(dom.find(css='indicators'))
                self.assertIsNone(self.chart.call_args.kwargs['history'])

    def test_refresh_failure_never_renders_cache_or_chart_and_hides_paths(self):
        payload = '<script>alert("refresh")</script>&\''
        self.analyze.side_effect = RefreshError('/private/cache.sqlite ' + payload,
            refresh_id='id-' + payload, evidence_dir='/private/output/proof', price_committed=True)
        html, dom = self.render(calc_dividend=True)
        self.assertEqual(dom.find(css='conclusion')[0].text, '⚠️ 更新失败，暂不下结论')
        self.assertIn(escape('id-' + payload, quote=True), html)
        self.assertNotIn('/private', html)
        self.assertNotIn('<script>', html)
        self.assertFalse(dom.find('img'))
        self.assertFalse(dom.find('table'))
        self.chart.assert_not_called()
        self.analyze.assert_called_once()

    def test_absolute_refresh_id_is_not_displayed(self):
        for identity in ('/private/proof', 'C:\\private\\proof', '\\\\host\\private'):
            self.analyze.side_effect = RefreshError('fail', refresh_id=identity)
            with self.subTest(identity=identity):
                html, _ = self.render()
                self.assertNotIn('private', html)

    def test_invalid_code_rejected_before_analysis(self):
        for code in (None, 1, b'000001', '', ' 000001', '000001 ', '000001\n', '０００００１',
                     '00000', '0000001', '../000001', '<img/>', '300001', '688001', '999999'):
            with self.subTest(code=code):
                html = web.render_analysis(code)
                self.assertIn('请输入6位沪深主板股票代码', html)
        self.analyze.assert_not_called()
        self.chart.assert_not_called()

    def test_every_mainboard_prefix_passes_code_validation(self):
        for prefix in ('600', '601', '603', '605', '000', '001', '002', '003'):
            code = prefix + '001'
            self.a['code'] = code
            self.a['history'] = None
            with self.subTest(code=code):
                html = web.render_analysis(code)
                self.assertIn('data:image/png;base64,', html)
                self.analyze.assert_called_with(code, holding=False)

    def test_names_reasons_warnings_and_notes_are_all_escaped(self):
        payloads = {key: f'<script>{key}("x")</script>&\'' for key in
                    ('name', 'reason', 'warning', 'fresh', 'note', 'limit', 'verdict', 'fill', 'provider')}
        self.a.update(name=payloads['name'], reason=payloads['reason'], warnings=[payloads['warning']])
        self.a['fresh'].update(warnings=[payloads['fresh']], provider=payloads['provider'])
        self.a['fresh']['actions']['metadata']['notes'] = [payloads['note']]
        self.a['history']['manifest']['limitations'] = [payloads['limit']]
        self.a['history']['summary']['verdict'] = payloads['verdict']
        self.a['history']['ledger'][-3]['reason'] = payloads['fill']
        html, dom = self.render(calc_dividend=True)
        for value in payloads.values():
            self.assertIn(escape(value, quote=True), html)
        self.assertFalse(dom.find('script'))
        self.assertNotIn('/never/', html)

    def test_malicious_css_or_mixed_analysis_identity_fails_closed(self):
        for key, value in (('css', 'buy" onclick="alert(1)'), ('code', '600001')):
            old = self.a[key]
            self.a[key] = value
            with self.subTest(key=key):
                html, dom = self.render()
                self.assertIn('分析数据不完整', dom.text)
                self.assertNotIn('onclick', html)
                self.chart.assert_not_called()
            self.a[key] = old

    def test_malformed_dates_arrays_and_votes_do_not_render_a_conclusion_or_chart(self):
        for kind in ('date', 'votes', 'nonternary', 'raw'):
            self.a = fixture()
            self.analyze.return_value = self.a
            if kind == 'date':
                self.a['fresh']['latest_date'] = '2026-09-10'
            elif kind == 'votes':
                self.a['votes'] = self.a['votes'][:, :33]
            elif kind == 'nonternary':
                self.a['votes'][-1, 0] = 2
            else:
                self.a['raw'][-1, 0] = float('nan')
            with self.subTest(kind=kind):
                _, dom = self.render()
                self.assertIn('分析数据不完整', dom.text)
                self.assertFalse(dom.find('img'))
        self.chart.assert_not_called()

    def test_generic_analysis_error_does_not_leak_exception_or_retry(self):
        self.analyze.side_effect = ValueError('<script>bad</script> /private/file')
        html, _ = self.render()
        self.assertIn('分析失败，暂不下结论', html)
        self.assertNotIn('/private', html)
        self.assertNotIn('<script>', html)
        self.analyze.assert_called_once()
        self.chart.assert_not_called()

    def test_chart_error_preserves_analysis_without_stale_image_or_retry(self):
        self.chart.side_effect = ValueError('<script>bad</script> /private/chart')
        html, dom = self.render()
        self.assertIn('图表暂不可用', dom.text)
        self.assertTrue(dom.find(css='indicators'))
        self.assertTrue(dom.find(css='historical-summary'))
        self.assertFalse(dom.find('img'))
        self.assertNotIn('/private', html)
        self.assertNotIn('<script>', html)
        self.analyze.assert_called_once()
        self.chart.assert_called_once()

    def test_image_attribute_is_escaped_even_for_mocked_bad_payload(self):
        self.chart.return_value = 'bad" onerror="alert(1)'
        _, dom = self.render()
        image = dom.find('img')[0]
        self.assertEqual(set(image.attrs), {'src', 'alt'})

    def test_frozen_risk_parameters_are_not_labeled_model_defaults(self):
        _, dom = self.render()
        self.assertIn('本轮冻结所选风险参数：年化波动目标 10.00%', dom.text)
        self.assertNotIn('波动目标 15.00%', dom.text)

    def test_render_has_no_mutation_or_temporary_output(self):
        before = deepcopy(self.a)
        self.render(calc_dividend=True)
        for key in ('dates', 'raw', 'adjusted', 'votes'):
            np.testing.assert_array_equal(before.pop(key), self.a[key])
        self.assertEqual(before, {key: value for key, value in self.a.items()
                                  if key not in ('dates', 'raw', 'adjusted', 'votes')})
        self.assertEqual(list(self.temp.iterdir()), [])

    def test_real_chart_one_bar_smoke_still_offline(self):
        self.a = fixture(1)
        self.a['history'] = None
        self.analyze.return_value = self.a
        self.chart.side_effect = real_render_chart
        _, dom = self.render()
        image = dom.find('img')[0]
        decoded = base64.b64decode(image.attrs['src'].split(',', 1)[1], validate=True)
        self.assertTrue(decoded.startswith(b'\x89PNG\r\n\x1a\n'))
        self.assertGreater(len(decoded), 1000)
        self.analyze.assert_called_once()
        self.chart.assert_called_once()
        self.assertEqual(list(self.temp.iterdir()), [])


if __name__ == '__main__':
    unittest.main()