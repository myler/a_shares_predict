"""Completed-run HTML contract; all artifacts are synthetic and temporary."""

from dataclasses import asdict
from html import escape
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pareto_web
from capital_account import AccountConfig
from pareto_strategy import OBJECTIVE_NAMES, STRATEGY_VERSION, RiskConfig


class TestParetoWeb(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        for name in ('accounts', 'features', 'ledgers'):
            (self.folder / name).mkdir()
        for target, kwargs in (
            ('pareto_web.RUN_DIR', {'new': self.folder}),
            ('socket.socket', {'side_effect': AssertionError('network forbidden')}),
            ('sqlite3.connect', {'side_effect': AssertionError('database forbidden')}),
        ):
            mock = patch(target, **kwargs)
            mock.start()
            self.addCleanup(mock.stop)
        self.calendar = ['2021-12-31', '2023-12-29', '2025-12-31', '2026-09-01', '2026-09-02', '2026-09-03',
                         '2026-09-04', '2026-09-07', '2026-09-08', '2026-09-09', '2026-09-10']
        self.manifest = dict(
            version=STRATEGY_VERSION, start='2021-01-04', requested_end='2026-09-10',
            effective_end='2026-09-10', calendar=self.calendar,
            # Deliberately not sorted: fronts use this order, display uses code order.
            universe=[dict(code='600001', name='模拟乙'), dict(code='000001', name='模拟甲')],
            objective_names=list(OBJECTIVE_NAMES), risk_defaults=asdict(RiskConfig()),
            account_defaults=asdict(AccountConfig()), limitations=['cached universe only'],
            train_end='2021-12-31', validation_end='2023-12-31', candidate_vol_targets=[.10, .20])
        self.summary = dict(
            version=STRATEGY_VERSION, start=self.manifest['start'],
            requested_end=self.manifest['requested_end'], effective_end=self.manifest['effective_end'],
            selected_risk=asdict(RiskConfig(annual_vol_target=.20)),
            totals=dict(account_count=2, initial_capital=2_000_000, final_cash=1_975_000,
                        residual_marked_value=50_000, final_equity=2_025_000, pnl=25_000,
                            fully_liquidated_count=1, fees=20, cash_dividends=10,
                            cash_minus_all_principal=-25_000, profitable_accounts=1, losing_accounts=1,
                            action_fetch_failures=0, rights_issue_accounts=0, feature_errors=0,
                            unexplained_gap_accounts=0, stale_accounts=0, late_start_accounts=0,
                            build_count=2, add_count=0, reduce_count=1, exit_count=1),
                  annual_results=[dict(year='2021', date='2021-12-31', final_equity=2_000_200,
                                   pnl=200, return_pct=.01),
                               dict(year='2023', date='2023-12-29', final_equity=2_000_000,
                                   pnl=-200, return_pct=(2_000_000/2_000_200-1)*100),
                               dict(year='2025', date='2025-12-31', final_equity=2_000_000,
                                 pnl=0, return_pct=0),
                            dict(year='2026', date='2026-09-10', final_equity=2_025_000,
                                 pnl=25_000, return_pct=1.25)],
            holdout_2024_onward_pnl=25_000,
              trials=[dict(vol_target=.10, train_pnl=100, validation_pnl=-50,
                        validation_end_nav=2_000_050, scope_end='2026-09-10'),
                    dict(vol_target=.20, train_pnl=200, validation_pnl=-200,
                        validation_end_nav=2_000_000, scope_end='2026-09-10')],
            limitations=['rights issues not modeled'], verdict='unverified research')
        self.accounts = {}
        for code, name, cash, shares, last_nav in (
            ('000001', '模拟甲', 900_000, 5000, 980_000),
            ('600001', '模拟乙', 1_075_000, 0, 1_020_000),
        ):
            nav = cash + shares * 10
            account = dict(code=code, name=name, initial_capital=1_000_000,
                           final_cash=cash, cash=cash, shares=shares, marked_nav=nav,
                           pnl=nav - 1_000_000, residual_value=shares * 10,
                           liquidation_complete=shares == 0, last_date=self.calendar[-1], last_price=10,
                           action_status='ok', rights_issue_present=False, unexplained_gap_count=0,
                           feature_error=None, late_start=False, stale=False, fees=10,
                           cash_dividends=10 if shares else 0, build_count=1, add_count=0,
                           reduce_count=1 if shares else 0, exit_count=0 if shares else 1,
                              average_exposure=.10, first_date=self.calendar[0], end=self.calendar[-1],
                              unfilled_count=1 if shares else 0, action_count=1 if shares else 0,
                              max_drawdown_observed=.05, fractional_share_approximation=False,
                              train_nav=1_000_100, validation_nav=1_000_000,
                              years={'2021': dict(date='2021-12-31', nav=1_000_100, cash=1_000_100, shares=0),
                                  '2023': dict(date='2023-12-29', nav=1_000_000, cash=1_000_000, shares=0),
                                  '2025': dict(date='2025-12-31', nav=last_nav, cash=last_nav, shares=0),
                                  '2026': dict(date=self.calendar[-1], nav=nav, cash=cash, shares=shares)})
            self.accounts[code] = account
            self.write_json(f'accounts/{code}.json', account)
            self.write_features(code)
            self.write_json(f'ledgers/{code}.json', [])
        self.ledger = [
            self.entry('2026-09-02', 'build', '2026-09-01', quantity=10000, price=15,
                       gross=150000, fee=5, cash=849995, shares=10000, target_weight=.15),
            self.entry('2026-09-03', 'action', None, quantity=0, price=None,
                       gross=10, fee=0, cash=850005, shares=10000, target_weight=None,
                       cash_per_share=.001),
            self.entry('2026-09-04', 'unfilled', '2026-09-03', reason='blocked',
                       block_reason='block_sell', cash=850005, shares=10000),
            self.entry('2026-09-10', 'reduce', '2026-09-09', quantity=5000, price=10,
                       gross=50000, fee=5, cash=900000, shares=5000, target_weight=.05),
        ]
        self.write_json('ledgers/000001.json', self.ledger)
        self.fronts = np.full((len(self.calendar), 2), 5, dtype=np.int8)
        self.fronts[0, 0] = 1
        self.fronts[-1, 1] = 1
        np.save(self.folder / 'fronts.npy', self.fronts)
        self.write_json('manifest.json', self.manifest)
        self.write_json('summary.json', self.summary)

    def write_json(self, name, data):
        (self.folder / name).write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')

    def write_features(self, code, **changes):
        n = len(self.calendar)
        data = dict(dates=np.array(self.calendar, dtype='U10'),
                    votes=np.tile(np.resize(np.array([-1, 0, 1], dtype=np.int8), 34), (n, 1)),
                    annual_vol=np.full(n, .4), daily_es=np.full(n, .08),
                    eligible=np.ones(n, dtype=bool), gates=np.ones(n, dtype=bool),
                    raw=np.tile([10., 11., 9., 10., 1000.], (n, 1)))
        data.update(changes)
        np.savez_compressed(self.folder / 'features' / (code + '.npz'), **data)

    @staticmethod
    def entry(day, event, signal_day, **changes):
        value = dict(day=day, signal_day=signal_day, event=event, reason='pareto_risk_target',
                     quantity=0, price=10, gross=0, fee=0, cash=900000, shares=5000, target_weight=0)
        value.update(changes)
        return value

    def assert_no_weighted_fallback(self, html):
        for text in ('S*', 'Sstar', '69 分', '40%', '融合评分'):
            self.assertNotIn(text, html)

    def test_missing_completion_marker_never_reads_other_files(self):
        (self.folder / 'summary.json').unlink()
        with patch('pareto_web._json', side_effect=AssertionError('must check completion first')):
            for render in (lambda: pareto_web.render_stock_report('000001'), pareto_web.render_population):
                html = render()
                self.assertIn('尚无完成的多维回测，请先运行', html)
                self.assertIn('pareto_backtest.py --output', html)
                self.assert_no_weighted_fallback(html)

    def test_unknown_or_invalid_code_is_friendly(self):
        html = pareto_web.render_stock_report('999999')
        self.assertIn('未知股票', html)
        self.assert_no_weighted_fallback(html)
        with patch('pareto_web._load_run', side_effect=AssertionError('invalid code must not read')):
            for code in ('../000001', '<script>', '１２３４５６', None):
                self.assertIn('6位股票代码', pareto_web.render_stock_report(code))

    def test_versions_and_objective_order_must_match(self):
        for filename, record in (('manifest.json', self.manifest), ('summary.json', self.summary)):
            with self.subTest(filename=filename):
                self.write_json(filename, dict(record, version='old'))
                for html in (pareto_web.render_stock_report('000001'), pareto_web.render_population()):
                    self.assertIn('版本不匹配', html)
                    self.assert_no_weighted_fallback(html)
                self.write_json(filename, record)
        self.write_json('manifest.json', dict(self.manifest, objective_names=list(reversed(OBJECTIVE_NAMES))))
        self.assertIn('34 维原子名称或顺序不同', pareto_web.render_stock_report('000001'))

    def test_stock_funds_34_independent_votes_and_exact_front_coordinates(self):
        html = pareto_web.render_stock_report('000001')
        for text in ('900,000.00 元', '50,000.00 元', '950,000.00 元', '-50,000.00 元',
                     '未完成清仓；残值不是现金', '有效截止日期 2026-09-10', '旧缓存并非实时推荐',
                     'F1（第一前沿）', '已进入：禁止买入，执行目标 0', '成交账本：共 2 次',
                     '20.00%', '25.00%', '未成交 / 权息等事件：共 2 次'):
            self.assertIn(text, html)
        self.assertIn('<td>信号风险上限（约束前）</td><td>25.00%</td>', html)
        self.assertIn('<td>2026</td><td>2026-09-10</td><td>950,000.00 元</td><td>-30,000.00 元</td><td>-3.06%</td>', html)
        for index, name in enumerate(OBJECTIVE_NAMES):
            vote = (-1, 0, 1)[index % 3]
            self.assertIn(f'<td>{name}</td><td>{vote:+d}</td>' if vote else f'<td>{name}</td><td>0</td>', html)
        self.assertEqual(len(re.findall(r'<td>(?:M|F|P|Q|A)_[^<]+</td>', html)), 34)
        self.assert_no_weighted_fallback(html)
        other = pareto_web.render_stock_report('600001')
        self.assertIn('不在 F1', other)
        self.assertNotIn('F1（第一前沿）', other)
        self.assertIn('<td>信号风险上限（约束前）</td><td>0.00%</td>', other)

    def test_fronts_are_not_forward_filled_or_aligned_by_position(self):
        self.accounts['000001']['last_date'] = '2026-09-09'
        self.write_json('accounts/000001.json', self.accounts['000001'])
        html = pareto_web.render_stock_report('000001')
        self.assertIn('<td>精确匹配日期</td><td>2026-09-09</td>', html)
        self.assertIn('不在 F1', html)
        self.assertNotIn('F1（第一前沿）', html)
        self.accounts['000001']['last_date'] = '2026-09-06'
        self.write_json('accounts/000001.json', self.accounts['000001'])
        html = pareto_web.render_stock_report('000001')
        self.assertIn('无精确同日匹配', html)
        self.assertNotIn('<td>信号风险上限', html)

    def test_gate_or_missing_risk_zeroes_cap_without_changing_stored_front(self):
        for changes in ({'gates': np.zeros(len(self.calendar), bool)},
                        {'eligible': np.zeros(len(self.calendar), bool)},
                        {'annual_vol': np.full(len(self.calendar), np.nan)}):
            with self.subTest(keys=list(changes)):
                self.write_features('000001', **changes)
                html = pareto_web.render_stock_report('000001')
                self.assertIn('F1（第一前沿）', html)
                self.assertIn('<td>信号风险上限（约束前）</td><td>0.00%</td>', html)

    def test_action_failure_unverified_or_rights_history_blocks_equity_snapshot(self):
        for status, rights in (('failed', False), ('cache_unverified', False), ('ok', True), ('failed', True)):
            for stored_eligible in (True, False):
                with self.subTest(status=status, rights=rights, stored_eligible=stored_eligible):
                    self.accounts['000001'].update(action_status=status, rights_issue_present=rights)
                    self.write_json('accounts/000001.json', self.accounts['000001'])
                    self.write_features('000001', eligible=np.full(len(self.calendar), stored_eligible, bool))
                    before = (self.folder / 'fronts.npy').read_bytes()
                    html = pareto_web.render_stock_report('000001')
                    self.assertIn('保守隔离：全期禁止买入，保留本金', html)
                    self.assertIn('<td>有效 eligible（含权息隔离）</td><td>false</td>', html)
                    self.assertIn('<td>资格 / 门禁</td><td>未通过 / 通过</td>', html)
                    self.assertIn('<td>信号风险上限（约束前）</td><td>0.00%</td>', html)
                    self.assertIn('冻结特征/前沿与权息隔离不一致', html)
                    self.assertEqual(len(re.findall(r'<td>(?:M|F|P|Q|A)_[^<]+</td>', html)), 34)
                    self.assertEqual(before, (self.folder / 'fronts.npy').read_bytes())
                    self.assert_no_weighted_fallback(html)

    def test_continuous_trial_and_holdout_contract_fails_closed(self):
        for change, message in (({'scope_end': '2023-12-31'}, '连续运行至同一有效截止日'),
                                ({'validation_end_nav': 2_000_100}, '资金不对齐')):
            original = dict(self.summary['trials'][1])
            self.summary['trials'][1].update(change)
            self.write_json('summary.json', self.summary)
            with self.subTest(change=change):
                self.assertIn(message, pareto_web.render_population())
            self.summary['trials'][1] = original
        self.summary['holdout_2024_onward_pnl'] += 1
        self.write_json('summary.json', self.summary)
        self.assertIn('资金不对齐', pareto_web.render_population())

    def test_validation_cannot_select_risk_parameters(self):
        self.summary['selected_risk']['annual_vol_target'] = .10
        self.write_json('summary.json', self.summary)
        self.assertIn('仅训练期选优结果不一致', pareto_web.render_population())

    def test_cash_minus_principal_contract(self):
        self.summary['totals']['cash_minus_all_principal'] += 10
        self.write_json('summary.json', self.summary)
        self.assertIn('资金不对齐', pareto_web.render_population())

    def test_annual_totals_and_account_end_must_match_the_completed_run(self):
        for key in ('pnl', 'return_pct', 'final_equity'):
            original = self.summary['annual_results'][-1][key]
            self.summary['annual_results'][-1][key] += 10
            self.write_json('summary.json', self.summary)
            with self.subTest(key=key):
                self.assertIn('资金不对齐', pareto_web.render_population())
            self.summary['annual_results'][-1][key] = original
        self.write_json('summary.json', self.summary)
        self.accounts['000001']['end'] = '2023-12-31'
        self.write_json('accounts/000001.json', self.accounts['000001'])
        self.assertIn('账户回放截止与完成运行不一致', pareto_web.render_stock_report('000001'))

    def test_invalid_snapshot_bar_cannot_have_a_positive_risk_cap(self):
        for value in (np.nan, 0., -1.):
            raw = np.tile([10., 11., 9., 10., 1000.], (len(self.calendar), 1))
            raw[-1, 3] = value
            self.write_features('000001', raw=raw)
            with self.subTest(value=value):
                html = pareto_web.render_stock_report('000001')
                self.assertIn('无效行情或无成交量', html)
                self.assertIn('<td>信号风险上限（约束前）</td><td>0.00%</td>', html)
                self.assertIn('900,000.00 元', html)

    def test_ledger_required_fields_and_signal_dates_fail_closed(self):
        for key in ('cash', 'price', 'quantity', 'signal_day'):
            record = dict(self.ledger[0]); del record[key]
            self.write_json('ledgers/000001.json', [record])
            with self.subTest(key=key):
                self.assertIn('账本缺少必需字段', pareto_web.render_stock_report('000001'))
        self.write_json('ledgers/000001.json', [dict(self.ledger[0], signal_day='2026-09-03')])
        self.assertIn('账本信号日晚于执行日', pareto_web.render_stock_report('000001'))

    def test_selected_risk_not_model_default_controls_snapshot_cap(self):
        self.write_features('000001', daily_es=np.full(len(self.calendar), .01))
        html = pareto_web.render_stock_report('000001')
        self.assertIn('<td>信号风险上限（约束前）</td><td>50.00%</td>', html)
        self.assertIn('σTarget=20.00%', html)
        self.assertNotIn('σTarget=15.00%', html)

    def test_population_truncates_by_code_not_pnl_and_keeps_total_funds(self):
        html = pareto_web.render_population(1)
        full = pareto_web.render_population(100)
        for text in ('2,000,000.00 元', '1,975,000.00 元', '2,025,000.00 元', '25,000.00 元',
                     '1.25%', '实际全部 2 个账户均参与', '2024 年起留出期盈亏'):
            self.assertIn(text, html)
            self.assertIn(text, full)
        self.assertIn('展示前 1 份账户', html)
        self.assertIn('模拟甲', html)  # The worse-performing account is first by code.
        self.assertNotIn('模拟乙', html)
        self.assertLess(full.index('<td>000001</td>'), full.index('<td>600001</td>'))
        self.assertIn('不是按收益排名、加权选择或推荐 Top-N', html)
        self.assert_no_weighted_fallback(html)
        self.assertIn('展示前 2 份账户', pareto_web.render_population('bad'))
        self.assertIn('展示前 1 份账户', pareto_web.render_population(-20))

    def test_money_misalignment_is_not_silently_rendered(self):
        self.summary['totals']['final_equity'] += 100
        self.write_json('summary.json', self.summary)
        self.assertIn('资金不对齐', pareto_web.render_population())
        self.summary['totals']['final_equity'] -= 100
        self.write_json('summary.json', self.summary)
        self.accounts['000001']['final_cash'] += 100
        self.write_json('accounts/000001.json', self.accounts['000001'])
        self.assertIn('资金不对齐', pareto_web.render_stock_report('000001'))

    def test_external_text_is_escaped_everywhere(self):
        payload = '<script>alert("x")</script>&\''
        for key in ('name', 'action_status', 'feature_error'):
            self.accounts['000001'][key] = payload
        self.write_json('accounts/000001.json', self.accounts['000001'])
        self.summary['limitations'] = [payload]
        self.summary['verdict'] = payload
        self.write_json('summary.json', self.summary)
        self.ledger[0]['reason'] = payload
        self.ledger[2]['block_reason'] = payload
        self.write_json('ledgers/000001.json', self.ledger)
        for html in (pareto_web.render_stock_report('000001'), pareto_web.render_population(1)):
            self.assertIn(escape(payload, quote=True), html)
            self.assertNotIn('<script>', html)
            self.assertNotIn(payload, html)
        self.assertIn(escape(payload), pareto_web.render_stock_report('000001'))

    def test_last_100_fills_not_100_mixed_events(self):
        fills = [dict(self.ledger[0], reason=f'fill-{i:03d}') for i in range(105)]
        others = [dict(self.ledger[2], reason=f'blocked-{i:03d}') for i in range(110)]
        self.write_json('ledgers/000001.json', fills + others)
        html = pareto_web.render_stock_report('000001')
        self.assertIn('成交账本：共 105 次，展示最近 100 次', html)
        self.assertNotIn('fill-004', html)
        self.assertIn('fill-005', html)
        self.assertIn('fill-104', html)
        self.assertNotIn('blocked-009', html)
        self.assertIn('blocked-010', html)
        self.assertIn('未成交 / 权息等事件：共 110 次，展示最近 100 次', html)

    def test_incomplete_or_unsafe_arrays_fail_closed(self):
        self.write_features('000001', votes=np.ones((len(self.calendar), 33), dtype=np.int8))
        self.assertIn('34 维特征形状', pareto_web.render_stock_report('000001'))
        self.write_features('000001', votes=np.full((len(self.calendar), 34), 'unsafe', dtype=object))
        self.assertIn('产物不完整或无法读取', pareto_web.render_stock_report('000001'))
        self.write_features('000001')
        np.save(self.folder / 'fronts.npy', np.ones((2, 2), dtype=np.int8))
        self.assertIn('前沿矩阵', pareto_web.render_stock_report('000001'))
        np.save(self.folder / 'fronts.npy', self.fronts)
        (self.folder / 'ledgers' / '000001.json').unlink()
        self.assertIn('产物不完整或无法读取', pareto_web.render_stock_report('000001'))

    def test_artifact_reads_are_side_effect_free(self):
        before = {p.relative_to(self.folder): p.read_bytes() for p in self.folder.rglob('*') if p.is_file()}
        for render in (lambda: pareto_web.render_stock_report('000001'), pareto_web.render_population):
            self.assertIn('pareto-report', render())
        after = {p.relative_to(self.folder): p.read_bytes() for p in self.folder.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_manifest_codes_cannot_escape_run_directory(self):
        self.manifest['universe'][0]['code'] = '../outside'
        self.write_json('manifest.json', self.manifest)
        with patch('pareto_web._load_account', side_effect=AssertionError('invalid manifest must fail first')):
            self.assertIn('股票清单不完整', pareto_web.render_population())

    def test_mixed_run_dates_are_rejected(self):
        self.summary['effective_end'] = '2026-09-09'
        self.write_json('summary.json', self.summary)
        self.assertIn('汇总与清单日期不一致', pareto_web.render_population())

    def test_shared_model_documentation(self):
        html = pareto_web.render_methodology()
        for text in ('34 个独立维度', '1,000,000.00 元', 'wcap = min(1, σTarget / σ, ESbudget / ES95)',
                     '0 ≤ w ≤ wcap', '目标为 0 强制 exit', '15.00%', '20 个市场日 cooldown',
                     '最后 5 个市场日', 'T+1', '5.00 元', '5bp', 'cash dividend on ex-date',
                     '下一市场交易日开盘', '配股未建模', '未计红利税', '保守隔离', 'eligible=false',
                     '全期禁止买入', '不重置本金、不前填交易信号'):
            self.assertIn(text, html)
        custom = pareto_web.render_methodology(RiskConfig(annual_vol_target=.10),
                                              AccountConfig(minimum_commission=8, cooldown_sessions=12))
        self.assertIn('σTarget=10.00%', custom)
        self.assertIn('每笔最低 8.00 元', custom)
        self.assertIn('12 个市场日 cooldown', custom)
        self.assert_no_weighted_fallback(html)

    def test_real_replay_summary_account_and_ledger_fields_render_read_only(self):
        # Real replay only, using these temporary features; never run collection,
        # prepare_stock, ProcessPoolExecutor or any production DB entry point.
        import pareto_backtest as runner
        records = []
        curve = np.zeros(len(self.calendar))
        for col, item in enumerate(self.manifest['universe']):
            code = item['code']
            account = self.accounts[code]
            meta = {key: account[key] for key in ('name', 'first_date', 'last_date', 'feature_error',
                    'late_start', 'stale', 'rights_issue_present', 'action_status', 'action_count',
                    'unexplained_gap_count')}
            self.write_json(f'features/{code}.json', meta)
            n = len(self.calendar)
            event_cash = np.zeros(n); event_cash[4] = .1
            event_mult = np.ones(n); event_mult[4] = 1.01
            self.write_features(code, event_cash=event_cash, event_mult=event_mult)
            self.fronts[:, col] = 1
            np.save(self.folder / 'fronts.npy', self.fronts)
            result, stock_curve, ledger = runner.replay_stock(self.folder, code, col,
                                                              RiskConfig(annual_vol_target=.20), record_ledger=True)
            curve += stock_curve
            records.append(result)
            self.write_json(f'accounts/{code}.json', result)
            self.write_json(f'ledgers/{code}.json', ledger)
        self.summary['totals'] = runner.aggregate(records)
        train_nav = sum(r['train_nav'] for r in records)
        validation_nav = sum(r['validation_nav'] for r in records)
        principal = self.summary['totals']['initial_capital']
        self.summary['annual_results'] = []
        previous = principal
        for year in sorted(set(day[:4] for day in self.calendar)):
            i = max(i for i, day in enumerate(self.calendar) if day.startswith(year))
            nav = float(curve[i])
            self.summary['annual_results'].append(dict(year=year, date=self.calendar[i],
                final_equity=nav, pnl=nav-previous, return_pct=(nav/previous-1)*100))
            previous = nav
        for trial in self.summary['trials']:
            trial.update(train_pnl=train_nav-principal-(1 if trial['vol_target'] == .10 else 0),
                         validation_pnl=validation_nav-train_nav+(1 if trial['vol_target'] == .10 else 0),
                         validation_end_nav=validation_nav)
        self.summary['holdout_2024_onward_pnl'] = self.summary['totals']['final_equity'] - validation_nav
        self.write_json('summary.json', self.summary)
        before = {p.relative_to(self.folder): p.read_bytes() for p in self.folder.rglob('*') if p.is_file()}
        html = pareto_web.render_stock_report('000001')
        overview = pareto_web.render_population(1)
        for rendered in (html, overview):
            self.assertNotIn('产物不完整或无法读取', rendered)
            self.assertNotIn('资金不对齐', rendered)
            self.assert_no_weighted_fallback(rendered)
        for text in ('买卖方向', '原始执行参考价（元）', '佣金（元）', '印花税（元）', '过户费（元）',
                     '送转倍数', '权息前有权股份', '零碎出售近似', '未成交尝试次数 / 权息事件数',
                     '观察到的最大净值回撤', '训练末权益 / 验证末权益', '(build)', '(action)', '(exit)'):
            self.assertIn(text, html)
        for text in ('全量数据质量与保守隔离', '验证末权益', '完整回放截止', '类别（可重叠，不相加）',
                     '权息失败或未核验账户（全期禁买，保留本金）', '配股历史账户（全期禁买，保留本金）'):
            self.assertIn(text, overview)
        after = {p.relative_to(self.folder): p.read_bytes() for p in self.folder.rglob('*') if p.is_file()}
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()