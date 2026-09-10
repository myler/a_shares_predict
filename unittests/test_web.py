"""Web 路由、页面参数与结构化引擎契约；所有数据使用模拟输入。"""

import inspect
import io
import json
from pathlib import Path
import re
import sys
import unittest
from datetime import date, timedelta
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine
import web


def make_handler(path='/'):
    handler = web.Handler.__new__(web.Handler)
    handler.path = path
    handler.wfile = io.BytesIO()
    handler.send_response = Mock()
    handler.send_header = Mock()
    handler.end_headers = Mock()
    return handler


def page_config(page):
    return json.loads(re.search(
        r'const COMPREHENSIVE_CONFIG = (\{[^\n]+\});', page).group(1))


class OfflineWebCase(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch('socket.socket', side_effect=AssertionError('network forbidden')))
        self.enterContext(patch('sqlite3.connect', side_effect=AssertionError('database forbidden')))


class TestPageContract(OfflineWebCase):
    def test_config_matches_engine_defaults(self):
        config = page_config(web.PAGE)
        defaults = inspect.signature(engine.backtest_comprehensive).parameters
        self.assertEqual(set(config), {
            'buy_score', 'stop_loss_pct', 'take_profit_pct',
            'profit_protect_pct', 'profit_protect_score', 'auxiliary_beta',
            'commission_rate', 'stamp_duty_rate',
        })
        for key, value in config.items():
            self.assertEqual(value, defaults[key].default, key)
        self.assertNotIn('__COMPREHENSIVE_CONFIG__', web.PAGE)

    def test_render_reads_updated_defaults(self):
        signature = inspect.signature(engine.backtest_comprehensive)

        def changed_backtest():
            pass

        changed_backtest.__signature__ = signature.replace(parameters=[
            param.replace(default=72) if name == 'buy_score' else param
            for name, param in signature.parameters.items()
        ])
        with patch('web.backtest_comprehensive', changed_backtest):
            self.assertEqual(page_config(web.render_page())['buy_score'], 72)

    def test_formula_uses_config_and_correct_comparisons(self):
        for text in (
            'S* ≥ ${C.buy_score}',
            '&lt; ${C.stop_loss_pct}%', '&gt; +${C.take_profit_pct}%',
            '&gt; ${C.profit_protect_pct}% 且 S* &lt; ${C.profit_protect_score}',
            '− ${C.auxiliary_beta}×A', 'S* &lt; 35',
            'OBV 5日净量能流 &lt; −60%', 'RSI &gt; 92',
            'M&lt;35 ∧ F&lt;40', '下一交易日开盘', '不计期末未平仓',
            '不代表分仓执行', '不是财务基本面',
        ):
            self.assertIn(text, web.PAGE)
        for obsolete in ('融合策略（双路径）', 'Z<sub>', '首仓25%',
                         '满仓100万', '减至50%', '止损−8%', '跌破5日线'):
            self.assertNotIn(obsolete, web.PAGE)

    def test_scan_and_other_strategy_boundaries(self):
        for text in ('多维回测总览', 'N 仅控制按代码顺序展示',
                     '实际全部账户', '旧缓存并非实时推荐',
                     '送转股份与分红日期上限尚未完善', '未严格验证年度连续性'):
            self.assertIn(text, web.PAGE)
        self.assertIn("const url='/pareto-summary?n='", web.PAGE)
        self.assertNotIn("const url='/scan?n='", web.PAGE)

    def test_pareto_is_default_and_uses_shared_methodology(self):
        options = re.findall(r'<option value="([^"]+)"', web.PAGE)
        self.assertEqual(options[0], 'pareto')
        self.assertIn('comprehensive', options)
        formula = json.loads(re.search(r'const PARETO_FORMULA = ("[^\n]+");', web.PAGE).group(1))
        self.assertEqual(formula, web.pareto_web.render_methodology())
        self.assertNotIn('__PARETO_FORMULA__', web.PAGE)
        self.assertIn("s==='value' || s==='pareto'", web.PAGE)
        self.assertIn("html:PARETO_FORMULA", web.PAGE)
        self.assertIn('三值化丢失幅度', formula)
        self.assertIn('不为本策略公式、参数或盈利背书', formula)
        with patch('web.pareto_web.render_methodology', return_value='<p></script>${evil}`</p>'):
            rendered = web.render_page()
        self.assertIn('\\u003c/script\\u003e', rendered)
        self.assertNotIn('<p></script>', rendered)

    def test_legacy_renderer_removed(self):
        self.assertFalse(hasattr(web.Handler, '_run_comprehensive_analysis_legacy'))

    def test_weighted_methods_are_not_presented_as_pareto(self):
        for text in ('核心评分 S*（加权标量，非帕累托）',
                     '此旧策略仍是加权标量化',
                     '18 项投票先合并成共识 A', '相反信号可能抵消'):
            self.assertIn(text, web.PAGE)
        self.assertNotIn('当前 Web 尚未接入帕累托非支配排序', web.PAGE)

    def test_fluid_layout_and_scrollable_result_tables(self):
        self.assertNotIn('max-width:920px', web.PAGE)
        self.assertNotIn('max-width:520px', web.PAGE)
        self.assertIn('grid-template-columns:minmax(240px,320px) minmax(0,1fr)', web.PAGE)
        self.assertIn('@media(max-width:900px)', web.PAGE)
        self.assertIn('overflow-x:auto', web.PAGE)
        self.assertIn('prepareResultTables(box)', web.PAGE)
        self.assertIn('prepareResultTables(result)', web.PAGE)

    def test_tabs_and_requests_keep_legacy_scan_explicit_and_separate(self):
        self.assertIn('data-tab="legacy-scan"', web.PAGE)
        self.assertIn('旧加权海选（非帕累托）', web.PAGE)
        self.assertIn('运行旧加权海选（可能联网）', web.PAGE)
        read_only = web.PAGE.split('async function runScan(){', 1)[1].split('async function runLegacyScan(){', 1)[0]
        legacy = web.PAGE.split('async function runLegacyScan(){', 1)[1].split('async function analyze(e){', 1)[0]
        self.assertIn('/pareto-summary?n=', read_only)
        self.assertNotIn('/scan?', read_only)
        self.assertIn('/scan?n=', legacy)
        self.assertNotIn('/pareto-summary', legacy)
        tab = web.PAGE.split('<div id="tab-scan"', 1)[1].split('<!-- 旧扫描独立保留', 1)[0]
        self.assertNotIn('S*', tab)
        self.assertIn('全期禁买但保留本金', tab)

    def test_strategy_changes_clear_previous_results_and_ignore_late_responses(self):
        change = web.PAGE.split('function onStrategyChange(){', 1)[1].split('function toggleDividend(){', 1)[0]
        self.assertIn('resetAnalysis();', change)
        self.assertIn("document.getElementById('holdingLabel').style.display=(s==='pareto' || s==='value')?'none':'flex'", change)
        self.assertIn('holding.checked=false;holding.disabled=true;', change)
        self.assertIn("document.getElementById('result').replaceChildren();", web.PAGE)
        analyze = web.PAGE.split('async function analyze(e){', 1)[1]
        self.assertIn("strategy==='pareto'?'':'&holding='", analyze)
        self.assertLess(analyze.index('if(revision!==analysisRevision)return;'), analyze.index('result.innerHTML=text'))
        self.assertIn('if(revision===analysisRevision)result.textContent=', analyze)


class TestWebRoutes(OfflineWebCase):
    def test_homepage_renders_config(self):
        handler = make_handler('/')
        handler.do_GET()
        handler.send_response.assert_called_once_with(200)
        self.assertEqual(handler.wfile.getvalue().decode(), web.PAGE)

    def test_default_strategy_is_pareto(self):
        handler = make_handler('/analyze?code=000651')
        with patch.object(handler, 'run_pareto_analysis', return_value='pareto') as run, \
             patch.object(handler, 'run_comprehensive_analysis') as legacy, \
             patch('web.fetch_kline') as fetch, patch('web.save_stock_name') as save:
            handler.do_GET()
        run.assert_called_once_with('000651')
        legacy.assert_not_called()
        fetch.assert_not_called()
        save.assert_not_called()
        handler.send_response.assert_called_once_with(200)
        self.assertEqual(handler.wfile.getvalue(), b'pareto')

    def test_explicit_routes_preserved(self):
        for strategy, method, args in (
            ('pareto', 'run_pareto_analysis', ('000651',)),
            ('comprehensive', 'run_comprehensive_analysis', ('000651', True, True)),
            ('buyhold', 'run_buyhold_analysis', ('000651', True)),
            ('value', 'run_value_analysis', ('000651',)),
            ('macd', 'run_analysis', ('000651', True, True)),
            ('multi', 'run_multifactor_analysis', ('000651', True, True)),
        ):
            with self.subTest(strategy=strategy):
                handler = make_handler(
                    f'/analyze?code=000651&strategy={strategy}&holding=1&dividend=1')
                with patch.object(handler, method, return_value='result') as run:
                    handler.do_GET()
                run.assert_called_once_with(*args)
                handler.send_response.assert_called_once_with(200)

    def test_unknown_strategy_rejected_before_loading_data(self):
        handler = make_handler('/analyze?code=000651&strategy=typo')
        with patch('web.fetch_kline') as fetch:
            handler.do_GET()
        handler.send_response.assert_called_once_with(400)
        fetch.assert_not_called()

    def test_invalid_stock_rejected(self):
        handler = make_handler('/analyze?code=abc')
        with patch('web.fetch_kline') as fetch:
            handler.do_GET()
        handler.send_response.assert_called_once_with(400)
        fetch.assert_not_called()

    def test_scan_defaults_match_form(self):
        for path in ('/scan', '/scan?n=bad'):
            with self.subTest(path=path):
                handler = make_handler(path)
                with patch.object(handler, 'run_scan_html', return_value='scan') as run:
                    handler.do_GET()
                run.assert_called_once_with(5, 100, 5)
                handler.send_response.assert_called_once_with(200)
        self.assertIn('id="scanN" value="5"', web.PAGE)

    def test_pareto_report_delegates_only_code(self):
        handler = make_handler('/analyze?code=000651&holding=1&dividend=1&path=/other&run_dir=/other&refresh=1&force=1')
        with patch('web.pareto_web.render_stock_report', return_value='local snapshot') as render, \
             patch('web.fetch_kline') as fetch, patch('web.save_stock_name') as save:
            handler.do_GET()
        render.assert_called_once_with('000651')
        fetch.assert_not_called()
        save.assert_not_called()
        self.assertEqual(handler.wfile.getvalue(), b'local snapshot')

    def test_route_prefixes_and_non_ascii_codes_cannot_trigger_data_access(self):
        for path in ('/analyze-refresh?code=000651', '/scan-refresh', '/pareto-summary-refresh'):
            with self.subTest(path=path):
                handler = make_handler(path)
                with patch.object(handler, 'run_pareto_analysis') as stock, \
                     patch.object(handler, 'run_scan_html') as scan:
                    handler.do_GET()
                handler.send_response.assert_called_once_with(404)
                stock.assert_not_called()
                scan.assert_not_called()
        handler = make_handler('/analyze?code=１２３４５６')
        with patch('web.pareto_web.render_stock_report') as render:
            handler.do_GET()
        handler.send_response.assert_called_once_with(400)
        render.assert_not_called()

    def test_pareto_summary_n_is_display_only_and_path_is_ignored(self):
        for query, expected in (('', 5), ('?n=bad', 5), ('?n=0', 1), ('?n=1000', 100),
                                ('?n=7&path=/other&run_dir=/other&PARETO_RUN_DIR=/other', 7)):
            with self.subTest(query=query):
                handler = make_handler('/pareto-summary' + query)
                with patch('web.pareto_web.render_population', return_value='all accounts') as render, \
                     patch.object(handler, 'run_scan_html') as old_scan:
                    handler.do_GET()
                render.assert_called_once_with(expected)
                old_scan.assert_not_called()
                handler.send_response.assert_called_once_with(200)
                self.assertEqual(handler.wfile.getvalue(), b'all accounts')

    def test_pareto_route_errors_are_escaped(self):
        for path, renderer in (('/analyze?code=000651', 'render_stock_report'),
                               ('/pareto-summary', 'render_population')):
            with self.subTest(path=path):
                handler = make_handler(path)
                with patch('web.pareto_web.' + renderer, side_effect=RuntimeError('<script>bad</script>')):
                    handler.do_GET()
                handler.send_response.assert_called_once_with(500)
                self.assertIn('&lt;script&gt;', handler.wfile.getvalue().decode())
                self.assertNotIn('<script>', handler.wfile.getvalue().decode())

    def test_legacy_scan_errors_are_escaped_and_never_call_pareto(self):
        handler = make_handler('/scan')
        with patch('web.run_scan', side_effect=RuntimeError('<script>bad</script>')), \
             patch('web.pareto_web.render_population') as pareto:
            handler.do_GET()
        handler.send_response.assert_called_once_with(500)
        pareto.assert_not_called()
        self.assertIn('旧加权海选失败', handler.wfile.getvalue().decode())
        self.assertNotIn('<script>', handler.wfile.getvalue().decode())

    def test_unknown_path(self):
        handler = make_handler('/not-found')
        handler.do_GET()
        handler.send_response.assert_called_once_with(404)


class TestWebRendering(OfflineWebCase):
    def test_scan_displays_each_factor_date(self):
        rows = [dict(code='000001', name='模拟甲', score=12.5, close=10,
                     pct=1, pe=20, mcap=100, date='2026-09-08'),
                dict(code='600001', name='模拟乙', score=11, close=12,
                     pct=-1, pe=30, mcap=200, date='2026-07-01')]
        stats = {'passed': 2, 'total_codes': 2, 'elapsed': 0.1}
        with patch('web.run_scan', return_value=(rows, stats)):
            html = make_handler().run_scan_html(5, 100, 5)
        self.assertIn('<th>因子日期</th>', html)
        for row in rows:
            self.assertIn(f'<td>{row["date"]}</td>', html)
        self.assertIn('未统一日期', html)
        self.assertIn('相对分不是融合 S*', html)
        self.assertIn('旧加权海选（非帕累托）', html)
        self.assertIn('加权相对分', html)
        self.assertIn('可能联网取报价', html)

    def test_scan_empty_result(self):
        with patch('web.run_scan', return_value=([], {
                'passed': 0, 'total_codes': 0, 'elapsed': 0})):
            html = make_handler().run_scan_html(5, 100, 5)
        self.assertIn('共 0 只通过过滤', html)

    def test_comprehensive_passes_opens_and_compounds_completed_trades(self):
        closes = 10 + np.sin(np.arange(320) / 10)
        data = [dict(day=(date(2020, 1, 1) + timedelta(days=i)).isoformat(),
                     open=float(close + 0.1), close=float(close),
                     high=float(close + 0.5), low=float(close - 0.5), volume=1000)
                for i, close in enumerate(closes)]
        trades = [dict(buy_date='2020-03-01', sell_date='2020-03-10',
                       buy_price=10, sell_price=11, profit_pct=10,
                       hold_days=9, buy_reason='模拟买入', sell_reason='模拟卖出'),
                  dict(buy_date='2020-04-01', sell_date='2020-04-10',
                       buy_price=10, sell_price=9, profit_pct=-10,
                       hold_days=9, buy_reason='模拟买入', sell_reason='模拟卖出')]
        with patch('web.fetch_kline', return_value=data), \
             patch('web.get_name', return_value='模拟股票'), \
             patch('web.save_stock_name'), \
             patch('web.backtest_comprehensive', return_value=trades) as backtest, \
             patch('web.predict_comprehensive', wraps=engine.predict_comprehensive) as predict, \
             patch('web.make_comprehensive_chart', return_value='chart'), \
             patch('web.fetch_dividends') as dividends:
            html = make_handler().run_comprehensive_analysis('000001', True)
        for call in (backtest.call_args, predict.call_args):
            np.testing.assert_allclose(call.kwargs['opens'], closes + 0.1)
        self.assertTrue(predict.call_args.kwargs['holding'])
        dividends.assert_not_called()
        self.assertIn('-1.0%', html)  # 1.1 × 0.9 − 1，而不是价差简单相加。
        self.assertIn('不计期末未平仓', html)
        self.assertIn('不代表分仓执行', html)


class TestDocumentationLinks(OfflineWebCase):
    def test_current_docs_do_not_reintroduce_retired_claims(self):
        for name in ('README.md', 'CLAUDE.md'):
            text = (Path(web.ROOT) / name).read_text(encoding='utf-8')
            for obsolete in ('62.1%', '+161%', '70个测试',
                             '--break-system-packages', 'factor_backtest5.py'):
                with self.subTest(document=name, obsolete=obsolete):
                    self.assertNotIn(obsolete, text)

    def test_current_local_document_links_exist(self):
        root = Path(web.ROOT)
        for name in ('README.md', 'CLAUDE.md', 'AUDIT_2026-09-09.md'):
            text = (root / name).read_text(encoding='utf-8')
            for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', text):
                if '://' in target or target.startswith('#'):
                    continue
                with self.subTest(document=name, target=target):
                    self.assertTrue((root / target.split('#')[0]).exists())


if __name__ == '__main__':
    unittest.main()