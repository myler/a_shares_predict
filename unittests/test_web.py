"""三策略 Web 路由、刷新委托与页面契约；所有数据使用模拟输入。"""

import ast
import io
import json
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import web


def make_handler(path='/'):
    handler = web.Handler.__new__(web.Handler)
    handler.path = path
    handler.wfile = io.BytesIO()
    handler.send_response = Mock()
    handler.send_header = Mock()
    handler.end_headers = Mock()
    return handler


class PageControls(HTMLParser):
    """检查真实 HTML 控件，不把脚本字符串误当作可见选项。"""
    def __init__(self, page):
        super().__init__()
        self.options = []
        self.tabs = []
        self.elements = {}
        self.current = None
        self.feed(page)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.elements[attrs['id']] = attrs
        if tag == 'option':
            self.current = [attrs, '']
            self.options.append(self.current)
        elif tag == 'button' and 'data-tab' in attrs:
            self.current = [attrs, '']
            self.tabs.append(self.current)

    def handle_data(self, text):
        if self.current is not None:
            self.current[1] += text

    def handle_endtag(self, tag):
        if tag in ('option', 'button'):
            self.current = None


class OfflineWebCase(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch('socket.socket', side_effect=AssertionError('network forbidden')))
        self.enterContext(patch('sqlite3.connect', side_effect=AssertionError('database forbidden')))


class TestPageContract(OfflineWebCase):
    def test_exact_three_visible_options_and_two_tabs(self):
        controls = PageControls(web.PAGE)
        self.assertEqual([(attrs['value'], text) for attrs, text in controls.options], [
            ('pareto', '融合策略'), ('buyhold', '长线持有'), ('value', '巴芒基本面研究'),
        ])
        self.assertIn('selected', controls.options[0][0])
        for attrs, _ in controls.options:
            self.assertNotIn('hidden', attrs)
            self.assertNotIn('disabled', attrs)
            self.assertNotIn('style', attrs)
        self.assertEqual([(attrs['data-tab'], text) for attrs, text in controls.tabs], [
            ('analyze', '策略分析'), ('scan', '融合策略回测总览'),
        ])
        self.assertEqual({key for key in controls.elements if key.startswith('tab-')},
                         {'tab-analyze', 'tab-scan'})
        self.assertNotIn('disabled', controls.elements['holding'])

    def test_homepage_uses_only_shared_default_methodology(self):
        formula = json.loads(re.search(r'const PARETO_FORMULA = ("[^\n]+");', web.PAGE).group(1))
        self.assertEqual(formula, web.pareto_web.render_methodology())
        self.assertNotIn('__PARETO_FORMULA__', web.PAGE)
        self.assertIn("pareto:{title:'融合策略',html:PARETO_FORMULA}", web.PAGE)
        self.assertIn('三值化丢失幅度', formula)
        self.assertIn('不为本策略公式、参数或盈利背书', formula)
        self.assertIn('模型默认参数（本次实际参数以分析结果为准）', formula)
        self.assertIn('不套用于当前单股分析', formula)
        with patch('web.pareto_web.render_methodology', return_value='<p></script>${evil}`</p>') as method, \
             patch('web.pareto_web.render_population') as history, \
             patch('web.fusion_web.render_analysis') as live:
            rendered = web.render_page()
        method.assert_called_once_with()
        history.assert_not_called()
        live.assert_not_called()
        self.assertIn('\\u003c/script\\u003e', rendered)
        self.assertNotIn('<p></script>', rendered)

    def test_retired_web_code_and_hidden_ui_are_removed(self):
        for obsolete in ('COMPREHENSIVE_CONFIG', 'comprehensive', 'legacy-scan',
                         'legacyN', 'legacyPE', 'legacyPrice', 'runLegacyScan',
                         '/scan?', '旧加权', 'paretoReadOnly'):
            self.assertNotIn(obsolete, web.PAGE)
        for name in ('run_scan_html', 'run_comprehensive_analysis',
                     '_run_comprehensive_analysis_legacy', 'run_analysis',
                     'run_multifactor_analysis', 'make_pred_table'):
            self.assertFalse(hasattr(web.Handler, name), name)
        tree = ast.parse(Path(web.__file__).read_text(encoding='utf-8'))
        imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        imported.update(alias.name for node in ast.walk(tree)
                        if isinstance(node, ast.Import) for alias in node.names)
        self.assertTrue({'engine', 'scan_composite', 'plotting', 'inspect'}.isdisjoint(imported))
        self.assertIn('fusion_web', imported)

    def test_live_refresh_and_history_boundaries_are_explicit(self):
        single = web.PAGE.split('<div id="tab-analyze">', 1)[1].split('<!-- ══════════ Tab 2:', 1)[0]
        for text in ('每次分析均联网刷新本股', '来源最新已完成日线',
                     '未成功刷新，不给出最新结论', '历史不足时仅作条件判断',
                     '不声称完整同日 F1', '历史回测期限'):
            self.assertIn(text, single)
        self.assertNotIn('只读', single)
        overview = web.PAGE.split('<div id="tab-scan"', 1)[1].split('<script>', 1)[0]
        for text in ('历史回测只读', '不联网、不刷新缓存，不重跑',
                     '请求区间和有效截止', '本轮冻结报告',
                     'N 仅控制按代码顺序展示', '实际全部账户', '全期禁买但保留本金'):
            self.assertIn(text, overview)
        for text in ('送转股份与分红日期上限尚未完善', '未严格验证年度连续性'):
            self.assertIn(text, web.PAGE)
        summary_request = web.PAGE.split('async function runScan(){', 1)[1].split('async function analyze(e){', 1)[0]
        self.assertIn("const url='/pareto-summary?n='", summary_request)
        self.assertNotIn('/analyze', summary_request)

    def test_fluid_layout_and_scrollable_result_tables(self):
        self.assertNotIn('max-width:920px', web.PAGE)
        self.assertNotIn('max-width:520px', web.PAGE)
        self.assertIn('grid-template-columns:minmax(240px,320px) minmax(0,1fr)', web.PAGE)
        self.assertIn('@media(max-width:900px)', web.PAGE)
        self.assertIn('overflow-x:auto', web.PAGE)
        self.assertIn('prepareResultTables(box)', web.PAGE)
        self.assertIn('prepareResultTables(result)', web.PAGE)
        self.assertIn('wrapper.tabIndex=0', web.PAGE)
        self.assertIn("wrapper.setAttribute('role','region')", web.PAGE)
        self.assertIn('width:100%;height:auto', web.PAGE)

    def test_holding_and_dividend_states(self):
        change = web.PAGE.split('function onStrategyChange(){', 1)[1].split('function toggleDividend(){', 1)[0]
        self.assertIn('resetAnalysis();', change)
        self.assertIn("document.getElementById('holdingLabel').style.display=s==='value'?'none':'flex'", change)
        buyhold = change.split("if(s==='buyhold'){", 1)[1].split("}else if(s==='value'){", 1)[0]
        self.assertIn('holding.checked=true;holding.disabled=true;', buyhold)
        self.assertIn("document.getElementById('divLabel').style.display='flex'", buyhold)
        value = change.split("}else if(s==='value'){", 1)[1].split('}else{', 1)[0]
        self.assertIn('holding.checked=false;holding.disabled=true;', value)
        self.assertIn("document.getElementById('calcDividend').checked=false", value)
        self.assertIn("document.getElementById('divLabel').style.display='none'", value)
        self.assertIn('}else{holding.disabled=false;toggleDividend();}', change)
        toggle = web.PAGE.split('function toggleDividend(){', 1)[1].split('function prepareResultTables', 1)[0]
        self.assertIn('resetAnalysis();', toggle)
        self.assertIn("style.display=holding?'flex':'none'", toggle)
        self.assertIn("if(!holding)document.getElementById('calcDividend').checked=false", toggle)

    def test_late_responses_symbol_safety_and_flags_on_every_request(self):
        controls = PageControls(web.PAGE)
        self.assertEqual(controls.elements['code']['oninput'], 'resetAnalysis()')
        self.assertEqual(controls.elements['code']['pattern'], '[0-9]{6}')
        self.assertEqual(controls.elements['calcDividend']['onchange'], 'resetAnalysis()')
        self.assertIn("document.getElementById('result').replaceChildren();", web.PAGE)
        analyze = web.PAGE.split('async function analyze(e){', 1)[1]
        self.assertIn('if(!/^[0-9]{6}$/.test(code))return;', analyze)
        self.assertIn('encodeURIComponent(code)', analyze)
        self.assertIn("+'&holding='+(holding?'1':'0')+'&dividend='+(calcDividend?'1':'0')", analyze)
        self.assertNotIn("strategy==='pareto'?'':'&holding='", analyze)
        self.assertIn("fetch(url,{cache:'no-store'})", analyze)
        self.assertIn('联网刷新本股→计算融合指标', analyze)
        self.assertIn('const revision=++analysisRevision;', analyze)
        self.assertLess(analyze.index('const text=await resp.text();'), analyze.index('if(revision!==analysisRevision)return;'))
        self.assertLess(analyze.index('if(revision!==analysisRevision)return;'), analyze.index('result.innerHTML=text'))
        self.assertIn('if(revision===analysisRevision)result.textContent=', analyze)
        self.assertIn("if(revision===analysisRevision){btn.disabled=false;btn.textContent='分析'}", analyze)


class TestWebRoutes(OfflineWebCase):
    def test_homepage_renders_formula(self):
        handler = make_handler('/')
        handler.do_GET()
        handler.send_response.assert_called_once_with(200)
        self.assertEqual(handler.wfile.getvalue().decode(), web.PAGE)

    def test_default_strategy_is_pareto(self):
        handler = make_handler('/analyze?code=000651')
        with patch('web.fusion_web.render_analysis', return_value='fusion') as run, \
             patch('web.pareto_web.render_stock_report') as history, \
             patch('web.fetch_kline') as fetch, patch('web.save_stock_name') as save:
            handler.do_GET()
        run.assert_called_once_with('000651', False, False)
        history.assert_not_called()
        fetch.assert_not_called()
        save.assert_not_called()
        handler.send_response.assert_called_once_with(200)
        self.assertEqual(handler.wfile.getvalue(), b'fusion')

    def test_explicit_routes_preserved(self):
        for strategy, method, args in (
            ('pareto', 'run_pareto_analysis', ('000651', True, True)),
            ('buyhold', 'run_buyhold_analysis', ('000651', True)),
            ('value', 'run_value_analysis', ('000651',)),
        ):
            with self.subTest(strategy=strategy):
                handler = make_handler(
                    f'/analyze?code=000651&strategy={strategy}&holding=1&dividend=1')
                with patch.object(handler, method, return_value='result') as run, \
                     patch('web.fusion_web.render_analysis') as refresh:
                    handler.do_GET()
                run.assert_called_once_with(*args)
                refresh.assert_not_called()
                handler.send_response.assert_called_once_with(200)

    def test_retired_and_unknown_strategies_rejected_before_data_access(self):
        for strategy in ('comprehensive', 'macd', 'multi', 'typo'):
            with self.subTest(strategy=strategy):
                handler = make_handler('/analyze?code=000651&strategy=' + strategy)
                with patch('web.fetch_kline') as fetch, \
                     patch('web.fetch_financial_summaries') as financials, \
                     patch('web.fusion_web.render_analysis') as refresh, \
                     patch('web.pareto_web.render_stock_report') as history:
                    handler.do_GET()
                handler.send_response.assert_called_once_with(400)
                for mocked in (fetch, financials, refresh, history):
                    mocked.assert_not_called()

    def test_invalid_stock_rejected_for_all_three_strategies(self):
        for strategy in ('pareto', 'buyhold', 'value'):
            for code in ('', 'abc', '12345', '1234567', '１２３４５６', '%3Csvg%3E', '00065%26'):
                with self.subTest(strategy=strategy, code=code):
                    handler = make_handler(f'/analyze?code={code}&strategy={strategy}')
                    with patch('web.fusion_web.render_analysis') as refresh, \
                         patch('web.fetch_kline') as fetch, \
                         patch('web.fetch_financial_summaries') as financials:
                        handler.do_GET()
                    handler.send_response.assert_called_once_with(400)
                    for mocked in (refresh, fetch, financials):
                        mocked.assert_not_called()

    def test_pareto_always_delegates_refresh_and_both_flags_not_http_paths(self):
        for holding, dividend in (('0', '0'), ('1', '0'), ('0', '1'), ('1', '1'), ('true', 'true')):
            with self.subTest(holding=holding, dividend=dividend):
                handler = make_handler(
                    f'/analyze?code=000651&strategy=pareto&holding={holding}&dividend={dividend}'
                    '&path=/other&run_dir=/other&PARETO_RUN_DIR=/other&refresh=0&force=0')
                with patch('web.fusion_web.render_analysis', return_value='refreshed analysis') as render, \
                     patch('web.pareto_web.render_stock_report') as history, \
                     patch('web.fetch_kline') as fetch, patch('web.save_stock_name') as save:
                    handler.do_GET()
                render.assert_called_once_with('000651', holding == '1', dividend == '1')
                for mocked in (history, fetch, save):
                    mocked.assert_not_called()
                handler.send_response.assert_called_once_with(200)
                self.assertEqual(handler.wfile.getvalue(), b'refreshed analysis')

    def test_pareto_method_defaults(self):
        with patch('web.fusion_web.render_analysis', return_value='analysis') as render:
            self.assertEqual(make_handler().run_pareto_analysis('000001'), 'analysis')
        render.assert_called_once_with('000001', False, False)

    def test_removed_scan_and_route_prefixes_return_404_without_data_access(self):
        for path in ('/scan', '/scan?n=5&max_pe=100', '/scan?n=bad',
                     '/analyze-refresh?code=000651', '/scan-refresh', '/pareto-summary-refresh'):
            with self.subTest(path=path):
                handler = make_handler(path)
                with patch('web.fusion_web.render_analysis') as stock, \
                     patch('web.fetch_kline') as fetch, \
                     patch('web.pareto_web.render_population') as overview:
                    handler.do_GET()
                handler.send_response.assert_called_once_with(404)
                for mocked in (stock, fetch, overview):
                    mocked.assert_not_called()

    def test_pareto_summary_n_is_display_only_and_path_is_ignored(self):
        for query, expected in (('', 5), ('?n=bad', 5), ('?n=0', 1), ('?n=1000', 100),
                                ('?n=7&path=/other&run_dir=/other&PARETO_RUN_DIR=/other', 7)):
            with self.subTest(query=query):
                handler = make_handler('/pareto-summary' + query)
                with patch('web.pareto_web.render_population', return_value='all accounts') as render, \
                     patch('web.fusion_web.render_analysis') as refresh, \
                     patch('web.fetch_kline') as fetch:
                    handler.do_GET()
                render.assert_called_once_with(expected)
                refresh.assert_not_called()
                fetch.assert_not_called()
                handler.send_response.assert_called_once_with(200)
                self.assertEqual(handler.wfile.getvalue(), b'all accounts')

    def test_pareto_route_errors_are_escaped(self):
        for path, renderer in (('/analyze?code=000651', 'web.fusion_web.render_analysis'),
                               ('/pareto-summary', 'web.pareto_web.render_population')):
            with self.subTest(path=path):
                handler = make_handler(path)
                with patch(renderer, side_effect=RuntimeError('<script>bad</script>')):
                    handler.do_GET()
                handler.send_response.assert_called_once_with(500)
                self.assertIn('&lt;script&gt;', handler.wfile.getvalue().decode())
                self.assertNotIn('<script>', handler.wfile.getvalue().decode())

    def test_refresh_failure_never_falls_back_to_cached_or_historical_result(self):
        handler = make_handler('/analyze?code=000651')
        with patch('web.fusion_web.render_analysis', side_effect=RuntimeError('刷新失败')), \
             patch('web.pareto_web.render_stock_report') as history, \
             patch('web.fetch_kline') as fetch:
            handler.do_GET()
        handler.send_response.assert_called_once_with(500)
        history.assert_not_called()
        fetch.assert_not_called()
        self.assertIn('刷新失败', handler.wfile.getvalue().decode())

    def test_unknown_path(self):
        handler = make_handler('/not-found')
        handler.do_GET()
        handler.send_response.assert_called_once_with(404)


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