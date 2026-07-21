"""run_cli.py 单元测试 —— 巴芒基本面研究输出必须保留非交易边界。"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_cli import format_bmfund_cli_research, parse_cli_args


class TestBmfundCliFormatting(unittest.TestCase):
    def test_research_output_includes_checks_reports_and_nontrading_boundary(self):
        research = {
            'status_label': '通过财务质量初筛',
            'conclusion': '可继续研究。',
            'source': '东方财富财务摘要',
            'scope': '非金融企业财务质量初筛。',
            'checks': [{
                'status': 'pass',
                'label': '扣非盈利连续性',
                'detail': '最近三年均为正',
            }],
            'annual_reports': [{
                'report_name': '2025年报',
                'report_date': '2025-12-31',
                'notice_date': '2026-04-29',
                'revenue': 100000000000,
                'core_profit': 10000000000,
                'operating_cash_flow': 12000000000,
                'roe': 15.0,
                'roic': 12.0,
                'debt_ratio': 50.0,
                'current_ratio': 1.5,
            }],
            'limitations': ['未做内在价值估值。'],
        }

        output = '\n'.join(format_bmfund_cli_research(research, '示例公司', '000001'))

        self.assertIn('巴芒基本面研究', output)
        self.assertIn('非买卖信号', output)
        self.assertIn('[通过] 扣非盈利连续性', output)
        self.assertIn('2025年报 | 披露 2026-04-29', output)
        self.assertIn('未做内在价值估值。', output)


class TestCliArgumentParsing(unittest.TestCase):
    def test_bmfund_accepts_one_six_digit_stock_code(self):
        self.assertEqual(parse_cli_args(['bmfund', '601888']),
                         ('bmfund', ['601888']))

    def test_six_digit_code_uses_legacy_chart_mode(self):
        self.assertEqual(parse_cli_args(['603893']),
                         ('chart', ['603893']))

    def test_typo_is_not_treated_as_a_stock_code(self):
        with self.assertRaisesRegex(ValueError, 'bmfund'):
            parse_cli_args(['mdfund'])

    def test_rejects_invalid_stock_code_before_analysis(self):
        with self.assertRaisesRegex(ValueError, '6位数字'):
            parse_cli_args(['bmfund', '60188'])


if __name__ == '__main__':
    unittest.main(verbosity=2)