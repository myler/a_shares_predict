"""fundamentals.py 单元测试 —— 财务质量初筛不应伪造完整估值结论。"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fundamentals import (
    STATUS_GO, STATUS_INSUFFICIENT, STATUS_NO_GO, STATUS_NOT_APPLICABLE,
    screen_value_quality,
)


def _annual(year, core_profit=100, operating_cash_flow=100, roe=15,
            debt_ratio=50, current_ratio=1.5, revenue=1000, **extra):
    return {
        'SECURITY_CODE': '000001',
        'SECURITY_NAME_ABBR': '示例公司',
        'REPORT_TYPE': '年报',
        'REPORT_DATE': f'{year}-12-31 00:00:00',
        'REPORT_DATE_NAME': f'{year}年报',
        'NOTICE_DATE': f'{year + 1}-04-30 00:00:00',
        'CURRENCY': 'CNY',
        'TOTALOPERATEREVE': revenue,
        'PARENTNETPROFIT': core_profit + 5,
        'KCFJCXSYJLR': core_profit,
        'NETCASH_OPERATE_PK': operating_cash_flow,
        'ROEJQ': roe,
        'ROIC': roe - 2,
        'ZCFZL': debt_ratio,
        'LD': current_ratio,
        **extra,
    }


class TestValueQualityScreen(unittest.TestCase):
    def test_stable_nonfinancial_company_passes_initial_screen(self):
        result = screen_value_quality([
            _annual(2023, core_profit=80, operating_cash_flow=90, roe=13,
                    debt_ratio=55, revenue=900),
            _annual(2025, core_profit=110, operating_cash_flow=120, roe=16,
                    debt_ratio=50, revenue=1100),
            _annual(2024, core_profit=100, operating_cash_flow=110, roe=15,
                    debt_ratio=52, revenue=1000),
        ])

        self.assertEqual(result['status'], STATUS_GO)
        self.assertEqual(result['annual_reports'][0]['report_date'], '2025-12-31')
        checks = {check['key']: check['status'] for check in result['checks']}
        self.assertEqual(checks['cash_conversion'], 'pass')
        self.assertEqual(checks['roe'], 'pass')

    def test_latest_core_loss_is_not_a_value_candidate(self):
        result = screen_value_quality([
            _annual(2025, core_profit=-10, operating_cash_flow=-5),
            _annual(2024, core_profit=100, operating_cash_flow=110),
            _annual(2023, core_profit=90, operating_cash_flow=95),
        ])

        self.assertEqual(result['status'], STATUS_NO_GO)
        profitability = next(check for check in result['checks']
                             if check['key'] == 'profitability')
        self.assertEqual(profitability['status'], 'fail')

    def test_fewer_than_three_annual_reports_is_insufficient(self):
        result = screen_value_quality([_annual(2025), _annual(2024)])

        self.assertEqual(result['status'], STATUS_INSUFFICIENT)
        self.assertEqual(len(result['checks']), 1)

    def test_financial_institution_uses_method_applicability_stop(self):
        result = screen_value_quality([
            _annual(2025, TOTALDEPOSITS=1000),
            _annual(2024, TOTALDEPOSITS=900),
            _annual(2023, TOTALDEPOSITS=800),
        ])

        self.assertEqual(result['status'], STATUS_NOT_APPLICABLE)


if __name__ == '__main__':
    unittest.main(verbosity=2)