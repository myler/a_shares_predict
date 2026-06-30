"""fetcher.py 单元测试 —— 验证分红计算、数据格式"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from fetcher import enrich_trades_with_dividends, get_name


class TestEnrichTrades(unittest.TestCase):
    def setUp(self):
        self.trades = [
            {'buy_date': '2023-01-15', 'sell_date': '2023-12-20',
             'buy_price': 10.0, 'sell_price': 12.0, 'profit_pct': 20.0,
             'buy_reason': '零轴上金叉', 'sell_reason': '顶背离', 'hold_days': 339},
            {'buy_date': '2024-03-01', 'sell_date': '2024-06-15',
             'buy_price': 8.0, 'sell_price': 9.0, 'profit_pct': 12.5,
             'buy_reason': '底背离', 'sell_reason': '死叉(熊市)', 'hold_days': 106},
            {'buy_date': '2024-10-01', 'sell_date': '2025-01-10',
             'buy_price': 20.0, 'sell_price': 22.0, 'profit_pct': 10.0,
             'buy_reason': '零轴上金叉', 'sell_reason': '顶背离', 'hold_days': 101},
        ]
        self.dividends = [
            {'ex_date': '2023-06-15', 'dividend_per_share': 0.50, 'dividend_10': 5.0},
            {'ex_date': '2024-05-20', 'dividend_per_share': 0.40, 'dividend_10': 4.0},
            {'ex_date': '2025-01-05', 'dividend_per_share': 0.30, 'dividend_10': 3.0},
        ]

    def test_no_dividends(self):
        """空分红列表不应修改交易"""
        trades = enrich_trades_with_dividends(self.trades.copy(), [])
        for t in trades:
            self.assertEqual(t['dividend_total'], 0)

    def test_dividend_matching(self):
        """分红应按除权日正确匹配到对应交易"""
        trades = enrich_trades_with_dividends([t.copy() for t in self.trades], self.dividends)
        # 第一笔: 除权日2023-06-15 在 2023-01-15~2023-12-20 之间
        self.assertEqual(trades[0]['dividend_total'], 0.50)
        # 第二笔: 除权日2024-05-20 在 2024-03-01~2024-06-15 之间
        self.assertEqual(trades[1]['dividend_total'], 0.40)
        # 第三笔: 除权日2025-01-05 在 2024-10-01~2025-01-10 之间
        self.assertEqual(trades[2]['dividend_total'], 0.30)

    def test_dividend_yield_pct(self):
        """分红收益率 = 累计分红 / 买入价 × 100"""
        trades = enrich_trades_with_dividends([t.copy() for t in self.trades], self.dividends)
        self.assertAlmostEqual(trades[0]['dividend_yield_pct'], 0.50 / 10.0 * 100)
        self.assertAlmostEqual(trades[1]['dividend_yield_pct'], 0.40 / 8.0 * 100)
        self.assertAlmostEqual(trades[2]['dividend_yield_pct'], 0.30 / 20.0 * 100)

    def test_total_return_pct(self):
        """总收益 = 价差 + 分红收益率"""
        trades = enrich_trades_with_dividends([t.copy() for t in self.trades], self.dividends)
        for t in trades:
            expected = t['profit_pct'] + t['dividend_yield_pct']
            self.assertAlmostEqual(t['total_return_pct'], expected)

    def test_no_matching_dividend(self):
        """除权日不在任何交易期内时不应影响任何交易"""
        divs = [{'ex_date': '2019-01-01', 'dividend_per_share': 1.0}]
        trades = enrich_trades_with_dividends([t.copy() for t in self.trades], divs)
        for t in trades:
            self.assertEqual(t['dividend_total'], 0)

    def test_boundary_excluded(self):
        """除权日等于卖出日时不应计入（卖出日已不持有了）"""
        divs = [{'ex_date': '2023-12-20', 'dividend_per_share': 1.0}]
        trades = enrich_trades_with_dividends([t.copy() for t in self.trades], divs)
        # 除权日=卖出日，不应计入第一笔交易
        self.assertEqual(trades[0]['dividend_total'], 0)

    def test_boundary_included(self):
        """除权日等于买入日时应计入"""
        divs = [{'ex_date': '2023-01-15', 'dividend_per_share': 1.0}]
        trades = enrich_trades_with_dividends([t.copy() for t in self.trades], divs)
        # 除权日=买入日，应计入
        self.assertEqual(trades[0]['dividend_total'], 1.0)


class TestGetName(unittest.TestCase):
    def test_format_6digit(self):
        """6位数字代码应尝试查询"""
        name = get_name('600519')
        self.assertIsInstance(name, str)
        self.assertTrue(len(name) > 0)

    def test_unknown_code(self):
        """不存在的代码应返回代码本身"""
        name = get_name('999999')
        self.assertEqual(name, '999999')


if __name__ == '__main__':
    unittest.main(verbosity=2)
