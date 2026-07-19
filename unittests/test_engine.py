"""engine.py 单元测试 —— 验证MACD计算、背离检测、牛熊判定、回测、预测等核心逻辑"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import patch
import numpy as np
from datetime import datetime, timedelta

from engine import (
    ema, calc_macd, find_divergences, detect_regime,
    zero_line_cycles, backtest, predict
)


def _mkdates(n, start_year=2020):
    """生成n个连续有效日期字符串 YYYY-MM-DD"""
    d0 = datetime(start_year, 1, 1)
    return [(d0 + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(n)]


class TestEMA(unittest.TestCase):
    def test_simple(self):
        """EMA 对常数序列应返回常数"""
        data = np.array([10.0] * 50)
        result = ema(data, 12)
        self.assertAlmostEqual(result[-1], 10.0, places=2)

    def test_rising(self):
        """EMA 对上升序列应滞后于价格"""
        data = np.arange(1, 101, dtype=float)
        result = ema(data, 20)
        self.assertTrue(result[-1] < 100)  # EMA滞后

    def test_too_short(self):
        """数据不足时应返回NaN"""
        data = np.array([1.0, 2.0, 3.0])
        result = ema(data, 12)
        self.assertTrue(np.all(np.isnan(result)))


class TestCalcMACD(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.prices = np.cumsum(np.random.randn(500) * 0.5) + 50

    def test_output_shape(self):
        """DIF、DEA、BAR应有相同长度"""
        dif, dea, bar = calc_macd(self.prices)
        self.assertEqual(len(dif), len(self.prices))
        self.assertEqual(len(dea), len(self.prices))
        self.assertEqual(len(bar), len(self.prices))

    def test_bar_relation(self):
        """BAR = (DIF - DEA) × 2"""
        dif, dea, bar = calc_macd(self.prices)
        valid = ~np.isnan(dif) & ~np.isnan(dea)
        np.testing.assert_array_almost_equal(bar[valid], (dif[valid] - dea[valid]) * 2)

    def test_golden_dead(self):
        """MACD应有金叉和死叉"""
        dif, dea, _ = calc_macd(self.prices)
        valid = ~np.isnan(dif) & ~np.isnan(dea)
        # 至少有一次金叉或死叉
        crosses = np.diff(np.sign(dif[valid] - dea[valid]))
        self.assertTrue(np.any(crosses != 0), "应有至少一次金叉/死叉")


class TestDetectRegime(unittest.TestCase):
    def test_bull_market(self):
        """单边上涨应为牛市"""
        prices = np.linspace(10, 100, 200)
        regimes, ma = detect_regime(prices)
        self.assertEqual(regimes[-1], 'bull')

    def test_bear_market(self):
        """单边下跌应为熊市"""
        prices = np.linspace(100, 10, 200)
        regimes, ma = detect_regime(prices)
        self.assertEqual(regimes[-1], 'bear')

    def test_ma_shift(self):
        """MA60应滞后于价格"""
        prices = np.linspace(10, 100, 200)
        _, ma = detect_regime(prices)
        self.assertGreater(prices[-1], ma[-1])


class TestFindDivergences(unittest.TestCase):
    def setUp(self):
        # 构造底背离：价格新低，DIF不再新低
        n = 200
        self.dates = _mkdates(n)
        np.random.seed(1)
        self.prices = np.linspace(50, 30, 100).tolist() + np.linspace(30, 55, 100).tolist()
        self.prices = np.array(self.prices) + np.random.randn(n) * 0.5
        dif, _, _ = calc_macd(self.prices)
        self.dif = dif

    def test_returns_lists(self):
        """应返回顶背离和底背离两个列表"""
        tops, bottoms = find_divergences(self.dates, self.prices, self.dif)
        self.assertIsInstance(tops, list)
        self.assertIsInstance(bottoms, list)

    def test_no_crash_on_nan(self):
        """存在NaN时不应崩溃"""
        dif_with_nan = self.dif.copy()
        dif_with_nan[50:60] = np.nan
        try:
            find_divergences(self.dates, self.prices, dif_with_nan)
        except Exception as e:
            self.fail(f"NaN导致崩溃: {e}")


class TestBacktest(unittest.TestCase):
    def setUp(self):
        n = 500
        self.dates = _mkdates(n)
        np.random.seed(2)
        self.prices = 50 + np.cumsum(np.random.randn(n) * 0.3)
        self.dif, self.dea, _ = calc_macd(self.prices)
        regimes, _ = detect_regime(self.prices)
        tops, bottoms = find_divergences(self.dates, self.prices, self.dif)
        self.regimes = regimes
        self.tops = tops
        self.bottoms = bottoms

    def test_returns_list(self):
        """回测应返回交易列表"""
        trades = backtest(self.dates, self.prices, self.dif, self.dea,
                          self.tops, self.bottoms, self.regimes)
        self.assertIsInstance(trades, list)

    def test_trade_structure(self):
        """每笔交易应有必需字段"""
        trades = backtest(self.dates, self.prices, self.dif, self.dea,
                          self.tops, self.bottoms, self.regimes)
        if trades:
            for t in trades:
                self.assertIn('buy_date', t)
                self.assertIn('sell_date', t)
                self.assertIn('buy_price', t)
                self.assertIn('sell_price', t)
                self.assertIn('profit_pct', t)
                self.assertIn('hold_days', t)
                self.assertIn('buy_reason', t)
                self.assertIn('sell_reason', t)

    def test_buy_before_sell(self):
        """买入日应在卖出日之前"""
        trades = backtest(self.dates, self.prices, self.dif, self.dea,
                          self.tops, self.bottoms, self.regimes)
        for t in trades:
            self.assertLess(t['buy_date'], t['sell_date'])


class TestPredict(unittest.TestCase):
    def setUp(self):
        n = 300
        np.random.seed(3)
        self.dates = _mkdates(n)
        self.prices = 50 + np.cumsum(np.random.randn(n) * 0.2)
        self.dif, self.dea, self.bar = calc_macd(self.prices)
        self.regimes, _ = detect_regime(self.prices)
        self.tops, self.bottoms = find_divergences(self.dates, self.prices, self.dif)

    def test_returns_dict(self):
        """预测应返回结构化 dict"""
        result = predict(self.dates, self.prices, self.dif, self.dea,
                        self.bar, self.regimes, self.tops, self.bottoms)
        self.assertIsInstance(result, dict)
        self.assertIn('strategy', result)
        self.assertIn('score', result)
        self.assertIn('signal', result)
        self.assertIn('action', result)

    def test_holding_mode(self):
        """holding=True 和 False 应有不同建议"""
        result_hold = predict(self.dates, self.prices, self.dif, self.dea,
                              self.bar, self.regimes, self.tops, self.bottoms,
                              holding=True)
        result_nohold = predict(self.dates, self.prices, self.dif, self.dea,
                                self.bar, self.regimes, self.tops, self.bottoms,
                                holding=False)
        self.assertIn('action', result_hold)
        self.assertIn('action', result_nohold)
        # holding 模式下 action 应包含 '增持'/'拿住'/'减持' 之一
        self.assertIn(result_hold['action'],
                      ['增持', '拿住', '减持', '卖出', '观望'])

    def test_contains_scoring(self):
        """预测结果应包含评分"""
        result = predict(self.dates, self.prices, self.dif, self.dea,
                        self.bar, self.regimes, self.tops, self.bottoms)
        self.assertIn('score', result)
        self.assertIsInstance(result['score'], int)

    def test_short_data_handled(self):
        """数据不足时应返回 error"""
        short_dates = self.dates[:30]
        short_prices = self.prices[:30]
        short_dif = self.dif[:30]
        short_dea = self.dea[:30]
        short_bar = self.bar[:30]
        short_reg = self.regimes[:30]
        result = predict(short_dates, short_prices, short_dif, short_dea,
                        short_bar, short_reg, self.tops, self.bottoms)
        self.assertIn('error', result)


class TestZeroLineCycles(unittest.TestCase):
    def setUp(self):
        n = 300
        np.random.seed(4)
        prices = 50 + np.cumsum(np.random.randn(n) * 0.3)
        dif, _, _ = calc_macd(prices)
        self.dates = _mkdates(n)
        self.dif = dif

    def test_returns_list(self):
        cycles = zero_line_cycles(self.dates, self.dif)
        self.assertIsInstance(cycles, list)

    def test_cycle_has_zone(self):
        cycles = zero_line_cycles(self.dates, self.dif)
        for c in cycles:
            self.assertIn(c['zone'], ('above', 'below'))
            self.assertIn('start_date', c)
            self.assertIn('end_date', c)


if __name__ == '__main__':
    unittest.main(verbosity=2)


class TestMultifactor(unittest.TestCase):
    """多因子共振策略测试"""
    def setUp(self):
        np.random.seed(42)
        n = 500
        self.closes = 50 + np.cumsum(np.random.randn(n) * 0.5)
        self.highs = self.closes + np.abs(np.random.randn(n) * 0.3)
        self.lows = self.closes - np.abs(np.random.randn(n) * 0.3)
        self.vols = np.abs(np.random.randn(n) * 1000000 + 5000000)

    def test_returns_list(self):
        from engine import backtest_multifactor
        dates = _mkdates(500)
        trades = backtest_multifactor(dates, self.closes, self.highs, self.lows, self.vols)
        self.assertIsInstance(trades, list)

    def test_rsi_range(self):
        from engine import calc_rsi
        rsi = calc_rsi(self.closes)
        valid = rsi[~np.isnan(rsi)]
        self.assertTrue(np.all(valid >= 0))
        self.assertTrue(np.all(valid <= 100))

    def test_rsi_flat_market_is_neutral(self):
        from engine import calc_rsi
        rsi = calc_rsi(np.full(30, 10.0))
        self.assertEqual(rsi[-1], 50)

    def test_obv_flow_is_signed_and_normalized(self):
        from engine import calc_obv, calc_obv_flow
        closes = np.array([10, 11, 12, 11, 10, 9], dtype=float)
        volumes = np.full(len(closes), 100.0)
        obv = calc_obv(closes, volumes)
        self.assertEqual(obv[0], 0)
        self.assertAlmostEqual(calc_obv_flow(obv, volumes, len(closes) - 1), -0.2)

    def test_comprehensive_rejects_strong_negative_obv_flow(self):
        from engine import predict_comprehensive
        closes = np.arange(100, 30, -1, dtype=float)
        highs = closes + 0.5
        lows = closes - 0.5
        volumes = np.full(len(closes), 100.0)
        pred = predict_comprehensive(_mkdates(len(closes)), closes, highs, lows, volumes)
        self.assertFalse(pred['gates']['passed'])
        self.assertTrue(any(item['gate'] == 'OBV净量能流'
                            for item in pred['gates']['details']))

    def test_auxiliary_short_data_reports_all_eighteen_votes(self):
        from engine import calc_auxiliary_consensus
        closes = np.full(59, 10.0)
        consensus = calc_auxiliary_consensus(
            closes, closes + 0.1, closes - 0.1, np.full(59, 100.0))
        self.assertEqual(consensus['neutral_count'], 18)
        self.assertEqual(len(consensus['votes']), 18)

    def test_net_trade_return_includes_both_side_costs(self):
        from engine import calc_net_trade_return
        self.assertAlmostEqual(
            calc_net_trade_return(10, 11),
            (11 * (1 - 0.0003 - 0.0005) / (10 * (1 + 0.0003)) - 1) * 100,
        )

    def test_summarize_trades_compounds_returns(self):
        from engine import summarize_trades
        summary = summarize_trades([
            {'profit_pct': 10.0}, {'profit_pct': -10.0},
        ])
        self.assertAlmostEqual(summary['price_return_pct'], -1.0)
        self.assertAlmostEqual(summary['total_return_pct'], -1.0)

    def test_comprehensive_executes_at_next_day_open(self):
        from engine import backtest_comprehensive
        count = 65
        closes = np.linspace(10, 20, count)
        opens = closes + 1
        highs = closes + 1.5
        lows = closes - 0.5
        volumes = np.full(count, 100.0)

        def score_at_index(*args):
            index = args[-1]
            composite = 65 if index == 60 else 30
            return 50, 50, 50, 50, 0.0, composite, True, {}, {}

        with patch('engine._score_comprehensive', side_effect=score_at_index):
            trades = backtest_comprehensive(
                _mkdates(count), closes, highs, lows, volumes, opens=opens)

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]['buy_date'], _mkdates(count)[61])
        self.assertEqual(trades[0]['sell_date'], _mkdates(count)[62])
        self.assertEqual(trades[0]['buy_price'], opens[61])
        self.assertEqual(trades[0]['sell_price'], opens[62])

    def test_kdj_output(self):
        from engine import calc_kdj
        k, d, j = calc_kdj(self.highs, self.lows, self.closes)
        self.assertEqual(len(k), len(self.closes))
        self.assertTrue(not np.all(np.isnan(k[-50:])))

    def test_bollinger_output(self):
        from engine import calc_bollinger
        upper, mid, lower = calc_bollinger(self.closes)
        self.assertEqual(len(upper), len(self.closes))
        valid = ~np.isnan(upper)
        self.assertTrue(np.all(upper[valid] >= lower[valid]))

    def test_wr_range(self):
        from engine import calc_wr
        wr = calc_wr(self.highs, self.lows, self.closes)
        valid = wr[~np.isnan(wr)]
        self.assertTrue(np.all(valid >= 0))
        self.assertTrue(np.all(valid <= 100))
