"""Pure synthetic account contracts: no market data, database or network."""

from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta
import json
import math
import unittest

from capital_account import Account, AccountConfig


def account(capital=10000., **overrides):
    settings = dict(initial_capital=capital, slippage_bps=0.)
    settings.update(overrides)
    return Account('600000', AccountConfig(**settings), record_ledger=True)


def fills(acc):
    return [row for row in acc.ledger if row['event'] in ('build', 'add', 'reduce', 'exit')]


class TestCapitalAccount(unittest.TestCase):
    def test_defaults_and_independent_accounts(self):
        config = AccountConfig()
        self.assertEqual((config.initial_capital, config.commission_rate,
                          config.minimum_commission, config.slippage_bps,
                          config.max_drawdown, config.cooldown_sessions),
                         (1000000., .0003, 5., 5., .15, 20))
        with self.assertRaises(FrozenInstanceError):
            config.initial_capital = 1.
        left, right = Account('600000'), Account('000001')
        self.assertTrue(left.rebalance('2026-01-05', 10., .3))
        self.assertEqual(right.cash, 1000000.)
        self.assertEqual(right.shares, 0.)
        self.assertIsNot(left.ledger, right.ledger)
        self.assertEqual(left.ledger, [])  # Opt-in logging, counters always active.
        self.assertEqual(left.build_count, 1)
        self.assertFalse(left.risk_blocked(0))

    def test_dynamic_build_add_reduce_exit(self):
        acc = account(1000000.)
        self.assertFalse(acc.rebalance('2026-01-05', 10., 0.))
        self.assertEqual(acc.unfilled_count, 0)
        for day, weight, event in (('2026-01-05', .3, 'build'),
                                   ('2026-01-06', .8, 'add'),
                                   ('2026-01-07', .2, 'reduce'),
                                   ('2026-01-08', 0., 'exit')):
            with self.subTest(event=event):
                before_nav = acc.nav(10.)
                target = math.floor(before_nav * weight / 10. / 100) * 100
                self.assertTrue(acc.rebalance(day, 10., weight, reason='daily target'))
                self.assertEqual(acc.shares, target)
                self.assertEqual(fills(acc)[-1]['event'], event)
                self.assertAlmostEqual(acc.nav(10.), before_nav - fills(acc)[-1]['fee'])
                self.assertGreaterEqual(acc.cash, 0.)
        result = acc.summary(10.)
        self.assertEqual([result[f'{event}_count'] for event in ('build', 'add', 'reduce', 'exit')],
                         [1, 1, 1, 1])
        self.assertAlmostEqual(result['cash'], 1000000. - result['fees'])
        self.assertEqual(result['marked_nav'], result['cash'])
        self.assertEqual(result['pnl'], result['cash'] - 1000000.)

    def test_cash_reused_same_day_over_multiple_rounds_and_periods(self):
        acc = account()
        start = date(2026, 1, 5)
        self.assertTrue(acc.rebalance(start, 10., .8))
        for i in range(1, 5):
            day = start + timedelta(days=i)
            self.assertTrue(acc.rebalance(day, 10., 0.))
            self.assertEqual(acc.shares, 0.)
            proceeds = acc.cash
            self.assertTrue(acc.rebalance(day, 10., .8))
            self.assertLess(acc.cash, proceeds)
            self.assertGreater(acc.shares, 500.)  # Not restricted to pre-sale cash.
            self.assertEqual(acc.sellable_shares, 0.)
            self.assertFalse(acc.observe_close(day, i, 10.))
        # Same object, capital and session indices continue into another period.
        self.assertTrue(acc.rebalance('2026-02-02', 10., 0.))
        self.assertFalse(acc.observe_close('2026-02-02', 20, 10.))
        self.assertEqual(acc.build_count, 5)
        self.assertEqual(acc.exit_count, 5)
        self.assertAlmostEqual(acc.cash, 10000. - acc.fees)

    def test_signal_day_and_execution_day_are_distinct_and_json_safe(self):
        acc = account()
        self.assertTrue(acc.rebalance(date(2026, 1, 6), 10., .5,
                                      signal_day='20260105', reason='prior close buy'))
        self.assertTrue(acc.rebalance(datetime(2026, 1, 8, 9, 30), 11., 0.,
                                      signal_day=date(2026, 1, 7), reason='prior close exit'))
        buy, sell = fills(acc)
        self.assertEqual((buy['signal_day'], buy['day']), ('2026-01-05', '2026-01-06'))
        self.assertEqual((sell['signal_day'], sell['day']), ('2026-01-07', '2026-01-08'))
        self.assertEqual(sell['reason'], 'prior close exit')
        json.dumps(acc.ledger, allow_nan=False)
        json.dumps(acc.summary(11.), allow_nan=False)

    def test_split_10_to_5_does_not_lose_equity_or_trigger_risk(self):
        acc = account()
        acc.rebalance('2026-01-05', 10., .8)
        acc.observe_close('2026-01-05', 0, 10.)
        old_shares, before = acc.shares, acc.nav(10.)
        acc.apply_action('2026-01-06', 0., 2.)
        self.assertEqual(acc.shares, old_shares * 2)
        self.assertEqual(acc.sellable_shares, acc.shares)
        self.assertAlmostEqual(acc.nav(5.), before)
        self.assertEqual(acc.dividends, 0.)
        self.assertFalse(acc.observe_close('2026-01-06', 1, 5.))

    def test_dividend_and_split_use_pre_event_shares_no_double_count(self):
        acc = account()
        acc.rebalance('2026-01-05', 10., .5)
        held, cash, before = acc.shares, acc.cash, acc.nav(10.)
        acc.apply_action('2026-01-06', 2., 2.)
        self.assertEqual(acc.dividends, held * 2.)
        self.assertEqual(acc.cash, cash + held * 2.)
        self.assertEqual(acc.shares, held * 2.)
        self.assertAlmostEqual(acc.nav(4.), before)  # (10 - 2) / 2, not 5.
        row = acc.ledger[-1]
        self.assertEqual(row['event'], 'action')
        self.assertEqual(row['eligible_shares'], held)
        self.assertEqual(row['gross'], held * 2.)
        self.assertIn('approximation', row['reason'])
        self.assertTrue(acc.rebalance('2026-01-06', 4., 0.))
        result = acc.summary(4.)
        self.assertAlmostEqual(result['cash'], 10000. - acc.fees)
        self.assertEqual(result['marked_nav'], result['cash'])
        self.assertAlmostEqual(result['pnl'], -acc.fees)
        self.assertEqual(result['cash_dividends'], held * 2.)

    def test_ex_date_buyer_gets_no_prior_holder_entitlements(self):
        acc = account()
        acc.apply_action('2026-01-06', 2., 2.)
        self.assertTrue(acc.rebalance('2026-01-06', 4., .5))
        self.assertEqual(acc.dividends, 0.)
        self.assertEqual(acc.ledger[0]['eligible_shares'], 0.)
        self.assertEqual(acc.new_shares, acc.shares)
        before = (acc.cash, acc.shares, acc.dividends, len(acc.ledger))
        with self.assertRaises(ValueError):
            acc.apply_action('2026-01-06', 2., 2.)
        self.assertEqual((acc.cash, acc.shares, acc.dividends, len(acc.ledger)), before)
        # Duplicate actions before any order must not credit twice either.
        acc.apply_action('2026-01-07', 1.)
        with self.assertRaises(ValueError):
            acc.apply_action('2026-01-07', 1.)

    def test_old_holder_adds_after_action_gets_cash_only_on_old_shares(self):
        acc = account()
        acc.rebalance('2026-01-05', 10., .3)
        held = acc.shares
        acc.apply_action('2026-01-06', 2.)
        self.assertTrue(acc.rebalance('2026-01-06', 8., .8))
        self.assertEqual(acc.dividends, held * 2.)
        self.assertEqual(acc.sellable_shares, held)
        self.assertEqual(acc.new_shares, acc.shares - held)

    def test_t_plus_one_sells_old_not_new_and_final_exit_leaves_residual(self):
        acc = account()
        acc.rebalance('2026-01-05', 10., .3)
        old = acc.shares
        acc.rebalance('2026-01-06', 10., .8)
        new = acc.new_shares
        self.assertEqual(acc.sellable_shares, old)
        self.assertTrue(acc.rebalance('2026-01-06', 10., 0., reason='scheduled end close'))
        self.assertEqual(acc.shares, new)
        self.assertEqual(acc.sellable_shares, 0.)
        self.assertEqual(acc.exit_count, 0)
        self.assertEqual(acc.reduce_count, 1)
        self.assertEqual(acc.ledger[-1]['block_reason'], 't_plus_one')
        self.assertEqual(acc.ledger[-1]['quantity'], new)
        cash, fees = acc.cash, acc.fees
        self.assertFalse(acc.rebalance('2026-01-06', 10., 0.))
        self.assertEqual((acc.cash, acc.fees), (cash, fees))
        self.assertGreater(acc.summary(10.)['marked_nav'], cash)
        self.assertEqual(acc.bought_day, '2026-01-06')
        self.assertTrue(acc.rebalance('2026-01-07', 10., 0.))
        self.assertEqual((acc.shares, acc.new_shares), (0., 0.))
        self.assertEqual(acc.exit_count, 1)

    def test_integer_odd_lot_reduction_and_complete_exit(self):
        acc = account(2000.)
        acc.rebalance('2026-01-05', 10., .5)
        acc.apply_action('2026-01-06', 0., 1.03)
        self.assertAlmostEqual(acc.shares, 103.)
        self.assertTrue(acc.rebalance('2026-01-06', 10., .1))
        self.assertEqual(fills(acc)[-1]['quantity'], 100.)
        self.assertAlmostEqual(acc.shares, 3.)
        before = (acc.cash, acc.shares, acc.fees)
        self.assertFalse(acc.rebalance('2026-01-06', 10., .1))
        self.assertEqual((acc.cash, acc.shares, acc.fees), before)
        self.assertEqual(acc.ledger[-1]['block_reason'], 'below_lot')
        self.assertTrue(acc.rebalance('2026-01-06', 10., 0.))
        self.assertEqual(acc.shares, 0.)
        self.assertAlmostEqual(fills(acc)[-1]['quantity'], 3.)

    def test_fractional_entitlement_retained_and_sale_flagged(self):
        acc = account(2000.)
        acc.rebalance('2026-01-05', 10., .5)
        before = acc.nav(10.)
        acc.apply_action('2026-01-06', 0., 1.005)
        self.assertAlmostEqual(acc.shares, 100.5)
        self.assertAlmostEqual(acc.nav(10. / 1.005), before)
        self.assertTrue(acc.ledger[-1]['fractional_entitlement'])
        self.assertTrue(acc.rebalance('2026-01-06', 10. / 1.005, 0.))
        self.assertEqual(acc.shares, 0.)
        self.assertTrue(fills(acc)[-1]['fractional_sale_approximation'])
        self.assertTrue(acc.summary(10.)['fractional_share_approximation'])

    def test_slippage_minimum_commission_and_separate_taxes(self):
        acc = account(2000., slippage_bps=5.)
        acc.rebalance('2026-01-05', 10., .5)
        buy = fills(acc)[-1]
        self.assertEqual(buy['quantity'], 100.)
        self.assertAlmostEqual(buy['price'], 10.005)
        self.assertEqual(buy['commission'], 5.)
        self.assertEqual(buy['stamp_tax'], 0.)
        self.assertAlmostEqual(buy['transfer_fee'], buy['gross'] * .00001)
        self.assertAlmostEqual(acc.cash, 2000. - buy['gross'] - buy['fee'])
        acc.rebalance('2026-01-06', 10., 0.)
        sell = fills(acc)[-1]
        self.assertAlmostEqual(sell['price'], 9.995)
        self.assertEqual(sell['commission'], 5.)
        self.assertAlmostEqual(sell['stamp_tax'], sell['gross'] * .0005)
        for row in (buy, sell):
            self.assertAlmostEqual(row['fee'], row['commission'] + row['stamp_tax'] + row['transfer_fee'])
        self.assertAlmostEqual(acc.fees, buy['fee'] + sell['fee'])
        self.assertAlmostEqual(acc.cash, 2000. - acc.fees - 1.)  # Round-trip slippage.

    def test_affordability_includes_minimum_fee_slippage_and_transfer(self):
        for capital, shares in ((1000., 0.), (1005.5, 0.), (1006., 100.),
                                (2006., 100.), (2006.1, 200.)):
            with self.subTest(capital=capital):
                acc = account(capital, slippage_bps=5.)
                self.assertEqual(acc.rebalance('2026-01-05', 10., 1.), shares > 0)
                self.assertEqual(acc.shares, shares)
                self.assertGreaterEqual(acc.cash, 0.)
                self.assertAlmostEqual(acc.cash + shares * 10.005 + acc.fees, capital)
                if shares == 0:
                    self.assertEqual(acc.fees, 0.)
                    self.assertEqual(acc.cash, capital)

    def test_tiny_sale_cannot_make_cash_negative_to_pay_minimum_fee(self):
        acc = account(1006.)
        self.assertTrue(acc.rebalance('2026-01-05', 10., 1.))
        before = (acc.cash, acc.shares, acc.fees)
        self.assertFalse(acc.rebalance('2026-01-06', .001, 0.))
        self.assertEqual((acc.cash, acc.shares, acc.fees), before)
        self.assertEqual(acc.ledger[-1]['block_reason'], 'insufficient_cash_for_fees')
        self.assertEqual(acc.exit_count, 0)
        self.assertGreater(acc.summary(.001)['marked_nav'], acc.cash)

    def test_proportional_commission_and_historical_rate_boundaries(self):
        for code in ('600000', '000001'):
            for day, transfer, stamp in (('2022-04-28', .00002, .001),
                                         ('2022-04-29', .00001, .001),
                                         ('2023-08-27', .00001, .001),
                                         ('2023-08-28', .00001, .0005)):
                with self.subTest(code=code, day=day):
                    acc = Account(code, AccountConfig(slippage_bps=0.), record_ledger=True)
                    buy_day = date.fromisoformat(day) - timedelta(days=1)
                    acc.rebalance(buy_day, 10., .3)
                    acc.rebalance(day, 10., 0.)
                    buy, sell = fills(acc)
                    self.assertAlmostEqual(buy['commission'], buy['gross'] * .0003)
                    buy_transfer = .00001 if buy_day >= date(2022, 4, 29) else .00002
                    self.assertAlmostEqual(buy['transfer_fee'], buy['gross'] * buy_transfer)
                    self.assertEqual(buy['stamp_tax'], 0.)
                    self.assertAlmostEqual(sell['transfer_fee'], sell['gross'] * transfer)
                    self.assertAlmostEqual(sell['stamp_tax'], sell['gross'] * stamp)

    def test_price_vector_sizes_from_current_equity_not_initial_cash(self):
        acc = account(100000., commission_rate=0., minimum_commission=0.)
        for i, price in enumerate((10., 20., 5., 12.)):
            day = date(2026, 1, 5) + timedelta(days=i)
            before = acc.nav(price)
            target = math.floor(before * .5 / price / 100) * 100
            self.assertTrue(acc.rebalance(day, price, .5))
            self.assertEqual(acc.shares, target)
            self.assertAlmostEqual(acc.nav(price), before - fills(acc)[-1]['fee'])
        self.assertEqual([r['side'] for r in fills(acc)], ['buy', 'sell', 'buy', 'sell'])

    def test_blocked_orders_no_cash_changes_and_explicit_retry_uses_new_price(self):
        acc = account()
        for options, why in ((dict(tradable=False), 'not_tradable'),
                             (dict(block_buy=True), 'block_buy')):
            self.assertFalse(acc.rebalance('2026-01-05', 10., .8,
                                           signal_day='2026-01-02', reason='pending', **options))
            self.assertEqual((acc.cash, acc.shares, acc.fees), (10000., 0., 0.))
            self.assertEqual(acc.ledger[-1]['block_reason'], why)
            self.assertEqual(acc.ledger[-1]['reason'], 'pending')
        self.assertTrue(acc.rebalance('2026-01-06', 20., .8, block_sell=True,
                                      signal_day='2026-01-02'))
        self.assertEqual(acc.shares, 400.)
        before = (acc.cash, acc.shares, acc.fees)
        self.assertFalse(acc.rebalance('2026-01-07', 20., 0., block_sell=True))
        self.assertFalse(acc.rebalance('2026-01-08', None, 0., tradable=False))
        self.assertEqual((acc.cash, acc.shares, acc.fees), before)
        self.assertIsNone(acc.ledger[-1]['price'])
        self.assertGreater(acc.summary(20.)['marked_nav'], acc.cash)
        self.assertEqual(acc.exit_count, 0)
        self.assertTrue(acc.rebalance('2026-01-09', 20., 0., block_buy=True))
        self.assertEqual(acc.unfilled_count, 4)

    def test_close_drawdown_latches_exit_without_extending_cooldown(self):
        acc = account(100000., commission_rate=0., minimum_commission=0., cooldown_sessions=3)
        acc.rebalance('2026-01-05', 10., .8)
        self.assertFalse(acc.observe_close('2026-01-05', 0, 10.))
        self.assertTrue(acc.observe_close('2026-01-06', 1, 8.))
        self.assertEqual(acc.cooldown_until_index, 4)
        self.assertEqual(acc.peak_nav, acc.nav(8.))
        self.assertTrue(acc.risk_blocked(1))
        self.assertTrue(acc.risk_blocked(4))
        self.assertFalse(acc.risk_blocked(5))
        for index, price in ((2, 7.), (3, 6.), (4, 6.), (5, 10.)):
            day = date(2026, 1, 5) + timedelta(days=index)
            self.assertFalse(acc.rebalance(day, price, 0., block_sell=True))
            self.assertTrue(acc.observe_close(day, index, price))
            self.assertEqual(acc.cooldown_until_index, 4)
        self.assertAlmostEqual(acc.max_drawdown_observed, (100000. - acc.nav(6.)) / 100000.)
        self.assertGreater(acc.shares, 0.)  # observe_close never pretends to execute.
        self.assertTrue(acc.rebalance('2026-01-12', 10., 0.))
        self.assertFalse(acc.observe_close('2026-01-12', 6, 10.))
        self.assertEqual(acc.cooldown_until_index, 4)
        # A subsequent position can trigger a NEW episode below the lifetime high.
        self.assertTrue(acc.rebalance('2026-01-13', 10., .8))
        self.assertFalse(acc.observe_close('2026-01-13', 7, 10.))
        self.assertTrue(acc.observe_close('2026-01-14', 7 + 1, 7.))
        self.assertEqual(acc.cooldown_until_index, 11)

    def test_exact_drawdown_threshold_and_zero_session_cooldown(self):
        acc = account(100001., commission_rate=0., minimum_commission=0., cooldown_sessions=0)
        acc.rebalance('2026-01-05', 10., 1.)
        self.assertEqual(acc.cash, 0.)  # 100000 gross + exactly 1 transfer fee.
        self.assertEqual(acc.shares, 10000.)
        # Exactly representable prices/NAV: 125000 -> 106250 is a 15% drawdown.
        self.assertFalse(acc.observe_close('2026-01-05', 0, 12.5))
        self.assertTrue(acc.observe_close('2026-01-06', 1, 10.625))
        self.assertTrue(acc.risk_blocked(1))
        self.assertFalse(acc.risk_blocked(2))
        self.assertTrue(acc.observe_close('2026-01-06', 1, 10.625))
        self.assertEqual(acc.cooldown_until_index, 1)
        self.assertAlmostEqual(acc.max_drawdown_observed, .15)

    def test_ledger_has_required_fields_and_summary_is_read_only(self):
        acc = account()
        acc.apply_action('2026-01-05', 0.)
        acc.rebalance('2026-01-05', 10., .3)
        acc.rebalance('2026-01-05', 10., 0.)
        required = {'day', 'signal_day', 'event', 'reason', 'quantity', 'price',
                    'gross', 'fee', 'cash', 'shares', 'target_weight'}
        for row in acc.ledger:
            self.assertTrue(required <= row.keys())
        before = (acc.cash, acc.shares, acc.fees, len(acc.ledger))
        first, second = acc.summary(10.), acc.summary(12.)
        self.assertEqual((acc.cash, acc.shares, acc.fees, len(acc.ledger)), before)
        self.assertAlmostEqual(second['marked_nav'] - first['marked_nav'], acc.shares * 2.)
        self.assertEqual(first['cash_dividends'], 0.)
        json.dumps(dict(summary=second, ledger=acc.ledger), allow_nan=False)


class TestValidation(unittest.TestCase):
    def test_invalid_config(self):
        for name in ('initial_capital', 'commission_rate', 'minimum_commission',
                     'slippage_bps', 'max_drawdown'):
            for value in (float('nan'), float('inf'), -float('inf'), -1., None, True):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    AccountConfig(**{name: value})
        for name, value in (('initial_capital', 0.), ('max_drawdown', 0.),
                            ('max_drawdown', 1.01), ('slippage_bps', 10000.),
                            ('cooldown_sessions', -1), ('cooldown_sessions', 1.5),
                            ('cooldown_sessions', True), ('cooldown_sessions', float('inf'))):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                AccountConfig(**{name: value})
        AccountConfig(commission_rate=0., minimum_commission=0., slippage_bps=0.,
                      max_drawdown=1., cooldown_sessions=0)

    def test_invalid_inputs_do_not_change_cash_shares_or_fees(self):
        acc = account()
        for value in (0., -1., float('nan'), float('inf'), None, True):
            with self.subTest(price=value):
                for operation in (lambda: acc.nav(value), lambda: acc.summary(value),
                                  lambda: acc.rebalance('2026-01-05', value, .5),
                                  lambda: acc.observe_close('2026-01-05', 0, value)):
                    with self.assertRaises(ValueError):
                        operation()
        for weight in (-.01, 1.01, float('nan'), float('inf'), None, True):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                acc.rebalance('2026-01-05', 10., weight)
        for cash, multiplier in ((-1., 1.), (float('nan'), 1.), (float('inf'), 1.),
                                  (0., 0.), (0., -1.), (0., float('nan')), (0., float('inf'))):
            with self.subTest(cash=cash, multiplier=multiplier), self.assertRaises(ValueError):
                acc.apply_action('2026-01-05', cash, multiplier)
        self.assertEqual((acc.cash, acc.shares, acc.fees, acc.dividends), (10000., 0., 0., 0.))
        self.assertEqual(acc.ledger, [])

    def test_dates_indices_and_preopen_order(self):
        acc = account()
        for day in ('2026-02-30', 'nonsense', None, 20260105):
            with self.subTest(day=day), self.assertRaises(ValueError):
                acc.rebalance(day, 10., .5)
        with self.assertRaises(ValueError):
            acc.rebalance('2026-01-05', 10., .5, signal_day='2026-01-06')
        acc.rebalance('2026-01-06', 10., .5)
        with self.assertRaises(ValueError):
            acc.rebalance('2026-01-05', 10., 0.)
        with self.assertRaises(ValueError):
            acc.apply_action('2026-01-06', 1.)
        with self.assertRaises(ValueError):
            acc.apply_action('2026-01-05', 1.)
        acc.observe_close('2026-01-06', 10, 10.)
        with self.assertRaises(ValueError):
            acc.observe_close('2026-01-07', 0, 10.)
        for index in (-1, 1.5, True, float('nan'), float('inf')):
            with self.subTest(index=index), self.assertRaises(ValueError):
                acc.risk_blocked(index)
        with self.assertRaises(ValueError):
            Account('')
        with self.assertRaises(ValueError):
            Account('600000', config=None)


if __name__ == '__main__':
    unittest.main()