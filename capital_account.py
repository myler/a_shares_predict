"""Independent, long-only single-stock cash/share account (no data access).

Runner contract: apply one combined corporate action BEFORE that day's orders;
execute causal signals at the supplied real open, then observe the real close.
Keep the same Account and increasing session indices across evaluation periods.
The runner owns the NAV curve, pending orders, trading calendar and buy blocks
during cooldown / before final liquidation. A scheduled final close is just an
explicit rebalance(..., 0): missing data, blocks and T+1 can leave residual stock.

Assumptions, NOT a fully historical exchange/tax model:
* Sell stamp duty: 0.1% before 2023-08-28, 0.05% from that date.
* Both-side SH/SZ transfer fee: 0.001% from 2022-04-29; a UNIFIED 0.002%
  assumption earlier (not a claim about all older exchange-specific schedules).
* Supplied cash dividends are paid on ex-date, with no extra withholding model.
  Bonus shares are also available on ex-date: actual payment/listing dates are
  unknown. Fractional entitlements remain float shares; liquidation including
  fractions is flagged as approximate, NOT a priced cash-in-lieu entitlement.
* No volume participation cap: volume units are unknown. Prices are unadjusted;
  actions must not be combined with already-adjusted prices.

rebalance returns bool: True means SOME quantity filled, not necessarily that
the target/exit completed. Unfilled attempts (including partial residuals and
sub-lot differences) are counted; no-change requests are not. With record_ledger
enabled every fill, unfilled attempt and action is recorded. No implicit retry.
All money/returns are in currency units; drawdown values are fractions.
"""

from dataclasses import dataclass
from datetime import date, datetime
import math
import operator


def _number(value, name, *, positive=False):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be finite and numeric') from exc
    if isinstance(value, bool) or not math.isfinite(result):
        raise ValueError(f'{name} must be finite and numeric')
    if result < 0 or (positive and result == 0):
        raise ValueError(f'{name} must be {"positive" if positive else "nonnegative"}')
    return result


def _index(value, name='index'):
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f'{name} must be a nonnegative integer') from exc
    if isinstance(value, bool) or result < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(result)


def _day(value):
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        if len(value) == 8 and value.isdigit():
            value = f'{value[:4]}-{value[4:6]}-{value[6:]}'
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError:
            pass
    raise ValueError('day must be a date or YYYY-MM-DD / YYYYMMDD string')


@dataclass(frozen=True)
class AccountConfig:
    initial_capital: float = 1000000.
    commission_rate: float = .0003
    minimum_commission: float = 5.
    slippage_bps: float = 5.
    max_drawdown: float = .15
    cooldown_sessions: int = 20

    def __post_init__(self):
        for name in ('initial_capital', 'commission_rate', 'minimum_commission',
                     'slippage_bps', 'max_drawdown'):
            value = _number(getattr(self, name), name,
                            positive=name in ('initial_capital', 'max_drawdown'))
            object.__setattr__(self, name, value)
        if self.slippage_bps >= 10000 or self.max_drawdown > 1:
            raise ValueError('slippage_bps must be < 10000; max_drawdown must be <= 1')
        object.__setattr__(self, 'cooldown_sessions',
                           _index(self.cooldown_sessions, 'cooldown_sessions'))


class Account:
    def __init__(self, code, config=AccountConfig(), record_ledger=False):
        if not isinstance(code, str) or not code.strip():
            raise ValueError('code must be a nonempty string')
        if not isinstance(config, AccountConfig):
            raise ValueError('config must be AccountConfig')
        self.code = code.strip()
        self.config = config  # Frozen: the shared default cannot be mutated.
        self.record_ledger = bool(record_ledger)
        self.cash = config.initial_capital
        self.shares = 0.
        self.fees = 0.
        self.dividends = 0.
        self.bought_day = None
        self.new_shares = 0.
        self.peak_nav = config.initial_capital
        self.cooldown_until_index = -1
        self.build_count = self.add_count = self.reduce_count = self.exit_count = 0
        self.unfilled_count = 0
        self.max_drawdown_observed = 0.
        self.ledger = []
        self.fractional_share_approximation = False
        self._current_day = None
        self._market_day = None
        self._action_day = None
        self._last_close_index = -1
        self._all_time_peak_nav = config.initial_capital
        self._risk_latched = False

    @property
    def sellable_shares(self):
        """Available on the last processed date, refreshed by dated methods."""
        return max(0., self.shares - self.new_shares)

    def _advance_day(self, day):
        if self._current_day is not None and day < self._current_day:
            raise ValueError('account events must be chronological')
        if day != self._current_day:
            self.new_shares = 0.
            self._current_day = day

    def nav(self, price):
        price = _number(price, 'price', positive=True)
        return _number(self.cash + self.shares * price, 'nav')

    def _record(self, day, event, *, signal_day=None, reason='', quantity=0.,
                price=None, gross=0., fee=0., target_weight=None, **extra):
        if self.record_ledger:
            self.ledger.append(dict(
                day=day, signal_day=signal_day, event=event, reason=reason,
                quantity=float(quantity), price=price, gross=gross, fee=fee,
                cash=self.cash, shares=self.shares, target_weight=target_weight,
                **extra))

    def _unfilled(self, day, price, weight, quantity, reason, signal_day, block_reason):
        self.unfilled_count += 1
        self._record(day, 'unfilled', price=price, target_weight=weight,
                     quantity=quantity, reason=reason or block_reason,
                     signal_day=signal_day, block_reason=block_reason)
        return False

    def apply_action(self, day, cash_per_share, share_multiplier=1.0):
        """One combined pre-open event; dividend is per PRE-event held share.

        Duplicate or late calls raise instead of accidentally crediting today's
        buyers. No event price exists, so fractional shares cannot be cashed out.
        """
        day = _day(day)
        cash_per_share = _number(cash_per_share, 'cash_per_share')
        share_multiplier = _number(share_multiplier, 'share_multiplier', positive=True)
        if day == self._market_day or day == self._action_day:
            raise ValueError('apply one combined action before any orders or close')
        held = self.shares
        dividend = _number(held * cash_per_share, 'dividend')
        shares = _number(held * share_multiplier, 'action shares')
        cash = _number(self.cash + dividend, 'action cash')
        dividends = _number(self.dividends + dividend, 'cash dividends')
        self._advance_day(day)
        self.cash, self.shares, self.dividends = cash, shares, dividends
        self._action_day = day
        fractional = not shares.is_integer()
        self.fractional_share_approximation |= fractional
        self._record(day, 'action', reason='ex_date_cash_and_share_availability_approximation',
                     quantity=shares - held, gross=dividend,
                     cash_per_share=cash_per_share, share_multiplier=share_multiplier,
                     eligible_shares=held, fractional_entitlement=fractional)

    def _costs(self, day, quantity, price, buy):
        gross = _number(quantity * price, 'gross', positive=True)
        commission = max(self.config.minimum_commission,
                         gross * self.config.commission_rate)
        stamp_tax = 0. if buy else gross * (.0005 if day >= '2023-08-28' else .001)
        transfer_fee = gross * (.00001 if day >= '2022-04-29' else .00002)
        fee = _number(commission + stamp_tax + transfer_fee, 'fee')
        return dict(gross=gross, fee=fee, commission=commission,
                    stamp_tax=stamp_tax, transfer_fee=transfer_fee)

    def rebalance(self, day, price, target_weight, tradable=True, block_buy=False,
                  block_sell=False, reason='', signal_day=None):
        """Size from current equity at raw execution price, never future closes.

        Buys and nonzero-target reductions use 100-share lots. Zero target sells
        ALL available shares, including odd/fractional lots, but never today's
        purchases. Sale proceeds are immediately reusable. price=None is allowed
        only with tradable=False to record genuinely missing execution data.
        """
        day = _day(day)
        weight = _number(target_weight, 'target_weight')
        if weight > 1:
            raise ValueError('target_weight must be in [0, 1]')
        raw = None if price is None and not tradable else _number(price, 'price', positive=True)
        signal_day = None if signal_day is None else _day(signal_day)
        if signal_day is not None and signal_day > day:
            raise ValueError('signal_day must not be after execution day')
        self._advance_day(day)
        self._market_day = day
        if not tradable:
            return self._unfilled(day, raw, weight, self.shares if weight == 0 else 0.,
                                  reason, signal_day, 'not_tradable')

        target_shares = _number(self.nav(raw) * weight / raw, 'target shares')
        target = math.floor(target_shares / 100) * 100
        diff = target - self.shares
        if diff == 0:
            return False
        buy = diff > 0
        if (buy and block_buy) or (not buy and block_sell):
            return self._unfilled(day, raw, weight, abs(diff), reason, signal_day,
                                  'block_buy' if buy else 'block_sell')
        requested = abs(diff) if weight == 0 else math.floor(abs(diff) / 100) * 100
        if requested == 0:
            return self._unfilled(day, raw, weight, abs(diff), reason, signal_day, 'below_lot')
        execution = _number(raw * (1 + (1 if buy else -1) * self.config.slippage_bps / 10000),
                            'execution price', positive=True)
        if buy:
            # Monotone binary search includes the minimum commission and all
            # buy charges. No epsilon/clamping can make a too-expensive lot fit.
            low, high = 0, int(requested // 100)
            while low < high:
                middle = (low + high + 1) // 2
                costs = self._costs(day, middle * 100, execution, True)
                if costs['gross'] + costs['fee'] <= self.cash:
                    low = middle
                else:
                    high = middle - 1
            quantity = float(low * 100)
            limiting_reason = 'insufficient_cash'
        else:
            quantity = min(requested, self.sellable_shares)
            if weight != 0:
                quantity = math.floor(quantity / 100) * 100
            limiting_reason = 't_plus_one'
        if quantity == 0:
            return self._unfilled(day, raw, weight, requested, reason, signal_day, limiting_reason)

        costs = self._costs(day, quantity, execution, buy)
        cash_delta = -(costs['gross'] + costs['fee']) if buy else costs['gross'] - costs['fee']
        next_cash = self.cash + cash_delta
        if next_cash < 0:
            return self._unfilled(day, raw, weight, requested, reason, signal_day,
                                  'insufficient_cash_for_fees')
        next_cash = _number(next_cash, 'cash')
        next_fees = _number(self.fees + costs['fee'], 'fees')
        before = self.shares
        next_shares = _number(before + quantity if buy else before - quantity, 'shares')
        self.cash, self.fees = next_cash, next_fees
        self.shares = next_shares
        if buy:
            self.new_shares += quantity
            self.bought_day = day
            event = 'build' if before == 0 else 'add'
        else:
            # Preserve the exact locked lot after selling all available float
            # entitlements; subtraction must not create a phantom T+1 residual.
            if quantity == max(0., before - self.new_shares):
                self.shares = self.new_shares
            event = 'exit' if self.shares == 0 else 'reduce'
            if event == 'exit':
                self._risk_latched = False
                self.peak_nav = self.cash
        setattr(self, f'{event}_count', getattr(self, f'{event}_count') + 1)
        self._record(day, event, signal_day=signal_day, reason=reason, quantity=quantity,
                     price=execution, raw_price=raw, target_weight=weight,
                     side='buy' if buy else 'sell',
                     fractional_sale_approximation=not buy and not float(quantity).is_integer(),
                     **costs)
        if quantity < requested:
            self._unfilled(day, raw, weight, requested - quantity, reason, signal_day, limiting_reason)
        return True

    def observe_close(self, day, index, price):
        """Return True while a risk exit is required (runner must schedule it).

        A trigger blocks buys through index + cooldown_sessions INCLUSIVE,
        i.e. the trigger and the next N sessions. Pending exits keep returning
        True without moving the deadline, even if the stock stays blocked.
        Risk peak resets on trigger/exit; observed drawdown uses a separate
        lifetime close-equity high, including dividends and share entitlements.
        """
        day, index = _day(day), _index(index)
        equity = self.nav(price)
        if index < self._last_close_index:
            raise ValueError('session indices must not move backwards across periods')
        self._advance_day(day)
        self._market_day = day
        self._last_close_index = index
        self._all_time_peak_nav = max(self._all_time_peak_nav, equity)
        drawdown = (self._all_time_peak_nav - equity) / self._all_time_peak_nav
        self.max_drawdown_observed = max(self.max_drawdown_observed, drawdown)
        if self.shares == 0:
            self.peak_nav = equity
            self._risk_latched = False
            return False
        self.peak_nav = max(self.peak_nav, equity)
        if self._risk_latched:
            return True
        drawdown = (self.peak_nav - equity) / self.peak_nav
        if drawdown >= self.config.max_drawdown:
            self._risk_latched = True
            self.peak_nav = equity
            self.cooldown_until_index = index + self.config.cooldown_sessions
            return True
        return False

    def risk_blocked(self, index):
        return _index(index) <= self.cooldown_until_index

    def summary(self, price):
        """Mark residual stock, without liquidation or double-counting dividends."""
        marked_nav = self.nav(price)
        return dict(code=self.code, initial_capital=self.config.initial_capital,
                    cash=self.cash, shares=self.shares, marked_nav=marked_nav,
                    pnl=marked_nav - self.config.initial_capital, fees=self.fees,
                    cash_dividends=self.dividends, build_count=self.build_count,
                    add_count=self.add_count, reduce_count=self.reduce_count,
                    exit_count=self.exit_count, unfilled_count=self.unfilled_count,
                    max_drawdown_observed=self.max_drawdown_observed,
                    fractional_share_approximation=self.fractional_share_approximation)