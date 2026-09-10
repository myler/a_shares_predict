"""Independent daily atoms, unweighted Pareto fronts and risk-based exposure.

All nondominated stocks remain in the first front, without Top-N truncation or
weighted aggregation of the 34 objectives. Each stock has an independent account:
weights are fractions of its own NAV, never normalized across stocks or funded
from another stock's cash. Callers select eligible rows before front extraction.

Mathematical policy (long-only, no leverage): for first-front stocks passing the
gate, desired exposure is min(max_weight, annual_vol_target / annual_vol,
es_budget / daily_es), with risk denominators floored at 1e-8. Historical daily ES
is the mean of the largest ceil(window * (1-confidence)) nonnegative daily losses;
annual volatility uses sample standard deviation of simple returns, times sqrt252.
Estimates include today's close, so they may only drive subsequent execution.

On an account's rebalance day, minimize over 0 <= w <= max_weight:
    0.5 * tracking_penalty * annual_vol**2 * (w-desired)**2
    + buy_cost * (w-current)_+ + sell_cost * (current-w)_+.
The scalar solution is asymmetric soft thresholding around current exposure,
then projection onto the bounds. Variance is floored at 1e-8 for the degenerate
near-zero-volatility case. Zero desired exposure / force_exit are hard constraints
and override costs. This tracking objective concerns only exposure risk and fees;
it does NOT scalarize or weight the 34 signal objectives.

References: the convex risk/trading-cost framework is described at
https://www.cvxportfolio.com/en/stable/optimization_policies.html . Moreira & Muir
(2017), Volatility-Managed Portfolios, https://doi.org/10.1111/jofi.12513 , motivates
volatility-managed exposure, not this exact formula or any claimed performance.

Pure NumPy computation: callers provide consistently preprocessed OHLCV (including
real opens), and implement execution, minimum commissions, lots, cash/shares and
corporate actions in their ledger. Cost inputs here are proportional rates.
"""

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from engine import (
    _INDICATOR_NAMES, _recent_price_divergence, calc_auxiliary_votes_series,
    calc_bollinger, calc_kdj, calc_macd, calc_obv, calc_obv_flow, calc_rsi,
    calc_wr,
)


STRATEGY_VERSION = 'pareto_daily_atoms_risk_v1'
OBJECTIVE_NAMES = (
    'M_DIF_DEA', 'M_DIF_ZERO', 'M_DIF_SLOPE_5', 'M_BAR_TREND_3',
    'M_LOW_REPAIR', 'M_TOP_WEAK',
    'F_RSI', 'F_K_D', 'F_J_EXTREME', 'F_BB_POSITION', 'F_WR_EXTREME',
    'P_POSITION_250', 'P_RETURN_250',
    'Q_OBV_FLOW', 'Q_PRICE_VOLUME_DIVERGENCE', 'Q_INSTITUTION_PROXY',
) + tuple('A_' + name for name in _INDICATOR_NAMES)

# The packed dominance graph costs U * ceil(U / 8) bytes (about 1.3 MiB at
# U=3300), not U*U*34 booleans. Bound temporary pairwise uint64 arrays as well.
_PAIRWISE_BLOCK_ROWS = 256
_MAX_PAIRWISE_CELLS = 262144


def _real_array(values, name, ndim):
    try:
        result = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a real numeric {ndim}D array') from exc
    if result.ndim != ndim or result.dtype.kind not in 'iuf':
        raise ValueError(f'{name} must be a real numeric {ndim}D array')
    if not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must contain only finite values')
    return result


def make_objectives(opens, highs, lows, closes, volumes):
    """Return votes (n,34) int8, eligible/gate_pass (n,) bool and two float64 arrays.

    Column order is OBJECTIVE_NAMES; +1 always means bullish. Core votes retain
    each original rule's sign/if priority, excluding its magnitude and baseline.
    Warm-up core votes are diagnostic only: eligible requires index >=250, a
    positive current volume and finite RSI. Auxiliary rows 0..58 are all zero.
    gate_pass is ONLY (raw RSI <=92) & (OBV flow >=-0.6), independent of eligible.
    Empty histories are allowed; invalid shape, OHLC or volume raises ValueError.
    """
    names = ('opens', 'highs', 'lows', 'closes', 'volumes')
    arrays = tuple(_real_array(values, name, 1).astype(np.float64, copy=False)
                   for name, values in zip(names, (opens, highs, lows, closes, volumes)))
    opens, highs, lows, closes, volumes = arrays
    n = len(closes)
    if any(len(values) != n for values in arrays):
        raise ValueError('OHLCV arrays must have equal lengths')
    if (any(np.any(values <= 0) for values in arrays[:4])
            or np.any(highs < np.maximum(opens, closes))
            or np.any(lows > np.minimum(opens, closes))
            or np.any(highs < lows) or np.any(volumes < 0)):
        raise ValueError('OHLC must be positive and low <= open/close <= high; volume >= 0')

    votes = np.zeros((n, len(OBJECTIVE_NAMES)), dtype=np.int8)
    rsi = calc_rsi(closes)
    obv_flow = np.zeros(n, dtype=np.float64)
    dif, dea, bar = calc_macd(closes)
    k, d, j = calc_kdj(highs, lows, closes)
    bb_u, _, bb_l = calc_bollinger(closes)
    wr = calc_wr(highs, lows, closes)
    obv = calc_obv(closes, volumes)
    returns = np.diff(closes) / closes[:-1]
    votes[:, 16:] = calc_auxiliary_votes_series(closes, highs, lows, volumes, opens)

    for i in range(n):
        row = votes[i]
        row[0] = 1 if dif[i] > dea[i] else -1
        row[1] = 1 if dif[i] > 0 else -1
        row[2] = 1 if i >= 5 and dif[i] > dif[i - 5] else -1
        row[3] = int(i >= 3 and bar[i] > bar[i - 3])
        bottom, top = _recent_price_divergence(closes, dif, i)
        row[4], row[5] = int(bottom), -int(top)

        rv = rsi[i] if not np.isnan(rsi[i]) else 50
        kv = k[i] if not np.isnan(k[i]) else 50
        dv = d[i] if not np.isnan(d[i]) else 50
        jv = j[i] if not np.isnan(j[i]) else 50
        wv = wr[i] if not np.isnan(wr[i]) else 50
        bb_pos = ((closes[i] - bb_l[i]) / (bb_u[i] - bb_l[i]) * 100
                  if not np.isnan(bb_u[i]) and bb_u[i] != bb_l[i] else 50)
        row[6] = 1 if rv <= 65 else (-1 if rv > 70 else 0)
        row[7] = 1 if kv > dv else (-1 if kv < dv else 0)
        row[8] = 1 if jv < 0 else (-1 if jv > 100 else 0)
        row[9] = 1 if bb_pos < 10 else (-1 if bb_pos > 90 else 0)
        row[10] = 1 if wv > 80 else (-1 if wv < 20 else 0)

        if i >= 249:
            window = closes[i - 249:i + 1]
            hi, lo = np.max(window), np.min(window)
            pos = (closes[i] - lo) / (hi - lo) * 100 if hi != lo else 50
            row[11] = 1 if pos < 40 else (-1 if pos > 80 else 0)
        if i >= 250:
            change = (closes[i] - closes[i - 250]) / closes[i - 250] * 100
            row[12] = 1 if change > 20 else (-1 if change < -20 else 0)

        flow = calc_obv_flow(obv, volumes, i)
        obv_flow[i] = flow
        row[13] = 1 if flow > 0.4 else (-1 if flow < -0.4 else 0)
        change5 = closes[i] - closes[max(0, i - 5)]
        if change5 > 0 and flow < -0.2:
            row[14] = -1
        elif change5 < 0 and flow > 0.2:
            row[14] = 1
        if i >= 60:
            ret5 = change5 / closes[i - 5] * 100
            ret20 = (closes[i] - closes[i - 20]) / closes[i - 20] * 100
            # Preserve the original inclusive j=i-60..i window (61 returns
            # after warm-up), rather than silently changing the proxy rule.
            volatility = np.std(returns[max(0, i - 61):i]) * np.sqrt(252) * 100
            if ret5 < -3 and volatility < 25:
                row[15] = 1
            elif ret20 < -10 and volatility < 25:
                row[15] = 1
            elif ret5 < -3 and volatility > 40:
                row[15] = -1

    return {
        'votes': votes,
        'eligible': (np.arange(n) >= 250) & (volumes > 0) & np.isfinite(rsi),
        'gate_pass': (rsi <= 92) & (obv_flow >= -0.6),
        'rsi': rsi,
        'obv_flow': obv_flow,
    }


def daily_front_layers(votes, max_layers=4):
    """Exact maximization fronts for a day's (stocks, objectives) ternary matrix.

    Returns int32 labels 1..max_layers, with 5 for everything not peeled.
    max_layers must be 1..4. Supports 1..64 dimensions (production uses 34).
    Equal vectors share a layer; strict dominance requires >= in every column
    and > in at least one. No tie-break, weights, crowding or Top-N truncation.
    Select eligible rows before calling; gates may independently zero targets.
    """
    values = _real_array(votes, 'votes', 2)
    if not 1 <= values.shape[1] <= 64:
        raise ValueError('votes must have 1..64 objectives')
    if not np.all((values == -1) | (values == 0) | (values == 1)):
        raise ValueError('votes must be ternary (-1, 0, 1)')
    if (isinstance(max_layers, (bool, np.bool_))
            or not isinstance(max_layers, (int, np.integer))
            or not 1 <= max_layers <= 4):
        raise ValueError('max_layers must be an integer in 1..4')
    if not len(values):
        return np.empty(0, dtype=np.int32)

    # a >= b iff both sets of b (nonnegative and positive) are subsets of a.
    bits = np.left_shift(np.uint64(1), np.arange(values.shape[1], dtype=np.uint64))
    masks = np.column_stack(((values >= 0).astype(np.uint64) @ bits,
                             (values > 0).astype(np.uint64) @ bits))
    unique, inverse = np.unique(masks, axis=0, return_inverse=True)
    nonnegative, positive = unique.T
    count = len(unique)
    rows = max(1, min(_PAIRWISE_BLOCK_ROWS, _MAX_PAIRWISE_CELLS // count))
    outgoing = np.empty((count, (count + 7) // 8), dtype=np.uint8)
    incoming = np.zeros(count, dtype=np.int64)
    for start in range(0, count, rows):
        stop = min(start + rows, count)
        missing = ((~nonnegative[start:stop, None] & nonnegative[None, :])
                   | (~positive[start:stop, None] & positive[None, :]))
        dominates = missing == 0
        # Uniqueness leaves only the diagonal equal, never a strict edge.
        dominates[np.arange(stop - start), np.arange(start, stop)] = False
        outgoing[start:stop] = np.packbits(dominates, axis=1)
        incoming += dominates.sum(axis=0)

    layers = np.full(count, 5, dtype=np.int32)
    for layer in range(1, max_layers + 1):
        front = np.flatnonzero((incoming == 0) & (layers == 5))
        if not len(front):
            break
        layers[front] = layer
        if layer == max_layers or np.all(layers != 5):
            break
        # Remove this entire front simultaneously, including all duplicates
        # through inverse at the end. Unpack only bounded row blocks.
        for start in range(0, len(front), rows):
            edges = np.unpackbits(outgoing[front[start:start + rows]],
                                  axis=1, count=count)
            incoming -= edges.sum(axis=0, dtype=np.int64)
    return layers[inverse].astype(np.int32, copy=False)


def _risk_scalar(value, name, allow_nonfinite=False):
    """Accept real numeric scalars, never boolean/string/vector coercions."""
    try:
        array = np.asarray(value)
        if array.ndim != 0 or array.dtype.kind not in 'iuf':
            raise ValueError(f'{name} must be a real numeric scalar')
        result = float(array)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a real numeric scalar') from exc
    if not allow_nonfinite and not np.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result


def _risk_window(window):
    if (isinstance(window, (bool, np.bool_))
            or not isinstance(window, (int, np.integer)) or window < 2):
        raise ValueError('risk window must be an integer >= 2')
    return int(window)


def _risk_confidence(confidence):
    confidence = _risk_scalar(confidence, 'confidence')
    if not 0 < confidence < 1:
        raise ValueError('confidence must be in (0, 1)')
    return confidence


@dataclass(frozen=True)
class RiskConfig:
    """Risk budgets in NAV fractions; annual vol is not a percent-point input.

    annual_vol_target and tracking_penalty must be positive and finite;
    es_budget and max_weight are in (0, 1] (long-only, unlevered accounts).
    risk_window is an integer >=2, and es_confidence is strictly between 0 and 1.
    Defaults budget 15% annual volatility and 2% NAV daily historical 95% ES.
    These configurable policy choices are not optimized-return claims.
    """

    annual_vol_target: float = .15
    es_budget: float = .02
    risk_window: int = 63
    es_confidence: float = .95
    tracking_penalty: float = 5.0
    max_weight: float = 1.0

    def __post_init__(self):
        for name in ('annual_vol_target', 'es_budget', 'tracking_penalty', 'max_weight'):
            value = _risk_scalar(getattr(self, name), name)
            if value <= 0 or (name in ('es_budget', 'max_weight') and value > 1):
                raise ValueError(f'{name} must be positive, and NAV fractions <= 1')
            object.__setattr__(self, name, value)
        object.__setattr__(self, 'risk_window', _risk_window(self.risk_window))
        object.__setattr__(self, 'es_confidence', _risk_confidence(self.es_confidence))


def risk_statistics(closes, window=63, confidence=.95) -> dict:
    """Return float64 annual_vol/daily_es arrays aligned with closes.

    Row i uses exactly the window simple returns ending at i, requiring window+1
    closes. The first window rows are NaN (all rows for shorter histories).
    Volatility is std(ddof=1)*sqrt(252). ES includes zero losses for positive
    returns, rather than conditioning on negative-return days. Window views and
    partition compute the tails without per-row Python sorting or future data.
    Closes must be a finite, positive, real 1D array; an empty array is valid.
    """
    window = _risk_window(window)
    confidence = _risk_confidence(confidence)
    prices = _real_array(closes, 'closes', 1).astype(np.float64, copy=False)
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError('closes must be positive and finite in float64')
    annual_vol = np.full(len(prices), np.nan, dtype=np.float64)
    daily_es = np.full(len(prices), np.nan, dtype=np.float64)
    if len(prices) > window:
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            returns = np.diff(prices) / prices[:-1]
            samples = sliding_window_view(returns, window)
            annual_vol[window:] = samples.std(axis=1, ddof=1) * np.sqrt(252)
            tail_count = int(np.ceil(window * (1 - confidence)))
            losses = np.maximum(0., -samples)
            tail = np.partition(losses, window - tail_count, axis=1)[:, -tail_count:]
            daily_es[window:] = tail.mean(axis=1)
        if (not np.all(np.isfinite(annual_vol[window:]))
                or not np.all(np.isfinite(daily_es[window:]))):
            raise ValueError('closes produce nonfinite risk statistics')
    return {'annual_vol': annual_vol, 'daily_es': daily_es}


def risk_target(front_layer, gate_pass, annual_vol, daily_es, config=RiskConfig()) -> float:
    """Scalar desired exposure for one independent account on a rebalance day.

    Only first-front membership plus a passing gate and finite risk qualifies.
    Missing/nonfinite risk (including warm-up NaNs) fails closed to zero. Finite
    risk must be nonnegative; malformed scalar/config inputs raise ValueError.
    Other front labels, including ineligible sentinels, have zero exposure.
    """
    if not isinstance(config, RiskConfig):
        raise ValueError('config must be a RiskConfig')
    if (isinstance(front_layer, (bool, np.bool_))
            or not isinstance(front_layer, (int, np.integer))):
        raise ValueError('front_layer must be an integer scalar')
    if not isinstance(gate_pass, (bool, np.bool_)):
        raise ValueError('gate_pass must be a boolean scalar')
    vol = _risk_scalar(annual_vol, 'annual_vol', allow_nonfinite=True)
    es = _risk_scalar(daily_es, 'daily_es', allow_nonfinite=True)
    if (np.isfinite(vol) and vol < 0) or (np.isfinite(es) and es < 0):
        raise ValueError('finite annual_vol and daily_es must be nonnegative')
    if front_layer != 1 or not gate_pass or not np.isfinite(vol) or not np.isfinite(es):
        return 0.0
    return float(max(0., min(config.max_weight,
                            config.annual_vol_target / max(vol, 1e-8),
                            config.es_budget / max(es, 1e-8))))


def optimal_rebalance_weight(current_weight, desired_weight, annual_vol,
                             buy_cost, sell_cost, config=RiskConfig(), force_exit=False,
                             risk_cap=None) -> float:
    """Bounded scalar quadratic-tracking/proportional-cost minimizer.

    Unconstrained solution: current + sign(desired-current) * max(
        abs(desired-current) - side_cost/(lambda*max(vol**2, 1e-8)), 0).
    Then clip to [0, max_weight]. With zero cost it tracks desired exactly; costs
    introduce an asymmetric no-trade band. Current exposure may exceed the cap
    after price appreciation: retain the actual current in the objective, then
    clip the solution, not the input. Nonnegative desired values above the cap
    are also valid, but cannot create leveraged output.

    All numeric inputs must be finite, weights/volatility nonnegative and cost
    rates in [0, 1]. Validation also applies to forced exits. force_exit or zero
    desired imposes w=0 regardless of costs. Minimum commission, trade lots and
    executable cash/share changes remain ledger responsibilities.
    """
    if not isinstance(config, RiskConfig):
        raise ValueError('config must be a RiskConfig')
    if not isinstance(force_exit, (bool, np.bool_)):
        raise ValueError('force_exit must be a boolean scalar')
    current = _risk_scalar(current_weight, 'current_weight')
    desired = _risk_scalar(desired_weight, 'desired_weight')
    vol = _risk_scalar(annual_vol, 'annual_vol')
    buy = _risk_scalar(buy_cost, 'buy_cost')
    sell = _risk_scalar(sell_cost, 'sell_cost')
    if min(current, desired, vol) < 0:
        raise ValueError('weights and annual_vol must be nonnegative')
    if not 0 <= buy <= 1 or not 0 <= sell <= 1:
        raise ValueError('buy_cost and sell_cost must be rates in [0, 1]')
    cap = config.max_weight if risk_cap is None else _risk_scalar(risk_cap, 'risk_cap')
    if not 0 <= cap <= config.max_weight:
        raise ValueError('risk_cap must be in [0, max_weight]')
    if force_exit or desired == 0:
        return 0.0
    side_cost = buy if desired > current else sell
    # Divide in stages to avoid squaring an extreme but finite volatility.
    risk_vol = max(vol, 1e-4)
    threshold = (side_cost / risk_vol) / config.tracking_penalty / risk_vol
    # Algebraically identical to soft thresholding, without current-current
    # cancellation when an out-of-bounds holding is much larger than desired.
    target = (max(current, desired - threshold) if desired > current
              else min(current, desired + threshold))
    return float(np.clip(target, 0., cap))