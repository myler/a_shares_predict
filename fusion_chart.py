"""In-memory Pareto34 display; no quotes, database, orders or output files.

Only historical ledger fills are marked, never inferred from votes. This module
does not import plotting/pyplot, select a global backend, register fonts or change
rcParams. A module lock covers font lookup, Figure construction, drawing and
cleanup because Matplotlib's font/rendering internals are not thread-safe.
"""

import base64
from datetime import date
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import re
from threading import Lock

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np

from engine import calc_macd
from pareto_strategy import OBJECTIVE_NAMES


_RENDER_LOCK = Lock()
_MAX_HEATMAP_COLUMNS = 1000
_FILLS = ('build', 'add', 'reduce', 'exit')
_BLUE, _ORANGE = '#1565C0', '#FF6F00'
_RED, _GRAY, _GREEN = '#ef5350', '#d6d9de', '#26a69a'
_FONT_PATHS = (
    Path(__file__).with_name('wqy-zenhei.ttf'),
    Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'),
    Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
    Path('C:/Windows/Fonts/msyh.ttc'),
    Path('C:/Windows/Fonts/simhei.ttf'),
)


def _iso_day(value, field):
    """Reject timestamps, month precision, NaT and coercion/truncation to U10."""
    if not isinstance(value, str) or re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', value) is None:
        raise ValueError(f'{field} must be an ISO YYYY-MM-DD string')
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f'{field} must be a valid calendar day') from exc


def _numeric(values, name, shape):
    try:
        array = np.asarray(values)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{name} must be a real numeric array of shape {shape}') from exc
    if array.shape != shape or array.dtype.kind not in 'iuf':
        raise ValueError(f'{name} must be a real numeric array of shape {shape}')
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must contain only finite values')
    return array


def _validate_inputs(dates, raw, adjusted, votes):
    try:
        days = np.asarray(dates)
    except (ValueError, TypeError) as exc:
        raise ValueError('dates must be a 1D array of ISO day strings') from exc
    if days.ndim != 1:
        raise ValueError('dates must be a 1D array of ISO day strings')
    if not len(days):
        raise ValueError('dates must not be empty')
    if days.dtype.kind != 'U':
        raise ValueError('dates must contain ISO day strings (not datetime64 or objects)')
    ordinals = np.array([_iso_day(day, f'dates[{i}]').toordinal()
                         for i, day in enumerate(days)], dtype=np.int64)
    if np.any(np.diff(ordinals) <= 0):
        raise ValueError('dates must be strictly increasing without duplicates')
    n = len(days)
    bars = []
    for name, values in (('raw', raw), ('adjusted', adjusted)):
        array = _numeric(values, name, (n, 5)).astype(np.float64, copy=False)
        opens, highs, lows, closes, volumes = array.T
        if (np.any(array[:, :4] <= 0) or np.any(volumes < 0)
                or np.any(lows > np.minimum(opens, closes))
                or np.any(highs < np.maximum(opens, closes)) or np.any(lows > highs)):
            raise ValueError(f'{name}: require positive OHLC, low <= open/close <= high, volume >= 0')
        bars.append(array)
    signals = _numeric(votes, 'votes', (n, 34))
    if not np.isin(signals, (-1, 0, 1)).all():
        raise ValueError('votes must be ternary (-1, 0, +1)')
    return days, ordinals, bars[0], bars[1], signals


def _positive_number(value, field):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value) or value <= 0):
        raise ValueError(f'{field} must be a finite positive number')
    return float(value)


def _history_markers(dates, history):
    """Return (cutoff, {event: (indices, execution_prices)}), without remapping.

    A minimal fill has day/event/price. Quantity, if supplied, must be positive;
    signal_day, if supplied, must strictly precede execution. Non-fill events
    (including unfilled/action/order) are ignored, but their day/event envelope
    is validated. Malformed history raises ValueError, not a partial chart.
    Valid fills after end or on dates absent from the chart are omitted. Caller
    is responsible for supplying a completed historical ledger, not live orders.
    """
    groups = {event: ([], []) for event in _FILLS}
    if history is None:
        return None, {event: (np.empty(0, dtype=int), np.empty(0)) for event in _FILLS}
    if not isinstance(history, dict):
        raise ValueError('history must be a dict with end and ledger, or None')
    end = _iso_day(history.get('end'), 'history.end').isoformat()
    if not isinstance(history.get('ledger'), list):
        raise ValueError('history.ledger must be a list')
    positions = {day: i for i, day in enumerate(dates)}
    for i, entry in enumerate(history['ledger']):
        field = f'history.ledger[{i}]'
        if not isinstance(entry, dict) or not isinstance(entry.get('event'), str):
            raise ValueError(f'{field} must be a dict with a string event')
        day = _iso_day(entry.get('day'), field + '.day').isoformat()
        event = entry['event']
        if event not in groups:
            continue
        price = _positive_number(entry.get('price'), field + '.price')
        if 'quantity' in entry:
            _positive_number(entry['quantity'], field + '.quantity')
        if entry.get('signal_day') is not None:
            signal = _iso_day(entry['signal_day'], field + '.signal_day').isoformat()
            if signal >= day:
                raise ValueError(f'{field}.signal_day must precede the fill day')
        if day <= end and day in positions:
            groups[event][0].append(positions[day])
            groups[event][1].append(price)
    return end, {event: (np.asarray(xs, dtype=int), np.asarray(ys, dtype=float))
                 for event, (xs, ys) in groups.items()}


def _sample_indices(n):
    """Only votes are sampled; retain both endpoints with gaps differing <= 1."""
    return np.linspace(0, n - 1, min(n, _MAX_HEATMAP_COLUMNS), dtype=int)


def _moving_average(closes, window):
    result = np.full(len(closes), np.nan)
    if len(closes) >= window:
        result[window - 1:] = np.convolve(closes, np.ones(window) / window, mode='valid')
    return result


@lru_cache(maxsize=1)
def _local_font():
    # Read known font files only; no addfont(), font installation or font search.
    for path in _FONT_PATHS:
        if path.is_file():
            prop = FontProperties(fname=str(path))
            try:
                prop.get_name()
            except (OSError, RuntimeError):
                continue
            return prop, True
    fallback = Path(matplotlib.get_data_path()) / 'fonts/ttf/DejaVuSans.ttf'
    return FontProperties(fname=str(fallback)), False


def _draw(fig, days, ordinals, raw, adjusted, votes, end, markers, font, chinese, action_continuous):
    text = lambda zh, en: zh if chinese else en
    price_ax, macd_ax, heat_ax = fig.subplots(
        3, 1, sharex=True, gridspec_kw={'height_ratios': [3, 2, 5]})
    # The 34-row heatmap is approximately five inches high, not a score subplot.
    fig.subplots_adjust(left=.18, right=.975, top=.94, bottom=.08, hspace=.30)
    x = np.arange(len(days))
    close = raw[:, 3]
    price_ax.plot(x, close, color=_BLUE, linewidth=1.1,
                  marker='.' if len(days) == 1 else None,
                  label=text('不复权收盘', 'Unadjusted close'))
    price_ax.plot(x, _moving_average(close, 20), color=_ORANGE, linewidth=1.1, label='MA20')
    price_ax.plot(x, _moving_average(close, 60), color='#FFB300', linewidth=1.1,
                  linestyle='--', label='MA60')
    styles = (('build', '^', _BLUE, '建仓'), ('add', 'P', '#42a5f5', '增持'),
              ('reduce', 'v', _ORANGE, '减持'), ('exit', 'X', '#b45309', '退出'))
    for event, marker, color, zh in styles:
        xs, ys = markers[event]
        if len(xs):
            # Four vectorized collections maximum, independent of trade count.
            price_ax.scatter(xs, ys, marker=marker, c=color, s=38, linewidths=.5,
                             edgecolors='white', zorder=5, label=text(zh, event))
    price_ax.set_title(text('不复权价格与历史实际成交（非实盘订单）',
                            'Unadjusted prices / historical fills (not live orders)'), loc='left')
    price_ax.set_ylabel(text('价格（元）', 'Price (CNY)'))
    price_ax.legend(loc='upper left', ncol=4, prop=font.copy(), fontsize=8)

    dif, dea, bar = calc_macd(adjusted[:, 3])
    macd_ax.plot(x, dif, color=_BLUE, linewidth=1, label='DIF (12,26)')
    macd_ax.plot(x, dea, color=_ORANGE, linewidth=1, label='DEA (9)')
    # Vectorized polygons instead of one Rectangle artist per daily bar.
    macd_ax.fill_between(x, 0, bar, where=bar >= 0, color=_RED, alpha=.65, step='mid')
    macd_ax.fill_between(x, 0, bar, where=bar < 0, color=_GREEN, alpha=.65, step='mid')
    macd_ax.axhline(0, color='#9e9e9e', linewidth=.7)
    macd_title = (text('权息连续价格MACD', 'Corporate-action-continuous price MACD') if action_continuous else
                  text('未连续化原价MACD（权息未通过，仅作诊断）',
                       'Unadjusted price MACD (actions unverified; diagnostics only)'))
    macd_ax.set_title(macd_title, loc='left')
    macd_ax.legend(loc='upper left', ncol=2, prop=font.copy(), fontsize=8)
    if not np.isfinite(bar).any():
        macd_ax.text(.99, .80, text('历史不足，MACD预热中', 'Insufficient history: MACD warming up'),
                     transform=macd_ax.transAxes, ha='right', fontsize=9, color='#64748b')

    sampled = _sample_indices(len(days))
    # Cell bounds reflect actual sampled trading-day indices on the shared axis.
    # No interpolation/aggregation of votes: each cell is ONE sampled day's vote.
    edges = np.r_[-.5, (sampled[:-1] + sampled[1:]) / 2, len(days) - .5]
    cmap = ListedColormap([_RED, _GRAY, _GREEN])
    heat_ax.pcolormesh(edges, np.arange(35) - .5, votes[sampled].T,
                       cmap=cmap, norm=BoundaryNorm([ -1.5, -.5, .5, 1.5], 3),
                       shading='flat', rasterized=True, antialiased=False)
    heat_ax.set_ylim(33.5, -.5)
    heat_ax.set_yticks(np.arange(34), labels=OBJECTIVE_NAMES)
    title = text('34维逐日信号（各列独立，不合成为总分）',
                 '34 daily signals (separate coordinates, no total score)')
    if len(sampled) < len(days):
        title += text(f'；仅热图等距抽样 {len(sampled)}/{len(days)} 日（含首尾）',
                      f'; heatmap only: {len(sampled)}/{len(days)} sampled days (both endpoints)')
    heat_ax.set_title(title, loc='left', fontsize=10)
    heat_ax.legend(handles=[Patch(color=_GREEN, label='+1'), Patch(color=_GRAY, label='0'),
                            Patch(color=_RED, label='-1')], loc='upper right', ncol=3,
                   bbox_to_anchor=(1, -.12), prop=font.copy(), frameon=False)
    heat_ax.set_xlabel(text('日期（交易日等距；色块仅代表所抽样当日）',
                            'Date (equally spaced sessions; cells show sampled days only)'))
    tick_indices = np.linspace(0, len(days) - 1, min(8, len(days)), dtype=int)
    heat_ax.set_xticks(tick_indices, labels=days[tick_indices])
    heat_ax.tick_params(axis='y', length=0, labelsize=8)
    heat_ax.tick_params(axis='x', labelsize=8, rotation=25)
    heat_ax.set_xlim(-.5, len(days) - .5)

    if end is not None:
        # Sessions are equally spaced: separate all dates <= end from later
        # dates at the cell edge, never interpolate calendar days across gaps.
        boundary = float(np.searchsorted(ordinals, date.fromisoformat(end).toordinal(), side='right') - .5)
        for ax in fig.axes:
            ax.axvline(boundary, color='#64748b', linestyle='--', linewidth=1, zorder=6)
        note = text(f'回测截止 {end}（虚线位于最后一根截止内日线之后）；之后无后续成交标记',
                    f'Backtest cutoff {end} (line after last included session); no subsequent fills')
        if end < days[0] or end > days[-1]:
            note += text('（截止在图示日期范围外，虚线置于边界）',
                         ' (cutoff outside displayed dates; line at chart edge)')
    else:
        note = text('无历史回测账本：仅行情/信号展示，不推断成交或收益',
                    'No historical ledger: quotes/signals only; no inferred fills or returns')
    fig.text(.18, .975, note, fontsize=10, color='#475569', va='top', fontproperties=font)
    for ax in fig.axes:
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_color('#cbd5e1')
        if ax is not heat_ax:
            ax.grid(True, alpha=.22, color='#94a3b8')
            ax.set_axisbelow(True)
            # ASCII minus even if caller's rcParams enables Unicode minus.
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        for label in (ax.title, ax._left_title, ax.xaxis.label, ax.yaxis.label,
                      *ax.texts, *ax.get_xticklabels(), *ax.get_yticklabels()):
            size = label.get_fontsize()
            label.set_fontproperties(font)
            label.set_fontsize(size)


def render_chart(dates, raw, adjusted, votes, history=None, *, action_continuous=True) -> str:
    """Return a bare base64 PNG string, without a data-URI prefix.

    dates: strictly ascending, unique ISO day strings (normally ndarray U10).
    raw/adjusted: finite (N,5) arrays in open/high/low/close/volume order.
    votes: (N,34) -1/0/+1 in pareto_strategy.OBJECTIVE_NAMES order.
    history: None or {end: ISO day, ledger: list}; see _history_markers.
    action_continuous: whether the caller verified/continuous-adjusted prices;
    pass False for raw-only diagnostics so the MACD title does not claim this.

    Empty input and malformed data raise ValueError for the UI to handle. One
    day still renders; unready MA/MACD values stay NaN (never backfilled). Price,
    indicators and ledger fills retain full resolution; only votes are sampled
    to <=1000 columns. No performance or future executions are calculated.
    """
    if not isinstance(action_continuous, bool):
        raise ValueError('action_continuous must be a bool')
    days, ordinals, raw, adjusted, votes = _validate_inputs(dates, raw, adjusted, votes)
    end, markers = _history_markers(days, history)
    with _RENDER_LOCK:
        font, chinese = _local_font()
        fig = Figure(figsize=(16, 14), dpi=110, facecolor='white')
        try:
            canvas = FigureCanvasAgg(fig)
            _draw(fig, days, ordinals, raw, adjusted, votes, end, markers, font, chinese, action_continuous)
            with BytesIO() as buffer:
                canvas.print_png(buffer)
                return base64.b64encode(buffer.getvalue()).decode('ascii')
        finally:
            # An OO Figure has no pyplot manager to close. Clear artists even
            # on PNG failures, and break the Figure/Canvas reference cycle.
            fig.clear()
            fig.set_canvas(None)