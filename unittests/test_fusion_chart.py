"""Offline contracts for the standalone in-memory chart; no other suite runs."""

import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import date
import io
import socket
import sqlite3
import unittest
from unittest.mock import patch
import warnings

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib._pylab_helpers import Gcf
import numpy as np
from PIL import Image

import fusion_chart as chart
from engine import calc_macd
from pareto_strategy import OBJECTIVE_NAMES


def market(n=90):
    days = (np.datetime64('2020-01-01', 'D') + np.arange(n)).astype('U10')
    close = 20 + np.arange(n) * .02 + np.sin(np.arange(n) / 5)
    raw = np.column_stack((close - .1, close + .5, close - .5, close, np.full(n, 1000.)))
    adjusted = raw.copy()
    adjusted[:, :4] *= np.linspace(.55, 1.4, n)[:, None]
    votes = ((np.arange(n)[:, None] + np.arange(34)) % 3 - 1).astype(np.int8)
    return days, raw, adjusted, votes


def fill(day, event='build', price=17.25, **extra):
    return dict(day=str(day), event=event, price=price, quantity=100, **extra)


class OfflineTest(unittest.TestCase):
    def setUp(self):
        self.guards = ExitStack()
        self.addCleanup(self.guards.close)
        for target in ('socket.create_connection', 'socket.socket.connect',
                       'socket.socket.connect_ex', 'socket.getaddrinfo', 'sqlite3.connect'):
            self.guards.enter_context(patch(target, side_effect=AssertionError('offline chart')))

    def assert_png(self, encoded):
        self.assertIsInstance(encoded, str)
        data = base64.b64decode(encoded, validate=True)
        self.assertTrue(data.startswith(b'\x89PNG\r\n\x1a\n'))
        with Image.open(io.BytesIO(data)) as image:
            self.assertEqual(image.format, 'PNG')
            self.assertEqual(image.size, (1760, 1540))
            image.verify()

    def capture(self, data, history=None, inspect=None, **kwargs):
        """Inspect actual axes before the required cleanup (no file snapshots)."""
        real_print = FigureCanvasAgg.print_png
        figures = []

        def observe(canvas, buffer, *args, **kwargs):
            figures.append(canvas.figure)
            if inspect is not None:
                inspect(canvas.figure)
            return real_print(canvas, buffer, *args, **kwargs)

        with patch.object(FigureCanvasAgg, 'print_png', new=observe):
            result = chart.render_chart(*data, history=history, **kwargs)
        self.assertEqual(len(figures), 1)
        self.assertEqual(figures[0].axes, [])
        self.assertIsNone(figures[0].canvas)
        return result


class TestInputValidation(OfflineTest):
    def test_empty_is_explicit_error_before_figure(self):
        with patch.object(chart, 'Figure') as figure:
            with self.assertRaisesRegex(ValueError, 'empty'):
                chart.render_chart(*market(0))
        figure.assert_not_called()

    def test_date_type_shape_order_and_calendar(self):
        data = market(2)
        invalid = [
            ['2020-01-02', '2020-01-01'], ['2020-01-01', '2020-01-01'],
            ['2020-02-30', '2020-03-01'], ['2020-01', '2020-02'],
            ['2020-1-01', '2020-01-02'], ['2020-01-01T00:00:00', '2020-01-02T00:00:00'],
            ['0000-01-01', '2020-01-02'], ['NaT', '2020-01-02'],
            np.array(['2020-01', '2020-02'], dtype='datetime64[M]'),
            np.array(['2020-01-01', '2020-01-02'], dtype='datetime64[D]'),
            [date(2020, 1, 1), date(2020, 1, 2)], [20200101, 20200102],
            np.array(['2020-01-01', '2020-01-02'], dtype=object),
            [b'2020-01-01', b'2020-01-02'], [['2020-01-01', '2020-01-02']],
        ]
        for days in invalid:
            with self.subTest(dates=days), self.assertRaisesRegex(ValueError, 'dates'):
                chart.render_chart(days, *data[1:])

    def test_ohlcv_shape_types_bounds_and_votes(self):
        data = market(2)
        for slot in (1, 2, 3):
            original = data[slot]
            bad = [original[:-1], original[:, :-1], original.ravel(),
                   original.astype(str), original.astype(complex), original.astype(bool)]
            for value in (np.nan, np.inf, -np.inf):
                arr = original.astype(float)
                arr[0, 0] = value
                bad.append(arr)
            if slot < 3:
                for col, value in ((0, 0), (1, 1), (2, 100), (3, -1), (4, -1)):
                    arr = original.copy()
                    arr[0, col] = value
                    bad.append(arr)
            else:
                for value in (.5, 2, -2, 255):
                    arr = original.astype(float)
                    arr[0, 0] = value
                    bad.append(arr)
                bad.append(np.zeros((2, 35)))
            for arr in bad:
                args = list(data)
                args[slot] = arr
                with self.subTest(slot=slot, shape=arr.shape), self.assertRaisesRegex(
                        ValueError, ('raw', 'adjusted', 'votes')[slot - 1]):
                    chart.render_chart(*args)

    def test_zero_volume_leap_day_and_one_point_render_without_warning(self):
        for n in (1, 2):
            data = list(market(n))
            data[0] = np.array(['2024-02-29', '2024-03-01'][:n], dtype='U10')
            data[1][:, 4] = data[2][:, 4] = 0
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                self.assert_png(self.capture(data))


class TestHistory(OfflineTest):
    def test_only_real_history_fills_at_execution_prices_no_future_or_date_remapping(self):
        days = np.array(['2024-03-01', '2024-03-04', '2024-03-05',
                         '2024-03-06', '2024-03-07'], dtype='U10')
        ledger = [fill(days[i], event, price=8 + i * .17)
                  for i, event in enumerate(chart._FILLS)]
        ledger += [fill(days[4], 'build', 900), fill('2024-03-02', 'add', 901),
                   fill('2024-02-28', 'exit', 902),
                   dict(day=days[2], event='unfilled', price=None),
                   dict(day=days[2], event='action', price=None),
                   dict(day=days[4], event='order', target_weight=1),
                   fill(days[1], 'build', 8.02, signal_day=days[0])]
        end, groups = chart._history_markers(days, dict(end=days[3], ledger=ledger))
        self.assertEqual(end, days[3])
        np.testing.assert_array_equal(groups['build'][0], [0, 1])
        np.testing.assert_allclose(groups['build'][1], [8, 8.02])
        for i, event in enumerate(chart._FILLS[1:], start=1):
            np.testing.assert_array_equal(groups[event][0], [i])
            np.testing.assert_allclose(groups[event][1], [8 + i * .17])

    def test_minimal_fill_allowed_none_or_empty_never_infers_votes(self):
        days = market(2)[0]
        for history in (None, dict(end=days[-1], ledger=[])):
            _, groups = chart._history_markers(days, history)
            self.assertTrue(all(len(xs) == len(ys) == 0 for xs, ys in groups.values()))
        _, groups = chart._history_markers(days, dict(end=days[-1], ledger=[
            dict(day=days[0], event='build', price=12.5)]))
        np.testing.assert_array_equal(groups['build'][1], [12.5])

    def test_malformed_history_rejected_with_field_context(self):
        days = market(2)[0]
        invalid = [(False, 'history'), ([], 'history'), ({}, 'history.end'),
                   (dict(end='2020-01', ledger=[]), 'history.end'),
                   (dict(end='2020-02-30', ledger=[]), 'history.end'),
                   (dict(end=days[-1], ledger=()), 'history.ledger')]
        bad_entries = [None, {}, dict(day=days[0], event=1),
                       dict(day=np.datetime64(days[0]), event='build', price=10),
                       dict(day='2020-01', event='build', price=10),
                       dict(day=days[0], event='build'),
                       fill(days[0], signal_day=days[0]),
                       fill(days[0], signal_day=days[-1]),
                       fill(days[0], signal_day='2020-01')]
        for value in (None, 0, -1, float('nan'), float('inf'), True, '12.5', []):
            bad_entries.append(fill(days[0], price=value))
            entry = fill(days[0])
            entry['quantity'] = value
            bad_entries.append(entry)
        invalid.extend((dict(end=days[-1], ledger=[entry]), r'history\.ledger\[0\]')
                       for entry in bad_entries)
        for history, pattern in invalid:
            with self.subTest(history=history), self.assertRaisesRegex(ValueError, pattern):
                chart._history_markers(days, history)

    def test_cutoff_after_last_included_session_without_calendar_interpolation(self):
        data = list(market(2))
        data[0] = np.array(['2024-03-01', '2024-03-04'], dtype='U10')
        for end, x in (('2024-02-29', -.5), ('2024-03-01', .5), ('2024-03-02', .5),
                       ('2024-03-03', .5), ('2024-03-04', 1.5), ('2024-03-05', 1.5)):
            def inspect(fig):
                self.assertIn(end, fig.texts[0].get_text())
                for ax in fig.axes:
                    np.testing.assert_allclose(ax.lines[-1].get_xdata(), [x, x])
            with self.subTest(end=end):
                self.assert_png(self.capture(data, dict(end=end, ledger=[]), inspect))


class TestRendering(OfflineTest):
    def test_unverified_action_title_is_raw_diagnostics_in_both_languages(self):
        font, _ = chart._local_font()
        data = list(market(2))
        data[2] = data[1].copy()
        for chinese, title in ((True, '未连续化原价MACD（权息未通过，仅作诊断）'),
                               (False, 'Unadjusted price MACD (actions unverified; diagnostics only)')):
            def inspect(fig):
                self.assertEqual(fig.axes[1].get_title(loc='left'), title)
                np.testing.assert_allclose(fig.axes[1].lines[0].get_ydata(),
                                           calc_macd(data[1][:, 3])[0], equal_nan=True)
            with self.subTest(chinese=chinese), patch.object(chart, '_local_font', return_value=(font, chinese)):
                self.assert_png(self.capture(data, inspect=inspect, action_continuous=False))

    def test_action_continuity_flag_requires_bool(self):
        for value in (None, 'false', 1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'action_continuous'):
                chart.render_chart(*market(2), action_continuous=value)

    def test_three_panels_raw_ma_adjusted_macd_and_34_separate_votes(self):
        data = market(90)
        days, raw, adjusted, votes = data
        history = dict(end=days[60], ledger=[fill(days[i + 10], event, 11 + i)
                                            for i, event in enumerate(chart._FILLS)]
                       + [fill(days[70], price=999)])

        def inspect(fig):
            self.assertEqual(len(fig.axes), 3)
            self.assertEqual(fig.get_facecolor(), to_rgba('white'))
            price, macd, heat = fig.axes
            np.testing.assert_array_equal(price.lines[0].get_ydata(), raw[:, 3])
            for line, window in zip(price.lines[1:3], (20, 60)):
                actual = line.get_ydata()
                self.assertTrue(np.isnan(actual[:window - 1]).all())
                np.testing.assert_allclose(actual[window - 1:],
                                           [raw[i - window + 1:i + 1, 3].mean()
                                            for i in range(window - 1, len(days))])
            self.assertEqual(len(price.collections), 4)
            for i, collection in enumerate(price.collections):
                np.testing.assert_allclose(collection.get_offsets(), [[10 + i, 11 + i]])
            expected = calc_macd(adjusted[:, 3])
            for line, values in zip(macd.lines[:2], expected[:2]):
                np.testing.assert_allclose(line.get_ydata(), values, equal_nan=True)
            mesh = heat.collections[0]
            np.testing.assert_array_equal(mesh.get_array(), votes.T)
            self.assertEqual(len(heat.get_yticks()), 34)
            self.assertEqual([t.get_text() for t in heat.get_yticklabels()], list(OBJECTIVE_NAMES))
            np.testing.assert_allclose(mesh.cmap(mesh.norm([-1, 0, 1])),
                                       [to_rgba(c) for c in (chart._RED, chart._GRAY, chart._GREEN)])
            self.assertGreater(heat.get_position().height * fig.get_figheight(), 4.9)
            self.assertEqual(len(heat.lines), 1)  # Cutoff only; NOT a sum score.
            self.assertEqual(len(heat.get_legend().get_texts()), 3)
            for ax in fig.axes:
                np.testing.assert_array_equal(ax.lines[-1].get_xdata(), [60.5, 60.5])

        self.assert_png(self.capture(data, history, inspect))

    def test_heatmap_cap_endpoint_selection_full_price_macd_and_batch_scatter(self):
        data = market(2405)
        days, _, adjusted, votes = data
        ledger = [fill(days[i % len(days)], chart._FILLS[i % 4], 18 + i % 7)
                  for i in range(12000)]
        sample = chart._sample_indices(len(days))
        self.assertEqual(len(sample), 1000)
        self.assertEqual((sample[0], sample[-1]), (0, len(days) - 1))
        self.assertTrue(np.all(np.diff(sample) > 0))
        self.assertLessEqual(np.ptp(np.diff(sample)), 1)

        def inspect(fig):
            price, macd, heat = fig.axes
            self.assertEqual(len(price.lines[0].get_xdata()), len(days))
            self.assertEqual(len(macd.lines[0].get_ydata()), len(days))
            self.assertEqual(len(price.collections), 4)
            self.assertEqual(sum(len(c.get_offsets()) for c in price.collections), len(ledger))
            self.assertEqual(len(macd.patches), 0)
            self.assertEqual(len(macd.collections), 2)
            mesh = heat.collections[0]
            np.testing.assert_array_equal(mesh.get_array(), votes[sample].T)
            np.testing.assert_array_equal(mesh.get_array()[:, -1], votes[-1])
            self.assertEqual(mesh.get_coordinates().shape, (35, 1001, 2))
            self.assertIn('1000/2405', heat.get_title(loc='left'))

        with patch.object(chart, 'calc_macd', wraps=calc_macd) as calc:
            self.assert_png(self.capture(data, dict(end=days[-1], ledger=ledger), inspect))
        calc.assert_called_once()
        np.testing.assert_array_equal(calc.call_args.args[0], adjusted[:, 3])

    def test_sampling_at_and_around_cap(self):
        for n in (1, 2, 999, 1000, 1001, 1999, 2001, 10001):
            sample = chart._sample_indices(n)
            with self.subTest(n=n):
                self.assertEqual(len(sample), min(n, 1000))
                self.assertEqual(sample[0], 0)
                self.assertEqual(sample[-1], n - 1)
                self.assertEqual(len(np.unique(sample)), len(sample))
                if n <= 1000:
                    np.testing.assert_array_equal(sample, np.arange(n))

    def test_causal_macd_ma_and_inputs_not_mutated(self):
        data = market(120)
        originals = [arr.copy() for arr in data]
        for n in (1, 2, 25, 26, 34, 60, 90):
            for prefix, full in zip(calc_macd(data[2][:n, 3]), calc_macd(data[2][:, 3])):
                np.testing.assert_allclose(prefix, full[:n], equal_nan=True)
            for window in (20, 60):
                np.testing.assert_allclose(chart._moving_average(data[1][:n, 3], window),
                                           chart._moving_average(data[1][:, 3], window)[:n],
                                           equal_nan=True)
        for array in data:
            array.setflags(write=False)
        self.assert_png(chart.render_chart(*data))
        for actual, expected in zip(data, originals):
            np.testing.assert_array_equal(actual, expected)

    def test_english_fallback_no_missing_glyphs_and_chinese_macd_label(self):
        fallback = FontProperties(fname=str(
            chart.Path(matplotlib.get_data_path()) / 'fonts/ttf/DejaVuSans.ttf'))
        with patch.object(chart, '_FONT_PATHS', ()):
            # Bypass cache without disturbing another thread's cached font.
            font, chinese = chart._local_font.__wrapped__()
        self.assertFalse(chinese)
        self.assertTrue(font.get_file().endswith('DejaVuSans.ttf'))

        def inspect_english(fig):
            self.assertEqual(fig.axes[1].get_title(loc='left'),
                             'Corporate-action-continuous price MACD')
            for ax in fig.axes:
                for label in (ax._left_title, ax.xaxis.label, ax.yaxis.label, *ax.texts,
                              *ax.get_xticklabels(), *ax.get_yticklabels()):
                    self.assertTrue(label.get_text().isascii(), label.get_text())
            self.assertTrue(fig.texts[0].get_text().isascii())

        with patch.object(chart, '_local_font', return_value=(fallback, False)):
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                self.assert_png(self.capture(market(1), inspect=inspect_english))
        font, chinese = chart._local_font()
        if chinese:
            def inspect_chinese(fig):
                self.assertEqual(fig.axes[1].get_title(loc='left'), '权息连续价格MACD')
            self.assert_png(self.capture(market(40), inspect=inspect_chinese))

    def test_no_writes_global_style_or_pyplot_managers(self):
        import builtins
        import os
        import sys

        before_style = dict(matplotlib.rcParams)
        before_managers = list(Gcf.get_all_fig_managers())
        before_imports = {name: sys.modules.get(name) for name in ('plotting', 'matplotlib.pyplot', 'db')}
        real_open, real_io_open, real_os_open = builtins.open, io.open, os.open

        def guard_open(original):
            def guarded(file, mode='r', *args, **kwargs):
                if any(flag in mode for flag in 'wax+'):
                    raise AssertionError(f'chart attempted a file write: {file}')
                return original(file, mode, *args, **kwargs)
            return guarded

        def guard_os_open(path, flags, *args, **kwargs):
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                raise AssertionError(f'chart attempted a file write: {path}')
            return real_os_open(path, flags, *args, **kwargs)

        with patch('builtins.open', side_effect=guard_open(real_open)), \
                patch('io.open', side_effect=guard_open(real_io_open)), \
                patch('os.open', side_effect=guard_os_open), \
                patch('os.mkdir', side_effect=AssertionError('no directories')):
            self.assert_png(chart.render_chart(*market(40)))
        self.assertEqual(before_style, dict(matplotlib.rcParams))
        self.assertEqual(before_managers, Gcf.get_all_fig_managers())
        for name, previous in before_imports.items():
            self.assertIs(sys.modules.get(name), previous)

    def test_failed_png_cleans_figure_and_releases_lock(self):
        figures = []

        def fail(canvas, *args, **kwargs):
            figures.append(canvas.figure)
            raise OSError('PNG failure')

        with patch.object(FigureCanvasAgg, 'print_png', new=fail):
            with self.assertRaisesRegex(OSError, 'PNG failure'):
                chart.render_chart(*market(2))
        self.assertEqual(figures[0].axes, [])
        self.assertIsNone(figures[0].canvas)
        self.assertTrue(chart._RENDER_LOCK.acquire(blocking=False))
        chart._RENDER_LOCK.release()
        self.assert_png(chart.render_chart(*market(2)))

    def test_concurrent_calls_are_serialized_and_cleanup_safe(self):
        real_print = FigureCanvasAgg.print_png
        before = list(Gcf.get_all_fig_managers())
        figures = []
        active = 0

        def observe(canvas, buffer, *args, **kwargs):
            nonlocal active
            active += 1
            self.assertEqual(active, 1)
            self.assertTrue(chart._RENDER_LOCK.locked())
            figures.append(canvas.figure)
            try:
                return real_print(canvas, buffer, *args, **kwargs)
            finally:
                active -= 1

        with patch.object(FigureCanvasAgg, 'print_png', new=observe):
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda n: chart.render_chart(*market(n)), (1, 2, 40, 90)))
        for result in results:
            self.assert_png(result)
        self.assertEqual(len(figures), 4)
        self.assertTrue(all(fig.axes == [] and fig.canvas is None for fig in figures))
        self.assertEqual(before, Gcf.get_all_fig_managers())


if __name__ == '__main__':
    unittest.main()