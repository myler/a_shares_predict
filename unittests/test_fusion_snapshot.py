"""Offline current-snapshot tests; all network and source reads are mocked."""

from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

import fusion_snapshot as snapshot
from unittests.test_fusion_analysis import fresh_fixture, TODAY, CODE, PEER


class TestFusionSnapshot(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for target in ('socket.socket.connect', 'socket.getaddrinfo', 'sqlite3.connect'):
            guard = patch(target, side_effect=AssertionError('offline only'))
            guard.start()
            self.addCleanup(guard.stop)
        root = patch.object(snapshot, 'SNAPSHOT_ROOT', self.root)
        root.start()
        self.addCleanup(root.stop)
        self.generation = '20260910T160000_' + 'a' * 32
        self.folder = self.root / self.generation
        self.folder.mkdir()
        self.rows = [dict(code=code, date=TODAY, votes=[0] * 34, bars=280,
                          price_status='ok', action_status='ok', rights_issue_present=False,
                          eligible=True, gate=True, feature_error=None) for code in (CODE, PEER)]
        self.payload = dict(schema_version=1, feature_version=snapshot.STRATEGY_VERSION,
                            date=TODAY, complete=True, checked_at=TODAY + 'T16:00:00+08:00',
                            actions_complete=True, stocks=self.rows)

    def publish(self):
        raw = json.dumps(self.payload).encode()
        (self.folder / 'snapshot.json').write_bytes(raw)
        (self.root / f'{TODAY}.json').write_text(json.dumps(dict(
            generation=self.generation, sha256=hashlib.sha256(raw).hexdigest())))

    def compare(self, vector=None):
        return snapshot.compare_current(CODE, TODAY, np.zeros(34) if vector is None else vector,
                                        [CODE, PEER])

    def test_missing_or_different_date_never_carries_forward(self):
        self.assertIsNone(self.compare())
        self.publish()
        self.assertIsNone(snapshot.compare_current(CODE, '2026-09-11', np.zeros(34)))

    def test_equal_vectors_frontier_and_strict_dominance(self):
        self.publish()
        self.assertEqual(self.compare()['status'], 'front1')
        self.rows[1]['votes'][0] = 1
        self.publish()
        self.assertEqual(self.compare()['status'], 'dominated')
        self.assertEqual(self.compare()['dominators'], [PEER])

    def test_failed_peer_cannot_be_a_trivial_frontier(self):
        self.rows[1]['price_status'] = 'error'
        self.publish()
        result = self.compare()
        self.assertEqual(result['status'], 'unknown')
        self.assertEqual(result['unavailable'], 1)

    def test_stale_and_ineligible_peers_are_not_compared(self):
        baseline = deepcopy(self.rows[1])
        for changes in (dict(date='2026-09-09'), dict(eligible=False), dict(gate=False),
                        dict(rights_issue_present=True), dict(action_status='error'),
                        dict(feature_error='failed')):
            self.rows[1] = dict(baseline, **changes)
            self.publish()
            with self.subTest(changes=changes):
                self.assertEqual(self.compare()['status'], 'unknown')
                self.assertEqual(self.compare()['compared'], 0)

    def test_incomplete_wrong_version_or_hash_fails_closed(self):
        for changes in (dict(complete=False), dict(actions_complete=False),
                        dict(feature_version='old'), dict(date='2026-09-09')):
            original = deepcopy(self.payload)
            self.payload.update(changes)
            self.publish()
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.compare()
            self.payload = original
        self.publish()
        (self.folder / 'snapshot.json').write_text('{}')
        with self.assertRaises(ValueError):
            self.compare()

    def test_universe_and_warmup_mismatch_rejected(self):
        self.publish()
        with self.assertRaises(ValueError):
            snapshot.compare_current(CODE, TODAY, np.zeros(34), [CODE])
        self.rows[1]['bars'] = 250
        self.publish()
        with self.assertRaises(ValueError):
            self.compare()

    def test_current_intraday_and_invalid_vectors_rejected(self):
        self.payload['checked_at'] = TODAY + 'T14:59:59+08:00'
        self.publish()
        with self.assertRaises(ValueError):
            self.compare()
        self.payload['checked_at'] = TODAY + 'T07:00:00+00:00'
        self.publish()
        self.assertEqual(self.compare()['status'], 'front1')
        for vector in (np.ones(34, dtype=bool), np.full(34, float('nan')), np.zeros(33)):
            with self.subTest(dtype=vector.dtype), self.assertRaises(ValueError):
                self.compare(vector)

    def test_partial_coverage_can_reject_f1_but_cannot_confirm_it(self):
        self.rows.append(dict(code='600001', price_status='error'))
        self.rows[1]['votes'][0] = 1
        self.publish()
        result = snapshot.compare_current(CODE, TODAY, np.zeros(34))
        self.assertEqual(result['status'], 'dominated')
        self.assertEqual(result['unavailable'], 1)

    def test_build_publishes_only_after_all_stocks_finish(self):
        context = (self.root, dict(universe=[dict(code=CODE), dict(code=PEER)]), {}, None, None)
        def actions(codes, folder, **kwargs):
            folder.mkdir()
            (folder / 'market_manifest.json').write_text('{}')
            return {code: fresh_fixture()['actions'] for code in codes}
        def peer(code, *args, **kwargs):
            self.assertFalse((self.root / f'{TODAY}.json').exists())
            return next(row for row in self.rows if row['code'] == code)
        with patch.object(snapshot.pareto_web, '_load_run', return_value=context), patch.object(
                snapshot.action_fallback, 'resolve_action_snapshot', side_effect=actions), patch.object(
                snapshot, '_peer', side_effect=peer):
            result = snapshot.build_snapshot(root=self.root, source=self.root / 'absent.sqlite',
                                             now=datetime.fromisoformat(self.payload['checked_at']))
        self.assertEqual(result['price_failures'], 0)
        self.assertEqual(self.compare()['status'], 'front1')

    def test_resume_wrong_date_or_traversal_cannot_request_network(self):
        self.publish()
        request = dict(universe=[dict(code=CODE), dict(code=PEER)], source=str(self.root / 'absent.sqlite'))
        (self.folder / 'request.json').write_text(json.dumps(request))
        context = (self.root, dict(universe=request['universe']), {}, None, None)
        with patch.object(snapshot.pareto_web, '_load_run', return_value=context), patch.object(
                snapshot.action_fallback, 'resolve_action_snapshot') as actions:
            for generation in ('../other', self.generation):
                with self.subTest(generation=generation), self.assertRaises(ValueError):
                    snapshot.build_snapshot(root=self.root, source=self.root / 'absent.sqlite',
                        now=datetime.fromisoformat('2026-09-11T16:00:00+08:00'), resume=generation)
            actions.assert_not_called()

    def test_resume_preserves_success_and_actions_without_network(self):
        self.publish()
        source = self.root / 'absent.sqlite'
        universe = [dict(code=CODE), dict(code=PEER)]
        (self.folder / 'request.json').write_text(json.dumps(dict(universe=universe, source=str(source))))
        action_folder = self.folder / 'actions'
        action_folder.mkdir()
        (action_folder / 'market_manifest.json').write_text(json.dumps(dict(status='ok',
            request=snapshot.action_fallback._spec(snapshot.live_data.START, TODAY))))
        with patch.object(snapshot.pareto_web, '_load_run', return_value=(self.root, dict(universe=universe), {}, None, None)), patch.object(
                snapshot.action_fallback, 'resolve_action_snapshot', side_effect=AssertionError('no action requests')), patch.object(
                snapshot.action_fallback, '_index', return_value={}), patch.object(
                snapshot.action_fallback, '_history', return_value=fresh_fixture()['actions']), patch.object(
                snapshot, '_peer', side_effect=AssertionError('no price requests')):
            state = snapshot.build_snapshot(root=self.root, source=source,
                now=datetime.fromisoformat(TODAY + 'T17:00:00+08:00'), resume=self.generation)
        self.assertNotEqual(state['generation'], self.generation)
        self.assertEqual(self.compare()['status'], 'front1')
        self.assertEqual(self.compare()['checked_at'], self.payload['checked_at'])
        self.assertTrue((self.folder / 'snapshot.json').is_file())

    def test_access_denial_cannot_publish_partial_generation(self):
        context = (self.root, dict(universe=[dict(code=CODE), dict(code=PEER)]), {}, None, None)
        def actions(codes, folder, **kwargs):
            folder.mkdir()
            (folder / 'market_manifest.json').write_text('{}')
            return {code: fresh_fixture()['actions'] for code in codes}
        def peer(code, source, folder, history, now, stopped):
            stopped.set()
            return dict(code=code, price_status='error')
        with patch.object(snapshot.pareto_web, '_load_run', return_value=context), patch.object(
                snapshot.action_fallback, 'resolve_action_snapshot', side_effect=actions), patch.object(
                snapshot, '_peer', side_effect=peer), self.assertRaisesRegex(ValueError, 'access denied'):
            snapshot.build_snapshot(root=self.root, source=self.root / 'absent.sqlite',
                                     now=datetime.fromisoformat(self.payload['checked_at']))
        self.assertFalse((self.root / f'{TODAY}.json').exists())
        self.assertEqual(json.loads((self.root / 'status.json').read_bytes())['status'], 'failed')

    def test_price_collection_is_read_only_and_matches_shared_math(self):
        fresh = fresh_fixture()
        old = dict(stock=None, klines=[dict(date=row['day'], **{key: row[key]
                   for key in snapshot.live_data.FIELDS}) for row in fresh['rows']])
        with patch.object(snapshot.live_data, '_read_target', return_value=old), patch.object(
                snapshot.live_data, '_request_bytes', return_value=json.dumps(fresh['rows']).encode()), patch.object(
                snapshot.live_data, '_commit_prices', side_effect=AssertionError('must not write source')):
            result = snapshot._peer(CODE, self.root / 'absent.sqlite', self.folder,
                                    fresh['actions'], datetime.fromisoformat(fresh['checked_at']), threading.Event())
        raw = np.array([[row[key] for key in snapshot.live_data.FIELDS] for row in fresh['rows']])
        self.assertEqual(result['votes'], snapshot.make_objectives(*raw.T)['votes'][-1].tolist())
        self.assertTrue(result['eligible'])
        self.assertEqual(result['date'], TODAY)
        self.assertTrue((self.folder / 'prices' / CODE / 'rows.json').is_file())

    def test_denial_stops_collection_without_fallback_or_retries(self):
        stopped = threading.Event()
        with patch.object(snapshot.live_data, '_read_target', return_value=dict(stock=None, klines=[])), patch.object(
                snapshot.live_data, '_request_bytes', side_effect=snapshot.live_data._HTTPFailure('HTTPError', 429)) as request:
            result = snapshot._peer(CODE, self.root / 'absent.sqlite', self.folder,
                                    fresh_fixture()['actions'], datetime.now(snapshot.SHANGHAI), stopped)
        self.assertEqual(request.call_count, 1)
        self.assertTrue(stopped.is_set())
        self.assertEqual(result['price_status'], 'error')

    def test_source_history_before_signal_window_is_stale_not_failed_collection(self):
        bars = [dict(day='2004-04-28', open='0', high='0', low='0', close='0', volume='0')]
        with patch.object(snapshot.live_data, '_read_target', return_value=dict(stock=None, klines=[])), patch.object(
                snapshot.live_data, '_request_bytes', return_value=json.dumps(bars).encode()) as request:
            result = snapshot._peer(CODE, self.root / 'absent.sqlite', self.folder,
                                    fresh_fixture()['actions'], datetime.now(snapshot.SHANGHAI), threading.Event())
        self.assertEqual(request.call_count, 1)
        self.assertEqual(result['price_status'], 'ok')
        self.assertEqual(result['date'], '2004-04-28')
        self.assertFalse(result['eligible'])
        self.assertTrue(result['history_only'])
        self.rows[1] = dict(result, code=PEER)
        self.publish()
        self.assertEqual(self.compare()['compared'], 0)
        self.assertEqual(self.compare()['unavailable'], 0)

    def test_incomplete_actions_never_start_prices_or_publish(self):
        context = (self.root, dict(universe=[dict(code=CODE), dict(code=PEER)]), {}, None, None)
        with patch.object(snapshot.pareto_web, '_load_run', return_value=context), patch.object(
                snapshot.action_fallback, 'resolve_action_snapshot', return_value={}), patch.object(
                snapshot, '_peer') as peer:
            with self.assertRaisesRegex(ValueError, 'action snapshot incomplete'):
                snapshot.build_snapshot(root=self.root, source=self.root / 'absent.sqlite')
            peer.assert_not_called()
        self.assertFalse((self.root / f'{TODAY}.json').exists())


if __name__ == '__main__':
    unittest.main()