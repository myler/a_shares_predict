"""db.py 单元测试 —— 验证SQLite缓存的增删改查"""

import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import sqlite3
import numpy as np

# 用临时DB替代真实DB
import db
_original_path = db.DB_PATH


class TestDB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        db.DB_PATH = os.path.join(cls.tmpdir, 'test_cache.db')

    @classmethod
    def tearDownClass(cls):
        db.DB_PATH = _original_path
        import shutil
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def setUp(self):
        # 每个测试前清空
        conn = db.get_db()
        conn.execute("DELETE FROM baseline_jobs")
        conn.execute("DELETE FROM baseline_runs")
        conn.execute("DELETE FROM klines")
        conn.execute("DELETE FROM dividends")
        conn.execute("DELETE FROM stocks")
        conn.commit()
        conn.close()

    def test_init_creates_tables(self):
        """初始化应自动建表"""
        conn = db.get_db()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        self.assertIn('stocks', table_names)
        self.assertIn('klines', table_names)
        self.assertIn('dividends', table_names)
        self.assertIn('baseline_runs', table_names)
        self.assertIn('baseline_jobs', table_names)
        conn.close()

    def test_save_and_load_klines(self):
        """保存K线后应能完整加载"""
        code = '000001'
        data = [
            {'day': '2024-01-02', 'open': '10.0', 'high': '11.0',
             'low': '9.5', 'close': '10.5', 'volume': '1000000'},
            {'day': '2024-01-03', 'open': '10.5', 'high': '12.0',
             'low': '10.0', 'close': '11.5', 'volume': '1500000'},
        ]
        db.save_klines(code, data)
        loaded = db.load_klines(code)
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]['day'], '2024-01-02')
        self.assertEqual(loaded[0]['close'], '10.5')

    def test_load_klines_respects_end_date(self):
        """历史重放不得读入冻结日期之后的缓存日线。"""
        code = '000010'
        db.save_klines(code, [
            {'day': '2026-07-18', 'open': '10', 'high': '11',
             'low': '9', 'close': '10', 'volume': '100'},
            {'day': '2026-07-19', 'open': '11', 'high': '12',
             'low': '10', 'close': '11', 'volume': '200'},
            {'day': '2026-07-20', 'open': '12', 'high': '13',
             'low': '11', 'close': '12', 'volume': '300'},
        ])

        loaded = db.load_klines(code, end_date='2026-07-19')

        self.assertEqual([row['day'] for row in loaded],
                         ['2026-07-18', '2026-07-19'])

    def test_deduplicate_klines(self):
        """重复保存不产生重复行"""
        code = '000002'
        data = [{'day': '2024-01-02', 'open': '10', 'high': '11',
                 'low': '9', 'close': '10', 'volume': '100'}]
        db.save_klines(code, data)
        db.save_klines(code, data)  # 重复保存
        loaded = db.load_klines(code)
        self.assertEqual(len(loaded), 1)

    def test_replace_klines_updates_existing_date(self):
        """强制刷新应覆盖同一交易日的旧 K 线值"""
        code = '000009'
        old = [
            {'day': '2024-01-01', 'open': '9', 'high': '10',
             'low': '8', 'close': '9', 'volume': '90'},
            {'day': '2024-01-02', 'open': '10', 'high': '11',
             'low': '9', 'close': '10', 'volume': '100'},
        ]
        new = [{'day': '2024-01-02', 'open': '11', 'high': '12',
                'low': '10', 'close': '11', 'volume': '200'}]
        db.save_klines(code, old)
        db.save_klines(code, new, replace=True)
        loaded = db.load_klines(code)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]['close'], '11.0')

    def test_load_nonexistent(self):
        """未缓存的股票返回None"""
        loaded = db.load_klines('999999')
        self.assertIsNone(loaded)

    def test_last_kline_date(self):
        """应正确追踪最后K线日期"""
        code = '000003'
        data = [
            {'day': '2024-01-02', 'open': '10', 'high': '11',
             'low': '9', 'close': '10', 'volume': '100'},
            {'day': '2024-06-30', 'open': '12', 'high': '13',
             'low': '11', 'close': '12', 'volume': '200'},
        ]
        db.save_klines(code, data)
        last = db.get_last_kline_date(code)
        self.assertEqual(last, '2024-06-30')

    def test_save_and_load_dividends(self):
        """保存分红后应能加载"""
        code = '000004'
        divs = [
            {'ex_date': '2024-06-15', 'date': '2024-06-01',
             'dividend_10': 5.0, 'dividend_per_share': 0.5,
             'bonus_share': 0, 'transfer_share': 0, 'status': '实施'},
        ]
        db.save_dividends(code, divs)
        loaded = db.load_dividends(code)
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]['dividend_10'], 5.0)
        self.assertEqual(loaded[0]['dividend_per_share'], 0.5)

    def test_dividends_sorted_desc(self):
        """分红应按除权日降序排列"""
        code = '000005'
        divs = [
            {'ex_date': '2023-06-01', 'date': '2023-05-01',
             'dividend_10': 3.0, 'dividend_per_share': 0.3,
             'bonus_share': 0, 'transfer_share': 0, 'status': '实施'},
            {'ex_date': '2024-06-01', 'date': '2024-05-01',
             'dividend_10': 5.0, 'dividend_per_share': 0.5,
             'bonus_share': 0, 'transfer_share': 0, 'status': '实施'},
        ]
        db.save_dividends(code, divs)
        loaded = db.load_dividends(code)
        self.assertEqual(loaded[0]['ex_date'], '2024-06-01')  # 最新的在前

    def test_save_stock_name(self):
        """保存股票名称"""
        db.save_stock_name('000006', '测试股票')
        conn = db.get_db()
        row = conn.execute("SELECT name FROM stocks WHERE code='000006'").fetchone()
        conn.close()
        self.assertEqual(row[0], '测试股票')

    def test_save_stock_name_update(self):
        """更新股票名称"""
        db.save_stock_name('000007', '旧名称')
        db.save_stock_name('000007', '新名称')
        conn = db.get_db()
        row = conn.execute("SELECT name FROM stocks WHERE code='000007'").fetchone()
        conn.close()
        self.assertEqual(row[0], '新名称')

    def test_load_stock_name(self):
        """已缓存的股票名称应可直接读取"""
        db.save_stock_name('000008', '缓存名称')
        self.assertEqual(db.load_stock_name('000008'), '缓存名称')
        self.assertIsNone(db.load_stock_name('999999'))

    def test_baseline_job_lifecycle(self):
        run_id = 'mainboard_20260719'
        db.create_baseline_run(run_id, 'cn_sh_sz_mainboard', 'test_v1', '2026-07-19')
        db.save_baseline_universe(run_id, [
            {'code': '600000', 'name': '浦发银行', 'exchange': 'SH',
             'list_date': '1999-11-10'},
            {'code': '000001', 'name': '平安银行', 'exchange': 'SZ',
             'list_date': '1991-04-03'},
        ])

        claimed = db.claim_baseline_jobs(run_id, 1)
        self.assertEqual(len(claimed), 1)
        self.assertEqual(claimed[0]['attempts'], 1)
        db.complete_baseline_job(
            run_id, claimed[0]['code'], 'eligible', 300, '2025-01-01',
            '2026-07-19', trade_count=np.int64(5), win_count=np.int64(3),
            win_rate_pct=np.float64(60.0),
            price_return_pct=12.0, composite_score=70.0, signal='偏多')

        claimed = db.claim_baseline_jobs(run_id, 1)
        db.fail_baseline_job(run_id, claimed[0]['code'], 'temporary error', 0)
        claimed = db.claim_baseline_jobs(run_id, 1)
        self.assertEqual(claimed[0]['attempts'], 2)
        db.complete_baseline_job(
            run_id, claimed[0]['code'], 'insufficient_history', 30,
            '2026-06-01', '2026-07-19')

        progress = db.get_baseline_progress(run_id)
        self.assertEqual(progress['status'], 'completed')
        self.assertEqual(progress['total_count'], 2)
        self.assertEqual(progress['success_count'], 2)
        self.assertEqual(progress['failed_count'], 0)
        self.assertEqual(progress['eligible_count'], 1)
        self.assertEqual(progress['insufficient_count'], 1)
        self.assertEqual(progress['stale_count'], 0)
        self.assertEqual(progress['fetch_failed_count'], 0)
        self.assertEqual(progress['total_trade_count'], 5)
        self.assertEqual(progress['total_win_count'], 3)
        self.assertAlmostEqual(progress['global_trade_win_rate_pct'], 60.0)
        self.assertEqual(db.list_baseline_jobs(run_id, 'success')[0]['win_count'], 3)
        self.assertEqual(len(db.list_baseline_jobs(run_id, 'success')), 2)
        self.assertEqual(db.list_baseline_jobs(run_id, 'failed'), [])

        self.assertEqual(db.requeue_baseline_jobs(run_id), 2)
        progress = db.get_baseline_progress(run_id)
        self.assertEqual(progress['success_count'], 0)
        self.assertEqual(progress['pending_count'], 2)

        db.supersede_baseline_run(run_id)
        self.assertNotEqual(db.find_latest_open_baseline_run('cn_sh_sz_mainboard'), run_id)
        self.assertIsNone(db.find_latest_baseline_run('cn_sh_sz_mainboard'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
