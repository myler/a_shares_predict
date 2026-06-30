"""db.py 单元测试 —— 验证SQLite缓存的增删改查"""

import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import sqlite3

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

    def test_deduplicate_klines(self):
        """重复保存不产生重复行"""
        code = '000002'
        data = [{'day': '2024-01-02', 'open': '10', 'high': '11',
                 'low': '9', 'close': '10', 'volume': '100'}]
        db.save_klines(code, data)
        db.save_klines(code, data)  # 重复保存
        loaded = db.load_klines(code)
        self.assertEqual(len(loaded), 1)

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


if __name__ == '__main__':
    unittest.main(verbosity=2)
