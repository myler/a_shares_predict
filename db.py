#!/usr/bin/env python3
"""SQLite本地缓存：K线和分红数据的增量存储"""

import sqlite3
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(ROOT, 'stock_cache.db')


def get_db():
    """获取数据库连接，自动建表"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _init_tables(conn)
    return conn


def _init_tables(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS stocks (
            code       TEXT PRIMARY KEY,
            name       TEXT,
            last_kline_date TEXT,
            last_div_date   TEXT
        );
        CREATE TABLE IF NOT EXISTS klines (
            code   TEXT,
            date   TEXT,
            open   REAL,
            high   REAL,
            low    REAL,
            close  REAL,
            volume REAL,
            PRIMARY KEY (code, date)
        );
        CREATE TABLE IF NOT EXISTS dividends (
            code             TEXT,
            ex_date          TEXT,
            announce_date    TEXT,
            dividend_10      REAL,
            dividend_per_share REAL,
            bonus_share      REAL,
            transfer_share   REAL,
            status           TEXT,
            PRIMARY KEY (code, ex_date)
        );
    """)


# ── K线操作 ──

def get_kline_dates(code):
    """返回库中已有的K线日期列表（升序）"""
    conn = get_db()
    rows = conn.execute(
        "SELECT date FROM klines WHERE code=? ORDER BY date", (code,)
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def save_klines(code, kline_list):
    """批量写入K线，已存在的跳过。
    kline_list: [{'day':'2024-01-01', 'open':'10.0', 'high':'11.0', ...}, ...]
    """
    if not kline_list:
        return
    conn = get_db()
    conn.executemany(
        """INSERT OR IGNORE INTO klines (code,date,open,high,low,close,volume)
           VALUES (?,?,?,?,?,?,?)""",
        [(code, d['day'], float(d['open']), float(d['high']),
          float(d['low']), float(d['close']), float(d['volume']))
         for d in kline_list]
    )
    # 更新最后日期
    last_date = max(d['day'] for d in kline_list)
    conn.execute(
        "INSERT INTO stocks(code,last_kline_date) VALUES (?,?) "
        "ON CONFLICT(code) DO UPDATE SET last_kline_date=MAX(last_kline_date,?)",
        (code, last_date, last_date)
    )
    conn.commit()
    conn.close()


def load_klines(code):
    """从DB加载完整K线，返回 [{'day':..., 'open':..., ...}, ...] 按日期升序"""
    conn = get_db()
    rows = conn.execute(
        "SELECT date,open,high,low,close,volume FROM klines "
        "WHERE code=? ORDER BY date", (code,)
    ).fetchall()
    conn.close()
    if not rows:
        return None
    return [
        {'day': r[0], 'open': str(r[1]), 'high': str(r[2]),
         'low': str(r[3]), 'close': str(r[4]), 'volume': str(r[5])}
        for r in rows
    ]


def get_last_kline_date(code):
    """获取库中某股最新K线日期"""
    conn = get_db()
    row = conn.execute(
        "SELECT last_kline_date FROM stocks WHERE code=?", (code,)
    ).fetchone()
    conn.close()
    return row[0] if row else None


# ── 分红操作 ──

def save_dividends(code, div_list):
    """批量写入分红，已存在的跳过。
    div_list: [{ex_date, announce_date, dividend_10, dividend_per_share, bonus_share, transfer_share, status}, ...]
    """
    if not div_list:
        return
    conn = get_db()
    conn.executemany(
        """INSERT OR IGNORE INTO dividends
           (code,ex_date,announce_date,dividend_10,dividend_per_share,bonus_share,transfer_share,status)
           VALUES (?,?,?,?,?,?,?,?)""",
        [(code, d.get('ex_date',''), d.get('date',''), d['dividend_10'],
          d['dividend_per_share'], d['bonus_share'], d['transfer_share'], d['status'])
         for d in div_list if d.get('ex_date')]
    )
    # 更新最后日期
    last_div = max((d.get('ex_date','') for d in div_list if d.get('ex_date')), default=None)
    if last_div:
        conn.execute(
            "INSERT INTO stocks(code,last_div_date) VALUES (?,?) "
            "ON CONFLICT(code) DO UPDATE SET last_div_date=MAX(last_div_date,?)",
            (code, last_div, last_div)
        )
    conn.commit()
    conn.close()


def load_dividends(code):
    """从DB加载分红，返回与 fetch_dividends 相同格式的列表"""
    conn = get_db()
    rows = conn.execute(
        "SELECT ex_date,announce_date,dividend_10,dividend_per_share,"
        "bonus_share,transfer_share,status "
        "FROM dividends WHERE code=? ORDER BY ex_date DESC", (code,)
    ).fetchall()
    conn.close()
    if not rows:
        return None
    return [
        {'ex_date': r[0], 'date': r[1], 'dividend_10': r[2],
         'dividend_per_share': r[3], 'bonus_share': r[4],
         'transfer_share': r[5], 'status': r[6]}
        for r in rows
    ]


# ── 股票名称 ──

def save_stock_name(code, name):
    conn = get_db()
    conn.execute(
        "INSERT INTO stocks(code,name) VALUES (?,?) "
        "ON CONFLICT(code) DO UPDATE SET name=?", (code, name, name)
    )
    conn.commit()
    conn.close()
