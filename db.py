#!/usr/bin/env python3
"""SQLite本地缓存：K线和分红数据的增量存储"""

import sqlite3
import os
from datetime import datetime, timedelta, timezone

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
        CREATE TABLE IF NOT EXISTS baseline_runs (
            run_id            TEXT PRIMARY KEY,
            universe_name     TEXT NOT NULL,
            strategy_version  TEXT NOT NULL,
            as_of_date        TEXT NOT NULL,
            status            TEXT NOT NULL,
            created_at        TEXT NOT NULL,
            updated_at        TEXT NOT NULL,
            completed_at      TEXT,
            total_count       INTEGER NOT NULL DEFAULT 0,
            success_count     INTEGER NOT NULL DEFAULT 0,
            failed_count      INTEGER NOT NULL DEFAULT 0,
            pending_count     INTEGER NOT NULL DEFAULT 0,
            running_count     INTEGER NOT NULL DEFAULT 0,
            eligible_count    INTEGER NOT NULL DEFAULT 0,
            insufficient_count INTEGER NOT NULL DEFAULT 0,
            stale_count       INTEGER NOT NULL DEFAULT 0,
            fetch_failed_count INTEGER NOT NULL DEFAULT 0,
            mean_stock_win_rate_pct REAL,
            mean_stock_price_return_pct REAL,
            total_trade_count INTEGER NOT NULL DEFAULT 0,
            total_win_count   INTEGER NOT NULL DEFAULT 0,
            global_trade_win_rate_pct REAL
        );
        CREATE TABLE IF NOT EXISTS baseline_jobs (
            run_id            TEXT NOT NULL,
            code              TEXT NOT NULL,
            name              TEXT,
            exchange          TEXT,
            list_date         TEXT,
            status            TEXT NOT NULL DEFAULT 'pending',
            attempts          INTEGER NOT NULL DEFAULT 0,
            next_retry_at     TEXT,
            last_attempt_at   TEXT,
            last_error        TEXT,
            kline_count       INTEGER,
            first_kline_date  TEXT,
            last_kline_date   TEXT,
            result_status     TEXT,
            trade_count       INTEGER,
            win_count         INTEGER,
            win_rate_pct      REAL,
            price_return_pct  REAL,
            composite_score   REAL,
            signal            TEXT,
            updated_at        TEXT NOT NULL,
            completed_at      TEXT,
            PRIMARY KEY (run_id, code),
            FOREIGN KEY (run_id) REFERENCES baseline_runs(run_id)
        );
        CREATE INDEX IF NOT EXISTS idx_baseline_jobs_claim
            ON baseline_jobs(run_id, status, next_retry_at, attempts, code);
    """)
    _ensure_baseline_columns(conn)


def _ensure_baseline_columns(conn):
    """兼容已创建过早期基线表的缓存数据库。"""
    expected = {
        'baseline_runs': {
            'stale_count': 'INTEGER NOT NULL DEFAULT 0',
            'fetch_failed_count': 'INTEGER NOT NULL DEFAULT 0',
            'mean_stock_win_rate_pct': 'REAL',
            'mean_stock_price_return_pct': 'REAL',
            'total_trade_count': 'INTEGER NOT NULL DEFAULT 0',
            'total_win_count': 'INTEGER NOT NULL DEFAULT 0',
            'global_trade_win_rate_pct': 'REAL',
        },
        'baseline_jobs': {
            'win_count': 'INTEGER',
        },
    }
    for table, columns in expected.items():
        existing = {
            row[1] for row in conn.execute(f'PRAGMA table_info({table})')
        }
        for name, definition in columns.items():
            if name not in existing:
                conn.execute(f'ALTER TABLE {table} ADD COLUMN {name} {definition}')


def _utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        '+00:00', 'Z')


def _utc_after(seconds):
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).replace(
        microsecond=0).isoformat().replace('+00:00', 'Z')


# ── K线操作 ──

def get_kline_dates(code):
    """返回库中已有的K线日期列表（升序）"""
    conn = get_db()
    rows = conn.execute(
        "SELECT date FROM klines WHERE code=? ORDER BY date", (code,)
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def save_klines(code, kline_list, replace=False):
    """批量写入 K 线；replace=True 时覆盖同日旧缓存。
    kline_list: [{'day':'2024-01-01', 'open':'10.0', 'high':'11.0', ...}, ...]
    """
    if not kline_list:
        return
    conn = get_db()
    if replace:
        conn.execute("DELETE FROM klines WHERE code=?", (code,))
    statement = (
        """INSERT INTO klines (code,date,open,high,low,close,volume)
           VALUES (?,?,?,?,?,?,?)
           ON CONFLICT(code,date) DO UPDATE SET
             open=excluded.open,high=excluded.high,low=excluded.low,
             close=excluded.close,volume=excluded.volume"""
        if replace else
        """INSERT OR IGNORE INTO klines (code,date,open,high,low,close,volume)
           VALUES (?,?,?,?,?,?,?)"""
    )
    conn.executemany(
        statement,
        [(code, d['day'], float(d['open']), float(d['high']),
          float(d['low']), float(d['close']), float(d['volume']))
         for d in kline_list]
    )
    # 更新最后日期
    last_date = max(d['day'] for d in kline_list)
    if replace:
        conn.execute(
            "INSERT INTO stocks(code,last_kline_date) VALUES (?,?) "
            "ON CONFLICT(code) DO UPDATE SET last_kline_date=excluded.last_kline_date",
            (code, last_date),
        )
    else:
        conn.execute(
            "INSERT INTO stocks(code,last_kline_date) VALUES (?,?) "
            "ON CONFLICT(code) DO UPDATE SET last_kline_date=MAX(last_kline_date,?)",
            (code, last_date, last_date),
        )
    conn.commit()
    conn.close()


def load_klines(code, end_date=None):
    """从DB加载K线，返回按日期升序的数据；end_date 可锁定历史快照。"""
    conn = get_db()
    query = (
        "SELECT date,open,high,low,close,volume FROM klines "
        "WHERE code=?"
    )
    params = [code]
    if end_date is not None:
        query += " AND date<=?"
        params.append(end_date)
    rows = conn.execute(query + " ORDER BY date", params).fetchall()
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


def load_stock_name(code):
    """读取已缓存的股票名称；未缓存时返回 None。"""
    conn = get_db()
    row = conn.execute(
        "SELECT name FROM stocks WHERE code=?", (code,)
    ).fetchone()
    conn.close()
    return row[0] if row and row[0] else None


# ── 全量基线任务 ──

def create_baseline_run(run_id, universe_name, strategy_version, as_of_date):
    """创建可恢复的基线运行；同一 run_id 已存在时保留原快照。"""
    now = _utc_now()
    conn = get_db()
    conn.execute(
        """INSERT OR IGNORE INTO baseline_runs
           (run_id,universe_name,strategy_version,as_of_date,status,created_at,updated_at)
           VALUES (?,?,?,?,?,?,?)""",
        (run_id, universe_name, strategy_version, as_of_date, 'running', now, now),
    )
    conn.commit()
    row = conn.execute(
        "SELECT run_id,universe_name,strategy_version,as_of_date,status "
        "FROM baseline_runs WHERE run_id=?", (run_id,)
    ).fetchone()
    conn.close()
    return {
        'run_id': row[0], 'universe_name': row[1],
        'strategy_version': row[2], 'as_of_date': row[3], 'status': row[4],
    }


def find_latest_open_baseline_run(universe_name):
    """返回最近一个未完成基线运行，便于后续命令断点续跑。"""
    conn = get_db()
    row = conn.execute(
        """SELECT run_id FROM baseline_runs
           WHERE universe_name=? AND status='running'
           ORDER BY created_at DESC LIMIT 1""",
        (universe_name,),
    ).fetchone()
    conn.close()
    return row[0] if row else None


def find_latest_baseline_run(universe_name):
    """返回最近一次非 superseded 基线，供完成后查询结果使用。"""
    conn = get_db()
    row = conn.execute(
        """SELECT run_id FROM baseline_runs
           WHERE universe_name=? AND status!='superseded'
           ORDER BY created_at DESC LIMIT 1""",
        (universe_name,),
    ).fetchone()
    conn.close()
    return row[0] if row else None


def supersede_baseline_run(run_id, reason='universe replaced'):
    """保留旧任务审计记录，但不再把其作为可恢复的活动基线。"""
    now = _utc_now()
    conn = get_db()
    conn.execute(
        """UPDATE baseline_runs
           SET status='superseded',updated_at=?,completed_at=?
           WHERE run_id=?""",
        (now, now, run_id),
    )
    conn.commit()
    conn.close()


def _refresh_baseline_run_conn(conn, run_id, now):
    row = conn.execute(
        """SELECT
             COUNT(*),
             SUM(CASE WHEN status='success' THEN 1 ELSE 0 END),
             SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END),
             SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END),
             SUM(CASE WHEN status='running' THEN 1 ELSE 0 END),
             SUM(CASE WHEN result_status='eligible' THEN 1 ELSE 0 END),
             SUM(CASE WHEN result_status='insufficient_history' THEN 1 ELSE 0 END),
             SUM(CASE WHEN result_status='stale_history' THEN 1 ELSE 0 END),
             SUM(CASE WHEN result_status='fetch_failed' THEN 1 ELSE 0 END),
             AVG(CASE WHEN status='success' AND result_status='eligible'
                      THEN win_rate_pct END),
             AVG(CASE WHEN status='success' AND result_status='eligible'
                      THEN price_return_pct END),
             SUM(CASE WHEN status='success' THEN COALESCE(trade_count, 0) ELSE 0 END),
             SUM(CASE WHEN status='success' THEN COALESCE(win_count, 0) ELSE 0 END)
           FROM baseline_jobs WHERE run_id=?""",
        (run_id,),
    ).fetchone()
    total, success, failed, pending, running, eligible, insufficient, stale, fetch_failed, mean_win, mean_return, total_trades, total_wins = [
        value or 0 for value in row
    ]
    status = 'completed' if total and success == total else 'running'
    completed_at = now if status == 'completed' else None
    global_win_rate = total_wins / total_trades * 100 if total_trades else 0.0
    conn.execute(
        """UPDATE baseline_runs
           SET status=?,updated_at=?,completed_at=?,total_count=?,success_count=?,
               failed_count=?,pending_count=?,running_count=?,eligible_count=?,
               insufficient_count=?,stale_count=?,
               fetch_failed_count=?,
               mean_stock_win_rate_pct=?,
               mean_stock_price_return_pct=?,total_trade_count=?,total_win_count=?,
               global_trade_win_rate_pct=?
           WHERE run_id=?""",
        (status, now, completed_at, total, success, failed, pending, running,
         eligible, insufficient, stale, fetch_failed, mean_win, mean_return,
         total_trades, total_wins,
         global_win_rate, run_id),
    )


def save_baseline_universe(run_id, stocks):
    """持久化一次冻结的股票池；已有任务不会被之后的名单刷新覆盖。"""
    now = _utc_now()
    conn = get_db()
    try:
        conn.execute('BEGIN IMMEDIATE')
        for stock in stocks:
            code = str(stock['code']).zfill(6)
            name = stock.get('name') or code
            conn.execute(
                """INSERT INTO stocks(code,name) VALUES (?,?)
                   ON CONFLICT(code) DO UPDATE SET name=excluded.name""",
                (code, name),
            )
            conn.execute(
                """INSERT OR IGNORE INTO baseline_jobs
                   (run_id,code,name,exchange,list_date,status,updated_at)
                   VALUES (?,?,?,?,?,'pending',?)""",
                (run_id, code, name, stock.get('exchange'), stock.get('list_date'), now),
            )
        _refresh_baseline_run_conn(conn, run_id, now)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def requeue_baseline_jobs(run_id, status='success'):
    """将指定状态的任务重新放回待处理池，用于统一数据口径后的重跑。"""
    now = _utc_now()
    conn = get_db()
    cursor = conn.execute(
        """UPDATE baseline_jobs
           SET status='pending',next_retry_at=NULL,last_error=NULL,
               kline_count=NULL,first_kline_date=NULL,last_kline_date=NULL,
               result_status=NULL,trade_count=NULL,win_count=NULL,
               win_rate_pct=NULL,price_return_pct=NULL,composite_score=NULL,
               signal=NULL,completed_at=NULL,updated_at=?
           WHERE run_id=? AND status=?""",
        (now, run_id, status),
    )
    _refresh_baseline_run_conn(conn, run_id, now)
    conn.commit()
    conn.close()
    return cursor.rowcount


def recover_running_baseline_jobs(run_id):
    """将异常退出时遗留的运行中任务放回失败重试池。"""
    now = _utc_now()
    conn = get_db()
    cursor = conn.execute(
        """UPDATE baseline_jobs
           SET status='failed',next_retry_at=?,last_error=COALESCE(last_error,?),
               updated_at=?
           WHERE run_id=? AND status='running'""",
        (now, 'worker interrupted before completion', now, run_id),
    )
    _refresh_baseline_run_conn(conn, run_id, now)
    conn.commit()
    conn.close()
    return cursor.rowcount


def claim_baseline_jobs(run_id, limit):
    """原子领取待处理或到期失败任务，避免多 worker 重复处理。"""
    now = _utc_now()
    conn = get_db()
    conn.execute('BEGIN IMMEDIATE')
    rows = conn.execute(
        """SELECT code,name,exchange,list_date,attempts
           FROM baseline_jobs
           WHERE run_id=? AND (
             status='pending' OR (status='failed' AND
             (next_retry_at IS NULL OR next_retry_at<=?)))
                     ORDER BY CASE status WHEN 'failed' THEN 0 ELSE 1 END,
                                        next_retry_at, attempts, code
           LIMIT ?""",
        (run_id, now, limit),
    ).fetchall()
    for code, _, _, _, _ in rows:
        conn.execute(
            """UPDATE baseline_jobs
               SET status='running',attempts=attempts+1,last_attempt_at=?,
                   updated_at=?,next_retry_at=NULL
               WHERE run_id=? AND code=?""",
            (now, now, run_id, code),
        )
    _refresh_baseline_run_conn(conn, run_id, now)
    conn.commit()
    conn.close()
    return [
        {'code': row[0], 'name': row[1], 'exchange': row[2],
         'list_date': row[3], 'attempts': row[4] + 1}
        for row in rows
    ]


def complete_baseline_job(run_id, code, result_status, kline_count,
                          first_kline_date, last_kline_date, trade_count=None,
                          win_count=None, win_rate_pct=None, price_return_pct=None,
                          composite_score=None, signal=None, last_error=None):
    """标记一个股票为已成功抓取并持久化其基线回测摘要。"""
    # NumPy 标量会被 sqlite3 当作 BLOB 绑定，先转换为原生 Python 数值。
    trade_count = int(trade_count) if trade_count is not None else None
    win_count = int(win_count) if win_count is not None else None
    win_rate_pct = float(win_rate_pct) if win_rate_pct is not None else None
    price_return_pct = float(price_return_pct) if price_return_pct is not None else None
    composite_score = float(composite_score) if composite_score is not None else None
    now = _utc_now()
    conn = get_db()
    conn.execute(
        """UPDATE baseline_jobs
              SET status='success',next_retry_at=NULL,last_error=?,kline_count=?,
               first_kline_date=?,last_kline_date=?,result_status=?,trade_count=?,
               win_count=?,win_rate_pct=?,price_return_pct=?,composite_score=?,signal=?,
               updated_at=?,completed_at=?
           WHERE run_id=? AND code=?""",
        (last_error, kline_count, first_kline_date, last_kline_date, result_status,
         trade_count, win_count, win_rate_pct, price_return_pct, composite_score, signal,
         now, now, run_id, code),
    )
    _refresh_baseline_run_conn(conn, run_id, now)
    conn.commit()
    conn.close()


def fail_baseline_job(run_id, code, error_message, retry_after_seconds):
    """记录失败并安排指数退避后的重试，不会丢弃失败股票。"""
    now = _utc_now()
    retry_at = _utc_after(retry_after_seconds)
    conn = get_db()
    conn.execute(
        """UPDATE baseline_jobs
           SET status='failed',last_error=?,next_retry_at=?,updated_at=?
           WHERE run_id=? AND code=?""",
        (str(error_message)[:1000], retry_at, now, run_id, code),
    )
    _refresh_baseline_run_conn(conn, run_id, now)
    conn.commit()
    conn.close()


def get_baseline_progress(run_id):
    """返回成功池、失败池和待处理池的持久化进度。"""
    conn = get_db()
    row = conn.execute(
        """SELECT run_id,universe_name,strategy_version,as_of_date,status,
                  total_count,success_count,failed_count,pending_count,
                  running_count,eligible_count,insufficient_count,updated_at,
                  stale_count,fetch_failed_count,
                  completed_at,mean_stock_win_rate_pct,
                  mean_stock_price_return_pct,total_trade_count,total_win_count,
                  global_trade_win_rate_pct
           FROM baseline_runs WHERE run_id=?""",
        (run_id,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    keys = ('run_id', 'universe_name', 'strategy_version', 'as_of_date', 'status',
            'total_count', 'success_count', 'failed_count', 'pending_count',
            'running_count', 'eligible_count', 'insufficient_count', 'updated_at',
            'stale_count', 'fetch_failed_count',
            'completed_at', 'mean_stock_win_rate_pct',
            'mean_stock_price_return_pct', 'total_trade_count', 'total_win_count',
            'global_trade_win_rate_pct')
    return dict(zip(keys, row))


def get_next_baseline_retry_at(run_id):
    conn = get_db()
    row = conn.execute(
        """SELECT MIN(next_retry_at) FROM baseline_jobs
           WHERE run_id=? AND status='failed'""",
        (run_id,),
    ).fetchone()
    conn.close()
    return row[0] if row else None


def list_baseline_jobs(run_id, status=None, limit=None):
    """列出基线成功池、失败池或全部任务，供进度查看和后续分析使用。"""
    query = (
        "SELECT code,name,exchange,list_date,status,attempts,next_retry_at,"
        "last_error,kline_count,result_status,trade_count,win_count,win_rate_pct,"
        "price_return_pct,composite_score,signal,updated_at "
        "FROM baseline_jobs WHERE run_id=?"
    )
    params = [run_id]
    if status:
        query += " AND status=?"
        params.append(status)
    query += " ORDER BY code"
    if limit is not None:
        query += " LIMIT ?"
        params.append(int(limit))

    conn = get_db()
    rows = conn.execute(query, params).fetchall()
    conn.close()
    keys = ('code', 'name', 'exchange', 'list_date', 'status', 'attempts',
            'next_retry_at', 'last_error', 'kline_count', 'result_status',
            'trade_count', 'win_count', 'win_rate_pct', 'price_return_pct',
            'composite_score', 'signal', 'updated_at')
    return [dict(zip(keys, row)) for row in rows]
