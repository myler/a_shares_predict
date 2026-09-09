#!/usr/bin/env python3
"""collect_lhb_resume.py: 龙虎榜续采（按月分段+超时+重试，不 DROP 表）。

用法（ashare-mcp venv）:
  /home/myl/ashare-mcp/.venv/bin/python3 collect_lhb_resume.py 2020-09 2026-09
"""
import os, sys, sqlite3, time, signal
import akshare as ak
import pandas as pd

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_cache.db")
MAINBOARD = ("600", "601", "603", "605", "000", "001", "002", "003")


def f(x):
    try:
        return float(x)
    except Exception:
        return None


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def collect_range(db, cur, start, end, timeout_s=25):
    """采集一个日期范围，带超时；返回条数或 None（失败）。"""
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout_s)
    try:
        df = ak.stock_lhb_detail_em(start_date=start, end_date=end)
    except Timeout:
        return None
    except Exception:
        return None
    finally:
        signal.alarm(0)

    n = 0
    for _, r in df.iterrows():
        code = str(r.get("代码", "")).zfill(6)
        if not code.startswith(MAINBOARD):
            continue
        cur.execute(
            "INSERT OR REPLACE INTO lhb VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (code, r.get("名称"), str(r.get("上榜日", ""))[:10], r.get("上榜原因"),
             r.get("解读"), f(r.get("收盘价")), f(r.get("涨跌幅")),
             f(r.get("龙虎榜净买额")), f(r.get("龙虎榜买入额")),
             f(r.get("龙虎榜卖出额")), f(r.get("龙虎榜成交额")),
             f(r.get("市场总成交额")), f(r.get("净买额占总成交比")),
             f(r.get("换手率")), f(r.get("流通市值")),
             f(r.get("上榜后1日")), f(r.get("上榜后2日")),
             f(r.get("上榜后5日")), f(r.get("上榜后10日"))))
        n += 1
    return n


def month_ranges(y1, m1, y2, m2):
    """生成 [月初, 月末] 的日期段列表。"""
    import calendar
    out = []
    y, m = y1, m1
    while (y, m) <= (y2, m2):
        last = calendar.monthrange(y, m)[1]
        out.append((f"{y}{m:02d}01", f"{y}{m:02d}{last:02d}"))
        m += 1
        if m > 12:
            y += 1
            m = 1
    return out


def main():
    start_ym, end_ym = sys.argv[1], sys.argv[2]
    y1, m1 = map(int, start_ym.split("-"))
    y2, m2 = map(int, end_ym.split("-"))
    ranges = month_ranges(y1, m1, y2, m2)

    db = sqlite3.connect(DB)
    cur = db.cursor()
    total = 0
    fails = []
    t0 = time.time()
    for start, end in ranges:
        n = None
        for attempt in range(3):
            n = collect_range(db, cur, start, end)
            if n is not None:
                break
            time.sleep(1.5)
        if n is None:
            fails.append(start[:6])
            print(f"  {start[:6]} 失败(3次重试)", flush=True)
        else:
            total += n
            db.commit()
            print(f"  {start[:6]} {n} 条 | 累计 {total} | {time.time()-t0:.0f}s", flush=True)
    db.close()
    print(f"\n续采完成: 新增 {total} 条 | 失败月份 {len(fails)}: {fails}")


if __name__ == "__main__":
    main()
