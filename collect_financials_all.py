#!/usr/bin/env python3
"""全市场财务采集（利润表 → financials 表），供基本面因子回测。

用法: /home/myl/ashare-mcp/.venv/bin/python3 collect_financials_all.py [--limit N]
"""
import sys, os, argparse, sqlite3, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_fundamentals import collect_financials, SCHEMAS

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_cache.db")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    db = sqlite3.connect(DB)
    cur = db.cursor()
    cur.execute(SCHEMAS["financials"])
    db.commit()

    codes = [r[0] for r in cur.execute(
        "SELECT DISTINCT code FROM klines").fetchall()]
    if args.limit:
        codes = codes[:args.limit]
    print(f"采集财务 {len(codes)} 只...")

    t0 = time.time()
    ok = fail = 0
    for idx, code in enumerate(codes, 1):
        try:
            n = collect_financials(cur, code)
            if n:
                ok += 1
            else:
                fail += 1
        except Exception as e:
            fail += 1
            if idx % 500 == 0:
                print(f"  ⚠ {code}: {e}", file=sys.stderr)
        if idx % 200 == 0:
            db.commit()
            el = time.time() - t0
            print(f"  进度 {idx}/{len(codes)} | 成功{ok} 失败{fail} | {el:.0f}s")

    db.commit()
    db.close()
    print(f"完成: 成功{ok} 失败{fail} | 耗时 {(time.time()-t0)/60:.1f}分钟")


if __name__ == "__main__":
    main()
