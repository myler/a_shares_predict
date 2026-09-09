#!/usr/bin/env python3
"""回填 adj_factors 表：从腾讯拉前复权(qfq)K线，算复权因子 = qfq_close/raw_close。

只存 factor != 1.0 的日期（有分红的股票才有非1因子；无分红股票默认1.0，不存行）。

用法: python3 build_adj_factors.py [--limit N]
"""
import sys, os, argparse, sqlite3, urllib.request, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_cache.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS adj_factors (
  code TEXT NOT NULL,
  date TEXT NOT NULL,
  factor REAL NOT NULL,
  PRIMARY KEY (code, date)
);
"""


def fetch_qfq(sc):
    """返回 {date: qfq_close}"""
    url = (f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
           f"?param={sc},day,,,2500,qfq")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=15).read().decode("utf-8")
    d = json.loads(raw)
    data = d.get("data", {}).get(sc, {})
    qfq = data.get("qfqday") or []
    return {row[0]: float(row[2]) for row in qfq if len(row) >= 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    db = sqlite3.connect(DB)
    cur = db.cursor()
    cur.execute(SCHEMA)

    codes = [r[0] for r in cur.execute(
        "SELECT DISTINCT code FROM klines").fetchall()]
    if args.limit:
        codes = codes[:args.limit]

    print(f"回填 adj_factors | 共 {len(codes)} 只")
    t0 = time.time()
    done = noadj = fail = 0
    for idx, code in enumerate(codes, 1):
        sc = f"sz{code}" if code.startswith(("0", "3")) else f"sh{code}"
        try:
            qfq = fetch_qfq(sc)
        except Exception as e:
            fail += 1
            if idx % 200 == 0:
                print(f"  ⚠ {code}: {e}", file=sys.stderr)
            continue
        if not qfq:
            fail += 1
            continue

        raw = {r[0]: r[1] for r in cur.execute(
            "SELECT date, close FROM klines WHERE code=?", (code,)).fetchall()}

        stored = 0
        for d, qc in qfq.items():
            pc = raw.get(d)
            if not pc:
                continue
            factor = qc / float(pc)
            if abs(factor - 1.0) > 1e-6:
                cur.execute("INSERT OR REPLACE INTO adj_factors VALUES (?,?,?)",
                            (code, d, round(factor, 6)))
                stored += 1
        if stored:
            done += 1
        else:
            noadj += 1
        if idx % 200 == 0:
            db.commit()
            print(f"  进度 {idx}/{len(codes)} | 有因子{done} 无因子{noadj} 失败{fail} | {time.time()-t0:.0f}s")

    db.commit()
    db.close()
    print(f"完成: 有因子{done}只 无因子{noadj}只 失败{fail}只 | 耗时 {(time.time()-t0)/60:.1f}分钟")


if __name__ == "__main__":
    main()
