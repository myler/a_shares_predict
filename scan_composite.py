#!/usr/bin/env python3
"""横截面多因子海选（新版，替换 S* 离散排名）：
算10个技术因子 → 横截面 percentile rank → ICIR 加权合成 → PE/主板过滤 → top N。

用法: python3 scan_composite.py [-n N] [--max-pe 100] [--min-price 5] [--out file.json]
"""
import sys, os, argparse, json, sqlite3, urllib.request, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factors import compute_technical_factors, composite_scores, FACTOR_WEIGHTS

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_cache.db")
MAINBOARD = ("600", "601", "603", "605", "000", "001", "002", "003")


def is_mainboard(c):
    return c.startswith(MAINBOARD)


def fetch_quotes(codes):
    out = {}
    for i in range(0, len(codes), 50):
        batch = codes[i:i + 50]
        q = ",".join(f"sz{c}" if c.startswith(("0", "3")) else f"sh{c}" for c in batch)
        try:
            req = urllib.request.Request(
                f"https://qt.gtimg.cn/q={q}", headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=10).read().decode("gbk")
        except Exception:
            continue
        for line in raw.strip().split("\n"):
            if "~" not in line:
                continue
            p = line.split("~")
            if len(p) < 46:
                continue

            def f(x):
                try:
                    return float(x)
                except Exception:
                    return 0.0
            out[p[2]] = {"name": p[1], "price": f(p[3]), "pct": f(p[32]),
                         "pe": f(p[39]), "mcap": f(p[45])}
        time.sleep(0.1)
    return out


def run_scan(n=20, max_pe=100, min_price=5):
    """跑横截面多因子海选，返回 (top, stats)。

    top: [{code,name,close,score,pe,mcap,pct,date}] 按 score 降序
    stats: {total_codes, passed, elapsed}
    """
    t0 = time.time()
    db = sqlite3.connect(DB)
    cur = db.cursor()
    codes = [r[0] for r in cur.execute(
        "SELECT DISTINCT code FROM klines").fetchall()]
    codes = [c for c in codes if is_mainboard(c)]

    rows = []
    for idx, code in enumerate(codes, 1):
        rec = cur.execute(
            "SELECT date,open,high,low,close,volume FROM klines "
            "WHERE code=? ORDER BY date", (code,)).fetchall()
        if len(rec) < 300:
            continue
        df = pd.DataFrame(rec, columns=["date", "open", "high", "low",
                                        "close", "volume"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df = df.set_index("date")
        f = compute_technical_factors(df)
        last = f.iloc[-1]
        rows.append((code, df.index[-1], last))

    codes2 = [r[0] for r in rows]
    dates = {r[0]: r[1] for r in rows}
    ff = pd.DataFrame({c: {r[0]: r[2][c] for r in rows} for c in FACTOR_WEIGHTS})
    ff = ff.reindex(codes2)
    score = composite_scores(ff)

    quotes = fetch_quotes(codes2)

    results = []
    for code in codes2:
        q = quotes.get(code)
        if not q or q["price"] <= 0:
            continue
        if q["pe"] <= 0 or q["pe"] > max_pe:
            continue
        if q["price"] < min_price:
            continue
        results.append({
            "code": code, "name": q["name"], "close": q["price"],
            "score": round(float(score[code]), 1),
            "pe": q["pe"], "mcap": q["mcap"], "pct": q["pct"],
            "date": dates[code],
        })

    results.sort(key=lambda r: -r["score"])
    top = results[:n]
    stats = {
        "total_codes": len(codes2),
        "passed": len(results),
        "elapsed": time.time() - t0,
    }
    return top, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--max-pe", type=float, default=100)
    ap.add_argument("--min-price", type=float, default=5)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    top, stats = run_scan(args.n, args.max_pe, args.min_price)

    print("\n" + "=" * 64)
    print(f"  横截面多因子海选 Top {len(top)}（低波+均值回归+反转）")
    print("=" * 64)
    for i, r in enumerate(top, 1):
        print(f"{i:2d}. {r['name']}({r['code']}) 得分{r['score']} 收{r['close']} "
              f"PE={r['pe']:.0f} 市值{r['mcap']:.0f}亿")

    print(f"\n共 {stats['passed']} 只通过过滤（{stats['total_codes']} 只有效因子）| "
          f"耗时 {stats['elapsed']/60:.1f}分钟")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(top, fh, ensure_ascii=False, indent=2)
        print(f"结果已写 {args.out}")


if __name__ == "__main__":
    main()
