#!/usr/bin/env python3
"""抓取主板全量股票当前总股本（亿股），存 shares.json。总股本 = 总市值/股价。"""
import sys, os, json, sqlite3, urllib.request, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_cache.db")
MAINBOARD = ("600", "601", "603", "605", "000", "001", "002", "003")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shares.json")


def fetch(codes):
    out = {}
    for i in range(0, len(codes), 50):
        batch = codes[i:i + 50]
        q = ",".join(f"sz{c}" if c.startswith(("0", "3")) else f"sh{c}" for c in batch)
        try:
            req = urllib.request.Request(
                f"https://qt.gtimg.cn/q={q}", headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=10).read().decode("gbk")
        except Exception as e:
            print(f"  batch err {e}", file=sys.stderr)
            continue
        for line in raw.strip().split("\n"):
            if "~" not in line:
                continue
            p = line.split("~")
            if len(p) < 46:
                continue
            try:
                price = float(p[3]); mcap = float(p[45])
            except Exception:
                continue
            if price <= 0 or mcap <= 0:
                continue
            out[p[2]] = mcap / price  # 亿元/元 = 亿股
        time.sleep(0.1)
    return out


def main():
    db = sqlite3.connect(DB)
    cur = db.cursor()
    codes = [r[0] for r in cur.execute(
        "SELECT DISTINCT code FROM klines").fetchall()]
    codes = [c for c in codes if c.startswith(MAINBOARD)]
    print(f"抓总股本 {len(codes)} 只...")
    shares = fetch(codes)
    with open(OUT, "w") as fh:
        json.dump(shares, fh)
    print(f"抓到 {len(shares)} 只 → {OUT}")


if __name__ == "__main__":
    main()
