#!/usr/bin/env python3
"""全量主板海选：从 features 表(技术快照) + 腾讯行情(PE/市值) 读 → 过滤 → top N。

依赖：build_features.py 已回填 features 表（refresh_klines.py 每日刷新日线后由 cron 联动重算）。

用法: python3 scan_top_n.py [-n N] [--require-gate] [--min-score S] [--out file.json]
"""
import sys, os, argparse, json, sqlite3, urllib.request, time
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_cache.db")

MAINBOARD_PREFIX = ("600", "601", "603", "605", "000", "001", "002", "003")


def is_mainboard(code):
    return code.startswith(MAINBOARD_PREFIX)


def fetch_quotes(codes):
    """腾讯批量行情：{code: {name,price,pe,mcap,turn,vol_ratio,pct}}"""
    out = {}
    for i in range(0, len(codes), 50):
        batch = codes[i:i + 50]
        q = ",".join(f"sz{c}" if c.startswith(("0", "3")) else f"sh{c}" for c in batch)
        url = f"https://qt.gtimg.cn/q={q}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
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
            out[p[2]] = {
                "name": p[1], "price": f(p[3]), "pct": f(p[32]),
                "pe": f(p[39]), "mcap": f(p[45]), "turn": f(p[38]),
                "vol_ratio": f(p[46]),
            }
        time.sleep(0.1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--require-gate", action="store_true")
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--max-days-stale", type=int, default=10,
                    help="features 最新日期距今超过 N 天则视为陈旧剔除")
    ap.add_argument("--max-pe", type=float, default=100,
                    help="PE 上限，>此值过滤(极端估值泡沫，华丰股份PE277教训)")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    db = sqlite3.connect(DB)
    db.row_factory = sqlite3.Row
    cur = db.cursor()

    # 新鲜度：features 最新日期
    latest_feat = cur.execute("SELECT MAX(date) FROM features").fetchone()[0]
    latest_kline = cur.execute("SELECT MAX(date) FROM klines").fetchone()[0]
    if latest_feat < latest_kline:
        print(f"⚠ features 最新 {latest_feat}，落后于 klines {latest_kline}。"
              f"请先跑 build_features.py。")

    # 陈旧阈值
    try:
        latest_d = date.fromisoformat(latest_feat)
    except Exception:
        latest_d = date.today()
    cutoff = (latest_d - timedelta(days=args.max_days_stale)).isoformat()

    # 从 features 读最新快照（每个 code 取最新一行）
    rows = cur.execute("""
        SELECT f.* FROM features f
        JOIN (SELECT code, MAX(date) d FROM features GROUP BY code) m
          ON f.code=m.code AND f.date=m.d
        WHERE f.date >= ?
    """, (cutoff,)).fetchall()
    codes = [r["code"] for r in rows if is_mainboard(r["code"])]
    print(f"features 快照 {len(rows)} 只(含陈旧过滤) | 主板 {len(codes)} 只 | 拉行情中...")

    quotes = fetch_quotes(codes)
    print(f"行情 {len(quotes)} 只 | 过滤排序中...")

    results = []
    for r in rows:
        code = r["code"]
        if not is_mainboard(code):
            continue
        q = quotes.get(code)
        if not q or q["price"] <= 0:
            continue
        # 硬过滤
        if q["pe"] <= 0:      # 亏损股(规则9)
            continue
        if q["pe"] > args.max_pe:  # 极端高PE(估值泡沫)
            continue
        if q["price"] < 5:    # 低价股
            continue
        if args.require_gate and not r["gate_passed"]:
            continue
        if r["s_star"] is not None and r["s_star"] < args.min_score:
            continue
        results.append({
            "code": code, "name": q["name"], "close": q["price"],
            "S": r["s_star"], "gate": r["gate_passed"], "signal": r["signal"],
            "pe": q["pe"], "mcap": q["mcap"], "pct": q["pct"],
            "ma20": r["ma20"], "ma60": r["ma60"],
            "macd_dif": r["macd_dif"], "macd_dea": r["macd_dea"],
            "m": r["m_score"], "f": r["f_score"], "p": r["p_score"], "q": r["q_score"],
            "feat_date": r["date"],
        })

    results.sort(key=lambda r: -(r["S"] or 0))
    top = results[:args.n]

    print("\n" + "=" * 66)
    print(f"  全市场海选 Top {args.n}（按融合策略 S* 排序，已过滤主板/PE>0/价≥5）")
    print("=" * 66)
    for i, r in enumerate(top, 1):
        gate = "✓" if r["gate"] else "✗门禁"
        print(f"{i:2d}. {r['name']}({r['code']}) S*={r['S']:.1f} {gate} {r['signal']} "
              f"收{r['close']} PE={r['pe']:.0f} 市值{r['mcap']:.0f}亿")
        print(f"     MACD={r['m']:.0f} 多因子={r['f']:.0f} 价={r['p']:.0f} 量={r['q']:.0f} "
              f"MA20={r['ma20']} MA60={r['ma60']}")

    print(f"\n共 {len(results)} 只通过过滤")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(top, fh, ensure_ascii=False, indent=2)
        print(f"结果已写 {args.out}")


if __name__ == "__main__":
    main()
