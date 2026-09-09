#!/usr/bin/env python3
"""lhb_pick.py: 每日龙虎榜选股 —— 抓龙头（连板高度 + 资金坚决买入 + 缩量）。

最优策略（1年 11403 条清洗样本回测，四季度全正）：
  净买占比>15% + 换手率<12.5% + 连板高度>=2（K线精确算连续涨停天数）
  1日 +5.70%/胜77% | 2日 +8.51%/胜67% | 5日 +8.96%/胜64%
  首板是陷阱(5日-2.11%)，2板起才是真龙头。

容错降级：龙虎榜接口失败时，自动用 db 缓存最近交易日数据，照常出结果。

用法（在 ashare-mcp venv 下跑）:
  /home/myl/ashare-mcp/.venv/bin/python3 lhb_pick.py                    # 严格(龙头精选)
  /home/myl/ashare-mcp/.venv/bin/python3 lhb_pick.py --loose            # 宽松(多候选)
  /home/myl/ashare-mcp/.venv/bin/python3 lhb_pick.py --date 20260908 --thr 20
"""
import sys, os, argparse, datetime, sqlite3
import akshare as ak

MAINBOARD = ("600", "601", "603", "605", "000", "001", "002", "003")
DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_cache.db")


def f(x):
    try:
        return float(x)
    except Exception:
        return None


def calc_heights_cur(codes):
    """算每只股票『当前』连板高度（用 klines 最新数据，连续涨停天数）。"""
    if not codes:
        return {}
    db = sqlite3.connect(DB)
    try:
        q = ",".join("?" * len(codes))
        k = db.execute(
            f"SELECT code,date,close FROM klines WHERE code IN ({q}) "
            f"ORDER BY code,date", codes).fetchall()
    finally:
        db.close()
    by_code = {}
    for code, date, close in k:
        by_code.setdefault(code, []).append(close)
    result = {}
    for code in codes:
        closes = by_code.get(code, [])
        if len(closes) < 2:
            result[code] = 0
            continue
        height = 0
        for i in range(len(closes) - 1, 0, -1):
            pct = (closes[i] / closes[i - 1] - 1) * 100
            if pct >= 9.5:
                height += 1
            else:
                break
        result[code] = height
    return result


def fetch_lhb(date: str):
    """拉当日龙虎榜，失败降级到 db 缓存最近交易日。返回 (records, source, actual_date)。"""
    try:
        df = ak.stock_lhb_detail_em(start_date=date, end_date=date)
        recs = []
        for _, r in df.iterrows():
            recs.append({
                "code": str(r.get("代码", "")).zfill(6),
                "name": r.get("名称"),
                "pct": f(r.get("涨跌幅")),
                "nbr": f(r.get("净买额占总成交比")),
                "net": f(r.get("龙虎榜净买额")),
                "turn": f(r.get("换手率")),
                "reason": str(r.get("上榜原因", "")),
                "interp": r.get("解读"),
            })
        return recs, "实时", date
    except Exception:
        pass
    try:
        db = sqlite3.connect(DB)
        cur = db.cursor()
        latest = cur.execute("SELECT MAX(list_date) FROM lhb").fetchone()[0]
        if not latest:
            db.close()
            return [], "无数据", ""
        rows = cur.execute(
            "SELECT code,name,reason,interpretation,pct_change,lhb_net_buy,"
            "net_buy_ratio,turnover_rate FROM lhb WHERE list_date=?",
            (latest,)).fetchall()
        db.close()
        recs = [{"code": c, "name": n, "pct": pct, "nbr": nbr, "net": net,
                 "turn": turn, "reason": reason or "", "interp": interp}
                for c, n, reason, interp, pct, net, nbr, turn in rows]
        return recs, "缓存", latest
    except Exception:
        return [], "无数据", ""


def pick(recs, thr: float, max_turn: float, need_lianban: bool, use_precise: bool):
    """use_precise: 用 K线精确连板高度(>=2)；否则用 reason 字段模糊判断。"""
    rows, seen = [], set()
    # 先过滤涨停 + 净买占比
    cands = []
    for r in recs:
        code = r["code"]
        if not code.startswith(MAINBOARD) or code in seen:
            continue
        seen.add(code)
        if r["pct"] is None or r["nbr"] is None or r["net"] is None:
            continue
        if r["pct"] < 9.5 or r["nbr"] <= thr:
            continue
        if r["turn"] is not None and r["turn"] > max_turn:
            continue
        cands.append(r)
    if need_lianban:
        if use_precise:
            heights = calc_heights_cur([c["code"] for c in cands])
            for c in cands:
                h = heights.get(c["code"], 0)
                if h >= 2:
                    c["height"] = h
                    rows.append(c)
        else:
            for c in cands:
                if "连续三个交易日" in c["reason"]:
                    c["height"] = None
                    rows.append(c)
    else:
        for c in cands:
            c["height"] = None
            rows.append(c)
    rows.sort(key=lambda x: -x["nbr"])
    return rows


def latest_date():
    for d in range(0, 5):
        dstr = (datetime.date.today() - datetime.timedelta(days=d)).strftime("%Y%m%d")
        try:
            if ak.stock_lhb_detail_em(start_date=dstr, end_date=dstr) is not None:
                return dstr
        except Exception:
            continue
    return datetime.date.today().strftime("%Y%m%d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, default="")
    ap.add_argument("--thr", type=float, default=15.0, help="净买占比阈值%")
    ap.add_argument("--turn", type=float, default=12.5, help="换手率上限%")
    ap.add_argument("--loose", action="store_true", help="宽松: 不加连板过滤")
    ap.add_argument("--fuzzy", action="store_true", help="用 reason 模糊判断连板(默认精确K线)")
    args = ap.parse_args()
    date = args.date or latest_date()
    need_lianban = not args.loose
    max_turn = 9999 if args.loose else args.turn
    thr = 10.0 if args.loose else args.thr

    recs, source, actual_date = fetch_lhb(date)
    rows = pick(recs, thr, max_turn, need_lianban, use_precise=not args.fuzzy)
    mode = "宽松" if args.loose else "严格(龙头精选)"
    src_note = "" if source == "实时" else f"  ⚠️数据源:{source}({actual_date})"
    lianban_note = "" if args.loose else ("精确K线" if not args.fuzzy else "模糊reason")
    print(f"龙虎榜选股 {actual_date or date} [{mode}]{src_note}")
    print(f"涨停+净买>{thr}%" + ("" if args.loose else f"+换手<{max_turn}%+连板({lianban_note})"))
    print(f"共 {len(rows)} 只 | 次日买入参考\n")
    if not rows:
        print("（今日无符合条件标的）")
        return
    print(f"{'代码':<7}{'名称':<9}{'涨幅':>7}{'净买占比':>9}{'换手':>7}{'连板':>5}  上榜原因")
    for r in rows:
        turn = f"{r['turn']:.1f}%" if r["turn"] is not None else "-"
        h = f"{r['height']}板" if r.get("height") is not None else "是"
        print(f"{r['code']:<7}{r['name']:<9}{r['pct']:>6.2f}%{r['nbr']:>8.2f}%{turn:>7}"
              f"{h:>5}  {r['reason'][:30]}")
    print("\n解读:")
    for r in rows:
        print(f"  {r['code']} {r['name']}: {r['interp']}")


if __name__ == "__main__":
    main()
