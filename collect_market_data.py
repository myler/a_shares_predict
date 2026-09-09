#!/usr/bin/env python3
"""collect_market_data.py: 采集龙虎榜 + 主力资金流入库，供短线强势股回测。

用法（在 ashare-mcp venv 下跑）:
  /home/myl/ashare-mcp/.venv/bin/python3 collect_market_data.py lhb --start 20250908 --end 20260908
  /home/myl/ashare-mcp/.venv/bin/python3 collect_market_data.py fundflow [--limit N]
"""
import sys, os, argparse, sqlite3, time
import requests
import akshare as ak

SINA_MF = "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/MoneyFlow.ssl_qsfx_zjlrqs"

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_cache.db")
MAINBOARD = ("600", "601", "603", "605", "000", "001", "002", "003")

LHB_SCHEMA = """
CREATE TABLE IF NOT EXISTS lhb (
  code TEXT, name TEXT, list_date TEXT, reason TEXT,
  interpretation TEXT, close REAL, pct_change REAL,
  lhb_net_buy REAL, lhb_buy REAL, lhb_sell REAL,
  lhb_turnover REAL, market_turnover REAL, net_buy_ratio REAL,
  turnover_rate REAL, float_market_cap REAL,
  after_1d REAL, after_2d REAL, after_5d REAL, after_10d REAL,
  PRIMARY KEY (code, list_date, reason)
);"""

FUND_SCHEMA = """
CREATE TABLE IF NOT EXISTS fund_flow (
  code TEXT NOT NULL, date TEXT NOT NULL,
  main_net REAL, main_pct REAL, xl_net REAL,
  lg_net REAL, md_net REAL, sm_net REAL,
  PRIMARY KEY (code, date)
);"""


def f(x):
    try:
        return float(x)
    except Exception:
        return None


def collect_lhb(start, end):
    db = sqlite3.connect(DB)
    cur = db.cursor()
    cur.execute("DROP TABLE IF EXISTS lhb")   # schema 变更后重建
    cur.execute(LHB_SCHEMA)
    df = ak.stock_lhb_detail_em(start_date=start, end_date=end)
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
    db.commit()
    db.close()
    return n


def collect_fund_flow(limit=0, pool="all"):
    db = sqlite3.connect(DB)
    cur = db.cursor()
    cur.execute(FUND_SCHEMA)
    if pool == "lhb":
        codes = [r[0] for r in cur.execute(
            "SELECT DISTINCT code FROM lhb").fetchall()]
    else:
        codes = [r[0] for r in cur.execute(
            "SELECT DISTINCT code FROM klines").fetchall()]
    codes = [c for c in codes if c.startswith(MAINBOARD)]
    done = set(r[0] for r in cur.execute(
        "SELECT DISTINCT code FROM fund_flow").fetchall())
    codes = [c for c in codes if c not in done]   # 跳过已采集
    if limit:
        codes = codes[:limit]
    if not codes:
        print("资金流: 无待采集股票（已全部入库）")
        db.close()
        return
    t0 = time.time()
    ok = fail = 0
    for i, code in enumerate(codes, 1):
        market = "sh" if code.startswith(("6", "5")) else "sz"
        got = False
        for attempt in range(3):
            try:
                df = ak.stock_individual_fund_flow(stock=code, market=market)
                if df is None or df.empty:
                    break
                for _, r in df.tail(100).iterrows():
                    cur.execute(
                        "INSERT OR REPLACE INTO fund_flow VALUES (?,?,?,?,?,?,?,?)",
                        (code, str(r.get("日期", ""))[:10], f(r.get("主力净流入-净额")),
                         f(r.get("主力净流入-净占比")), f(r.get("超大单净流入-净额")),
                         f(r.get("大单净流入-净额")), f(r.get("中单净流入-净额")),
                         f(r.get("小单净流入-净额"))))
                ok += 1
                got = True
                break
            except Exception:
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        if not got:
            fail += 1
        time.sleep(0.15)   # 节流，避免东财限流
        if i % 100 == 0:
            db.commit()
            el = time.time() - t0
            print(f"  进度 {i}/{len(codes)} | 成功{ok} 失败{fail} | {el:.0f}s "
                  f"ETA {el/i*(len(codes)-i):.0f}s", flush=True)
    db.commit()
    db.close()
    print(f"资金流完成: 成功{ok} 失败{fail} | 耗时 {(time.time()-t0)/60:.1f}分钟")


def collect_fund_flow_sina(limit=0, pool="all"):
    """新浪资金流（替代东财，东财接口常挂）。字段：netamount→main_net,
    ratioamount→main_pct(小数), r0_net→xl_net。历史约200交易日。"""
    db = sqlite3.connect(DB)
    cur = db.cursor()
    cur.execute(FUND_SCHEMA)
    if pool == "lhb":
        codes = [r[0] for r in cur.execute(
            "SELECT DISTINCT code FROM lhb").fetchall()]
    else:
        codes = [r[0] for r in cur.execute(
            "SELECT DISTINCT code FROM klines").fetchall()]
    codes = [c for c in codes if c.startswith(MAINBOARD)]
    done = set(r[0] for r in cur.execute(
        "SELECT DISTINCT code FROM fund_flow").fetchall())
    codes = [c for c in codes if c not in done]
    if limit:
        codes = codes[:limit]
    if not codes:
        print("资金流: 无待采集股票（已全部入库）")
        db.close()
        return
    t0 = time.time()
    ok = fail = 0
    for i, code in enumerate(codes, 1):
        mkt = "sh" if code.startswith(("6", "5")) else "sz"
        try:
            r = requests.get(SINA_MF, params={
                "page": 1, "num": 200, "sort": "opendate",
                "asc": 0, "daima": mkt + code}, timeout=10)
            d = r.json()
            if isinstance(d, list) and d:
                for x in d:
                    cur.execute(
                        "INSERT OR REPLACE INTO fund_flow VALUES (?,?,?,?,?,?,?,?)",
                        (code, str(x.get("opendate", ""))[:10],
                         f(x.get("netamount")), f(x.get("ratioamount")),
                         f(x.get("r0_net")), None, None, None))
                ok += 1
            else:
                fail += 1
        except Exception:
            fail += 1
        time.sleep(0.1)
        if i % 100 == 0:
            db.commit()
            el = time.time() - t0
            print(f"  进度 {i}/{len(codes)} | 成功{ok} 失败{fail} | {el:.0f}s "
                  f"ETA {el/i*(len(codes)-i):.0f}s", flush=True)
    db.commit()
    db.close()
    print(f"新浪资金流完成: 成功{ok} 失败{fail} | 耗时 {(time.time()-t0)/60:.1f}分钟")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["lhb", "fundflow"])
    ap.add_argument("--start", type=str, default="")
    ap.add_argument("--end", type=str, default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pool", type=str, default="all",
                    help="股票池: all=全市场, lhb=近一年龙虎榜股票池")
    ap.add_argument("--src", type=str, default="sina",
                    help="资金流源: sina=新浪(推荐), akshare=东财(常挂)")
    args = ap.parse_args()

    if args.mode == "lhb":
        n = collect_lhb(args.start, args.end)
        print(f"龙虎榜入库 {n} 条 ({args.start}~{args.end})")
    else:
        if args.src == "akshare":
            collect_fund_flow(args.limit, args.pool)
        else:
            collect_fund_flow_sina(args.limit, args.pool)


if __name__ == "__main__":
    main()
