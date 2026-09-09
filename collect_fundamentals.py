#!/usr/bin/env python3
"""Phase2 采集：财务/资金流/筹码 三张表，按代码列表采集（akshare）。

用法（用 ashare-mcp 的 venv 跑，因其装了 akshare）：
  /home/myl/ashare-mcp/.venv/bin/python3 collect_fundamentals.py --codes 000977,002371
  /home/myl/ashare-mcp/.venv/bin/python3 collect_fundamentals.py --init   # 仅建表
"""
import sys, os, argparse, sqlite3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_cache.db")

SCHEMAS = {
    "financials": """
CREATE TABLE IF NOT EXISTS financials (
  code TEXT NOT NULL,
  report_date TEXT NOT NULL,
  revenue REAL, revenue_yoy REAL,
  net_profit REAL, net_profit_yoy REAL,
  gross_margin REAL, net_margin REAL,
  PRIMARY KEY (code, report_date)
);""",
    "fund_flow": """
CREATE TABLE IF NOT EXISTS fund_flow (
  code TEXT NOT NULL,
  date TEXT NOT NULL,
  main_net REAL, main_pct REAL,
  xl_net REAL, lg_net REAL, md_net REAL, sm_net REAL,
  PRIMARY KEY (code, date)
);""",
    "chips": """
CREATE TABLE IF NOT EXISTS chips (
  code TEXT NOT NULL,
  date TEXT NOT NULL,
  profit_ratio REAL, avg_cost REAL,
  conc_90 REAL, conc_70 REAL,
  PRIMARY KEY (code, date)
);""",
}


def init_tables(cur):
    for s in SCHEMAS.values():
        cur.execute(s)


def collect_financials(cur, code):
    import akshare as ak
    prefix = f"sh{code}" if code.startswith(("6", "5")) else f"sz{code}"
    df = ak.stock_financial_report_sina(stock=prefix, symbol="利润表")
    if df is None or df.empty:
        return 0
    n = 0
    for _, row in df.iterrows():
        rd = str(row.get("报告日", ""))
        if not rd:
            continue
        rev = row.get("营业总收入")
        cost = row.get("营业成本")  # 营业成本=COGS，非营业总成本
        np_ = row.get("归属于母公司所有者的净利润") or row.get("净利润")
        gm = (rev - cost) / rev * 100 if rev and cost else None
        nm = np_ / rev * 100 if np_ and rev else None

        def f(x):
            try:
                return float(x)
            except Exception:
                return None
        cur.execute(
            "INSERT OR REPLACE INTO financials VALUES (?,?,?,?,?,?,?,?)",
            (code, rd[:10], f(rev), None, f(np_), None, f(gm), f(nm)))
        n += 1
    return n


def collect_fund_flow(cur, code):
    import akshare as ak
    market = "sh" if code.startswith(("6", "5")) else "sz"
    df = ak.stock_individual_fund_flow(stock=code, market=market)
    if df is None or df.empty:
        return 0
    n = 0
    for _, row in df.tail(100).iterrows():
        d = str(row.get("日期", ""))[:10]
        if not d:
            continue

        def f(x):
            try:
                return float(x)
            except Exception:
                return None
        cur.execute(
            "INSERT OR REPLACE INTO fund_flow VALUES (?,?,?,?,?,?,?,?)",
            (code, d, f(row.get("主力净流入-净额")), f(row.get("主力净流入-净占比")),
             f(row.get("超大单净流入-净额")), f(row.get("大单净流入-净额")),
             f(row.get("中单净流入-净额")), f(row.get("小单净流入-净额"))))
        n += 1
    return n


def collect_chips(cur, code):
    import akshare as ak
    df = ak.stock_cyq_em(symbol=code, adjust="")
    if df is None or df.empty:
        return 0
    n = 0
    for _, row in df.tail(60).iterrows():
        d = str(row.get("日期", ""))[:10]
        if not d:
            continue

        def f(x):
            try:
                return float(x)
            except Exception:
                return None
        cur.execute(
            "INSERT OR REPLACE INTO chips VALUES (?,?,?,?,?,?)",
            (code, d, f(row.get("获利比例")), f(row.get("平均成本")),
             f(row.get("90成本-集中度")), f(row.get("70成本-集中度"))))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--codes", type=str, default="")
    args = ap.parse_args()

    db = sqlite3.connect(DB)
    cur = db.cursor()
    init_tables(cur)
    db.commit()

    if not args.codes:
        print("已建表(未采集)。用 --codes 000977,002371 指定采集。")
        db.close()
        return

    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    for code in codes:
        line = []
        for name, fn in [("财务", collect_financials),
                         ("资金流", collect_fund_flow),
                         ("筹码", collect_chips)]:
            try:
                n = fn(cur, code)
                line.append(f"{name}{n}")
            except Exception as e:
                line.append(f"{name}失败({type(e).__name__})")
        db.commit()
        print(f"{code}: " + " | ".join(line))
    db.close()


if __name__ == "__main__":
    main()
