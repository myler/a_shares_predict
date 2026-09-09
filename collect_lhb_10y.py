#!/usr/bin/env python3
"""collect_lhb_10y.py: 补采龙虎榜 10 年（2016-09 ~ 2026-09）入库。

一次性脚本，分年采集（akshare 内部按月分批，1年约7秒），
DROP 重建 lhb 表，清空旧的 1 年数据。

用法（ashare-mcp venv）:
  /home/myl/ashare-mcp/.venv/bin/python3 collect_lhb_10y.py
"""
import os, sqlite3, time
import akshare as ak

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


def f(x):
    try:
        return float(x)
    except Exception:
        return None


def collect_year(db, cur, start, end):
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
    return n


def main():
    db = sqlite3.connect(DB)
    cur = db.cursor()
    cur.execute("DROP TABLE IF EXISTS lhb")
    cur.execute(LHB_SCHEMA)
    db.commit()

    # 分年采集 2016-09-08 ~ 2026-09-08
    years = [(2016, 2017), (2017, 2018), (2018, 2019), (2019, 2020),
             (2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024),
             (2024, 2025), (2025, 2026)]
    total = 0
    t0 = time.time()
    for y1, y2 in years:
        start = f"{y1}0908"
        end = f"{y2}0908"
        try:
            n = collect_year(db, cur, start, end)
            total += n
            db.commit()
            print(f"  {start}~{end}: {n} 条 | 累计 {total} | {time.time()-t0:.0f}s",
                  flush=True)
        except Exception as e:
            print(f"  {start}~{end}: 失败 {type(e).__name__} {str(e)[:80]}",
                  flush=True)
    db.close()
    print(f"\n龙虎榜 10 年补采完成: 共 {total} 条 | 耗时 {(time.time()-t0)/60:.1f}分钟")


if __name__ == "__main__":
    main()
