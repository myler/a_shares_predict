#!/usr/bin/env python3
"""回填 features 表：从 klines 日线实时算技术指标快照 + 融合策略 S*，存最新一行。

用法: python3 build_features.py [--limit N] [--fresh]
  --limit  只算 N 只（调试）
  --fresh  重建表（DROP 后 CREATE）
"""
import sys, os, argparse, sqlite3, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import predict_comprehensive

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_cache.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS features (
  code TEXT NOT NULL,
  date TEXT NOT NULL,
  close REAL,
  ma5 REAL, ma10 REAL, ma20 REAL, ma60 REAL, ma120 REAL, ma250 REAL,
  macd_dif REAL, macd_dea REAL, macd_bar REAL,
  rsi14 REAL, kdj_k REAL, kdj_d REAL, kdj_j REAL, wr10 REAL,
  boll_upper REAL, boll_mid REAL, boll_lower REAL, boll_pos REAL,
  obv_flow REAL,
  s_star REAL, base_composite REAL, aux_adjust REAL,
  m_score REAL, f_score REAL, p_score REAL, q_score REAL,
  gate_passed INTEGER, signal TEXT,
  aux_consensus REAL,
  PRIMARY KEY (code, date)
);
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    db = sqlite3.connect(DB)
    cur = db.cursor()
    if args.fresh:
        cur.execute("DROP TABLE IF EXISTS features")
    cur.execute(SCHEMA)

    codes = [r[0] for r in cur.execute(
        "SELECT DISTINCT code FROM klines").fetchall()]
    if args.limit:
        codes = codes[:args.limit]

    print(f"回填 features | 共 {len(codes)} 只")
    t0 = time.time()
    done = skip = err = 0
    for idx, code in enumerate(codes, 1):
        rows = cur.execute(
            "SELECT date,open,high,low,close,volume FROM klines "
            "WHERE code=? ORDER BY date", (code,)).fetchall()
        if len(rows) < 60:
            skip += 1
            continue
        try:
            dates = [r[0] for r in rows]
            opens = np.array([float(r[1]) for r in rows])
            highs = np.array([float(r[2]) for r in rows])
            lows = np.array([float(r[3]) for r in rows])
            closes = np.array([float(r[4]) for r in rows])
            vols = np.array([float(r[5]) for r in rows])
        except Exception as e:
            err += 1
            continue

        pred = predict_comprehensive(dates, closes, highs, lows, vols, opens=opens)
        if "error" in pred:
            err += 1
            continue

        def ma(n):
            return float(np.mean(closes[-n:])) if len(closes) >= n else None

        s = pred["scores"]
        m = s["macd"]; f = s["multifactor"]; p = s["fundamental"]; q = s["game"]
        kl = pred["key_levels"]
        bb_mid = (kl["bb_upper"] + kl["bb_lower"]) / 2
        aux = pred["auxiliary_consensus"].get("consensus_score")

        cur.execute(
            """INSERT OR REPLACE INTO features
               (code,date,close,ma5,ma10,ma20,ma60,ma120,ma250,
                macd_dif,macd_dea,macd_bar,rsi14,kdj_k,kdj_d,kdj_j,wr10,
                boll_upper,boll_mid,boll_lower,boll_pos,obv_flow,
                s_star,base_composite,aux_adjust,m_score,f_score,p_score,q_score,
                gate_passed,signal,aux_consensus)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                code, dates[-1], pred["close"],
                ma(5), ma(10), ma(20), ma(60), ma(120), ma(250),
                m["dif"], m["dea"], m["bar"],
                f["rsi"], f["k"], f["d"], f["j"], f["wr"],
                kl["bb_upper"], bb_mid, kl["bb_lower"], pred["details"]["bb_position"],
                q["obv_flow"],
                pred["composite"], pred["base_composite"], pred["auxiliary_adjustment"],
                m["score"], f["score"], p["score"], q["score"],
                1 if pred["gates"]["passed"] else 0, pred["signal"],
                aux,
            ),
        )
        done += 1
        if idx % 500 == 0:
            db.commit()
            print(f"  进度 {idx}/{len(codes)} | 完成{done} 跳过{skip} 错误{err} | {time.time()-t0:.0f}s")

    db.commit()
    db.close()
    print(f"完成: 写入{done} 跳过{skip} 错误{err} | 耗时 {(time.time()-t0)/60:.1f}分钟")


if __name__ == "__main__":
    main()
