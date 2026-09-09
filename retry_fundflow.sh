#!/bin/bash
# 后台重试资金流采集：先探测接口是否恢复，恢复了才全量采龙虎榜股票池。
cd /home/myl/a_shares_predict

probe() {
  /home/myl/ashare-mcp/.venv/bin/python3 -c "
import akshare as ak
try:
    df = ak.stock_individual_fund_flow(stock='600519', market='sh')
    print('OK' if df is not None and len(df) > 0 else 'EMPTY')
except Exception:
    print('DOWN')
" 2>/dev/null | tail -1
}

while true; do
  PENDING=$(python3 -c "
import sqlite3
db = sqlite3.connect('stock_cache.db')
cur = db.cursor()
lhb = set(r[0] for r in cur.execute('SELECT DISTINCT code FROM lhb').fetchall())
done = set(r[0] for r in cur.execute('SELECT DISTINCT code FROM fund_flow').fetchall())
mb = ('600','601','603','605','000','001','002','003')
pending = [c for c in lhb if c.startswith(mb) and c not in done]
print(len(pending))
")
  if [ "$PENDING" = "0" ]; then
    echo "[$(date +%F\ %T)] 资金流采集完成，全部入库，退出"
    exit 0
  fi
  S=$(probe)
  if [ "$S" = "OK" ]; then
    echo "[$(date +%F\ %T)] 接口已恢复，开始采集 $PENDING 只..."
    /home/myl/ashare-mcp/.venv/bin/python3 collect_market_data.py fundflow --pool lhb 2>&1 | tail -2
  else
    echo "[$(date +%F\ %T)] 接口仍挂($S)，待采 $PENDING 只，10分钟后重探"
  fi
  sleep 600
done
