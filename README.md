# A股策略分析

MACD择时策略 + 长线持有策略，含分红回测、SQLite本地缓存、Web界面。

## 安装

```bash
pip install -r requirements.txt
```

PEP 668 限制（Debian/Ubuntu）：

```bash
pip install -r requirements.txt --break-system-packages
```

可选：中文字体（图表中文显示）

```bash
sudo apt install fonts-wqy-zenhei
```

## 策略

### MACD择时策略

基于 MACD(12,26,9) 三维分析，牛市追涨、熊市空仓，背离信号触发买卖。

- **买入**：底背离（熊市抄底）| 零轴上金叉（牛市追涨）
- **卖出**：顶背离（牛市见顶）| 死叉（熊市止损）
- **牛熊判定**：价格 > MA60 且 MA60(10日斜率)↑ → 牛市，连续5日确认
- **评分**：金叉+2/死叉-2 | 零轴上+1/零轴下-1 | 牛+2/熊-2 | DIF↑+1/DIF↓-1 | 顶背离-3

### 长线持有策略

买入持有不动，展示价差收益、累计分红、股息率走势。适合与 MACD 择时策略对比。

- **买入**：数据起始日一次性买入
- **卖出**：不卖出，持有至今
- **分红**：持有期间所有已实施分红累加
- **对比**：可直观看到"择时交易 vs 买入不动"的收益差异

## 分红回测

自动拉取 A 股历史分红数据（送股、转增、派息），计算每笔交易持股期间到手的每股分红金额。

- 数据源：新浪财经 F10
- 回测交易表显示每笔交易的价差收益 + 分红收益（元/股）
- 分红历史表展示历年派息方案与股息率

## 本地缓存

首次查询某只股票时从网络拉取全量数据并存入 SQLite，之后秒读。

- 数据库：`stock_cache.db`（已加入 .gitignore）
- K 线和分红数据均缓存，搜过的股票不用再等网络

## 测试

```bash
python3 -m unittest discover -s unittests -v
```

覆盖 engine（MACD/背离/回测/预测）、fetcher（分红计算/边界条件）、db（缓存增删改查），共38个测试用例。每次改代码后跑一次。

## 命令行

```bash
./run_cli.py 603893              # 全部: 回测 + 图表 + 预测
./run_cli.py backtest 600329     # 仅回测（含分红明细）
./run_cli.py predict 000651      # 仅预测
./run_cli.py help                # 帮助
```

## Web 界面

```bash
./web.py 8099                # 默认端口 8080
```

浏览器打开 `http://localhost:8099`：

- 输入6位股票代码
- 选择策略：MACD择时 / 长线持有
- 勾选"已持仓"后可选择"计算分红"
- 长线持有策略自动锁定已持仓

## 输出

```
output/YYYYMMDD_CODE_NN/
├── report.txt
├── XXXX_macd_cycle.png
└── predict.txt
```

## 文件结构

```
a_shares_predict/
├── run_cli.py          # CLI入口（参数解析+格式化输出）
├── web.py              # Web界面（HTTP服务）
├── engine.py           # 核心引擎（MACD、背离、回测、预测、画图）
├── fetcher.py          # 数据层（K线抓取、分红抓取、多源降级）
├── db.py               # SQLite缓存模块
├── templates/
│   └── page.html       # Web页面模板
├── requirements.txt
└── README.md
```
