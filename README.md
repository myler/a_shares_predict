# A股预测 — MACD 三维分析 + 背离回测

月线 / 周线 / 日线 MACD 图表生成，牛熊分界，背离信号回测，近期走势预测。

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

## 命令行

```bash
./run.py 603893              # 全部: 回测 + 图表 + 预测
./run.py backtest 002371     # 仅回测
./run.py predict 000651      # 仅预测
./run.py help                # 帮助
```

## Web 界面

```bash
nohup ./web.py 8080 > web.log 2>&1 &
```

浏览器打开 `http://localhost:8080`，输入股票代码，可选勾选"已持仓"。

## 输出

```
output/YYYYMMDD_CODE_01/
├── report.txt
├── XXXX_macd_cycle.png
└── predict.txt
```

## 规则

- **买入**：底背离（熊市抄底）| 零轴上金叉（牛市追涨）
- **卖出**：顶背离（牛市见顶）| 死叉（熊市止损）
- **牛熊**：价格 > MA60 且 MA60 斜率↑ → 牛市，5 日确认切换
- **持仓模式**：拿住 / 增持 / 减持 / 卖出
- **未持仓模式**：买入 / 观望 / 不买
