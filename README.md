# A 股多维策略研究 · Pareto34 风险预算 v1

完整结果与逐股可复核材料见 [本轮独立核账报告](reports/pareto_20260910/README.md)、[全部账户](reports/pareto_20260910/accounts.csv)和[核验记录](reports/pareto_20260910/verification.json)。

34 个独立三值原子、每日非支配前沿、单股独立现金/股份账户，以及完成回测的只读 Web 展示。保留旧加权分析、海选、CLI、API、长线展示和财务初筛，不将它们一概标为废弃。

> **当前状态（2026-09-10）**：已实现 `pareto_daily_atoms_risk_v1`，Web 默认 `pareto`，仅读取本地完成的历史报告，不生成实时推荐。**本轮实证失败：33.3 亿元本金，期末含残值亏损约 3.8735 亿元（−11.6320%），未实现保本盈利目标。** 技术实现完成不等于策略有效；历史缺陷证据仍见 [AUDIT_2026-09-09.md](AUDIT_2026-09-09.md)。

## 入口边界

| 用途 | 实际入口 | 行为与边界 |
|---|---|---|
| 默认多维历史报告 | Web `/analyze?code=000651&strategy=pareto`；省略策略也默认 `pareto` | [pareto_web.py](pareto_web.py) 只读冻结特征、前沿、账户与账本；不联网、不刷新、不回测，不接受当前持仓/分红开关 |
| 多维回测总览 | Web `/pareto-summary?n=5` | 全部独立账户资金汇总；N 仅按代码顺序截断展示，不排名、不裁掉 F1、不改变总本金 |
| 多维研究运行 | [pareto_backtest.py](pareto_backtest.py) | 快照行情、解析权息、计算每日 34 维 F1、连续回放独立账户 |
| 保留旧加权融合 | Web `comprehensive`、CLI `predict`、API 默认 `comprehensive` | [engine.py](engine.py) 四维加权＋18 项辅助共识；**不是 Pareto34**，各入口成交/汇总口径仍有差异 |
| 保留旧加权海选 | Web「旧加权海选（非帕累托）」`/scan`、[scan_composite.py](scan_composite.py) | [factors.py](factors.py) 横截面排名加权；可能联网取报价，不是 S* 或 Pareto34 |
| 保留独立研究 | [money_backtest4.py](money_backtest4.py)、[money_pareto_backtest.py](money_pareto_backtest.py) 等 | 旧双路径资金/5 因子 Pareto 实验，未接入默认多维入口，不能当作新策略业绩 |

默认报告目录由服务端 `PARETO_RUN_DIR` 配置，默认指向本轮输出；HTTP 参数不能指定文件路径。缺少完成标记、版本不兼容或产物不完整时报告错误，**不回落旧加权策略**。只读承诺仅适用于 Pareto 页面，不适用于显式选择的旧入口；项目没有自动下单功能。

## 当前多维方法：34 原子 → 同日前沿 → 风险仓位

### 独立维度与非支配关系

[pareto_strategy.py](pareto_strategy.py) 的 `make_objectives()` 输出 34 列 +1/0/−1（看多/中性/看空），不是先聚合 M/F/P/Q 或辅助共识后再做 Pareto：

| 来源 | 列数 | 原子规则 |
|---|---:|---|
| MACD | 6 | DIF/DEA、零轴、5 日 DIF 斜率、3 日 BAR 趋势、底部修复、顶部衰弱 |
| 多因子 | 5 | RSI、K/D、J 极值、布林位置、WR 极值 |
| 价格位置/趋势 | 2 | 250 日价格区间位置、250 日涨跌幅 |
| 量能/价格代理 | 3 | OBV 净流、量价背离、波动率/价格机构行为代理；不是实际资金流 |
| 辅助指标 | 18 | DMI、CCI、BIAS(6)、BIAS(12)、EXPMA、BBI、TRIX、VR、BR、AR、CR、DMA、DPO、MTM、SKDJ、LWR、ENE、LON，各自保留方向，不反向合并成 A |

“独立”指单独作为比较坐标，**不指统计独立**。三值化保留原规则方向及分支优先级，但丢失原始幅度和强弱差异，不能称“保留所有信息”。

令 $\mathcal U_t$ 为同日有有效行情、含当日至少 251 根日线、正成交量且通过资格与门禁的股票集合。门禁为 RSI≤92、OBV 5 日净流≥−60%；不再使用旧 M/F 加权分数门禁。权息隔离和跳空隔离先限制资格。统一按各维越大越优：

$$
\mathbf x_{i,t}\in\{-1,0,+1\}^{34},\qquad
i\succ_t j\iff(\forall k,\ x_{i,t,k}\ge x_{j,t,k})\land(\exists k,\ x_{i,t,k}>x_{j,t,k})
$$

$$
F_{1,t}=\{i\in\mathcal U_t:\nexists j\in\mathcal U_t,\ j\succ_t i\}
$$

`daily_front_layers()` 可计算前四层，**本轮运行器只计算 F1**，其他标签不代表已精确分层。相同向量同层，全部 F1 保留，不加权、不做拥挤距离或 Top-N 截断。高维中不可比较的股票可能很多，前沿不提供唯一最优解，也不保证优于旧加权策略。

### 波动率、ES 与成本跟踪

用截至当日收盘的权息连续化价格计算简单日收益 $r$，窗口 $L=63$，置信度 $\alpha=0.95$：

$$
\sigma_{i,t}=\sqrt{252}\,\operatorname{std}_{ddof=1}(r_{i,t-L+1:t}),\qquad
ES_{i,t}=\frac1K\sum_{k=1}^{K}\ell_{(k)},\quad
\ell=\max(0,-r),\quad K=\lceil L(1-\alpha)\rceil=4
$$

$\ell_{(k)}$ 按亏损从大到小排序，盈利日保留零亏损，不是仅对下跌日取均值。仅 F1、资格/门禁通过、风险值有效且未触发账户退出/冷静期时，目标风险上限为：

$$
\bar w_{i,t}=\min\left(w_{\max},\frac{\sigma_{\rm target}}{\max(\sigma_{i,t},10^{-8})},\frac{b_{ES}}{\max(ES_{i,t},10^{-8})}\right)
$$

其他情况目标为 0。每只股票的 $w$ 是其**自身账户权益比例**，不跨股归一化、不借用其他账户现金，不加杠杆。默认 $w_{\max}=1$、$b_{ES}=0.02$；`RiskConfig` 默认年化波动目标 15%，**本轮训练选择 10%**，二者不能混称。

运行器调用 `optimal_rebalance_weight(..., risk_cap=desired)`，将收盘实际仓位 $w_{\rm prev}$ 调向风险上限：

$$
w^*_{i,t}=\underset{0\le w\le\bar w_{i,t}}{\arg\min}\left[
\frac{\lambda}{2}\max(\sigma_{i,t}^2,10^{-8})(w-\bar w_{i,t})^2
+c_b(w-w_{\rm prev})_++c_s(w_{\rm prev}-w)_+\right],\quad\lambda=5
$$

其中 $(z)_+=\max(z,0)$。比例费用形成不对称不交易带，但目标为 0 是硬退出，不因省费用继续持仓。费用代理包括佣金、滑点、固定过户费代理及分期卖出印花税；实际最低佣金、日期费率、整手与现金约束由账本结算。此处标量优化的是**仓位跟踪与交易成本**，不是将 34 个信号再次加权。

概念参考：[Moreira 与 Muir《Volatility Managed Portfolios》，NBER w22208](https://www.nber.org/papers/w22208) 支持高波动时降低风险暴露的研究思路；[cvxportfolio 优化策略文档](https://www.cvxportfolio.com/en/stable/optimization_policies.html) 说明风险、成本和约束的联合建模。它们不是本项目 ES 公式、参数或收益的验证；本项目是 NumPy 实现，不是直接运行这些文献/库的策略，**引用不构成有效性背书**。

### 模块职责与执行约束

| 模块 | 职责 |
|---|---|
| [pareto_strategy.py](pareto_strategy.py) | 34 维原子、非支配层、滚动风险统计、风险目标、成本跟踪；纯计算 |
| [capital_account.py](capital_account.py) | `Account` 现金/股份账本、整手、T+1、费用、权息、回撤状态与逐笔记录；不取数据 |
| [pareto_backtest.py](pareto_backtest.py) | 只读备份源库、冻结股票池/日历/参数，预处理、每日 F1、训练选参、全期连续账户及汇总产物 |
| [action_fallback.py](action_fallback.py) / [action_snapshot.py](action_snapshot.py) | 前者批量冻结东方财富（EM）权息并交叉核验配股，后者保留独立新浪快照工具；运行器不调用新浪下载 |
| [pareto_web.py](pareto_web.py)、[web.py](web.py)、[templates/page.html](templates/page.html) | 只读产物与资金恒等式校验、HTML 路由与界面；旧策略另走旧路径 |

- **时序**：收盘信号→下一市场交易日真实开盘价尝试成交；停牌后的过期非零目标不续用，退出请求可续行。成交用不复权价，权息连续化价格仅供信号，不回写过去信号。
- **账户**：每只股票独立 100 万元，闲置现金零利息；买入/非零目标调减按 100 股整手，目标 0 尝试卖出全部可卖股份。净值回撤 15% 触发退出及随后 20 个市场日禁买，不保证最大实际回撤止于 15%；退出后风险峰值可重置，不能称全期本金底线。
- **截止**：预先规定最后 5 个市场日禁买并尝试退出，公共截止日另可尝试收盘清仓；不把陈旧个股末日当作预知的清仓日。停牌、T+1、成交限制可能留下股份，按最后有效价标记残值，不当现金。
- **费用**：双边佣金 0.03%、每笔最低 5 元，买卖各 5bp 不利滑点；卖出印花税 2023-08-28 前 0.1%、此后 0.05%；双边过户费 2022-04-29 前统一假设 0.002%、此后 0.001%，不是完整历史费率复原。
- **权息**：现金分红近似除权日到账、送转近似当日可用；缺失日事件映射到下一有效日线，未计红利税，零碎股出售为近似。配股未建模；权息错误/未核验或有配股历史的账户全期禁买但保留本金。既有新浪成功现金事件保留、EM 校验配股，不等于双源分红一致或历史完整。
- **数据/成交限制**：未解释的绝对开盘跳空>25% 后禁止新信号；缺失/无效行情只延续估值、不前填信号。开盘不利跳空≥4.8% 作为保守涨跌停代理，无历史 ST/官方限价、无成交量参与上限，日线不能证明真实可成交。

## 本轮结果：公开失败，不以筛选口径隐藏亏损

已完整读取 [output/pareto_mainboard_20260910/summary.json](output/pareto_mainboard_20260910/summary.json)，并与全部 3330 份账户、[output/pareto_mainboard_20260910/accounts.csv](output/pareto_mainboard_20260910/accounts.csv)、[output/pareto_mainboard_20260910/manifest.json](output/pareto_mainboard_20260910/manifest.json) 核对。以下为该次完成运行，不是实时或实盘业绩。

- 请求区间 **2016-09-10～2026-09-10**；价缓存有效截止 **2026-09-08**，市场日历从 2016-09-12 起共 **2425 日**，不是已取得 9 月 10 日价格。
- **3330** 个独立账户，每户 **1000000 元**，总本金 **3330000000 元（33.3 亿元）**；实际有成交 **3124** 户，其中盈利 **397**、亏损 **2727**（397+2727=3124）。其余 **206** 户无交易，**206000000 元本金仍在分母和期末现金中**。
- 权息状态 **3305 ok / 25 error**；**71** 户有配股历史，全期隔离。配股与错误实际交叉 **1** 户，合计禁买 **95** 户；这不等于下项跳空的 95 户。另有未解释跳空 **95** 户（发生后禁新信号）、陈旧末日 **144** 户（缓存末日早于有效截止）、晚起始 **1041** 户（缓存首日晚于请求起点）、特征错误 **0** 户。各质量标签可交叉，不能相加或剔除后重算本金。

| 全量资金项 | 元（保留汇总精度） |
|---|---:|
| 初始本金 | 3330000000.0 |
| 期末现金 | 2941535512.222679 |
| 残余股份标记值（非现金） | 1117595.0 |
| 期末权益＝现金＋残值 | 2942653107.222679 |
| 总盈亏＝权益−全部本金 | -387346892.77732086 |
| 现金−全部本金（不含残值） | -388464487.77732086 |
| 已扣交易费用 | 201464774.82044384 |
| 已入账现金分红 | 13065946.28145 |

总收益率 **−11.632038822141768%**；3320 户已清仓，10 户仍有残值。费用字段为佣金/印花税/过户费，滑点已体现在成交价；**费用已扣、股息已进入现金及权益，不再扣一次费用或加一次股息**。这只是现有模型假设下的账户总收益，不宣称完整历史总收益已验证。

### 设计分段与风险参数试验

代码规定训练期截至 **2021-12-31**，验证期 **2022～2023**，**2024 年起为设计上的留出分段**。三档试验均连续运行至同一有效截止日；分段切点只读取净值，不清仓、不重置账户。仅训练期货币盈亏选优，同分取较低波动目标。

| 年化波动目标 | 训练期盈亏（元） | 验证期盈亏（元） | 选择 |
|---|---:|---:|---|
| 10% | -215783854.52 | -90694532.34 | 所选，亏损较少 |
| 15% | -304996876.24 | -128201881.40 | 未选 |
| 20% | -334226256.27 | -146421989.05 | 未选 |

三档训练及验证均亏损，10% 优于 15%/20% 不等于盈利。所选参数 2024 年起盈亏为 **-80868505.9126463 元**，起点权益 **3023521613.1353254 元**，与期末权益差额一致（并行浮点求和存在微小尾差）。

**本轮已多次查看 2024 年后结果，因此不是完全独立盲测。** 选参函数只读取训练净值，不能消除研究过程已看到后段数据的影响；看过后再调参必须另记版本，不能继续把同一后段冒充未见过的 holdout。股票池来自现有缓存、有幸存者偏差，权息全历史隔离也是事后质量筛查，不是当时可投资股票池。

### 年度收益与高交易频率

按相邻年末权益计算，首段相对初始本金；2016、2026 为不足一年的区间，金额四舍五入到分，收益含标记残值。

| 年份 | 本期盈亏（元） | 本期收益率 |
|---|---:|---:|
| 2016（起始后） | -823079.75 | -0.0247% |
| 2017 | -43997847.55 | -1.3216% |
| 2018 | -89657290.12 | -2.7291% |
| 2019 | -25925483.28 | -0.8113% |
| 2020 | -32302713.80 | -1.0191% |
| 2021 | -23077440.03 | -0.7356% |
| 2022 | -50868068.00 | -1.6334% |
| 2023 | -39826464.35 | -1.3001% |
| 2024 | -37255879.41 | -1.2322% |
| 2025 | -11956381.83 | -0.4004% |
| 2026（截至 09-08） | -31656244.68 | -1.0643% |

建仓 **549295**、增持 **210418**、减持 **113739**、退出 **549285** 次，合计 **1422737 次成交事件**，不是这么多笔完整往返交易，也不包含未成交尝试。每日 F1/资格切换导致频繁进出，零目标硬退出不会被成本不交易带阻止；高交易频率与约 2.0146 亿元显式费用是突出问题，但未经归因检验不能断言它们解释全部亏损。**风险预算没有实现用户的保本盈利目标，本轮结果应作为失败实验保留。**

## 安装与启动

推荐在 Linux / WSL 的项目目录内创建独立环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python web.py 8099
```

打开 **http://localhost:8099**。Web 默认端口是 8080，上例指定 8099。当前服务监听所有网卡，没有鉴权；仅用于可信环境，不要直接暴露公网。

如果 Ubuntu 提示缺少 `ensurepip`，需先补齐与 Python 版本匹配的 venv 支持。已有系统 pip（≥22.3）时，也可无提权安装到独立环境：

```bash
python3 -m venv --without-pip .venv
python3 -m pip --python .venv/bin/python install pip -r requirements.txt
```

- 核心、图表及 Web 海选依赖：[requirements.txt](requirements.txt)（NumPy、Matplotlib、pandas）。
- 财务/龙虎榜/资金流采集另需 [requirements-research.txt](requirements-research.txt)（requests、AkShare）：

```bash
.venv/bin/python -m pip install -r requirements-research.txt
```

系统没有中文字体时图表可能缺字。依赖暂未锁定版本；安装成功不等于历史实验可复现。

### 多维回测与只读报告

以下在 Linux / WSL 仓库根目录执行。复用本轮冻结权息目录的完整命令：

```bash
.venv/bin/python -B pareto_backtest.py --output output/pareto_mainboard_20260910 --action-snapshot output/action_fallback_market_2015_20260910 --start 2016-09-10 --end 2026-09-10 --workers 8
PARETO_RUN_DIR=output/pareto_mainboard_20260910 .venv/bin/python web.py 8099
```

Windows PowerShell 对应回测命令：

```powershell
wsl -d Ubuntu01 --cd /home/myl/a_shares_predict -- .venv/bin/python -B pareto_backtest.py --output output/pareto_mainboard_20260910 --action-snapshot output/action_fallback_market_2015_20260910 --start 2016-09-10 --end 2026-09-10 --workers 8
```

- `--action-snapshot` **可选**：完整冻结目录复用时不发 HTTP；覆盖日期须匹配 2015-01-01～请求截止日。省略时自动在运行输出内建立 EM 市场批量快照，一次分页取全市场后按代码匹配，**不触发新浪逐股采集，不靠重试冲击新浪限流**。目录尚未完整冻结时也可能继续公开 EM 请求，不能称无条件离线。
- EM 遇访问拒绝/限流保留失败证据，不绕过限制、不自动重试；权息未核验仍按隔离口径保留本金。旧新浪成功快照及原失败证据不覆盖，解析结果另存。
- 行情来自源缓存的只读一致性备份，不刷新生产库。`--source` 可指定源库，`--limit` 仅供小样本冒烟；日期/股票池已冻结的目录续跑不因新参数自动换样本。新数据或策略版本应使用新输出目录。
- 新运行记录源路径、limit 和 2015-01-01 特征/权息覆盖起点；不支持更早起始。续跑日期、limit、已记录源路径不符会拒绝，冻结源库字节哈希也须匹配。原生产库同路径内容变动不影响已冻结输入，续跑不重新读取它；旧清单缺少的路径字段不能追溯验证。
- `--cached-actions` 只是禁用本轮批量解析的诊断选项；缓存补出的权息标记 `cache_unverified`，不能据此开展可信交易。正常运行不要用它代替冻结权息。
- 运行器会重建完成汇总，不能把“重跑”当作只读浏览。等待 [output/pareto_mainboard_20260910/summary.json](output/pareto_mainboard_20260910/summary.json) 写完再读报告；已有报告只需启动 Web，无须重跑。首页方法说明读取模型默认值，具体报告读取该次所选参数。

### 保留的旧 CLI、海选与独立初筛

```bash
# 旧加权融合预测：不是 Pareto34；不回测、不出图、不抓分红
.venv/bin/python run_cli.py predict 000651

# 独立财务质量初筛：非买卖信号
.venv/bin/python run_cli.py bmfund 601888

# 旧加权横截面海选：先有本地 K 线缓存，再按报价过滤
.venv/bin/python scan_composite.py -n 5 --max-pe 100 --min-price 5

# 保留旧 JSON 服务，默认 comprehensive，不由 web.py 提供
.venv/bin/python api.py 8100
```

注意：[run_cli.py](run_cli.py) **直接传股票代码、`chart`、`backtest`、`json` 仍是旧 MACD 路径**；`multi` 是旧多因子共振，`predict` 是旧加权融合，均未改为 Pareto34。

## 保留旧加权融合：comprehensive / CLI predict

实现：[engine.py](engine.py) 的 `_score_comprehensive()`、`predict_comprehensive()`、`backtest_comprehensive()`。显式选择旧 Web `comprehensive` 时读取结构化评分并回测；CLI `predict` 只预测。以下评分、门槛与建议表**均不适用于 Pareto34**，没有资金流 Z、龙虎榜双路径或分级仓位。

### 评分

$$S=0.40M+0.30F+0.15P+0.15Q$$

$$A=\frac{N_{看多}-N_{看空}}{18}\times100,\qquad S^*=S-0.08A$$

四维各从 50 分起算，总分不额外裁剪。A 范围 −100～100，最大修正 ±8 分：偏空共识加分、偏多共识减分。分数不是概率。

| 维度 | 权重 | 输入与含义 |
|---|---:|---|
| M：MACD 核心 | 40% | DIF/DEA 多空、零轴、5 日斜率、3 日 BAR 趋势、前高/前低动能比较 |
| F：多因子确认 | 30% | RSI(14)、KDJ(9,3,3)、Bollinger(20,2)、WR(10) |
| P：价格位置/趋势 | 15% | 250 日价格区间位置与价格涨幅，**不是财报或估值** |
| Q：量能 | 15% | OBV 5 日净量能流、量价背离、年化波动率代理，**不是实际主力资金流** |

18 项辅助指标：DMI、CCI、BIAS(6/12)、EXPMA、BBI、TRIX、VR、BR、AR、CR、DMA、DPO、MTM、SKDJ、LWR、ENE、LON。各投 +1/0/−1，不改变质量门禁。

预测至少需要 60 根日线；完整价格位置项需要更长历史。具体加减分见 CLI 输出或 Web「分析过程」，以引擎实现为准。

### 买入门禁

任一条件触发即否决买入：

- OBV 5 日净量能流 **<−60%**；
- RSI **>92**；
- M **<35** 且 F **<40**。

### 回测规则：收盘决策，下一交易日开盘执行

| 动作 | 旧加权条件 |
|---|---|
| 买入 | S* **≥69** 且通过全部门禁 |
| 止损卖出 | 含费用的浮动价差收益 **<−12%** |
| 止盈卖出 | 含费用的浮动价差收益 **>+15%** |
| 低分卖出 | S* **<35** |
| 获利回吐卖出 | 浮动价差收益 **>12%** 且 S* **<50** |
| 背离/形态卖出 | 顶背离；持仓超过 10 个交易日后每 5 日检查 M 顶/头肩顶破颈线、看跌吞没 |

- 买卖都是收盘信号、次日开盘成交，**不是盘中触价立即成交**，实际卖价不保证等于阈值。
- 比例费用：双边佣金各 0.03%、卖出印花税 0.05%。未完整建模历史税率、最低佣金、滑点和涨跌停成交限制。
- 按单笔持仓顺序配对并汇总已完成交易复利，**没有 25%→100% 分级仓位**。
- 期末未平仓不强制清仓、不计入已完成交易汇总；这个汇总不是完整资金账户净值。

### 旧加权建议不等于回测订单

预测根据最新评分和“已持仓”布尔值给建议，**没有实际买入成本或仓位输入**。“减持/增持”不代表实施了分仓，也不能替代成本相关止损。

| 前提 | S* | 未持仓 | 已持仓 |
|---|---:|---|---|
| 通过门禁 | ≥70 | 买入 | 增持 |
| 通过门禁 | 69～<70 | 买入 | 拿住 |
| 通过门禁 | 50～<69 | 观望 | 减持 |
| 通过门禁 | 40～<50 | 不买 | 减持 |
| 通过门禁 | <40 | 不买 | 卖出 |
| 门禁否决 | 任意 | 观望 | 观望 |

## 保留旧加权 Top-N 横截面海选

[scan_composite.py](scan_composite.py) 的 `run_scan()` 同时供旧海选 CLI 和 Web `/scan` 使用，不由默认 Pareto 报告调用：

1. 从缓存选择沪深主板代码（600/601/603/605/000/001/002/003），至少 300 根日线。
2. [factors.py](factors.py) 计算 **13 个技术因子，其中 12 个有权重**。
3. 每个因子横截面百分位排名，缺失项填中性 50；按带正负号的固定 ICIR 权重加权，再除以权重绝对值之和。
4. 获取腾讯报价，过滤无有效价格、PE≤0 或超上限、低于最低价的股票，按分数取前 N 只。

- 分数是**横截面相对分**，不是 S*、不是 0～100 概率，不能套用融合 69 分门槛。
- 低波、低 ATR、低成交额、均值回归等项参与评分。低成交额是流动性代理，不能直接当真实小市值。
- 旧海选 Web 默认 N=5，独立 CLI 默认 N=20；PE 默认上限 100、最低价默认 5 元，均可调整。
- 海选不自动刷新全市场 K 线；不同股票的因子日期、报价日期可能不一致，应逐行查看因子日期。
- 权重来自历史样本内研究；配套资金回测尚有调仓时点等问题，**未证明能取得稳定收益**。
- `/scan` 返回 HTML 片段，不是 JSON API，请从首页「旧加权海选（非帕累托）」标签使用。

## 保留长线持有与独立财务初筛

### 长线持有

从缓存起始日收盘买入，展示到缓存末日的价格变化；勾选分红后额外展示每股现金分红。

**旧长线路径限制**：尚未正确累乘送转股份、完善分红日期上限；涉及送转或缓存陈旧时，总收益可能错误，不能作为经校验的持有基准。

旧加权融合已完成交易可以事后补记现金分红与送转股份，但尚未将权益事件纳入其回测中的成本/止损状态。此问题不能与新独立账户账本混称；旧汇总现金分红以元/初始股解释，不复权价不能与前复权收益重复相加。

### 巴芒基本面研究

[fundamentals.py](fundamentals.py) 使用东方财富财务摘要年报，检查扣非盈利、经营现金流转化、ROE、杠杆和收入趋势。

不参与 Pareto34 或旧加权融合评分，不输出买卖信号或 DCF 目标价；金融机构需专用方法。少于三份年报不作结论，但尚未严格检查三份是否连续三年。初筛不能替代原始年报、审计、治理与护城河研究。

## 保留旧 JSON API 与小程序的边界

启动 [api.py](api.py) 后可访问：

- http://localhost:8100/api/analyze?code=000651&strategy=comprehensive
- http://localhost:8100/api/analyze?code=000651&strategy=buyhold&dividend=1
- http://localhost:8100/api/analyze?code=601888&strategy=value
- http://localhost:8100/api/health

API 支持 `holding=1`、`dividend=1`，默认策略仍为旧加权 `comprehensive`，没有 Pareto34 JSON 接口。新 Web 报告路由返回 HTML。

**尚未统一**：API 旧融合路径未传真实 `opens`，回测退化为次日收盘成交；汇总用逐笔价差简单相加，未采用旧 Web `comprehensive` 的复利/含分红汇总，也未使用 Pareto 独立账户。因此不能假定 API 与任一 Web 回测数值相同。小程序使用独立旧 JavaScript 引擎，不是 Pareto34 或 Python 旧加权策略的等价实现。

## 保留的脚本与用途

| 类别 | 文件 | 状态 |
|---|---|---|
| 入口 | [run_cli.py](run_cli.py)、[web.py](web.py)、[api.py](api.py) | 模式差异见上文 |
| 核心 | [engine.py](engine.py)、[fetcher.py](fetcher.py)、[db.py](db.py)、[plotting.py](plotting.py)、[fundamentals.py](fundamentals.py) | 计算、数据、缓存、绘图、初筛 |
| 保留旧加权海选 | [scan_composite.py](scan_composite.py)、[factors.py](factors.py) | Web 旧海选标签实际使用，非默认多维总览 |
| 旧 S* 快照海选 | [scan_top_n.py](scan_top_n.py)、[build_features.py](build_features.py) | 独立工具，不是 Web 旧 Top-N 算法 |
| 基线 | [batch_backtest.py](batch_backtest.py)、[mainboard_baseline.py](mainboard_baseline.py) | 100 股面板/主板全量任务，非实盘业绩 |
| 双路径实验 | [money_backtest4.py](money_backtest4.py) | 保留最新版本；加仓抢占止损、费用/权息等问题待修 |
| 海选资金实验 | [money_scan_backtest.py](money_scan_backtest.py)、[money_pareto_backtest.py](money_pareto_backtest.py) | 调仓资金重叠、样本内权重等问题待修 |
| 龙虎榜研究 | [lhb_pick.py](lhb_pick.py)、[lhb_factor_analysis.py](lhb_factor_analysis.py)、[lhb_lianban_height.py](lhb_lianban_height.py) | 事件/标签/历史时点待校验 |
| 资金流研究 | [fund_flow_analysis.py](fund_flow_analysis.py)、[fund_flow_backtest.py](fund_flow_backtest.py) | 次日收益等问题待修 |
| 财务采集 | [collect_fundamentals.py](collect_fundamentals.py)、[collect_financials_all.py](collect_financials_all.py)、[fetch_shares.py](fetch_shares.py) | 研究输入，不直接参与融合评分 |
| 市场采集 | [collect_market_data.py](collect_market_data.py)、[collect_lhb_10y.py](collect_lhb_10y.py)、[collect_lhb_resume.py](collect_lhb_resume.py)、[retry_fundflow.sh](retry_fundflow.sh) | 有破坏性入口，见下方警告 |
| 复权原型 | [build_adj_factors.py](build_adj_factors.py) | 未接入当前策略，不表示数据已复权 |
| 小程序 | [wechat_app/app.js](wechat_app/app.js) | 独立旧策略，未与 Python 对齐 |

新多维模块职责见前文。旧 [money_backtest4.py](money_backtest4.py) 保留评分叠加资金流、龙虎榜路径与分级仓位实验；旧 [money_pareto_backtest.py](money_pareto_backtest.py) 的 `non_dominated_set()` / `crowding_distance()` 比较低波、低 ATR、低成交额、接近 MA20、60 日反转这 5 项目标。后者 F1 不足 N 时跳过整期、未补后续层，调仓资金时序待修；不是每日 34 原子策略。低成交额不等于实际小市值；未被默认 Web 导入不代表工具废弃。

### 已删除的中间实验

2026-09-10 清理了因子回测 v1–v7、资金回测 v1–v3、旧 Web 融合渲染方法和已跟踪运行日志，不保留另一份归档副本。可从 [清理前提交 42673a8](https://github.com/myler/a_shares_predict/tree/42673a8) 按需恢复。

它们不是当前入口依赖。旧实验收益及“收益天花板”“某方法已证伪”“帕累托已证明优越”等结论已从首页移除；删除脚本不等于修复共用引擎缺陷。

## 数据安全与待修问题

**不要为尝试命令而直接运行全量采集或删除缓存。** 本地 SQLite 包含行情、财务、龙虎榜、资金流、冻结股票池和任务元数据，不保证能无损重抓。

- [collect_market_data.py](collect_market_data.py) 的 `lhb` 模式和 [collect_lhb_10y.py](collect_lhb_10y.py) **会先 DROP 龙虎榜表**，网络失败也可能丢旧数据。改成增量前先作一致性备份。
- K 线/分红缓存不自动更新。强制刷新会替换该股旧历史，可能缩短窗口、破坏基线可复现性。
- 旧行情获取仍按缓存→新浪→腾讯不复权→东方财富不复权回退；腾讯字段错位、旧 API 汇总、旧长线/融合权益风控、旧 MACD 前视、财务可得时点等仍待修。新账本并未修复所有旧路径，新 Pareto 的权息/成交近似也须保留披露。
- [db.py](db.py) 的读函数会走读写建表连接。旧基线 `--replay-cached` 虽不联网、不改回测摘要，底层仍可能建表/设置 WAL，**不是真正只读连接**；严格审查使用 SQLite URI `mode=ro` 和 `query_only`，不调用生产初始化，也不能恢复已覆盖的历史。
- 不把样本内回放当样本外验证；费用、权息、未成交、退市、期末持仓和资金占用都需明确统计口径。

详细证据及优先级见 [AUDIT_2026-09-09.md](AUDIT_2026-09-09.md)。

## 开发与验证

本轮最终隔离验证 **328 项测试通过**，阻断生产数据库与真实网络；另独立核验全部 3330 账户及 1,455,005 条流水。真实 HTTP 页面验证了首页、000001 和 N=5 全量总览，320/390/768/1024/1920 宽度无页面横向溢出，宽表在容器内滚动。这些是实现与账务验证，不是盈利证明。

```bash
.venv/bin/python -m unittest discover -s unittests -v
```

单元测试使用临时数据库与模拟网络，完整验证需阻止真实网络和生产缓存连接。测试覆盖原子/前沿、风险目标、账户、权息解析、运行器及 Web 契约；测试通过不代表策略盈利。

维护约定见 [CLAUDE.md](CLAUDE.md)。Pareto 首页说明读取模型默认参数、报告读取冻结参数；旧 `comprehensive` 文案仍读取旧回测默认值。修改时同步相应 README、Web 解释与契约测试，不混用新旧规则。历史审计证据链接固定到当时提交，不修改为当前代码链接。
