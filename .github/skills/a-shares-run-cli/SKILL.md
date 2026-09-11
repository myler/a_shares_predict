---
name: a-shares-run-cli
description: "Use when: running, explaining, or troubleshooting this repository's A-share entry points; current multidimensional or fusion analysis, 策略分析, 融合策略回测总览, live_data.refresh_stock, fresh daily quotes/actions, Pareto34 historical reports, pareto_backtest.py, legacy weighted fusion, run_cli.py predict, stock code analysis, MACD backtests, or charts. Distinguish targeted online analysis from read-only history and legacy CLI/API; current analysis must not default to predict."
argument-hint: "[fresh fusion analysis, historical report, or explicit CLI mode] [6-digit-stock-code]"
user-invocable: true
---

# A-Shares Analysis: Select the Entry Point First

Use this skill from the repository root. Read [../../../README.md](../../../README.md) for the current contracts; command syntax must agree with the actual entry-point source, not just its help text.

## Mode Selection — Before Any Command

| User intent | Entry point and boundary |
|---|---|
| Current multidimensional / 当前多维 / 融合策略 single-stock analysis | First Web tab「策略分析」, default `pareto`: refresh the target online on every request, then calculate 34-atom conditional diagnostics; targeted DB updates, not read-only or intraday quotes. Never substitute CLI `predict`. |
| Read existing Pareto results / 融合策略回测总览 | Second tab or completed local artifacts; `/pareto-summary` is read-only. Do not visit `/analyze` for a historical-only request. No backtest or refresh is needed. |
| Explicitly run a full Pareto research replay | Use the separate Pareto runner below, only when that full run is requested. This is not a CLI stock-prediction mode. |
| Explicit CLI `predict` or legacy weighted fusion | Run `predict`: old four-dimensional weighted scoring plus 18 auxiliary votes, not Pareto34. |
| Explicit MACD chart/backtest/JSON or legacy multi-factor analysis | Use the corresponding legacy CLI mode below; none is Pareto34. |
| Long hold / financial quality research | Web `buyhold` / `value`, or explicit CLI `bmfund` for financial research; separate legacy data paths, not the new fusion refresh chain or a validated holding benchmark. |

Use the supplied intent to distinguish current analysis from history; clarify only if genuinely ambiguous. A request for current fusion analysis permits its documented single-stock refresh, not whole-market collection. Under no-network/read-only constraints, explain that fresh analysis cannot run and inspect existing artifacts/source instead. Missing Pareto artifacts never justify substituting old weighted output. Viewing all results is not permission to rerun all accounts.

## Preconditions

- Use a user-supplied six-digit A-share code for a single-stock report or CLI analysis; preserve leading zeroes. A population overview/full replay does not require a single code.
- Run commands from the repository root using the existing project `.venv/bin/python`; from Windows PowerShell invoke `Ubuntu01` WSL explicitly. Do not use system Python or silently install dependencies/change the environment.
- Initial actions are mode selection and local inspection, not full-market replay, collection, or refresh. Run only the requested operation; a valid first-tab fusion request itself always performs a targeted online refresh.
- Read the actual implementation before changing entry-point documentation: [../../../web.py](../../../web.py), [../../../templates/page.html](../../../templates/page.html), [../../../live_data.py](../../../live_data.py), [../../../fusion_analysis.py](../../../fusion_analysis.py), [../../../fusion_web.py](../../../fusion_web.py), and [../../../fusion_chart.py](../../../fusion_chart.py). Do not carry forward the former read-only first-tab contract.
- Legacy CLI data/name/financial lookups can access the network and write caches. “No dividend fetch” does not mean “offline/read-only.” Under a no-network or strict-read-only request, inspect existing artifacts/source instead of running those entry points; do not call database helpers that initialize writable connections.

## Current Web: Exactly Two Tabs and Three Options

- Tabs are exactly「策略分析」and「融合策略回测总览」. Options are exactly「融合策略」(`pareto`, default),「长线持有」(`buyhold`),「巴芒基本面研究」(`value`).
- Current version remains `pareto_daily_atoms_risk_v1`: 34 separate ternary coordinates, same-day F1 and risk budgeting. No old four-score weights, auxiliary consensus or $S^*$ in current analysis.
- Web `comprehensive` / MACD / `multi` dispatch and old handler methods are removed (unsupported strategy values return 400); `/scan` returns 404. Internal template names `tab-scan` / `runScan()` now request only `/pareto-summary`. Do not restore hidden legacy controls or confuse these names with an active scan route.
- CLI, API, engine and scan/research scripts remain independently useful. Removing their Web entry points is not permission to delete scripts. API still defaults to old `comprehensive`, with no Pareto34 JSON interface; current Web responses are HTML.

### First Tab: Fresh Single-Stock Fusion Diagnostics

`/analyze?code=000651&strategy=pareto&holding=0&dividend=0` (strategy optional) calls `fusion_web.render_analysis()` → `fusion_analysis.analyze_stock()` → `live_data.refresh_stock()`. Each valid mainboard request refreshes first; `refresh=0` / `force=0` cannot make this route read-only.

- Only the requested stock is refreshed. The user's revised requirement explicitly permits targeted source-DB writes for this chain, superseding its former read-only rule; it does not permit whole-market refresh, unrelated table writes, deletion, or changing frozen reports.
- Save target-only old prices/metadata using a read-only connection; freeze raw responses, SHA256 and per-attempt/normalization evidence in a unique local refresh directory. After validation, `BEGIN IMMEDIATE` rechecks concurrent changes and UPSERTs only target prices/stock metadata, preserving history outside the response window. Read full committed target history afterward. Do not initialize schema, clear tables or write legacy dividend caches.
- Public price sources are Sina → unadjusted Tencent → unadjusted EM, one attempt per source. All fail: report「更新失败，暂不下结论」, never substitute stale cache, historical analysis or an old chart. A successful fresh source may be tried after another source fails; this is not stale fallback.
- `checked_at` is an aware request/check clock converted to Asia/Shanghai, not the quote date. `latest_date` / `date` is the actual latest completed daily bar: discard today's provisional bar before 15:00, reject future/invalid OHLCV, require provider latest ≥ cached actual `MAX(date)`. Same-date verification/correction is valid, not a new trading day. No verified holiday/suspension calendar: disclose the actual date, never rename an older bar as today's quote.
- Volume with existing history requires ≥3 positive overlapping pairs excluding cached latest day, all supporting one cached/provider factor from 1 / 100 / .01 within 1%. Insufficient, mixed or unknown units reject that source. Without history, normalize known shares/lots to shares; with history preserve cached units, do not relabel the whole DB.
- Every successful price refresh freshly requests per-code EM dated dividends (2015-01-01 through Shanghai check date), ALL undated implemented dividends, and ALL rights lifecycles without a date filter. Validate pagination and coverage; do not reuse frozen market action snapshots or fetch Sina actions. Failed/unverified actions or unknown/present rights block trading conclusions; raw-price diagnostics must be marked as such. Action failure does not undo an already committed price update; post-commit I/O failure must not be called a rollback.
- Exact EM empty exception: page 1 only, integer `code=9201`, `success=false`, explicit `result=null`, exact message「返回数据为空」. Normalize to an empty source dataset while retaining raw bytes, SHA256 and page proof. Other failures/missing fields/mismatches or later-page sentinels are not empty history. Source-empty does not prove historical absence. Action endpoint denial/rate limits stop remaining action requests, with no retry or access workaround.
- Analysis uses the same 34 atoms, eligibility/gates, causal action-continuous signal prices, volatility and historical ES as the backtest. Use frozen selected risk when a complete report exists (this run 10%); if the report is unavailable, clearly label model-default 15%, unavailable history and unknown F1, not selected 10% or old scoring.
- Compare only exact-date eligible, warmed-up, gate-passing frozen peers. This is a frozen-universe comparison, not a refreshed whole-market cross-section. Beyond the historical calendar/effective end (this run 2026-09-08), F1 is unknown and cannot yield a build/hold candidate. Never forward-fill peers' last rows/fronts, fabricate a one-stock F1, or truncate new target prices at that old cutoff. Historical final-five-day exits and historical account positions do not control the current holding flag.
- 「已持仓」is clickable and changes the next analysis's build/hold candidate or no-buy/exit-signal interpretation. It is only a boolean: no share quantity, cost, NAV peak or cooldown state, so no actual add/reduce amounts, executable target or personal drawdown-exit advice. Conditional risk cap is not actual position and cannot justify buying with unknown F1.
- UI shows「计算分红」when held; it adds the freshly retrieved per-share action table and a twelve-calendar-month simple cash sum. Rendering can also show it independently of holding. Action verification remains mandatory with the switch off. Neither checkbox changes frozen account profits/dividends. Cash is pre-tax yuan per pre-event share, not a normalized initial-share return or yield.
- Output order: condition symbol ✅/❌/⚠️ with non-probability legend; current analysis plus separate historical summary (check time, quote date, requested/effective backtest dates kept separate); all 34 details grouped M6/F5/P2/Q3/Aux18 and diagnostic statistics; historical annual NAV results and last 20 actual filled events, not 20 mixed ledger rows or live orders; optional new dividend table; final three-panel chart.
- Chart: unadjusted price/MA20/MA60 and available historical fills, action-continuous MACD, 34-row heatmap with no aggregate score. Only heatmap may sample to ≤1000 columns with disclosure. No inferred fills after history cutoff; action failure uses raw MACD with a warning. Chart failure keeps diagnostics without a stale image or second refresh.

### Second Tab: Read-Only Completed Historical Reports

- `/pareto-summary?n=5` shows full-population totals; N only truncates display in code order, not a ranking, F1 selection or change to total principal. No network, cache refresh or account replay.
- [../../../pareto_web.py](../../../pareto_web.py) retains `render_stock_report()` as an internal historical helper, not the first-tab HTTP handler. For individual history-only questions, read completed local artifacts/helper, not `/analyze`.
- Report directory is only the server's `PARETO_RUN_DIR`, never an HTTP path parameter; default is the completed 2026-09-10 run. HTTP cannot set DB/evidence directories either.
- Require final summary completion marker and compatible complete artifacts; report missing/incomplete history as unavailable, never regenerate automatically or fall back to weighted scoring. Current fresh diagnostics may continue separately without history, with the limitations above.

When Web viewing is requested and no suitable server is running, start it against the existing completed run. Inside WSL:

```bash
PARETO_RUN_DIR=output/pareto_mainboard_20260910 .venv/bin/python web.py 8099
```

From Windows PowerShell:

```powershell
wsl -d Ubuntu01 --cd /home/myl/a_shares_predict -- env PARETO_RUN_DIR=output/pareto_mainboard_20260910 .venv/bin/python web.py 8099
```

Use http://localhost:8099 with the intended route: homepage and `/pareto-summary` do not refresh; `/analyze` with `pareto` does. The server listens on all interfaces without authentication; use only in a trusted environment, not on the public Internet. Do not automatically open an analysis URL just to verify a historical-only server startup.

### Full Pareto Replay — Only on Explicit Request

Do not execute this as an initial action, for an ordinary stock analysis, to view all results, or to repair a missing report automatically. It creates/rebuilds outputs and can make public EM requests; it is not read-only browsing.

Example full-population invocation, using a **new** output directory and an existing frozen Eastmoney (EM) action snapshot:

```bash
.venv/bin/python -B pareto_backtest.py --output output/pareto_mainboard_new_run --action-snapshot output/action_fallback_market_2015_20260910 --start 2016-09-10 --end 2026-09-10 --workers 8
```

Windows PowerShell equivalent:

```powershell
wsl -d Ubuntu01 --cd /home/myl/a_shares_predict -- .venv/bin/python -B pareto_backtest.py --output output/pareto_mainboard_new_run --action-snapshot output/action_fallback_market_2015_20260910 --start 2016-09-10 --end 2026-09-10 --workers 8
```

- Replace the example output directory with a new run-specific directory. Preserve prior evidence; changing data/strategy versions requires a new directory, not silently changing a frozen sample on resume. A rerun rebuilds the completion summary.
- `--action-snapshot` is **optional**. Reusing a complete frozen EM directory with coverage matching **2015-01-01 through the requested end date** sends no HTTP. Omission automatically freezes a batch EM snapshot inside the run output; an incomplete supplied directory can also continue public EM requests. Merely specifying this option does not guarantee offline execution.
- Under a no-network constraint, do not start until a matching complete snapshot is verified locally. The runner does not invoke Sina downloads. On denial/rate limiting, retain failure evidence; do not bypass restrictions or automatically retry.
- Market prices come from a read-only consistent backup of the source cache; production prices are not refreshed. `--source` selects the source database. No `--limit` means the full eligible cached mainboard universe; `--limit` is only a small-sample smoke-test option, not full-market results.
- Do not use `--cached-actions` as a trusted offline substitute: unverified cached actions are marked `cache_unverified`, and affected accounts retain cash rather than trade.
- Wait for successful completion and the final summary before reading reports. Full-population totals must include all principal, including failed, quarantined and no-trade accounts; residual shares are not cash.

## Explicit Legacy CLI Commands

### Old Weighted Fusion Prediction

Inside `Ubuntu01` WSL at the repository root:

```bash
.venv/bin/python ./run_cli.py predict 601888
```

From Windows PowerShell:

```powershell
wsl -d Ubuntu01 --cd /home/myl/a_shares_predict -- .venv/bin/python ./run_cli.py predict 601888
```

Replace `601888` with the requested code. `predict` calls `predict_comprehensive()` and remains **old weighted fusion**, not the current Pareto default. It does not run a backtest, fetch dividends, create an output directory, or generate a chart. It does fetch/read K-lines and the stock name and save the name in the cache; do not claim it is strictly read-only or always offline.

Report these sections from its output:

1. Explicitly labelled old weighted fusion signal, recommendation, data date, close, and data range.
2. Four-dimensional scores: MACD core (40%), multi-factor (30%), price position/trend (15%), and volume (15%).
3. Base score $S$, auxiliary-consensus adjustment, and effective score $S^*$.
4. Quality-gate result and every veto reason when a gate fails.
5. All 18 auxiliary-indicator votes, including consensus and bullish, bearish, and neutral counts.
6. The rule-by-rule scoring details for all four dimensions.

Do not add backtest performance, dividend returns, Pareto rankings, or chart commentary to a `predict` response unless separately requested and obtained from the appropriate entry point.

### Other Supported Legacy Modes

```bash
.venv/bin/python ./run_cli.py chart <股票代码>
.venv/bin/python ./run_cli.py backtest <股票代码>
.venv/bin/python ./run_cli.py json <股票代码> [更多股票代码]
.venv/bin/python ./run_cli.py multi <股票代码>
.venv/bin/python ./run_cli.py bmfund <股票代码>
```

The angle/square-bracket arguments are documentation placeholders; substitute actual codes, do not type the brackets. From PowerShell use the same WSL prefix and project interpreter as the `predict` example, replacing the mode/code arguments.

- `chart`, and passing a code with no mode, run the legacy MACD backtest, dividend enrichment, chart and MACD prediction workflow. Despite the help text, `chart` is not chart-only.
- `backtest` runs the legacy MACD backtest with dividend enrichment and creates an output directory/report, but no chart.
- `json` emits legacy MACD predictions and supports multiple six-digit codes; it is not weighted fusion or Pareto JSON. Other modes accept exactly one code.
- `multi` runs the legacy multi-factor resonance backtest and prediction, including dividend enrichment when available.
- `bmfund` runs the independent Buffett-Munger-style financial quality screen for non-financial companies. It reports a research state, not a buy, sell, target price, or backtest result.
- None of these CLI modes provides current Pareto analysis or reports. The separate JSON API defaults to old `comprehensive`, has no Pareto34 JSON interface and does not use the new refresh chain; do not assume API/Web/CLI performance numbers are interchangeable. API still omits `opens` and simply sums trade price returns; do not describe it by comparison to a now-removed legacy Web renderer.

## Operating Procedure

1. Select fresh single-stock fusion, read-only Pareto history, full replay, or an explicitly requested legacy mode. Use the supplied code/intent; do not ask redundant questions or silently route current analysis to `predict`.
2. Read completed artifacts for historical-only questions; no `/analyze` request. Fresh fusion performs exactly the target refresh, not whole-market collection. Do not repeat a request merely to toggle presentation or recover a chart; UI resubmission is another fresh request.
3. Preserve check time, actual completed quote date, requested/effective historical end and data ranges separately. Never describe online verification as today's intraday price or reuse the old historical deadline for new diagnostics.
4. Summarize in Chinese using actual outputs: current conditions and F1 uncertainty, selected/default risk source, then independent historical money/actions. For old `predict`, show actual scores/gates/model output, not invented fills. Do not interpret a holding checkbox as an actual position or condition symbols as profit probabilities.
5. On failure, report the error and check `Ubuntu01`, repository working directory and project virtual environment before proposing changes. Missing Pareto output is not permission to fall back to old scoring or rerun a population. Do not install dependencies or alter code just to hide a mode mismatch.

## Maintenance and Validation Boundaries

- New module/test responsibilities are listed in [../../../README.md](../../../README.md): live data refresh/evidence, shared fusion analysis/exact-day F1, HTML/history/dividend rendering, in-memory three-panel chart, Web routing/controls, Tencent mapping and read-only report helpers. Read actual files before documenting them.
- The shared Tencent parser now correctly maps date/open/close/high/low/volume; do not keep listing that mapping bug as unfixed. Legacy `fetch_kline(force_refresh=True)` still replaces history and can truncate it; the new targeted UPSERT guarantee does not extend to all legacy refresh paths.
- 328 passing tests describe the pre-redesign historical validation. On 2026-09-11, the redesigned implementation passed 470 fully isolated tests; separately authorized live 000001 acceptance extended quotes from September 8 to September 10 without truncating older history, with verified actions but unknown current F1. Later changes require fresh verification; neither test counts nor this one-stock acceptance validate market-wide data or profitability. Check obsolete route assertions rather than restoring incorrect claims.
- Tests must use temporary databases and mocked/blocked network, not the authorized live production-refresh path. If the task is documentation-only/no-network/no-testing, do not execute analysis, browser routes that refresh, or tests. Preserve frozen report numbers and fixed-commit historical audit links.

## Interpretation Boundaries

- Old weighted fusion only: $S^* = S - 0.08A$; $A$ is auxiliary consensus, so negative consensus can increase the effective score. Scores are not probabilities; passing a gate does not guarantee returns. Price-position and volume dimensions use price/volume proxies, not financial statements or actual main-fund flows.
- Pareto keeps 34 separate ternary coordinates, not four weighted aggregates. “Independent” does not mean statistically independent; ternarization loses magnitude, and F1 neither picks a unique optimum nor means full investment.
- Current Pareto accounts are independent, normally 1 million yuan each; use the frozen run configuration. Equity = cash + residual marked shares; P/L = equity − all initial principal. Fees are already deducted and cash dividends already credited: do not count either twice. Report dividends in yuan/share with the relevant share basis.
- Trades use unadjusted prices with corporate actions separately accounted for; Pareto signal prices are corporate-action-continuous. Do not imply all legacy backtests/long-hold paths have complete total-return accounting. Disclose rights-issue quarantine, corporate-action approximations and non-executable orders.
- `RiskConfig` defaults to 15% annual volatility, while the documented 2026-09-10 run selected 10%. Historical reports and current analysis with a complete report use frozen selected parameters; current analysis without that report must disclose default 15%. Historical 15% drawdown exit and 20-market-day cooldown do not guarantee principal protection or cap realized drawdown and cannot be applied to a current holding boolean.
- The documented current experiment failed the principal-protection/profit objective. Do not present implementation, tests, training selection or literature references as profit evidence. Read completed results before quoting amounts, retain no-trade principal, disclose residuals and report annual results rather than only annualized returns. The post-2024 segment has already been inspected; it is not an untouched blind holdout.