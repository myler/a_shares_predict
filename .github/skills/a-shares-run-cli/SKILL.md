---
name: a-shares-run-cli
description: "Use when: running, explaining, or troubleshooting this repository's A-share analysis entry points; requests mentioning current multidimensional analysis, Pareto34 historical reports, pareto_backtest.py, legacy weighted fusion, run_cli.py predict, stock code analysis, MACD backtests, or charts. Select current Pareto reports versus legacy CLI explicitly; current multidimensional requests must not default to predict."
argument-hint: "[Pareto report or explicit CLI mode] [6-digit-stock-code]"
user-invocable: true
---

# A-Shares Analysis: Select the Entry Point First

Use this skill from the repository root. Read [../../../README.md](../../../README.md) for the current contracts; command syntax must agree with the actual entry-point source, not just its help text.

## Mode Selection — Before Any Command

| User intent | Entry point and boundary |
|---|---|
| Current multidimensional analysis / 当前多维 / Pareto34 | Current Web `pareto`: read a completed historical replay, not a live recommendation. Never automatically route to CLI `predict`. |
| Read existing Pareto results or the full-account overview | Read completed local artifacts or the Pareto Web routes below. No backtest or data collection is needed. |
| Explicitly run a full Pareto research replay | Use the separate Pareto runner below, only when that full run is requested. This is not a CLI stock-prediction mode. |
| Explicit CLI `predict` or legacy weighted fusion | Run `predict`: old four-dimensional weighted scoring plus 18 auxiliary votes, not Pareto34. |
| Explicit MACD chart/backtest/JSON or legacy multi-factor analysis | Use the corresponding legacy CLI mode below; none is Pareto34. |
| Financial quality research | Independent `bmfund` screen; not a technical strategy or trading signal. |

For an ambiguous stock-code-only or “current recommendation” request, establish the intended mode before executing a command. Explain that the current default provides historical reports, not current buy/sell recommendations. Do not silently substitute old weighted output, even if Pareto artifacts are missing. A request to view all results is not permission to rerun all accounts.

## Preconditions

- Use a user-supplied six-digit A-share code for a single-stock report or CLI analysis; preserve leading zeroes. A population overview/full replay does not require a single code.
- Run commands from the repository root using the existing project `.venv/bin/python`; from Windows PowerShell invoke `Ubuntu01` WSL explicitly. Do not use system Python or silently install dependencies/change the environment.
- Initial actions are mode selection and local inspection, not full-market replay, collection, or refresh. Only perform the requested operation.
- Legacy CLI data/name/financial lookups can access the network and write caches. “No dividend fetch” does not mean “offline/read-only.” Under a no-network or strict-read-only request, inspect existing artifacts/source instead of running those entry points; do not call database helpers that initialize writable connections.

## Current Pareto: Completed Historical Reports

- Current version: `pareto_daily_atoms_risk_v1`, 34 separate ternary coordinates, same-day F1 and risk-budgeted independent accounts. Do not explain it using the old four-score weights or auxiliary consensus.
- Web `/analyze?code=000651&strategy=pareto` (also the default when strategy is omitted) reads frozen features, fronts, accounts and ledgers. `/pareto-summary?n=5` shows full-population totals; N only truncates the display in code order, not a ranking or F1 selection.
- Pareto pages do not fetch data, refresh caches, rerun accounts, or fall back to weighted scoring. Holding/dividend switches do not alter frozen accounts. These guarantees do not extend to legacy Web modes or `/scan`.
- The report directory is controlled only by the server's `PARETO_RUN_DIR`, never an HTTP path parameter. The default is the completed 2026-09-10 run shown below.
- Require the runner's final summary completion marker and compatible, complete artifacts. Missing, incomplete or incompatible reports must be reported as unavailable, not automatically regenerated. An existing report requires only reading it or starting Web, not rerunning the backtest.

When Web viewing is requested and no suitable server is running, start it against the existing completed run. Inside WSL:

```bash
PARETO_RUN_DIR=output/pareto_mainboard_20260910 .venv/bin/python web.py 8099
```

From Windows PowerShell:

```powershell
wsl -d Ubuntu01 --cd /home/myl/a_shares_predict -- env PARETO_RUN_DIR=output/pareto_mainboard_20260910 .venv/bin/python web.py 8099
```

Use http://localhost:8099 with the routes above. The server listens on all interfaces without authentication; use only in a trusted environment, not on the public Internet.

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
- None of these CLI modes provides current Pareto reports. The separate JSON API also defaults to old `comprehensive` and has no Pareto34 JSON interface; do not assume API/Web/CLI performance numbers are interchangeable.

## Operating Procedure

1. Select current Pareto history versus an explicitly requested legacy mode before running anything; confirm a six-digit code only when needed.
2. Read existing completed Pareto artifacts for historical questions. Otherwise run exactly one targeted command unless comparison, batch analysis or full replay was explicitly requested. Do not refresh caches or start collection as a convenience.
3. Preserve requested/effective end dates and actual data ranges. Never describe cached or historical prices as live intraday prices.
4. Summarize in Chinese, label the strategy/version, and report actual outputs only. For Pareto, show frozen risk parameters and real account actions; for old `predict`, show actual scores, gates and model recommendation, not invented fills. Distinguish model output from investment advice.
5. On failure, report the error and check `Ubuntu01`, repository working directory and project virtual environment before proposing changes. Missing Pareto output is not permission to fall back to old scoring or rerun a population. Do not install dependencies or alter code just to hide a mode mismatch.

## Interpretation Boundaries

- Old weighted fusion only: $S^* = S - 0.08A$; $A$ is auxiliary consensus, so negative consensus can increase the effective score. Scores are not probabilities; passing a gate does not guarantee returns. Price-position and volume dimensions use price/volume proxies, not financial statements or actual main-fund flows.
- Pareto keeps 34 separate ternary coordinates, not four weighted aggregates. “Independent” does not mean statistically independent; ternarization loses magnitude, and F1 neither picks a unique optimum nor means full investment.
- Current Pareto accounts are independent, normally 1 million yuan each; use the frozen run configuration. Equity = cash + residual marked shares; P/L = equity − all initial principal. Fees are already deducted and cash dividends already credited: do not count either twice. Report dividends in yuan/share with the relevant share basis.
- Trades use unadjusted prices with corporate actions separately accounted for; Pareto signal prices are corporate-action-continuous. Do not imply all legacy backtests/long-hold paths have complete total-return accounting. Disclose rights-issue quarantine, corporate-action approximations and non-executable orders.
- `RiskConfig` defaults to 15% annual volatility, while the documented 2026-09-10 run selected 10%; report frozen selected parameters, not defaults. A 15% drawdown exit and 20-market-day cooldown do not guarantee principal protection or cap realized drawdown.
- The documented current experiment failed the principal-protection/profit objective. Do not present implementation, tests, training selection or literature references as profit evidence. Read completed results before quoting amounts, retain no-trade principal, disclose residuals and report annual results rather than only annualized returns. The post-2024 segment has already been inspected; it is not an untouched blind holdout.