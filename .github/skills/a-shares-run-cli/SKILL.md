---
name: a-shares-run-cli
description: "Use when: running, explaining, or troubleshooting this repository's run_cli.py A-share analysis commands; requests mentioning CLI stock prediction, fusion strategy, run_cli.py predict, stock code analysis, MACD backtests, or charts."
argument-hint: "[predict|chart|backtest|multi] <6-digit-stock-code>"
user-invocable: true
---

# A-Shares CLI Analysis

Use this skill to run the project's `run_cli.py` commands from the repository root and report the resulting market-analysis data accurately.

## Preconditions

- Use a six-digit A-share stock code supplied by the user.
- Run commands from the repository root.

## Direct Use in WSL

When the terminal is already inside the `Ubuntu01` WSL distribution and the repository root, run:

```bash
python3 ./run_cli.py predict 601888
```

Replace `601888` with the requested six-digit stock code, and replace `predict` with `chart`, `backtest`, or `multi` when needed.

## Use from Windows PowerShell

When the terminal is Windows PowerShell, invoke the workspace through WSL explicitly:

```powershell
wsl.exe -d Ubuntu01 --cd /home/myl/a_shares_predict -- python3 ./run_cli.py predict 601888
```

Keep the `/home/myl/a_shares_predict` working directory and replace the mode or stock code as required.

## Commands

### Fusion strategy prediction

```bash
python3 ./run_cli.py predict <股票代码>
```

`predict` is the preferred analysis command. It calls the structured fusion-strategy engine and does not run a backtest, fetch dividends, create an output directory, or generate a chart.

Report these sections from its output:

1. Fusion strategy signal, recommendation, date, close, and data range.
2. Four-dimensional scores: MACD core (40%), multi-factor (30%), price position/trend (15%), and volume (15%).
3. Base score $S$, auxiliary-consensus adjustment, and effective score $S^*$.
4. Quality-gate result and every veto reason when a gate fails.
5. All 18 auxiliary-indicator votes, including consensus and bullish, bearish, and neutral counts.
6. The rule-by-rule scoring details for all four dimensions.

Do not add historical backtest performance or chart commentary to a `predict` response unless the user separately requests it.

### Other supported modes

```bash
python3 ./run_cli.py chart <股票代码>
python3 ./run_cli.py backtest <股票代码>
python3 ./run_cli.py multi <股票代码>
```

- `chart` runs the legacy MACD workflow and creates a chart.
- `backtest` runs the legacy MACD backtest workflow.
- `multi` runs the multi-factor resonance workflow.
- Do not present output from these modes as the current fusion strategy.

## Operating Procedure

1. Confirm the requested stock code and mode. Default a request for a current recommendation to `predict`.
2. Run exactly one targeted command unless comparison or batch analysis was requested.
3. Preserve the returned data date. Do not describe cached or historical prices as live intraday prices.
4. Summarize the result in Chinese with the actual scores, gate status, and recommendation. Distinguish model output from investment advice.
5. For a command failure, report the error text and first check whether the command ran through `Ubuntu01` WSL before changing code or dependencies.

## Interpretation Boundaries

- The fusion score is $S^* = S - \beta A$; $A$ is the auxiliary-indicator consensus. A negative consensus can increase the effective score.
- A passed quality gate is necessary for a buy signal but does not guarantee a future return.
- The price-position/trend dimension uses price-series proxies, not financial-statement fundamentals.
- This project uses unadjusted prices; cash dividends and share distributions are accounted for separately in backtests, not in `predict` output.