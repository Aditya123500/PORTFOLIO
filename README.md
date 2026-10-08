# Nifty 200 Contrarian Strategy

Simple long-only strategy using historical Nifty 200 stock lists.

## Files

- `DATA/_NIFTY_200_FETCHER.py` - Reads market-cap Excel files, keeps top 200 stocks, and downloads daily Yahoo Finance price data.
- `DATA/NIFTY_LIST/` - Original market-cap Excel files.
- `DATA/NIFTY_200_FROM_MCAP/` - Dated top-200 stock lists.
- `DATA/TICKER_DATA/` - Daily OHLCV data, split by valid list period.
- `STRATEGY/Contrarian.py` - Creates long-only trading signals.
- `STRATEGY/OUTPUT/CONTRARIAN/` - Generated signals and diagnostics.
- `BACKTESTER/backtest.py` - Tests signals using historical prices.
- `BACKTESTER/OUTPUT/CONTRARIAN/` - Metrics, trades, equity curve, open positions, and audit files.

## Strategy

1. Use only stocks present in top-200 list on that date.
2. Calculate each stock's 60-trading-day return.
3. Every 21 trading days, select 10 worst performers.
4. Create signal after market close.
5. Buy at next trading day's open.
6. At each monthly rebalance, close all positions and buy the new bottom-10 basket.

This is contrarian: buy recent losers expecting recovery.

## Backtester

- Invests up to INR 10,000 per trade using whole shares.
- Executes entries and exits at next available open.
- Charges 0.1% transaction cost on entry and exit.
- Squares off every position at each monthly rebalance, then opens the new basket.
- Also exits a stock when it leaves the top-200 list.
- Never uses future prices to create signals.
- Produces return, CAGR, Sharpe ratio, drawdown, win rate, trade log, and equity curve.

## Latest Backtest Result

Period: 8 April 2020 to 8 October 2026.

- Initial capital: INR 400,000
- Final equity: INR 537,840.80
- Total return: 34.46%
- CAGR: 4.66%
- Sharpe ratio: 1.13
- Maximum drawdown: -8.82%
- Win rate: 53.41%
- Profit factor: 1.72
- Closed trades: 747
- Transaction costs: INR 14,327.71
- Open positions at end: 10

## Run

```powershell
python DATA/_NIFTY_200_FETCHER.py
python STRATEGY/Contrarian.py
python BACKTESTER/backtest.py
```
