# Portfolio Optimization

## Overview

This project applies modern portfolio theory to construct and compare diversified equity portfolios using historical market data.

The analysis estimates expected returns, volatility, and asset covariances, then simulates alternative portfolio allocations to identify the minimum-volatility and maximum-Sharpe portfolios.

## Objectives

- Retrieve and process historical market data
- Calculate annualized returns and the covariance matrix
- Generate random portfolio allocations
- Estimate portfolio return, volatility, and Sharpe ratio
- Identify the minimum-volatility portfolio
- Identify the maximum-Sharpe portfolio
- Visualize the portfolio opportunity set

## Assets

- Apple — AAPL
- Microsoft — MSFT
- NVIDIA — NVDA
- JPMorgan Chase — JPM
- SPDR S&P 500 ETF Trust — SPY

## Methodology

Daily adjusted prices are converted into simple returns.

Expected annual returns are estimated using the historical mean daily return multiplied by 252 trading days.

The annualized covariance matrix is estimated from daily returns and multiplied by 252.

For each simulated portfolio, random non-negative asset weights are generated and normalized so that their sum equals one.

Portfolio volatility is calculated using the covariance matrix. The Sharpe ratio assumes a 2% annual risk-free rate.

## Technologies

- Python
- pandas
- NumPy
- Matplotlib
- yfinance

## Outputs

- Annualized asset returns
- Annualized covariance matrix
- Simulated portfolio return and volatility
- Maximum-Sharpe portfolio
- Minimum-volatility portfolio
- Portfolio opportunity-set visualization

## Limitations

- Expected returns are based on historical averages
- Short selling is not allowed
- Transaction costs and taxes are excluded
- Portfolio weights are assumed to remain constant
- Historical covariance may not represent future market relationships

## Future Improvements

- Numerical optimization with SciPy
- Efficient frontier construction
- Alternative return-estimation methods
- Regularized covariance matrices
- Portfolio constraints and transaction costs
- Out-of-sample backtesting
