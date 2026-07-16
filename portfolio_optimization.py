import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Configuration

TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "SPY"]
START_DATE = "2020-01-01"
RISK_FREE_RATE = 0.02
TRADING_DAYS = 252
NUMBER_OF_PORTFOLIOS = 20_000
RANDOM_SEED = 42

# Market data

prices = yf.download(
    TICKERS,
    start=START_DATE,
    auto_adjust=True,
    progress=False,
)["Close"]

prices = prices.dropna()

returns = prices.pct_change().dropna()

annual_returns = returns.mean() * TRADING_DAYS
annual_covariance = returns.cov() * TRADING_DAYS


# Portfolio simulation

rng = np.random.default_rng(RANDOM_SEED)

portfolio_results = []

for _ in range(NUMBER_OF_PORTFOLIOS):
    weights = rng.random(len(TICKERS))
    weights = weights / weights.sum()

    portfolio_return = np.dot(weights, annual_returns)

    portfolio_variance = np.dot(
        weights.T,
        np.dot(annual_covariance, weights),
    )

    portfolio_volatility = np.sqrt(portfolio_variance)

    sharpe_ratio = (
        portfolio_return - RISK_FREE_RATE
    ) / portfolio_volatility

    portfolio_results.append(
        {
            "return": portfolio_return,
            "volatility": portfolio_volatility,
            "sharpe_ratio": sharpe_ratio,
            **{
                f"weight_{ticker}": weight
                for ticker, weight in zip(TICKERS, weights)
            },
        }
    )

results = pd.DataFrame(portfolio_results)

# Optimal simulated portfolios

max_sharpe_index = results["sharpe_ratio"].idxmax()
min_volatility_index = results["volatility"].idxmin()

max_sharpe_portfolio = results.loc[max_sharpe_index]
min_volatility_portfolio = results.loc[min_volatility_index]


# Summary tables

def build_summary(portfolio: pd.Series) -> pd.Series:
    summary = pd.Series(
        {
            "Annual Return": portfolio["return"],
            "Annual Volatility": portfolio["volatility"],
            "Sharpe Ratio": portfolio["sharpe_ratio"],
        }
    )

    for ticker in TICKERS:
        summary[f"{ticker} Weight"] = portfolio[f"weight_{ticker}"]

    return summary


summary = pd.DataFrame(
    {
        "Maximum Sharpe": build_summary(max_sharpe_portfolio),
        "Minimum Volatility": build_summary(min_volatility_portfolio),
    }
)

print("\nAnnualized Asset Returns")
print(annual_returns.round(4))

print("\nAnnualized Covariance Matrix")
print(annual_covariance.round(4))

print("\nOptimal Portfolio Summary")
print(summary.round(4))


# Portfolio opportunity set

plt.figure(figsize=(11, 7))

scatter = plt.scatter(
    results["volatility"],
    results["return"],
    c=results["sharpe_ratio"],
    alpha=0.6,
)

plt.scatter(
    max_sharpe_portfolio["volatility"],
    max_sharpe_portfolio["return"],
    marker="*",
    s=250,
    label="Maximum Sharpe",
)

plt.scatter(
    min_volatility_portfolio["volatility"],
    min_volatility_portfolio["return"],
    marker="X",
    s=180,
    label="Minimum Volatility",
)

plt.xlabel("Annualized Volatility")
plt.ylabel("Expected Annual Return")
plt.title("Simulated Portfolio Opportunity Set")
plt.colorbar(scatter, label="Sharpe Ratio")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Portfolio weights

weight_comparison = pd.DataFrame(
    {
        "Maximum Sharpe": [
            max_sharpe_portfolio[f"weight_{ticker}"]
            for ticker in TICKERS
        ],
        "Minimum Volatility": [
            min_volatility_portfolio[f"weight_{ticker}"]
            for ticker in TICKERS
        ],
    },
    index=TICKERS,
)

weight_comparison.plot(
    kind="bar",
    figsize=(10, 6),
)

plt.title("Optimal Portfolio Weights")
plt.xlabel("Asset")
plt.ylabel("Portfolio Weight")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.tight_layout()
plt.show()
