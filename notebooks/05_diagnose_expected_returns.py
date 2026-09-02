from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.data.reader import load_universe
from src.features.returns import calculate_return_matrix
from src.features.statistics import rolling_mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------
# Load configuration
# ---------------------------------------------------------

with open(PROJECT_ROOT / "configs" / "universe.yaml", "r") as f:
    config = yaml.safe_load(f)

tickers = config["universe"]["tickers"]


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

price_data = load_universe(tickers)

returns = calculate_return_matrix(
    price_data,
    price_column="Close",
    method="simple",
)

expected_returns = rolling_mean(
    returns,
    window=60,
)


# ---------------------------------------------------------
# Basic diagnostics
# ---------------------------------------------------------

print("=" * 60)
print("EXPECTED RETURN DIAGNOSTICS")
print("=" * 60)

print(f"\nExpected returns shape: {expected_returns.shape}")
print(
    f"Expected return period: "
    f"{expected_returns.index.min().date()} "
    f"to "
    f"{expected_returns.index.max().date()}"
)


# ---------------------------------------------------------
# Distribution of expected returns
# ---------------------------------------------------------

flat_expected_returns = expected_returns.stack()

print("\nDistribution of daily expected returns")
print("--------------------------------------")

print(flat_expected_returns.describe())

print("\nAnnualized equivalent expected returns")
print("--------------------------------------")

annualized = flat_expected_returns * 252

print(annualized.describe())


# ---------------------------------------------------------
# Cross-sectional ranking stability
# ---------------------------------------------------------

rankings = expected_returns.rank(
    axis=1,
    ascending=False,
    method="min",
)

rank_changes = rankings.diff().abs()

print("\nRanking stability")
print("-----------------")

print(
    "Average absolute rank change:",
    rank_changes.stack().mean(),
)

print(
    "Median absolute rank change:",
    rank_changes.stack().median(),
)

print(
    "Maximum absolute rank change:",
    rank_changes.stack().max(),
)


# ---------------------------------------------------------
# Average expected return by ticker
# ---------------------------------------------------------

mean_expected_returns = expected_returns.mean().sort_values(
    ascending=False
)

print("\nAverage expected return by ticker")
print("--------------------------------")

for ticker, value in mean_expected_returns.items():
    print(
        f"{ticker:5s} "
        f"{value: .6f} daily "
        f"({value * 252: .2%} annualized)"
    )


# ---------------------------------------------------------
# Expected-return volatility by ticker
# ---------------------------------------------------------

expected_return_volatility = (
    expected_returns.std()
    .sort_values(ascending=False)
)

print("\nExpected-return estimate volatility")
print("-----------------------------------")

for ticker, value in expected_return_volatility.items():
    print(
        f"{ticker:5s} "
        f"{value: .6f} daily "
        f"({value * 252: .2%} annualized)"
    )


# ---------------------------------------------------------
# Plot selected stocks
# ---------------------------------------------------------

selected_tickers = [
    "AAPL",
    "NVDA",
    "JPM",
    "KO",
    "COST",
]

ax = expected_returns[selected_tickers].plot(
    figsize=(12, 6)
)

ax.axhline(0, linestyle="--")

ax.set_title("60-Day Rolling Expected Returns")
ax.set_xlabel("Date")
ax.set_ylabel("Expected Daily Return")

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# Rolling cross-sectional spread
# ---------------------------------------------------------

cross_sectional_spread = (
    expected_returns.max(axis=1)
    - expected_returns.min(axis=1)
)

print("\nCross-sectional expected-return spread")
print("--------------------------------------")

print(cross_sectional_spread.describe())

print(
    "\nAverage spread:",
    cross_sectional_spread.mean(),
)

print(
    "Maximum spread:",
    cross_sectional_spread.max(),
)


# ---------------------------------------------------------
# Top-ranked stocks over time
# ---------------------------------------------------------

valid_expected_returns = expected_returns.dropna(how="all")

top_tickers = valid_expected_returns.idxmax(axis=1)

top_frequency = (
    top_tickers.value_counts(normalize=True)
    .sort_values(ascending=False)
)

print("\nMost frequently highest-ranked stocks")
print("--------------------------------------")

for ticker, frequency in top_frequency.items():
    print(
        f"{ticker:5s} "
        f"{frequency: .2%}"
    )


print("\nDiagnostic complete.")