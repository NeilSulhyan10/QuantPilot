import pandas as pd
import yaml

from src.data.reader import load_universe
from src.features.returns import calculate_return_matrix
from src.features.statistics import rolling_mean
from src.features.covariance import rolling_covariance

from src.backtesting.config import BacktestConfig
from src.backtesting.engine import WalkForwardBacktester


# ============================================================
# Load Universe Configuration
# ============================================================

with open("configs/universe.yaml", "r") as file:
    config = yaml.safe_load(file)

tickers = config["universe"]["tickers"]


# ============================================================
# Load Data
# ============================================================

prices = load_universe(tickers)

returns = calculate_return_matrix(
    prices,
    price_column="Close",
    method="simple",
)


# ============================================================
# Calculate Rolling Features
# ============================================================

expected_returns = rolling_mean(
    returns,
    window=60,
)

covariance_matrices = rolling_covariance(
    returns,
    window=60,
)


# ============================================================
# Run QuantPilot
# ============================================================

backtest_config = BacktestConfig()

backtester = WalkForwardBacktester(
    config=backtest_config,
)

result = backtester.run(
    returns=returns,
    expected_returns=expected_returns,
    covariance_matrices=covariance_matrices,
)


weights = result.weights.copy()


# ============================================================
# Basic Weight Statistics
# ============================================================

weight_summary = pd.DataFrame({
    "mean_weight": weights.mean(),
    "median_weight": weights.median(),
    "min_weight": weights.min(),
    "max_weight": weights.max(),
    "std_weight": weights.std(),
})


print("\nWeight Summary")
print("==============")

print(
    weight_summary
    .sort_values("mean_weight", ascending=False)
    .to_string()
)


# ============================================================
# Position Activity
# ============================================================

tolerance = 1e-6

active_positions = (weights > tolerance).sum(axis=1)

positions_at_cap = (
    (weights >= backtest_config.max_weight - tolerance)
    .sum(axis=1)
)


print("\nPortfolio Structure")
print("===================")

print(
    f"Average active positions: "
    f"{active_positions.mean():.2f}"
)

print(
    f"Minimum active positions: "
    f"{active_positions.min()}"
)

print(
    f"Maximum active positions: "
    f"{active_positions.max()}"
)

print(
    f"Average positions at {backtest_config.max_weight:.0%} cap: "
    f"{positions_at_cap.mean():.2f}"
)

print(
    f"Maximum positions at cap: "
    f"{positions_at_cap.max()}"
)


# ============================================================
# How Often Each Stock Hits the Maximum Weight
# ============================================================

cap_frequency = (
    weights >= backtest_config.max_weight - tolerance
).mean()

cap_frequency = (
    cap_frequency
    .sort_values(ascending=False)
    .rename("cap_frequency")
)


print("\nPosition Cap Frequency")
print("======================")

print(cap_frequency.to_string())


# ============================================================
# Portfolio Concentration
# ============================================================

herfindahl = (
    weights.pow(2)
    .sum(axis=1)
)

effective_number = 1.0 / herfindahl


print("\nConcentration")
print("=============")

print(
    f"Average Herfindahl Index: "
    f"{herfindahl.mean():.4f}"
)

print(
    f"Average Effective Number of Stocks: "
    f"{effective_number.mean():.2f}"
)

print(
    f"Minimum Effective Number of Stocks: "
    f"{effective_number.min():.2f}"
)

print(
    f"Maximum Effective Number of Stocks: "
    f"{effective_number.max():.2f}"
)


# ============================================================
# Top 10 Stocks by Average Allocation
# ============================================================

print("\nTop Stocks by Average Allocation")
print("================================")

print(
    weight_summary["mean_weight"]
    .sort_values(ascending=False)
    .head(10)
    .to_string()
)


# ============================================================
# Weight Snapshot
# ============================================================

print("\nFirst Five Rebalance Weight Snapshots")
print("=====================================")

print(
    weights.head().round(4).to_string()
)


print("\nLast Five Rebalance Weight Snapshots")
print("====================================")

print(
    weights.tail().round(4).to_string()
)