import pandas as pd
import yaml

from src.data.reader import load_universe
from src.features.returns import calculate_return_matrix
from src.features.statistics import rolling_mean
from src.features.covariance import rolling_covariance

from src.backtesting.config import BacktestConfig
from src.backtesting.engine import WalkForwardBacktester
from src.backtesting.benchmark import EqualWeightBacktester


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


# ============================================================
# Run Equal-Weight Benchmark
# ============================================================

benchmark_backtester = EqualWeightBacktester(
    config=backtest_config,
)

benchmark_result = benchmark_backtester.run(
    returns=returns,
    rebalance_dates=result.weights.index,
)


# ============================================================
# Turnover Statistics
# ============================================================

quant_turnover = result.turnover
equal_turnover = benchmark_result.turnover


def turnover_summary(series: pd.Series) -> pd.Series:
    return pd.Series(
        {
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "minimum": series.min(),
            "maximum": series.max(),
            "90th_percentile": series.quantile(0.90),
            "95th_percentile": series.quantile(0.95),
        }
    )


print("\nQuantPilot Turnover Statistics")
print("==============================")

print(
    turnover_summary(quant_turnover)
    .to_string()
)


print("\nEqual-Weight Turnover Statistics")
print("================================")

print(
    turnover_summary(equal_turnover)
    .to_string()
)


# ============================================================
# Rebalance-by-Rebalance Comparison
# ============================================================

turnover_comparison = pd.DataFrame(
    {
        "QuantPilot": quant_turnover,
        "Equal-Weight": equal_turnover,
    }
)

turnover_comparison["difference"] = (
    turnover_comparison["QuantPilot"]
    - turnover_comparison["Equal-Weight"]
)

turnover_comparison["ratio"] = (
    turnover_comparison["QuantPilot"]
    / turnover_comparison["Equal-Weight"].replace(0, pd.NA)
)


print("\nTurnover Comparison")
print("===================")

print(
    turnover_comparison.round(4).to_string()
)


# ============================================================
# Largest QuantPilot Rebalances
# ============================================================

largest_rebalances = (
    quant_turnover
    .sort_values(ascending=False)
    .head(10)
)


print("\nLargest QuantPilot Rebalances")
print("=============================")

print(
    largest_rebalances.round(4).to_string()
)


# ============================================================
# Transaction Costs
# ============================================================

quant_transaction_costs = result.transaction_costs
equal_transaction_costs = benchmark_result.transaction_costs

cost_comparison = pd.DataFrame(
    {
        "QuantPilot": quant_transaction_costs,
        "Equal-Weight": equal_transaction_costs,
    }
).fillna(0.0)


print("\nTransaction Cost Summary")
print("========================")

print(
    f"QuantPilot total transaction costs: "
    f"{quant_transaction_costs.sum():.6f}"
)

print(
    f"Equal-Weight total transaction costs: "
    f"{equal_transaction_costs.sum():.6f}"
)

print(
    f"QuantPilot total slippage: "
    f"{result.slippage.sum():.6f}"
)

print(
    f"Equal-Weight total slippage: "
    f"{benchmark_result.slippage.sum():.6f}"
)


# ============================================================
# Cost Contribution
# ============================================================

quant_total_cost = (
    result.transaction_costs.sum()
    + result.slippage.sum()
)

equal_total_cost = (
    benchmark_result.transaction_costs.sum()
    + benchmark_result.slippage.sum()
)


print("\nTotal Trading Friction")
print("======================")

print(
    f"QuantPilot: "
    f"{quant_total_cost:.6f}"
)

print(
    f"Equal-Weight: "
    f"{equal_total_cost:.6f}"
)

print(
    f"QuantPilot / Equal-Weight cost ratio: "
    f"{quant_total_cost / equal_total_cost:.2f}x"
)


# ============================================================
# Rebalance Frequency
# ============================================================

print("\nRebalance Frequency")
print("===================")

print(
    f"Number of QuantPilot rebalances: "
    f"{len(quant_turnover)}"
)

print(
    f"Number of Equal-Weight rebalances: "
    f"{len(equal_turnover)}"
)

print(
    f"Rebalances with QuantPilot turnover > 20%: "
    f"{(quant_turnover > 0.20).sum()}"
)

print(
    f"Rebalances with QuantPilot turnover > 30%: "
    f"{(quant_turnover > 0.30).sum()}"
)

print(
    f"Rebalances with QuantPilot turnover > 50%: "
    f"{(quant_turnover > 0.50).sum()}"
)