import pandas as pd
import yaml

from src.data.reader import load_universe
from src.data.benchmark_reader import load_benchmark

from src.features.returns import calculate_return_matrix
from src.features.covariance import rolling_covariance

from src.backtesting.config import BacktestConfig
from src.backtesting.engine import WalkForwardBacktester
from src.backtesting.benchmark import EqualWeightBacktester
from src.backtesting.buy_and_hold import BuyAndHoldBacktester

from src.evaluation.metrics import evaluate_returns
from src.evaluation.alignment import align_return_series

from src.models.expected_returns import calculate_expected_returns


# ============================================================
# Load Universe Configuration
# ============================================================

with open("configs/universe.yaml", "r") as file:
    config = yaml.safe_load(file)

tickers = config["universe"]["tickers"]


# ============================================================
# Load Price Data
# ============================================================

prices = load_universe(tickers)

spy_prices = load_benchmark("SPY")


# ============================================================
# Calculate Asset Returns
# ============================================================

returns = calculate_return_matrix(
    prices,
    price_column="Close",
    method="simple",
)

spy_returns = calculate_return_matrix(
    {"SPY": spy_prices},
    price_column="Close",
    method="simple",
)["SPY"]


# ============================================================
# Calculate Rolling Features
# ============================================================

expected_returns = calculate_expected_returns(
    returns,
    window=60,
    alpha=0.5,
)

covariance_matrices = rolling_covariance(
    returns,
    window=60,
)


print("Tickers:", len(tickers))
print("Returns shape:", returns.shape)
print("Expected returns shape:", expected_returns.shape)
print("Covariance matrices:", len(covariance_matrices))
print("SPY returns:", spy_returns.shape)


# ============================================================
# Backtest Configuration
# ============================================================

backtest_config = BacktestConfig()


# ============================================================
# QuantPilot Backtest
# ============================================================

backtester = WalkForwardBacktester(
    config=backtest_config,
)

result = backtester.run(
    returns=returns,
    expected_returns=expected_returns,
    covariance_matrices=covariance_matrices,
)

quantpilot_metrics = evaluate_returns(
    result.returns,
)


print("\nQuantPilot Performance")
print("----------------------")

for name, value in quantpilot_metrics.items():
    print(f"{name}: {value:.4f}")

print(
    f"Backtest return observations: "
    f"{len(result.returns)}"
)

print(
    f"Rebalance dates: "
    f"{len(result.weights)}"
)

print(
    f"Average turnover: "
    f"{result.turnover.mean():.4f}"
)

print(
    f"Total transaction costs: "
    f"{result.transaction_costs.sum():.6f}"
)

print(
    f"Total slippage: "
    f"{result.slippage.sum():.6f}"
)


# ============================================================
# Equal-Weight Benchmark
# ============================================================

benchmark_backtester = EqualWeightBacktester(
    config=backtest_config,
)

benchmark_result = benchmark_backtester.run(
    returns=returns,
    rebalance_dates=result.weights.index,
)

benchmark_metrics = evaluate_returns(
    benchmark_result.returns,
)


print("\nEqual-Weight Benchmark")
print("----------------------")

for name, value in benchmark_metrics.items():
    print(f"{name}: {value:.4f}")

print(
    f"Benchmark return observations: "
    f"{len(benchmark_result.returns)}"
)

print(
    f"Benchmark rebalance dates: "
    f"{len(benchmark_result.weights)}"
)

print(
    f"Benchmark average turnover: "
    f"{benchmark_result.turnover.mean():.4f}"
)

print(
    f"Benchmark total transaction costs: "
    f"{benchmark_result.transaction_costs.sum():.6f}"
)

print(
    f"Benchmark total slippage: "
    f"{benchmark_result.slippage.sum():.6f}"
)


# ============================================================
# Buy-and-Hold Benchmark
# ============================================================

buy_hold_backtester = BuyAndHoldBacktester()

buy_hold_result = buy_hold_backtester.run(
    returns
)

buy_hold_metrics = evaluate_returns(
    buy_hold_result.returns
)


print("\nBuy-and-Hold Benchmark")
print("----------------------")

for name, value in buy_hold_metrics.items():
    print(f"{name}: {value:.4f}")

print(
    f"Buy-and-Hold return observations: "
    f"{len(buy_hold_result.returns)}"
)

print(
    f"Buy-and-Hold transaction costs: "
    f"{buy_hold_result.transaction_costs.sum():.6f}"
)

print(
    f"Buy-and-Hold slippage: "
    f"{buy_hold_result.slippage.sum():.6f}"
)


# ============================================================
# Align All Strategies to Common Evaluation Period
# ============================================================

quantpilot_returns, equal_weight_returns = align_return_series(
    result.returns,
    benchmark_result.returns,
)

quantpilot_returns, buy_hold_returns = align_return_series(
    quantpilot_returns,
    buy_hold_result.returns,
)

quantpilot_returns, spy_returns = align_return_series(
    quantpilot_returns,
    spy_returns,
)


# ============================================================
# Ensure Every Strategy Uses Exactly the Same Dates
# ============================================================

common_index = (
    quantpilot_returns.index
    .intersection(equal_weight_returns.index)
    .intersection(buy_hold_returns.index)
    .intersection(spy_returns.index)
)

quantpilot_returns = quantpilot_returns.loc[common_index]
equal_weight_returns = equal_weight_returns.loc[common_index]
buy_hold_returns = buy_hold_returns.loc[common_index]
spy_returns = spy_returns.loc[common_index]


# ============================================================
# Evaluate Aligned Returns
# ============================================================

aligned_quantpilot_metrics = evaluate_returns(
    quantpilot_returns
)

aligned_equal_weight_metrics = evaluate_returns(
    equal_weight_returns
)

aligned_buy_hold_metrics = evaluate_returns(
    buy_hold_returns
)

aligned_spy_metrics = evaluate_returns(
    spy_returns
)


# ============================================================
# Final Comparison
# ============================================================

comparison = pd.DataFrame(
    {
        "QuantPilot": aligned_quantpilot_metrics,
        "Equal-Weight": aligned_equal_weight_metrics,
        "Buy-and-Hold": aligned_buy_hold_metrics,
        "SPY": aligned_spy_metrics,
    }
)


print("\nAligned Performance Comparison")
print("------------------------------")

print(comparison)


# ============================================================
# Evaluation Period
# ============================================================

print(
    "\nEvaluation observations:",
    len(common_index),
)

print(
    "Evaluation start:",
    common_index[0].date(),
)

print(
    "Evaluation end:",
    common_index[-1].date(),
)