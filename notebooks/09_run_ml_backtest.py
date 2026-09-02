import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.backtesting.engine import (
    BacktestConfig,
    WalkForwardBacktester,
)
from src.backtesting.ml_engine import (
    MLWalkForwardBacktester,
)
from src.backtesting.benchmark import (
    EqualWeightBacktester,
)
from src.backtesting.buy_and_hold import (
    BuyAndHoldBacktester,
)
from src.evaluation.metrics import evaluate_returns
from src.data.reader import load_universe
from src.data.benchmark_reader import load_benchmark
from src.features.returns import calculate_return_matrix
from src.features.covariance import rolling_covariance
from src.features.statistics import rolling_mean
from src.models.ml_expected_returns import (
    calculate_ml_expected_returns,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AVGO",
    "GOOGL",
    "AMZN",
    "META",
    "JPM",
    "V",
    "MA",
    "JNJ",
    "UNH",
    "XOM",
    "CVX",
    "PG",
    "KO",
    "COST",
    "CAT",
    "WMT",
    "HD",
]


# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------

price_data = load_universe(TICKERS)

spy = load_benchmark("SPY")


# ---------------------------------------------------------
# 2. Calculate common return matrix
# ---------------------------------------------------------

returns = calculate_return_matrix(
    price_data,
    price_column="Close",
    method="simple",
)


# ---------------------------------------------------------
# 3. Existing covariance model
# ---------------------------------------------------------

covariance_matrices = rolling_covariance(
    returns,
    window=60,
)


# ---------------------------------------------------------
# 4. Existing V2 expected returns
# ---------------------------------------------------------

v2_expected_returns = rolling_mean(
    returns,
    window=60,
)


# ---------------------------------------------------------
# 5. ML expected returns
# ---------------------------------------------------------

print("Generating ML expected returns...")

ml_expected_returns = calculate_ml_expected_returns(
    returns,
    min_train_size=252,
    horizon=21,
    alpha=1.0,
)

print("ML predictions generated.")


# ---------------------------------------------------------
# 6. Backtest configuration
# ---------------------------------------------------------

config = BacktestConfig(
    estimation_window=60,
    risk_aversion=10,
    max_weight=0.10,
    transaction_cost_rate=0.001,
    slippage_rate=0.0005,
)


# ---------------------------------------------------------
# 7. V2 QuantPilot
# ---------------------------------------------------------

v2_backtester = WalkForwardBacktester(
    config
)

v2_result = v2_backtester.run(
    returns=returns,
    expected_returns=v2_expected_returns,
    covariance_matrices=covariance_matrices,
)


# ---------------------------------------------------------
# 8. ML QuantPilot
# ---------------------------------------------------------

ml_backtester = MLWalkForwardBacktester(
    config
)

ml_result = ml_backtester.run_ml(
    returns=returns,
    ml_expected_returns=ml_expected_returns,
    covariance_matrices=covariance_matrices,
)


# ---------------------------------------------------------
# 9. Equal Weight
# ---------------------------------------------------------

equal_weight_backtester = EqualWeightBacktester(
    config
)

equal_weight_result = (
    equal_weight_backtester.run(
        returns
    )
)


# ---------------------------------------------------------
# 10. Buy and Hold
# ---------------------------------------------------------

buy_hold_backtester = BuyAndHoldBacktester()

buy_hold_result = buy_hold_backtester.run(
    returns
)


# ---------------------------------------------------------
# 11. SPY
# ---------------------------------------------------------

spy_returns = (
    spy["Close"]
    .pct_change()
    .dropna()
)


# ---------------------------------------------------------
# 12. Align evaluation period
# ---------------------------------------------------------

strategy_returns = {
    "ML QuantPilot": ml_result.returns,
    "V2 QuantPilot": v2_result.returns,
    "Equal Weight": equal_weight_result.returns,
    "Buy & Hold": buy_hold_result.returns,
    "SPY": spy_returns,
}

# ML is the limiting strategy because it requires
# sufficient historical training data.
ml_start_date = ml_result.returns.dropna().index[0]

common_index = None

for series in strategy_returns.values():

    index = series.dropna().index

    # Restrict every strategy to the ML start date.
    index = index[index >= ml_start_date]

    if common_index is None:
        common_index = index
    else:
        common_index = common_index.intersection(index)

common_index = common_index.sort_values()

aligned_returns = {
    name: series.loc[common_index]
    for name, series in strategy_returns.items()
}


# ---------------------------------------------------------
# 13. Evaluation
# ---------------------------------------------------------

results = {}

for name, series in aligned_returns.items():

    results[name] = evaluate_returns(
        series
    )

comparison = pd.DataFrame(results).T

print("\n=== ML QuantPilot Comparison ===")
print(comparison)


# ---------------------------------------------------------
# 14. Trading statistics
# ---------------------------------------------------------

print("\n=== Trading Statistics ===")

print(
    f"ML average turnover: "
    f"{ml_result.turnover.mean():.4f}"
)

print(
    f"ML transaction costs: "
    f"{ml_result.transaction_costs.sum():.6f}"
)

print(
    f"ML slippage: "
    f"{ml_result.slippage.sum():.6f}"
)

print(
    f"ML rebalances: "
    f"{len(ml_result.turnover)}"
)

print(
    f"\nEvaluation period: "
    f"{common_index[0].date()} → "
    f"{common_index[-1].date()}"
)

print(
    f"Observations: {len(common_index)}"
)