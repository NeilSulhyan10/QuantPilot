from pathlib import Path

import pandas as pd
import yaml

from src.data.reader import load_universe
from src.features.returns import calculate_return_matrix
from src.features.statistics import rolling_mean
from src.features.covariance import rolling_covariance
from src.backtesting.engine import BacktestConfig, WalkForwardBacktester
from src.evaluation.metrics import evaluate_returns


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


# ---------------------------------------------------------
# Estimate covariance
# ---------------------------------------------------------

covariances = rolling_covariance(
    returns,
    window=60,
)


# ---------------------------------------------------------
# Baseline expected returns
# ---------------------------------------------------------

base_expected_returns = rolling_mean(
    returns,
    window=60,
)


# ---------------------------------------------------------
# Backtest configuration
# ---------------------------------------------------------

backtest_config = BacktestConfig(
    estimation_window=60,
    risk_aversion=10,
    max_weight=0.10,
    transaction_cost_rate=0.001,
    slippage_rate=0.0005,
)


# ---------------------------------------------------------
# Shrinkage function
# ---------------------------------------------------------

def shrink_expected_returns(
    expected_returns: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:

    cross_sectional_mean = expected_returns.mean(axis=1)

    shrunk = (
        cross_sectional_mean.to_numpy()[:, None]
        + alpha
        * (
            expected_returns.to_numpy()
            - cross_sectional_mean.to_numpy()[:, None]
        )
    )

    return pd.DataFrame(
        shrunk,
        index=expected_returns.index,
        columns=expected_returns.columns,
    )


# ---------------------------------------------------------
# Run experiments
# ---------------------------------------------------------

alphas = [1.00, 0.75, 0.50, 0.25]

results = {}
summary_rows = []


for alpha in alphas:

    print("\n" + "=" * 70)
    print(f"RUNNING SHRINKAGE EXPERIMENT: ALPHA = {alpha:.2f}")
    print("=" * 70)

    expected_returns = shrink_expected_returns(
        base_expected_returns,
        alpha,
    )

    backtester = WalkForwardBacktester(
        config=backtest_config,
    )

    result = backtester.run(
        returns=returns,
        expected_returns=expected_returns,
        covariance_matrices=covariances,
    )

    metrics = evaluate_returns(
        result.returns
    )

    results[alpha] = result

    summary_rows.append(
        {
            "alpha": alpha,
            "cumulative_return": metrics["cumulative_return"],
            "cagr": metrics["cagr"],
            "annualized_volatility": metrics["annualized_volatility"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "sortino_ratio": metrics["sortino_ratio"],
            "maximum_drawdown": metrics["maximum_drawdown"],
            "average_turnover": result.turnover.mean(),
            "total_transaction_costs": result.transaction_costs.sum(),
            "total_slippage": result.slippage.sum(),
        }
    )

    print("\nPerformance")
    print("-----------")

    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    print(
        f"Average turnover: "
        f"{result.turnover.mean():.6f}"
    )

    print(
        f"Total transaction costs: "
        f"{result.transaction_costs.sum():.6f}"
    )

    print(
        f"Total slippage: "
        f"{result.slippage.sum():.6f}"
    )


# ---------------------------------------------------------
# Comparison table
# ---------------------------------------------------------

comparison = pd.DataFrame(summary_rows)

print("\n")
print("=" * 90)
print("SHRINKAGE BACKTEST COMPARISON")
print("=" * 90)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ---------------------------------------------------------
# Relative comparison against baseline
# ---------------------------------------------------------

baseline = comparison[
    comparison["alpha"] == 1.00
].iloc[0]

comparison["CAGR_vs_baseline"] = (
    comparison["cagr"]
    - baseline["cagr"]
)

comparison["Sharpe_vs_baseline"] = (
    comparison["sharpe_ratio"]
    - baseline["sharpe_ratio"]
)

comparison["Turnover_vs_baseline"] = (
    comparison["average_turnover"]
    - baseline["average_turnover"]
)

comparison["Trading_cost_vs_baseline"] = (
    comparison["total_transaction_costs"]
    + comparison["total_slippage"]
    - baseline["total_transaction_costs"]
    - baseline["total_slippage"]
)


print("\n")
print("=" * 90)
print("RELATIVE TO BASELINE (ALPHA = 1.00)")
print("=" * 90)

print(
    comparison[
        [
            "alpha",
            "CAGR_vs_baseline",
            "Sharpe_vs_baseline",
            "Turnover_vs_baseline",
            "Trading_cost_vs_baseline",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


print("\nExperiment complete.")