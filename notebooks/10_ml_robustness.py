import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.backtesting.engine import BacktestConfig
from src.backtesting.ml_engine import MLWalkForwardBacktester
from src.data.reader import load_universe
from src.features.returns import calculate_return_matrix
from src.features.covariance import rolling_covariance
from src.models.ml_expected_returns import calculate_ml_expected_returns
from src.evaluation.metrics import evaluate_returns


TICKERS = [
    "AAPL", "MSFT", "NVDA", "AVGO", "GOOGL",
    "AMZN", "META", "JPM", "V", "MA",
    "JNJ", "UNH", "XOM", "CVX", "PG",
    "KO", "COST", "CAT", "WMT", "HD",
]


price_data = load_universe(TICKERS)

returns = calculate_return_matrix(
    price_data,
    price_column="Close",
    method="simple",
)

covariance_matrices = rolling_covariance(
    returns,
    window=60,
)

config = BacktestConfig(
    estimation_window=60,
    risk_aversion=10,
    max_weight=0.10,
    transaction_cost_rate=0.001,
    slippage_rate=0.0005,
)

backtester = MLWalkForwardBacktester(config)


experiments = [
    (252, 21, 0.1),
    (252, 21, 1.0),
    (252, 21, 10.0),

    (252, 42, 0.1),
    (252, 42, 1.0),
    (252, 42, 10.0),

    (504, 21, 0.1),
    (504, 21, 1.0),
    (504, 21, 10.0),

    (504, 42, 0.1),
    (504, 42, 1.0),
    (504, 42, 10.0),
]


results = []


for min_train_size, horizon, alpha in experiments:

    print(
        f"\nRunning: "
        f"train={min_train_size}, "
        f"horizon={horizon}, "
        f"alpha={alpha}"
    )

    ml_expected_returns = calculate_ml_expected_returns(
        returns,
        min_train_size=min_train_size,
        horizon=horizon,
        alpha=alpha,
    )

    result = backtester.run_ml(
        returns=returns,
        ml_expected_returns=ml_expected_returns,
        covariance_matrices=covariance_matrices,
    )

    # Evaluate only the strategy's own valid period.
    strategy_returns = result.returns.dropna()

    metrics = evaluate_returns(
        strategy_returns
    )

    results.append(
        {
            "train_window": min_train_size,
            "horizon": horizon,
            "alpha": alpha,
            "cagr": metrics["cagr"],
            "volatility": metrics[
                "annualized_volatility"
            ],
            "sharpe": metrics["sharpe_ratio"],
            "sortino": metrics["sortino_ratio"],
            "max_drawdown": metrics[
                "maximum_drawdown"
            ],
            "turnover": result.turnover.mean(),
            "transaction_costs": (
                result.transaction_costs.sum()
            ),
            "slippage": result.slippage.sum(),
            "observations": len(strategy_returns),
        }
    )


results_df = pd.DataFrame(results)

print("\n=== ML Robustness Results ===")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)