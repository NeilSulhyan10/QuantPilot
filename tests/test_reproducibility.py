from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_final_research_outputs_exist():
    required_files = [
        PROJECT_ROOT / "results/comparisons/strategy_comparison.csv",
        PROJECT_ROOT / "results/backtests/quantpilot_v2_returns.csv",
        PROJECT_ROOT / "results/backtests/quantpilot_v2_weights.csv",
        PROJECT_ROOT / "results/backtests/quantpilot_v2_turnover.csv",
        PROJECT_ROOT / "results/backtests/quantpilot_v2_transaction_costs.csv",
        PROJECT_ROOT / "results/backtests/quantpilot_v2_slippage.csv",
    ]

    for path in required_files:
        assert path.exists(), f"Missing research output: {path}"


def test_strategy_comparison_is_valid():
    path = (
        PROJECT_ROOT
        / "results"
        / "comparisons"
        / "strategy_comparison.csv"
    )

    comparison = pd.read_csv(path, index_col="strategy")

    expected_strategies = {
        "QuantPilot V2",
        "Equal Weight",
        "Buy & Hold",
        "SPY",
    }

    assert set(comparison.index) == expected_strategies

    required_columns = {
        "cumulative_return",
        "cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "maximum_drawdown",
    }

    assert required_columns.issubset(comparison.columns)

    assert comparison.notna().all().all()


def test_quantpilot_returns_are_valid():
    path = (
        PROJECT_ROOT
        / "results"
        / "backtests"
        / "quantpilot_v2_returns.csv"
    )

    returns = pd.read_csv(path, index_col="date", parse_dates=True)

    assert len(returns) > 0
    assert returns["return"].notna().all()
    assert returns.index.is_monotonic_increasing


def test_quantpilot_weights_are_valid():
    path = (
        PROJECT_ROOT
        / "results"
        / "backtests"
        / "quantpilot_v2_weights.csv"
    )

    weights = pd.read_csv(path, index_col="date", parse_dates=True)

    assert len(weights) > 0
    assert weights.notna().all().all()

    # Fully invested portfolio.
    weight_sums = weights.sum(axis=1)
    assert (weight_sums - 1.0).abs().max() < 1e-6

    # Long-only.
    assert weights.min().min() >= -1e-8

    # 10% position cap.
    assert weights.max().max() <= 0.10 + 1e-8
