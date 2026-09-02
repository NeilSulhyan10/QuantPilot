import numpy as np
import pandas as pd
import pytest

from src.backtesting.ml_engine import (
    MLWalkForwardBacktester,
)
from src.backtesting.engine import (
    BacktestConfig,
)


def make_inputs():

    dates = pd.date_range(
        "2020-01-01",
        periods=80,
        freq="D",
    )

    returns = pd.DataFrame(
        {
            "A": np.full(80, 0.001),
            "B": np.full(80, 0.0005),
        },
        index=dates,
    )

    expected_returns = pd.DataFrame(
        {
            "A": np.full(80, 0.001),
            "B": np.full(80, 0.0005),
        },
        index=dates,
    )

    covariance = pd.DataFrame(
        [
            [0.0001, 0.0],
            [0.0, 0.0001],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )

    covariance_matrices = {
        date: covariance
        for date in dates
    }

    return (
        returns,
        expected_returns,
        covariance_matrices,
    )


def test_ml_engine_runs():

    (
        returns,
        expected_returns,
        covariance_matrices,
    ) = make_inputs()

    backtester = MLWalkForwardBacktester(
        BacktestConfig(
            risk_aversion=10,
            max_weight=0.9,
        )
    )

    result = backtester.run_ml(
        returns,
        expected_returns,
        covariance_matrices,
    )

    assert not result.returns.empty


def test_ml_engine_returns_have_no_nan():

    (
        returns,
        expected_returns,
        covariance_matrices,
    ) = make_inputs()

    backtester = MLWalkForwardBacktester(
        BacktestConfig(
            risk_aversion=10,
            max_weight=0.9,
        )
    )

    result = backtester.run_ml(
        returns,
        expected_returns,
        covariance_matrices,
    )

    assert result.returns.notna().all()


def test_ml_engine_preserves_assets():

    (
        returns,
        expected_returns,
        covariance_matrices,
    ) = make_inputs()

    backtester = MLWalkForwardBacktester(
        BacktestConfig(
            risk_aversion=10,
            max_weight=0.9,
        )
    )

    result = backtester.run_ml(
        returns,
        expected_returns,
        covariance_matrices,
    )

    assert list(result.weights.columns) == [
        "A",
        "B",
    ]


def test_empty_ml_predictions_fail():

    (
        returns,
        expected_returns,
        covariance_matrices,
    ) = make_inputs()

    expected_returns.iloc[:, :] = np.nan

    backtester = MLWalkForwardBacktester(
        BacktestConfig(
            risk_aversion=10,
            max_weight=0.9,
        )
    )

    with pytest.raises(ValueError):

        backtester.run_ml(
            returns,
            expected_returns,
            covariance_matrices,
        )


def test_mismatched_indices_fail():

    (
        returns,
        expected_returns,
        covariance_matrices,
    ) = make_inputs()

    expected_returns.index = pd.date_range(
        "2021-01-01",
        periods=80,
        freq="D",
    )

    backtester = MLWalkForwardBacktester(
        BacktestConfig(
            risk_aversion=10,
            max_weight=0.9,
        )
    )

    with pytest.raises(ValueError):

        backtester.run_ml(
            returns,
            expected_returns,
            covariance_matrices,
        )


def test_mismatched_columns_fail():

    (
        returns,
        expected_returns,
        covariance_matrices,
    ) = make_inputs()

    expected_returns = expected_returns.rename(
        columns={"A": "X"}
    )

    backtester = MLWalkForwardBacktester(
        BacktestConfig(
            risk_aversion=10,
            max_weight=0.9,
        )
    )

    with pytest.raises(ValueError):

        backtester.run_ml(
            returns,
            expected_returns,
            covariance_matrices,
        )