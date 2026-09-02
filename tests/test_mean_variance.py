import numpy as np
import pytest

from src.optimization.mean_variance import optimize_mean_variance


def test_weights_are_fully_invested():
    expected_returns = np.array([0.01, 0.02, 0.015])

    covariance = np.array([
        [0.04, 0.01, 0.01],
        [0.01, 0.05, 0.01],
        [0.01, 0.01, 0.03],
    ])

    tickers = ["A", "B", "C"]

    weights = optimize_mean_variance(
        expected_returns=expected_returns,
        covariance=covariance,
        tickers=tickers,
        risk_aversion=10.0,
        max_weight=0.5,
    )

    assert np.isclose(weights.sum(), 1.0)


def test_weights_are_long_only():
    expected_returns = np.array([0.01, 0.02, 0.015])

    covariance = np.array([
        [0.04, 0.01, 0.01],
        [0.01, 0.05, 0.01],
        [0.01, 0.01, 0.03],
    ])

    tickers = ["A", "B", "C"]

    weights = optimize_mean_variance(
        expected_returns=expected_returns,
        covariance=covariance,
        tickers=tickers,
        risk_aversion=10.0,
        max_weight=0.5,
    )

    assert (weights >= -1e-8).all()


def test_weights_respect_maximum():
    expected_returns = np.array([0.01, 0.02, 0.015])

    covariance = np.array([
        [0.04, 0.01, 0.01],
        [0.01, 0.05, 0.01],
        [0.01, 0.01, 0.03],
    ])

    tickers = ["A", "B", "C"]

    weights = optimize_mean_variance(
        expected_returns=expected_returns,
        covariance=covariance,
        tickers=tickers,
        risk_aversion=10.0,
        max_weight=0.5,
    )

    assert (weights <= 0.5 + 1e-8).all()


def test_mismatched_tickers_raise_error():
    expected_returns = np.array([0.01, 0.02, 0.015])

    covariance = np.eye(3)

    tickers = ["A", "B"]

    with pytest.raises(ValueError):
        optimize_mean_variance(
            expected_returns=expected_returns,
            covariance=covariance,
            tickers=tickers,
        )


def test_invalid_risk_aversion_raises_error():
    expected_returns = np.array([0.01, 0.02, 0.015])
    covariance = np.eye(3)
    tickers = ["A", "B", "C"]

    with pytest.raises(ValueError):
        optimize_mean_variance(
            expected_returns=expected_returns,
            covariance=covariance,
            tickers=tickers,
            risk_aversion=0,
        )


def test_invalid_max_weight_raises_error():
    expected_returns = np.array([0.01, 0.02, 0.015])
    covariance = np.eye(3)
    tickers = ["A", "B", "C"]

    with pytest.raises(ValueError):
        optimize_mean_variance(
            expected_returns=expected_returns,
            covariance=covariance,
            tickers=tickers,
            max_weight=0.2,
        )