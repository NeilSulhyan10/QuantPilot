import numpy as np
import pandas as pd
import pytest

from src.goals.optimizer import optimize_for_goal


@pytest.fixture
def expected_returns():
    return pd.Series(
        {
            "AAPL": 0.08,
            "MSFT": 0.12,
            "NVDA": 0.18,
            "JPM": 0.10,
        }
    )


@pytest.fixture
def covariance():
    return pd.DataFrame(
        [
            [0.0001, 0.00002, 0.00003, 0.00002],
            [0.00002, 0.00012, 0.00003, 0.00002],
            [0.00003, 0.00003, 0.00020, 0.00003],
            [0.00002, 0.00002, 0.00003, 0.00011],
        ],
        index=["AAPL", "MSFT", "NVDA", "JPM"],
        columns=["AAPL", "MSFT", "NVDA", "JPM"],
    )


def test_target_return_is_achieved(
    expected_returns,
    covariance,
):
    result = optimize_for_goal(
        expected_returns,
        covariance,
        target_return=0.10,
        risk_tolerance="aggressive",
        max_weight=0.50,
    )

    assert result.status == "optimal"
    assert result.expected_return >= 0.10 - 1e-6


def test_weights_sum_to_one(
    expected_returns,
    covariance,
):
    result = optimize_for_goal(
        expected_returns,
        covariance,
        target_return=0.10,
        risk_tolerance="aggressive",
        max_weight=0.50,
    )

    assert result.weights.sum() == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_long_only_constraint(
    expected_returns,
    covariance,
):
    result = optimize_for_goal(
        expected_returns,
        covariance,
        target_return=0.10,
        risk_tolerance="aggressive",
        max_weight=0.50,
    )

    assert result.weights.min() >= -1e-6


def test_max_weight_constraint(
    expected_returns,
    covariance,
):
    result = optimize_for_goal(
        expected_returns,
        covariance,
        target_return=0.10,
        risk_tolerance="aggressive",
        max_weight=0.50,
    )

    assert result.weights.max() <= 0.50 + 1e-6


def test_risk_limit_is_respected(
    expected_returns,
    covariance,
):
    result = optimize_for_goal(
        expected_returns,
        covariance,
        target_return=0.10,
        risk_tolerance="aggressive",
        max_weight=0.50,
    )

    assert (
        result.expected_volatility
        <= result.maximum_allowed_volatility + 1e-6
    )


def test_risk_profile_is_recorded(
    expected_returns,
    covariance,
):
    result = optimize_for_goal(
        expected_returns,
        covariance,
        target_return=0.10,
        risk_tolerance="moderate",
        max_weight=0.50,
    )

    assert result.risk_tolerance == "moderate"
    assert result.maximum_allowed_volatility == pytest.approx(
        0.18
    )


def test_aggressive_allows_more_risk_than_conservative(
    expected_returns,
    covariance,
):
    conservative = optimize_for_goal(
        expected_returns,
        covariance,
        target_return=0.08,
        risk_tolerance="conservative",
        max_weight=0.50,
    )

    aggressive = optimize_for_goal(
        expected_returns,
        covariance,
        target_return=0.08,
        risk_tolerance="aggressive",
        max_weight=0.50,
    )

    assert (
        aggressive.expected_volatility
        >= conservative.expected_volatility - 1e-6
    )


def test_infeasible_target_is_rejected(
    expected_returns,
    covariance,
):
    with pytest.raises(ValueError, match="infeasible"):
        optimize_for_goal(
            expected_returns,
            covariance,
            target_return=0.50,
            risk_tolerance="aggressive",
            max_weight=0.50,
        )


def test_infeasible_risk_return_combination_is_rejected(
    expected_returns,
    covariance,
):
    with pytest.raises(ValueError, match="infeasible"):
        optimize_for_goal(
            expected_returns,
            covariance,
            target_return=0.18,
            risk_tolerance="conservative",
            max_weight=0.50,
        )


def test_too_restrictive_max_weight_is_rejected(
    expected_returns,
    covariance,
):
    with pytest.raises(ValueError, match="too restrictive"):
        optimize_for_goal(
            expected_returns,
            covariance,
            target_return=0.10,
            risk_tolerance="aggressive",
            max_weight=0.20,
        )


def test_nan_inputs_are_rejected(
    expected_returns,
    covariance,
):
    bad_returns = expected_returns.copy()
    bad_returns["AAPL"] = np.nan

    with pytest.raises(ValueError, match="NaN"):
        optimize_for_goal(
            bad_returns,
            covariance,
            target_return=0.10,
            risk_tolerance="aggressive",
            max_weight=0.50,
        )


def test_index_mismatch_is_rejected(
    expected_returns,
    covariance,
):
    bad_covariance = covariance.copy()

    bad_covariance.index = [
        "AAPL",
        "MSFT",
        "NVDA",
        "KO",
    ]

    with pytest.raises(
        ValueError,
        match="indices must match",
    ):
        optimize_for_goal(
            expected_returns,
            bad_covariance,
            target_return=0.10,
            risk_tolerance="aggressive",
            max_weight=0.50,
        )