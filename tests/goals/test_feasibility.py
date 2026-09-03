import pandas as pd
import pytest

from src.goals.feasibility import (
    assess_goal_feasibility,
    calculate_maximum_feasible_return,
    calculate_recommended_target_return,
)
from src.goals.risk_profile import get_risk_profile


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


def test_maximum_feasible_return_is_positive(
    expected_returns,
    covariance,
):
    maximum_return = calculate_maximum_feasible_return(
        expected_returns,
        covariance,
        risk_tolerance="aggressive",
        max_weight=0.50,
    )

    assert maximum_return > 0


def test_feasible_target_is_identified(
    expected_returns,
    covariance,
):
    result = assess_goal_feasibility(
        target_return=0.10,
        expected_returns=expected_returns,
        covariance=covariance,
        risk_tolerance="aggressive",
        max_weight=0.50,
    )

    assert result.feasible is True
    assert result.maximum_feasible_return >= 0.10


def test_infeasible_target_is_identified(
    expected_returns,
    covariance,
):
    result = assess_goal_feasibility(
        target_return=0.30,
        expected_returns=expected_returns,
        covariance=covariance,
        risk_tolerance="aggressive",
        max_weight=0.50,
    )

    assert result.feasible is False
    assert result.maximum_feasible_return < 0.30


def test_aggressive_has_at_least_as_much_feasible_return(
    expected_returns,
    covariance,
):
    conservative = calculate_maximum_feasible_return(
        expected_returns,
        covariance,
        risk_tolerance="conservative",
        max_weight=0.50,
    )

    aggressive = calculate_maximum_feasible_return(
        expected_returns,
        covariance,
        risk_tolerance="aggressive",
        max_weight=0.50,
    )

    assert aggressive >= conservative - 1e-6


def test_result_contains_risk_information(
    expected_returns,
    covariance,
):
    result = assess_goal_feasibility(
        target_return=0.10,
        expected_returns=expected_returns,
        covariance=covariance,
        risk_tolerance="moderate",
        max_weight=0.50,
    )

    assert result.risk_tolerance == "moderate"
    assert result.maximum_allowed_volatility == pytest.approx(
        0.18
    )


def test_invalid_target_is_rejected(
    expected_returns,
    covariance,
):
    with pytest.raises(ValueError):
        assess_goal_feasibility(
            target_return=-1.0,
            expected_returns=expected_returns,
            covariance=covariance,
            risk_tolerance="moderate",
            max_weight=0.50,
        )


def test_invalid_max_weight_is_rejected(
    expected_returns,
    covariance,
):
    with pytest.raises(ValueError, match="restrictive"):
        calculate_maximum_feasible_return(
            expected_returns,
            covariance,
            risk_tolerance="moderate",
            max_weight=0.20,
        )


def test_mismatched_indices_are_rejected(
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
        calculate_maximum_feasible_return(
            expected_returns,
            bad_covariance,
            risk_tolerance="moderate",
            max_weight=0.50,
        )

def test_recommended_return_is_below_maximum():
    maximum = 0.20

    conservative = calculate_recommended_target_return(
        maximum,
        "conservative",
    )

    moderate = calculate_recommended_target_return(
        maximum,
        "moderate",
    )

    aggressive = calculate_recommended_target_return(
        maximum,
        "aggressive",
    )

    assert conservative == pytest.approx(0.10)
    assert moderate == pytest.approx(0.14)
    assert aggressive == pytest.approx(0.17)


def test_recommended_return_increases_with_risk():
    maximum = 0.20

    conservative = calculate_recommended_target_return(
        maximum,
        "conservative",
    )

    moderate = calculate_recommended_target_return(
        maximum,
        "moderate",
    )

    aggressive = calculate_recommended_target_return(
        maximum,
        "aggressive",
    )

    assert conservative < moderate < aggressive


def test_recommended_return_accepts_risk_profile():
    profile = get_risk_profile("moderate")

    result = calculate_recommended_target_return(
        0.20,
        profile,
    )

    assert result == pytest.approx(0.14)
