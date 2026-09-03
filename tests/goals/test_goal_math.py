import pytest

from src.goals.goal_math import (
    calculate_future_value,
    calculate_required_cagr,
    calculate_required_initial_investment,
    calculate_required_monthly_contribution,
    calculate_required_return,	
)


def test_required_cagr():
    cagr = calculate_required_cagr(
        present_value=100_000,
        future_value=200_000,
        years=10,
    )

    assert cagr == pytest.approx(0.0717734625)


def test_future_value():
    future_value = calculate_future_value(
        present_value=100_000,
        annual_return=0.10,
        years=10,
    )

    assert future_value == pytest.approx(259_374.246)


def test_initial_investment_is_inverse_of_future_value():
    future_value = calculate_future_value(
        present_value=100_000,
        annual_return=0.10,
        years=10,
    )

    initial = calculate_required_initial_investment(
        future_value=future_value,
        annual_return=0.10,
        years=10,
    )

    assert initial == pytest.approx(100_000)


def test_required_monthly_contribution_zero_return():
    contribution = calculate_required_monthly_contribution(
        future_value=120_000,
        annual_return=0.0,
        years=10,
    )

    assert contribution == pytest.approx(1_000)


def test_required_monthly_contribution():
    contribution = calculate_required_monthly_contribution(
        future_value=1_000_000,
        annual_return=0.10,
        years=10,
    )

    assert contribution == pytest.approx(5_003.4059, rel=1e-4)


@pytest.mark.parametrize(
    "function,args",
    [
        (
            calculate_required_cagr,
            (0, 100_000, 10),
        ),
        (
            calculate_future_value,
            (-1, 0.10, 10),
        ),
        (
            calculate_required_initial_investment,
            (100_000, 0.10, 0),
        ),
        (
            calculate_required_monthly_contribution,
            (100_000, -1.0, 10),
        ),
    ],
)
def test_invalid_inputs(function, args):
    with pytest.raises(ValueError):
        function(*args)
def test_required_return_with_initial_and_monthly_contribution():
    required_return = calculate_required_return(
        future_value=1_000_000,
        initial_investment=100_000,
        monthly_contribution=2_000,
        years=10,
    )

    assert required_return > 0
    assert required_return < 1.0


def test_required_return_zero_when_contributions_already_reach_goal():
    required_return = calculate_required_return(
        future_value=120_000,
        initial_investment=0,
        monthly_contribution=1_000,
        years=10,
    )

    assert required_return == 0.0


def test_required_return_rejects_zero_investment():
    with pytest.raises(ValueError):
        calculate_required_return(
            future_value=1_000_000,
            initial_investment=0,
            monthly_contribution=0,
            years=10,
        )
