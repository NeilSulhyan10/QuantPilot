import pytest

from src.goals.scenarios import (
    build_scenario_set,
    calculate_goal_scenario,
)


def test_goal_scenario():
    scenario = calculate_goal_scenario(
        name="Expected",
        future_value=1_000_000,
        annual_return=0.10,
        years=10,
    )

    assert scenario.name == "Expected"
    assert scenario.annual_return == pytest.approx(0.10)
    assert scenario.initial_investment > 0
    assert scenario.monthly_contribution > 0
    assert scenario.future_value == pytest.approx(
        1_000_000
    )


def test_scenario_set_contains_three_scenarios():
    scenarios = build_scenario_set(
        target_amount=1_000_000,
        years=10,
        conservative_return=0.06,
        expected_return=0.10,
        optimistic_return=0.14,
    )

    assert scenarios.conservative.name == "Conservative"
    assert scenarios.expected.name == "Expected"
    assert scenarios.optimistic.name == "Optimistic"


def test_higher_return_requires_less_initial_capital():
    scenarios = build_scenario_set(
        target_amount=1_000_000,
        years=10,
        conservative_return=0.06,
        expected_return=0.10,
        optimistic_return=0.14,
    )

    assert (
        scenarios.conservative.initial_investment
        > scenarios.expected.initial_investment
        > scenarios.optimistic.initial_investment
    )


def test_higher_return_requires_less_monthly_contribution():
    scenarios = build_scenario_set(
        target_amount=1_000_000,
        years=10,
        conservative_return=0.06,
        expected_return=0.10,
        optimistic_return=0.14,
    )

    assert (
        scenarios.conservative.monthly_contribution
        > scenarios.expected.monthly_contribution
        > scenarios.optimistic.monthly_contribution
    )


def test_scenario_returns_must_be_ordered():
    with pytest.raises(ValueError, match="conservative"):
        build_scenario_set(
            target_amount=1_000_000,
            years=10,
            conservative_return=0.12,
            expected_return=0.10,
            optimistic_return=0.14,
        )


def test_invalid_target():
    with pytest.raises(ValueError):
        build_scenario_set(
            target_amount=0,
            years=10,
            conservative_return=0.06,
            expected_return=0.10,
            optimistic_return=0.14,
        )


def test_invalid_horizon():
    with pytest.raises(ValueError):
        build_scenario_set(
            target_amount=1_000_000,
            years=0,
            conservative_return=0.06,
            expected_return=0.10,
            optimistic_return=0.14,
        )


def test_invalid_return():
    with pytest.raises(ValueError):
        calculate_goal_scenario(
            name="Expected",
            future_value=1_000_000,
            annual_return=-1.0,
            years=10,
        )


def test_empty_scenario_name():
    with pytest.raises(ValueError):
        calculate_goal_scenario(
            name="",
            future_value=1_000_000,
            annual_return=0.10,
            years=10,
        )