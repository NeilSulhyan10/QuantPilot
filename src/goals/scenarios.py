"""Goal scenario calculations for QuantPilot V2."""

from __future__ import annotations

from dataclasses import dataclass

from src.goals.goal_math import (
    calculate_future_value,
    calculate_required_initial_investment,
    calculate_required_monthly_contribution,
)


@dataclass(frozen=True)
class GoalScenario:
    """Projected outcome for a single annual-return assumption."""

    name: str
    annual_return: float
    future_value: float
    initial_investment: float
    monthly_contribution: float


@dataclass(frozen=True)
class ScenarioSet:
    """Conservative, expected, and optimistic goal scenarios."""

    conservative: GoalScenario
    expected: GoalScenario
    optimistic: GoalScenario


def _validate_inputs(
    annual_return: float,
    years: float,
) -> None:
    if annual_return <= -1:
        raise ValueError(
            "annual_return must be greater than -100%."
        )

    if years <= 0:
        raise ValueError(
            "years must be greater than 0."
        )


def calculate_goal_scenario(
    name: str,
    future_value: float,
    annual_return: float,
    years: float,
) -> GoalScenario:
    """Calculate a scenario for a target future amount."""

    if not name.strip():
        raise ValueError("Scenario name cannot be empty.")

    if future_value <= 0:
        raise ValueError(
            "future_value must be greater than 0."
        )

    _validate_inputs(
        annual_return,
        years,
    )

    initial_investment = (
        calculate_required_initial_investment(
            future_value=future_value,
            annual_return=annual_return,
            years=years,
        )
    )

    monthly_contribution = (
        calculate_required_monthly_contribution(
            future_value=future_value,
            annual_return=annual_return,
            years=years,
        )
    )

    projected_future_value = calculate_future_value(
        present_value=initial_investment,
        annual_return=annual_return,
        years=years,
    )

    return GoalScenario(
        name=name,
        annual_return=annual_return,
        future_value=projected_future_value,
        initial_investment=initial_investment,
        monthly_contribution=monthly_contribution,
    )


def build_scenario_set(
    target_amount: float,
    years: float,
    conservative_return: float,
    expected_return: float,
    optimistic_return: float,
) -> ScenarioSet:
    """Build three scenarios around a target future amount."""

    if target_amount <= 0:
        raise ValueError(
            "target_amount must be greater than 0."
        )

    if years <= 0:
        raise ValueError(
            "years must be greater than 0."
        )

    returns = [
        conservative_return,
        expected_return,
        optimistic_return,
    ]

    if not (
        conservative_return
        <= expected_return
        <= optimistic_return
    ):
        raise ValueError(
            "Scenario returns must satisfy "
            "conservative <= expected <= optimistic."
        )

    return ScenarioSet(
        conservative=calculate_goal_scenario(
            name="Conservative",
            future_value=target_amount,
            annual_return=conservative_return,
            years=years,
        ),
        expected=calculate_goal_scenario(
            name="Expected",
            future_value=target_amount,
            annual_return=expected_return,
            years=years,
        ),
        optimistic=calculate_goal_scenario(
            name="Optimistic",
            future_value=target_amount,
            annual_return=optimistic_return,
            years=years,
        ),
    )