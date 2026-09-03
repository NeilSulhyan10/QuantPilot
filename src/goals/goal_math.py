"""Goal-based investment mathematics for QuantPilot V2."""

from __future__ import annotations

import math


def calculate_required_cagr(
    present_value: float,
    future_value: float,
    years: float,
) -> float:
    """Calculate the CAGR required to grow present_value to future_value."""

    if present_value <= 0:
        raise ValueError("present_value must be greater than 0.")

    if future_value <= 0:
        raise ValueError("future_value must be greater than 0.")

    if years <= 0:
        raise ValueError("years must be greater than 0.")

    return (future_value / present_value) ** (1.0 / years) - 1.0


def calculate_future_value(
    present_value: float,
    annual_return: float,
    years: float,
) -> float:
    """Calculate future value of a lump-sum investment."""

    if present_value < 0:
        raise ValueError("present_value must be non-negative.")

    if annual_return <= -1:
        raise ValueError("annual_return must be greater than -100%.")

    if years < 0:
        raise ValueError("years must be non-negative.")

    return present_value * (1.0 + annual_return) ** years


def calculate_required_initial_investment(
    future_value: float,
    annual_return: float,
    years: float,
) -> float:
    """Calculate the initial investment required to reach a future value."""

    if future_value <= 0:
        raise ValueError("future_value must be greater than 0.")

    if annual_return <= -1:
        raise ValueError("annual_return must be greater than -100%.")

    if years <= 0:
        raise ValueError("years must be greater than 0.")

    return future_value / (1.0 + annual_return) ** years


def calculate_required_monthly_contribution(
    future_value: float,
    annual_return: float,
    years: float,
) -> float:
    """Calculate the monthly contribution required to reach a future value.

    Contributions are assumed to occur at the end of each month.
    The annual return is converted to an effective monthly return.
    """

    if future_value <= 0:
        raise ValueError("future_value must be greater than 0.")

    if annual_return <= -1:
        raise ValueError("annual_return must be greater than -100%.")

    if years <= 0:
        raise ValueError("years must be greater than 0.")

    months = round(years * 12)

    if months <= 0:
        raise ValueError("years must represent at least one month.")

    monthly_return = (1.0 + annual_return) ** (1.0 / 12.0) - 1.0

    if math.isclose(monthly_return, 0.0, abs_tol=1e-15):
        return future_value / months

    return (
        future_value
        * monthly_return
        / ((1.0 + monthly_return) ** months - 1.0)
    )

def calculate_required_return(
    future_value: float,
    initial_investment: float,
    monthly_contribution: float,
    years: float,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> float:
    """
    Calculate the effective annual return required to reach a target
    future value using an initial investment and end-of-month
    contributions.

    Returns:
        Effective annual return as a decimal.

    Raises:
        ValueError: If the goal cannot be reached under the supported
        return search range or inputs are invalid.
    """
    if future_value <= 0:
        raise ValueError("Future value must be positive.")

    if initial_investment < 0:
        raise ValueError("Initial investment must be non-negative.")

    if monthly_contribution < 0:
        raise ValueError("Monthly contribution must be non-negative.")

    if years <= 0:
        raise ValueError("Years must be positive.")

    if initial_investment == 0 and monthly_contribution == 0:
        raise ValueError(
            "Initial investment or monthly contribution must be positive."
        )

    months = int(round(years * 12))

    if months <= 0:
        raise ValueError("Investment horizon must contain at least one month.")

    def future_value_at_rate(annual_return: float) -> float:
        monthly_return = (1.0 + annual_return) ** (1.0 / 12.0) - 1.0

        if abs(monthly_return) < 1e-14:
            contribution_growth = float(months)
        else:
            contribution_growth = (
                (1.0 + monthly_return) ** months - 1.0
            ) / monthly_return

        return (
            initial_investment
            * (1.0 + annual_return) ** years
            + monthly_contribution * contribution_growth
        )

    # If the goal is already reachable without investment growth.
    if future_value_at_rate(0.0) >= future_value:
        return 0.0

    # Search between -99% and +1000% annual return.
    lower = -0.99
    upper = 10.0

    if future_value_at_rate(upper) < future_value:
        raise ValueError(
            "Required return exceeds the supported calculation range."
        )

    for _ in range(max_iterations):
        midpoint = (lower + upper) / 2.0
        value = future_value_at_rate(midpoint)

        if abs(value - future_value) <= tolerance:
            return midpoint

        if value < future_value:
            lower = midpoint
        else:
            upper = midpoint

    return (lower + upper) / 2.0