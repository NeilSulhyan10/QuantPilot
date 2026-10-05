"""Goal-based investment mathematics for QuantPilot V2."""

from __future__ import annotations

import math
import re


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


_HORIZON_PATTERN = re.compile(
    r"""
    ^\s*
    (?:(?P<years>\d+(?:\.\d+)?)\s*(?:y|yr|year|years)\s*)?
    (?:(?P<months>\d+(?:\.\d+)?)\s*(?:m|mo|month|months)\s*)?
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_horizon(horizon_str: str) -> float:
    """
    Parse a horizon string into total years (float).

    Supported formats:
        - "10" or "10 years" or "10y" -> 10.0
        - "6 months" or "6m" -> 0.5
        - "1 year 6 months" or "1y 6m" -> 1.5
        - "18 months" -> 1.5
        - "2.5 years" -> 2.5

    Args:
        horizon_str: String representation of investment horizon.

    Returns:
        Total years as float.

    Raises:
        ValueError: If the string cannot be parsed.
    """
    if not isinstance(horizon_str, str):
        raise TypeError("horizon_str must be a string.")

    s = horizon_str.strip()
    if not s:
        raise ValueError("Horizon string cannot be empty.")

    # Try parsing as a plain number (assumed to be years)
    try:
        val = float(s)
        if val <= 0:
            raise ValueError("Horizon must be positive.")
        return val
    except ValueError:
        pass

    # Try parsing structured format
    match = _HORIZON_PATTERN.match(s)
    if not match:
        raise ValueError(
            f"Invalid horizon format: '{horizon_str}'. "
            f"Use formats like '10', '10 years', '6 months', '1 year 6 months', '1y 6m'."
        )

    years = float(match.group("years") or 0)
    months = float(match.group("months") or 0)

    if years == 0 and months == 0:
        raise ValueError("Horizon must specify at least one year or month.")

    total_years = years + months / 12.0

    if total_years <= 0:
        raise ValueError("Horizon must be positive.")

    return total_years


def format_horizon(years: float) -> str:
    """
    Format years as a human-readable string.

    Args:
        years: Total years as float.

    Returns:
        Formatted string like "1 year 6 months" or "10 years".
    """
    if years <= 0:
        return "0"

    total_months = round(years * 12)
    y = total_months // 12
    m = total_months % 12

    parts = []
    if y:
        parts.append(f"{y} year{'s' if y != 1 else ''}")
    if m:
        parts.append(f"{m} month{'s' if m != 1 else ''}")

    return " ".join(parts) if parts else f"{years:.1f} years"