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