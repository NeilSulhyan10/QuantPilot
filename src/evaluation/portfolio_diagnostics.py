from __future__ import annotations

import numpy as np
import pandas as pd


def clean_weights(
    weights: pd.DataFrame,
    tolerance: float = 1e-8,
) -> pd.DataFrame:
    """
    Remove numerical solver noise around zero.

    Values whose absolute magnitude is below tolerance
    are treated as zero.
    """
    if weights.empty:
        raise ValueError("weights cannot be empty")

    if weights.isna().any().any():
        raise ValueError("weights cannot contain NaN values")

    cleaned = weights.copy()
    cleaned = cleaned.where(cleaned.abs() >= tolerance, 0.0)

    return cleaned


def average_weights(
    weights: pd.DataFrame,
) -> pd.Series:
    """Average portfolio weight for each asset."""
    weights = clean_weights(weights)
    return weights.mean(axis=0).sort_values(ascending=False)


def maximum_weights(
    weights: pd.DataFrame,
) -> pd.Series:
    """Maximum observed portfolio weight for each asset."""
    weights = clean_weights(weights)
    return weights.max(axis=0).sort_values(ascending=False)


def active_positions(
    weights: pd.DataFrame,
    tolerance: float = 1e-8,
) -> pd.Series:
    """Number of materially non-zero holdings at each rebalance."""
    weights = clean_weights(weights, tolerance)
    return (weights.abs() > tolerance).sum(axis=1)


def effective_number_of_stocks(
    weights: pd.DataFrame,
) -> pd.Series:
    """
    Herfindahl-based effective number of holdings:

        1 / sum(w_i^2)
    """
    weights = clean_weights(weights)

    denominator = (weights ** 2).sum(axis=1)

    if (denominator <= 0).any():
        raise ValueError(
            "Cannot calculate effective number of stocks "
            "for zero-weight portfolios"
        )

    return 1.0 / denominator


def cap_hit_frequency(
    weights: pd.DataFrame,
    cap: float = 0.10,
    tolerance: float = 1e-6,
) -> pd.Series:
    """
    Percentage of rebalance dates where each stock is
    at or effectively at the portfolio weight cap.
    """
    weights = clean_weights(weights)

    hits = weights >= (cap - tolerance)

    return (
        hits.mean()
        .sort_values(ascending=False)
    )


def turnover_statistics(
    turnover: pd.Series,
) -> pd.Series:
    """Summary statistics for portfolio turnover."""
    if turnover.empty:
        raise ValueError("turnover cannot be empty")

    if turnover.isna().any():
        raise ValueError("turnover cannot contain NaN values")

    return pd.Series(
        {
            "mean": turnover.mean(),
            "median": turnover.median(),
            "std": turnover.std(),
            "minimum": turnover.min(),
            "maximum": turnover.max(),
            "p75": turnover.quantile(0.75),
            "p90": turnover.quantile(0.90),
            "p95": turnover.quantile(0.95),
        }
    )


def portfolio_diagnostics(
    weights: pd.DataFrame,
    turnover: pd.Series,
    max_weight: float = 0.10,
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Calculate the complete QuantPilot portfolio diagnostic set.
    """
    weights = clean_weights(weights)

    avg_weights = average_weights(weights)
    max_weights = maximum_weights(weights)
    active = active_positions(weights)
    effective = effective_number_of_stocks(weights)
    cap_frequency = cap_hit_frequency(
        weights,
        cap=max_weight,
    )
    turnover_summary = turnover_statistics(turnover)

    return {
        "average_weights": avg_weights,
        "maximum_weights": max_weights,
        "active_positions": active,
        "effective_number_of_stocks": effective,
        "cap_hit_frequency": cap_frequency,
        "turnover_statistics": turnover_summary,
    }
