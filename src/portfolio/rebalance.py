import pandas as pd

from src.portfolio.drift import drift_weights


def calculate_rebalance_turnover(
    current_weights: pd.Series,
    target_weights: pd.Series,
) -> float:
    """
    Calculate one-way turnover required to move from current
    portfolio weights to target weights.
    """
    if current_weights.empty:
        raise ValueError("current_weights is empty.")

    if target_weights.empty:
        raise ValueError("target_weights is empty.")

    if not current_weights.index.equals(target_weights.index):
        raise ValueError(
            "current_weights and target_weights must have identical indices."
        )

    if current_weights.isna().any():
        raise ValueError("current_weights contains NaN values.")

    if target_weights.isna().any():
        raise ValueError("target_weights contains NaN values.")

    return float(0.5 * (target_weights - current_weights).abs().sum())


def simulate_rebalance_period(
    starting_weights: pd.Series,
    asset_returns: pd.DataFrame,
    target_weights: pd.Series,
) -> tuple[pd.Series, float]:
    """
    Simulate one period between rebalances.

    Returns:
        ending_weights:
            Portfolio weights immediately before the next rebalance.

        turnover:
            One-way turnover required to rebalance to target_weights.
    """
    if starting_weights.empty:
        raise ValueError("starting_weights is empty.")

    if asset_returns.empty:
        raise ValueError("asset_returns is empty.")

    if not starting_weights.index.equals(asset_returns.columns):
        raise ValueError(
            "starting_weights and asset_returns columns must match."
        )

    if not target_weights.index.equals(asset_returns.columns):
        raise ValueError(
            "target_weights and asset_returns columns must match."
        )

    current_weights = starting_weights.copy()

    for date in asset_returns.index:
        current_weights = drift_weights(
            current_weights,
            asset_returns.loc[date],
        )

    turnover = calculate_rebalance_turnover(
        current_weights,
        target_weights,
    )

    return current_weights, turnover