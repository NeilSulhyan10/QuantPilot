import pandas as pd


def calculate_weight_changes(
    previous_weights: pd.Series,
    target_weights: pd.Series,
) -> pd.Series:
    """
    Calculate changes required to move from previous to target weights.
    """

    if previous_weights.empty:
        raise ValueError("previous_weights is empty.")

    if target_weights.empty:
        raise ValueError("target_weights is empty.")

    if not previous_weights.index.equals(target_weights.index):
        raise ValueError("Weight series must have identical ticker indices.")

    if previous_weights.isna().any():
        raise ValueError("previous_weights contains missing values.")

    if target_weights.isna().any():
        raise ValueError("target_weights contains missing values.")

    return target_weights - previous_weights

def calculate_turnover(
    previous_weights: pd.Series,
    target_weights: pd.Series,
) -> float:
    """
    Calculate one-way portfolio turnover.

    Turnover is half the sum of absolute weight changes.
    """

    weight_changes = calculate_weight_changes(
        previous_weights,
        target_weights,
    )

    return 0.5 * weight_changes.abs().sum()