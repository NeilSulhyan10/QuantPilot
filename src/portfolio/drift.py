import pandas as pd


def drift_weights(
    weights: pd.Series,
    asset_returns: pd.Series,
) -> pd.Series:
    """
    Update portfolio weights after one period of asset returns.

    The portfolio is not rebalanced during this calculation, so
    weights drift according to relative asset performance.
    """
    if weights.empty:
        raise ValueError("weights is empty.")

    if asset_returns.empty:
        raise ValueError("asset_returns is empty.")

    if not weights.index.equals(asset_returns.index):
        raise ValueError("weights and asset_returns must have identical indices.")

    if weights.isna().any():
        raise ValueError("weights contains NaN values.")

    if asset_returns.isna().any():
        raise ValueError("asset_returns contains NaN values.")

    portfolio_return = float((weights * asset_returns).sum())

    denominator = 1.0 + portfolio_return

    if denominator <= 0:
        raise ValueError("Portfolio value became non-positive.")

    new_weights = weights * (1.0 + asset_returns) / denominator

    return new_weights