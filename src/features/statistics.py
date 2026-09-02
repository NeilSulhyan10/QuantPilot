import pandas as pd


def rolling_mean(
    returns: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """
    Calculate rolling mean returns for each asset.
    """
    if returns.empty:
        raise ValueError("Returns dataframe is empty.")

    if window <= 0:
        raise ValueError("Window must be positive.")

    return returns.rolling(window=window).mean()


def rolling_volatility(
    returns: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """
    Calculate rolling standard deviation of returns for each asset.
    """
    if returns.empty:
        raise ValueError("Returns dataframe is empty.")

    if window <= 0:
        raise ValueError("Window must be positive.")

    return returns.rolling(window=window).std()