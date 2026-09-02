import pandas as pd


def historical_mean_expected_returns(
    returns: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """
    Calculate rolling historical mean expected returns.

    Parameters
    ----------
    returns:
        Daily asset returns.
    window:
        Rolling estimation window.

    Returns
    -------
    pd.DataFrame
        Rolling expected-return estimates.
    """
    if returns.empty:
        raise ValueError("Returns dataframe is empty.")

    if window <= 0:
        raise ValueError("window must be positive.")

    if returns.isna().any().any():
        raise ValueError("Returns dataframe contains missing values.")

    return returns.rolling(window=window).mean()


def shrink_expected_returns(
    expected_returns: pd.DataFrame,
    alpha: float = 0.5,
) -> pd.DataFrame:
    """
    Shrink stock-specific expected returns toward
    the cross-sectional mean.

    alpha = 1.0:
        No shrinkage.

    alpha = 0.0:
        All stocks receive the same expected return.

    Parameters
    ----------
    expected_returns:
        Cross-sectional expected-return estimates.
    alpha:
        Weight assigned to the stock-specific estimate.

    Returns
    -------
    pd.DataFrame
        Shrunk expected returns.
    """
    if expected_returns.empty:
        raise ValueError("Expected returns dataframe is empty.")

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    cross_sectional_mean = expected_returns.mean(axis=1)

    shrunk = (
        cross_sectional_mean.to_numpy()[:, None]
        + alpha
        * (
            expected_returns.to_numpy()
            - cross_sectional_mean.to_numpy()[:, None]
        )
    )

    return pd.DataFrame(
        shrunk,
        index=expected_returns.index,
        columns=expected_returns.columns,
    )


def calculate_expected_returns(
    returns: pd.DataFrame,
    window: int = 60,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """
    Calculate rolling historical expected returns
    with optional cross-sectional shrinkage.
    """
    estimates = historical_mean_expected_returns(
        returns=returns,
        window=window,
    )

    return shrink_expected_returns(
        expected_returns=estimates,
        alpha=alpha,
    )