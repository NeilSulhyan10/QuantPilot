import pandas as pd


def rolling_covariance(
    returns: pd.DataFrame,
    window: int = 60,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """
    Calculate a rolling covariance matrix for each date.

    Returns
    -------
    dict[pd.Timestamp, pd.DataFrame]
        A dictionary mapping each date to its covariance matrix.
    """
    if returns.empty:
        raise ValueError("Returns dataframe is empty.")

    if window <= 0:
        raise ValueError("Window must be positive.")

    rolling_cov = returns.rolling(window=window).cov()

    covariance_matrices = {}

    for date in returns.index:
        matrix = rolling_cov.loc[date]

        if matrix.isna().any().any():
            continue

        covariance_matrices[date] = matrix

    return covariance_matrices