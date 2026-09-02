import pandas as pd


def monthly_rebalance_dates(
    dates: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """
    Return the last available trading date of each calendar month.
    """

    if not isinstance(dates, pd.DatetimeIndex):
        raise TypeError("dates must be a pandas DatetimeIndex.")

    if len(dates) == 0:
        raise ValueError("dates cannot be empty.")

    dates = dates.sort_values().unique()

    date_series = pd.Series(dates, index=dates)

    rebalance_dates = date_series.groupby(
        date_series.index.to_period("M")
    ).last()

    return pd.DatetimeIndex(rebalance_dates.values)