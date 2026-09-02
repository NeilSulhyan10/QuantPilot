import pandas as pd


def equal_weight(tickers: list[str]) -> pd.Series:
    """Return equal portfolio weights for the supplied tickers."""
    if not tickers:
        raise ValueError("Tickers list is empty.")

    if len(set(tickers)) != len(tickers):
        raise ValueError("Tickers must be unique.")

    weight = 1.0 / len(tickers)

    return pd.Series(
        weight,
        index=tickers,
        name="weight",
    )