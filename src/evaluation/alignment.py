import pandas as pd


def align_return_series(
    first: pd.Series,
    second: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Align two return series to their common dates."""
    if first.empty:
        raise ValueError("First return series is empty.")

    if second.empty:
        raise ValueError("Second return series is empty.")

    common_index = first.index.intersection(second.index)

    if len(common_index) == 0:
        raise ValueError("Return series have no common dates.")

    common_index = common_index.sort_values()

    return (
        first.loc[common_index],
        second.loc[common_index],
    )