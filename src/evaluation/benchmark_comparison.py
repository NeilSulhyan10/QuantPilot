from __future__ import annotations

import pandas as pd

from src.evaluation.alignment import align_return_series
from src.evaluation.metrics import evaluate_returns


def evaluate_strategy_comparison(
    strategy_returns: dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Evaluate multiple strategies on their common evaluation period.

    The common period is the intersection of all supplied return
    series. This prevents one strategy from receiving a longer
    evaluation window than another.
    """

    if not strategy_returns:
        raise ValueError("No strategy return series provided.")

    for name, returns in strategy_returns.items():
        if returns.empty:
            raise ValueError(
                f"Strategy '{name}' has an empty return series."
            )

    common_index = None

    for returns in strategy_returns.values():
        if common_index is None:
            common_index = returns.index
        else:
            common_index = common_index.intersection(returns.index)

    if common_index is None or len(common_index) == 0:
        raise ValueError(
            "Strategies have no common evaluation dates."
        )

    common_index = common_index.sort_values()

    rows = {}

    for name, returns in strategy_returns.items():
        aligned_returns = returns.loc[common_index]

        rows[name] = evaluate_returns(aligned_returns)

    result = pd.DataFrame.from_dict(rows, orient="index")

    result.index.name = "strategy"

    return result
