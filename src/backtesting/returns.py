import pandas as pd


def calculate_portfolio_return(
    asset_returns: pd.Series,
    weights: pd.Series,
) -> float:
    """
    Calculate the gross portfolio return for one trading period.
    """

    if asset_returns.empty:
        raise ValueError("asset_returns is empty.")

    if weights.empty:
        raise ValueError("weights is empty.")

    if not asset_returns.index.equals(weights.index):
        raise ValueError(
            "Asset returns and weights must have identical ticker indices."
        )

    if asset_returns.isna().any():
        raise ValueError("asset_returns contains missing values.")

    if weights.isna().any():
        raise ValueError("weights contains missing values.")

    return float((asset_returns * weights).sum())


def calculate_net_portfolio_return(
    gross_return: float,
    transaction_cost: float = 0.0,
    slippage: float = 0.0,
) -> float:
    """
    Calculate portfolio return after trading costs and slippage.
    """

    if transaction_cost < 0:
        raise ValueError("transaction_cost cannot be negative.")

    if slippage < 0:
        raise ValueError("slippage cannot be negative.")

    return gross_return - transaction_cost - slippage

def apply_rebalance_costs(
    gross_returns: pd.Series,
    rebalance_dates: pd.DatetimeIndex,
    transaction_costs: pd.Series,
    slippage: pd.Series,
) -> pd.Series:
    if gross_returns.empty:
        raise ValueError("gross_returns is empty.")

    net_returns = gross_returns.copy()

    for date in rebalance_dates:
        if date in net_returns.index:
            net_returns.loc[date] -= (
                transaction_costs.loc[date]
                + slippage.loc[date]
            )
        else:
            # Costs are charged on the rebalance date even though
            # portfolio returns begin on the following trading day.
            next_dates = net_returns.index[net_returns.index > date]

            if len(next_dates) > 0:
                first_return_date = next_dates[0]

                net_returns.loc[first_return_date] -= (
                    transaction_costs.loc[date]
                    + slippage.loc[date]
                )

    return net_returns
