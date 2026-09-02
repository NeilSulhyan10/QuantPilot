import pandas as pd


def calculate_ml_features(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate predictive features from historical daily returns.

    Features are calculated independently for each asset and use
    only information available up to the current date.
    """

    if returns.empty:
        raise ValueError("Returns dataframe is empty.")

    if returns.isna().any().any():
        raise ValueError("Returns dataframe contains missing values.")

    features = {}

    for ticker in returns.columns:
        asset_returns = returns[ticker]

        features[(ticker, "return_5d")] = (
            asset_returns.rolling(5).sum()
        )

        features[(ticker, "return_20d")] = (
            asset_returns.rolling(20).sum()
        )

        features[(ticker, "volatility_20d")] = (
            asset_returns.rolling(20).std()
        )

        features[(ticker, "volatility_60d")] = (
            asset_returns.rolling(60).std()
        )

        features[(ticker, "mean_60d")] = (
            asset_returns.rolling(60).mean()
        )

    result = pd.concat(
        features,
        axis=1,
    )

    result.columns = pd.MultiIndex.from_tuples(
        result.columns,
        names=["ticker", "feature"],
    )

    return result

def calculate_forward_return_target(
    returns: pd.DataFrame,
    horizon: int = 21,
) -> pd.DataFrame:
    if returns.empty:
        raise ValueError("Returns cannot be empty.")

    if horizon <= 0:
        raise ValueError("Horizon must be positive.")

    if returns.isna().any().any():
        raise ValueError("Returns contain NaN values.")

    forward_returns = (
        (1.0 + returns)
        .rolling(window=horizon)
        .apply(lambda x: x.prod() - 1.0, raw=True)
        .shift(-horizon)
    )

    return forward_returns