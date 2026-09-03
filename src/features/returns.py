import numpy as np
import pandas as pd


def simple_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate simple percentage returns.

    R_t = P_t / P_{t-1} - 1
    """
    return prices.pct_change()


def log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate continuously compounded log returns.

    r_t = ln(P_t / P_{t-1})
    """
    return np.log(prices / prices.shift(1))


def validate_prices(prices: pd.Series) -> None:
    """Validate a price series before calculating returns."""
    if prices.empty:
        raise ValueError("Price series is empty.")

    if prices.isna().any():
        raise ValueError("Price series contains missing values.")

    if (prices <= 0).any():
        raise ValueError("Price series contains non-positive values.")


def calculate_returns(
    prices: pd.Series,
    method: str = "simple",
) -> pd.Series:
    """
    Calculate returns using the requested method.

    Parameters
    ----------
    prices : pd.Series
        Price series indexed by date.
    method : str
        Either "simple" or "log".

    Returns
    -------
    pd.Series
        Return series.
    """
    validate_prices(prices)

    if method == "simple":
        return simple_returns(prices)

    if method == "log":
        return log_returns(prices)

    raise ValueError(
        f"Unknown return method: {method}. "
        "Use 'simple' or 'log'."
    )

def calculate_return_matrix(
    price_data: dict[str, pd.DataFrame],
    price_column: str = "Close",
    method: str = "simple",
) -> pd.DataFrame:
    """
    Calculate an aligned cross-sectional return matrix.

    Parameters
    ----------
    price_data : dict[str, pd.DataFrame]
        Mapping from ticker to processed OHLCV data.
    price_column : str
        Price column used to calculate returns.
    method : str
        Either "simple" or "log".

    Returns
    -------
    pd.DataFrame
        Date-indexed return matrix with one column per ticker.
    """
    if not price_data:
        raise ValueError("price_data is empty.")

    returns = {}

    for ticker, data in price_data.items():
        if price_column not in data.columns:
            raise ValueError(
                f"{price_column} column missing for {ticker}."
            )

        returns[ticker.upper()] = calculate_returns(
            data[price_column],
            method=method,
        )

    return pd.concat(returns, axis=1, sort=False,).dropna(how="any")