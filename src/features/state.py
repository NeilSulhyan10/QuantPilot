from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FeatureState:
    """
    Statistical information available to the portfolio model at one date.
    """

    date: pd.Timestamp
    tickers: list[str]
    expected_returns: np.ndarray
    covariance: np.ndarray
    volatility: np.ndarray

def build_feature_state(
    date: pd.Timestamp,
    mean_returns: pd.DataFrame,
    volatility: pd.DataFrame,
    covariance_matrices: dict[pd.Timestamp, pd.DataFrame],
) -> FeatureState:
    """
    Build a FeatureState for a specific date.
    """

    if date not in mean_returns.index:
        raise ValueError(f"No expected-return data available for {date}.")

    if date not in volatility.index:
        raise ValueError(f"No volatility data available for {date}.")

    if date not in covariance_matrices:
        raise ValueError(f"No covariance matrix available for {date}.")

    expected_returns = mean_returns.loc[date]
    vol = volatility.loc[date]
    covariance = covariance_matrices[date]

    tickers = list(expected_returns.index)

    if list(vol.index) != tickers:
        raise ValueError("Ticker ordering mismatch between mean returns and volatility.")

    if list(covariance.index) != tickers:
        raise ValueError("Ticker ordering mismatch in covariance matrix.")

    if list(covariance.columns) != tickers:
        raise ValueError("Ticker ordering mismatch in covariance columns.")

    if expected_returns.isna().any():
        raise ValueError("Expected returns contain missing values.")

    if vol.isna().any():
        raise ValueError("Volatility contains missing values.")

    if covariance.isna().any().any():
        raise ValueError("Covariance matrix contains missing values.")

    return FeatureState(
        date=date,
        tickers=tickers,
        expected_returns=expected_returns.to_numpy(),
        covariance=covariance.to_numpy(),
        volatility=vol.to_numpy(),
    )