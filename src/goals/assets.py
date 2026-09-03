"""Dynamic asset selection for QuantPilot V2."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.reader import load_stock


@dataclass(frozen=True)
class AssetSelection:
    """Validated collection of assets selected for a goal."""

    tickers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.tickers:
            raise ValueError("At least one ticker must be selected.")

        normalized = tuple(ticker.upper().strip() for ticker in self.tickers)

        if any(not ticker for ticker in normalized):
            raise ValueError("Ticker symbols cannot be empty.")

        if len(set(normalized)) != len(normalized):
            raise ValueError("Duplicate ticker symbols are not allowed.")

        object.__setattr__(self, "tickers", normalized)


def validate_asset_selection(
    tickers: list[str] | tuple[str, ...],
    available_tickers: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    """Validate and normalize user-selected assets."""

    if not tickers:
        raise ValueError("At least one ticker must be selected.")

    available = {
        ticker.upper().strip()
        for ticker in available_tickers
    }

    selected = tuple(
        ticker.upper().strip()
        for ticker in tickers
    )

    if any(not ticker for ticker in selected):
        raise ValueError("Ticker symbols cannot be empty.")

    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate ticker symbols are not allowed.")

    unknown = sorted(set(selected) - available)

    if unknown:
        raise ValueError(
            f"Unknown ticker(s): {', '.join(unknown)}"
        )

    return selected


def load_selected_assets(
    tickers: list[str] | tuple[str, ...],
    available_tickers: list[str] | tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    """Load processed historical data for the selected assets."""

    selected = validate_asset_selection(
        tickers,
        available_tickers,
    )

    return {
        ticker: load_stock(ticker)
        for ticker in selected
    }


def build_selected_return_matrix(
    asset_data: dict[str, pd.DataFrame],
    price_column: str = "Close",
    method: str = "simple",
) -> pd.DataFrame:
    """Build an aligned return matrix for selected assets.

    Only dates with valid observations for every selected asset
    are retained. No forward-filling is performed.
    """

    if not asset_data:
        raise ValueError("asset_data cannot be empty.")

    if price_column not in {
        column
        for data in asset_data.values()
        for column in data.columns
    }:
        raise ValueError(
            f"Price column '{price_column}' is not available."
        )

    from src.features.returns import calculate_return_matrix

    return calculate_return_matrix(
        asset_data,
        price_column=price_column,
        method=method,
    )

def validate_minimum_history(
    returns: pd.DataFrame,
    minimum_observations: int = 252,
) -> None:
    """Validate that selected assets have sufficient common history."""

    if minimum_observations <= 0:
        raise ValueError(
            "minimum_observations must be greater than 0."
        )

    if returns.empty:
        raise ValueError(
            "No historical observations are available."
        )

    if returns.shape[0] < minimum_observations:
        raise ValueError(
            f"Insufficient historical data: "
            f"{returns.shape[0]} observations available, "
            f"{minimum_observations} required."
        )