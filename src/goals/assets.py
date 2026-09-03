"""Dynamic, market-aware asset selection for QuantPilot."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.market_data import MarketDataAdapter
from src.data.markets import Market, normalize_tickers


@dataclass(frozen=True)
class AssetSelection:
    """Validated collection of assets selected for a goal."""

    tickers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.tickers:
            raise ValueError(
                "At least one ticker must be selected."
            )

        normalized = tuple(
            ticker.upper().strip()
            for ticker in self.tickers
        )

        if any(not ticker for ticker in normalized):
            raise ValueError(
                "Ticker symbols cannot be empty."
            )

        if len(set(normalized)) != len(normalized):
            raise ValueError(
                "Duplicate ticker symbols are not allowed."
            )

        object.__setattr__(
            self,
            "tickers",
            normalized,
        )


def validate_asset_selection(
    tickers: list[str] | tuple[str, ...],
    available_tickers: list[str] | tuple[str, ...] | None = None,
    *,
    market: Market | str = Market.US,
    min_assets: int = 1,
    max_assets: int = 20,
) -> tuple[str, ...]:
    """
    Validate and normalize user-selected assets.

    Backward-compatible behavior:
        validate_asset_selection(tickers, available_tickers)

    V3 market-aware behavior:
        validate_asset_selection(
            tickers,
            market=Market.INDIA,
        )

    If ``available_tickers`` is supplied, selected assets must
    belong to that collection.
    """

    if min_assets <= 0:
        raise ValueError(
            "min_assets must be greater than 0."
        )

    if max_assets < min_assets:
        raise ValueError(
            "max_assets must be greater than or equal to min_assets."
        )

    if not tickers:
        raise ValueError(
            "At least one ticker must be selected."
        )

    # Check duplicates BEFORE normalize_tickers(), because
    # normalize_tickers() intentionally deduplicates.
    raw_selected = tuple(
        ticker.upper().strip()
        for ticker in tickers
    )

    if any(not ticker for ticker in raw_selected):
        raise ValueError(
            "Ticker symbols cannot be empty."
        )

    if len(set(raw_selected)) != len(raw_selected):
        raise ValueError(
            "Duplicate ticker symbols are not allowed."
        )

    normalized = tuple(
        normalize_tickers(
            raw_selected,
            market,
        )
    )

    if len(normalized) < min_assets:
        raise ValueError(
            f"At least {min_assets} assets must be selected."
        )

    if len(normalized) > max_assets:
        raise ValueError(
            f"No more than {max_assets} assets may be selected."
        )

    if available_tickers is not None:
        available = set(
            normalize_tickers(
                available_tickers,
                market,
            )
        )

        unknown = sorted(
            set(normalized) - available
        )

        if unknown:
            raise ValueError(
                f"Unknown ticker(s): {', '.join(unknown)}"
            )

    return normalized


def load_selected_assets(
    tickers: list[str] | tuple[str, ...],
    available_tickers: list[str] | tuple[str, ...] | None = None,
    *,
    market: Market | str = Market.US,
    data_adapter: MarketDataAdapter | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    allow_download: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load historical data for user-selected assets."""

    selected = validate_asset_selection(
        tickers,
        available_tickers,
        market=market,
    )

    adapter = data_adapter or MarketDataAdapter(
        market=market,
    )

    return adapter.load_assets(
        selected,
        allow_download=allow_download,
        start_date=start_date,
        end_date=end_date,
    )


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
        raise ValueError(
            "asset_data cannot be empty."
        )

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