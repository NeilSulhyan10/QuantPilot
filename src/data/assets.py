from dataclasses import dataclass

from .markets import Market, normalize_ticker, normalize_tickers


@dataclass(frozen=True)
class Asset:
    """A user-selectable investment asset."""

    ticker: str
    name: str


def build_assets(
    tickers: list[str],
    market: Market | str,
) -> list[Asset]:
    """
    Convert user-entered tickers into normalized Asset objects.

    Names are initially derived from the normalized ticker. A later
    data/search layer can replace these with human-readable company names.
    """
    normalized = normalize_tickers(tickers, market)

    return [
        Asset(
            ticker=ticker,
            name=ticker,
        )
        for ticker in normalized
    ]


def validate_asset_selection(
    tickers: list[str],
    market: Market | str,
    *,
    min_assets: int = 2,
    max_assets: int = 20,
) -> list[str]:
    """
    Validate and normalize a user-selected asset universe.

    This function deliberately does not check whether the tickers
    actually exist. That belongs to the market-data layer.
    """
    if min_assets < 1:
        raise ValueError("min_assets must be at least 1.")

    if max_assets < min_assets:
        raise ValueError("max_assets must be >= min_assets.")

    normalized = normalize_tickers(tickers, market)

    if len(normalized) < min_assets:
        raise ValueError(
            f"Select at least {min_assets} assets."
        )

    if len(normalized) > max_assets:
        raise ValueError(
            f"Select at most {max_assets} assets."
        )

    return normalized