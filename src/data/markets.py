from dataclasses import dataclass
from enum import Enum


class Market(str, Enum):
    """Supported QuantPilot markets."""

    US = "us"
    INDIA = "india"


@dataclass(frozen=True)
class MarketConfig:
    """Market-specific configuration used by the data layer."""

    market: Market
    name: str
    currency: str
    benchmark: str
    ticker_suffix: str = ""


MARKET_CONFIGS: dict[Market, MarketConfig] = {
    Market.US: MarketConfig(
        market=Market.US,
        name="United States",
        currency="USD",
        benchmark="SPY",
        ticker_suffix="",
    ),
    Market.INDIA: MarketConfig(
        market=Market.INDIA,
        name="India",
        currency="INR",
        benchmark="^NSEI",
        ticker_suffix=".NS",
    ),
}


def get_market_config(market: Market | str) -> MarketConfig:
    """Return configuration for a supported market."""
    try:
        market = Market(market)
    except ValueError as exc:
        supported = ", ".join(m.value for m in Market)
        raise ValueError(
            f"Unsupported market: {market!r}. "
            f"Supported markets: {supported}."
        ) from exc

    return MARKET_CONFIGS[market]


def normalize_ticker(ticker: str, market: Market | str) -> str:
    """
    Normalize a user-entered ticker for the selected market.

    US:
        AAPL -> AAPL
        AAPL.US -> AAPL

    India:
        RELIANCE -> RELIANCE.NS
        RELIANCE.NS -> RELIANCE.NS
    """
    if not isinstance(ticker, str):
        raise TypeError("Ticker must be a string.")

    ticker = ticker.strip().upper()

    if not ticker:
        raise ValueError("Ticker must not be empty.")

    config = get_market_config(market)

    if config.market is Market.INDIA:
        if ticker.endswith(".NS"):
            return ticker
        return f"{ticker}.NS"

    return ticker.removesuffix(".US")


def normalize_tickers(
    tickers: list[str],
    market: Market | str,
) -> list[str]:
    """Normalize tickers, remove duplicates, and preserve input order."""
    normalized: list[str] = []

    for ticker in tickers:
        value = normalize_ticker(ticker, market)
        if value not in normalized:
            normalized.append(value)

    if not normalized:
        raise ValueError("At least one ticker is required.")

    return normalized
