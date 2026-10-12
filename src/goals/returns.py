"""Multi-horizon returns and trend analysis for QuantPilot."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class Trend(str, Enum):
    """Trend classification."""

    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"


@dataclass(frozen=True)
class MultiHorizonReturns:
    """Realized returns over multiple horizons."""

    return_30d: float | None = None
    return_90d: float | None = None
    return_180d: float | None = None
    return_1y: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "30d": self.return_30d,
            "90d": self.return_90d,
            "180d": self.return_180d,
            "1y": self.return_1y,
        }


@dataclass(frozen=True)
class TechnicalIndicators:
    """Technical indicators for trend analysis."""

    sma_50: float | None = None
    sma_200: float | None = None
    price_vs_sma50: str | None = None  # "above" / "below" / None
    price_vs_sma200: str | None = None

    def to_dict(self) -> dict[str, float | str | None]:
        return {
            "sma_50": self.sma_50,
            "sma_200": self.sma_200,
            "price_vs_sma50": self.price_vs_sma50,
            "price_vs_sma200": self.price_vs_sma200,
        }


@dataclass(frozen=True)
class StockAnalysis:
    """Complete stock analysis combining recent performance, trend, and technicals."""

    ticker: str
    current_price: float
    multi_horizon: MultiHorizonReturns
    trend: Trend
    technical: TechnicalIndicators
    trend_score: int

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "current_price": self.current_price,
            "multi_horizon": self.multi_horizon.to_dict(),
            "trend": self.trend.value,
            "trend_score": self.trend_score,
            "technical": self.technical.to_dict(),
        }


def calculate_multi_horizon_returns(
    prices: pd.Series,
    horizons: tuple[int, ...] = (30, 90, 180, 252),
) -> dict[int, float | None]:
    """
    Calculate realized returns over specified horizons.

    Returns decimals (e.g., 0.05 for 5%), NOT annualized.

    Args:
        prices: Price series (chronological, oldest first)
        horizons: Trading day horizons

    Returns:
        Dict mapping horizon -> return (decimal) or None if insufficient data
    """
    if len(prices) < 2:
        return {h: None for h in horizons}

    current_price = prices.iloc[-1]
    results = {}

    for h in horizons:
        if len(prices) > h:
            past_price = prices.iloc[-h - 1]
            if past_price > 0:
                results[h] = float(current_price / past_price - 1.0)
            else:
                results[h] = None
        else:
            results[h] = None

    return results


def calculate_moving_averages(
    prices: pd.Series,
    windows: tuple[int, ...] = (50, 200),
) -> dict[int, float | None]:
    """
    Calculate simple moving averages.

    Args:
        prices: Price series
        windows: Window sizes in trading days

    Returns:
        Dict mapping window -> SMA value or None if insufficient data
    """
    results = {}
    for w in windows:
        if len(prices) >= w:
            results[w] = float(prices.tail(w).mean())
        else:
            results[w] = None
    return results


def calculate_trend_score(
    multi_horizon: MultiHorizonReturns,
    technical: TechnicalIndicators,
) -> int:
    """
    Calculate trend score from multiple signals.

    Signals:
    - 30D return: +1 if positive, -1 if negative
    - 90D return: +1 if positive, -1 if negative
    - Price vs 50DMA: +1 if above, -1 if below
    - Price vs 200DMA: +1 if above, -1 if below

    Returns:
        Score from -4 to +4
    """
    score = 0

    # 30D return signal
    if multi_horizon.return_30d is not None:
        if multi_horizon.return_30d > 0:
            score += 1
        elif multi_horizon.return_30d < 0:
            score -= 1

    # 90D return signal
    if multi_horizon.return_90d is not None:
        if multi_horizon.return_90d > 0:
            score += 1
        elif multi_horizon.return_90d < 0:
            score -= 1

    # Price vs 50DMA
    if technical.price_vs_sma50 == "above":
        score += 1
    elif technical.price_vs_sma50 == "below":
        score -= 1

    # Price vs 200DMA
    if technical.price_vs_sma200 == "above":
        score += 1
    elif technical.price_vs_sma200 == "below":
        score -= 1

    return score


def classify_trend(score: int) -> Trend:
    """Classify trend from score."""
    if score <= -2:
        return Trend.BEARISH
    elif score >= 2:
        return Trend.BULLISH
    else:
        return Trend.NEUTRAL


def analyze_stock(
    prices: pd.Series,
    ticker: str,
    horizons: tuple[int, ...] = (30, 90, 180, 252),
) -> StockAnalysis:
    """
    Complete stock analysis: multi-horizon returns, technicals, trend.

    Args:
        prices: Price series (chronological)
        ticker: Ticker symbol
        horizons: Horizons for return calculation

    Returns:
        StockAnalysis with all computed fields
    """
    if len(prices) < 2:
        raise ValueError(f"Insufficient price data for {ticker}: {len(prices)} observations")

    current_price = float(prices.iloc[-1])

    # Multi-horizon returns
    horizon_returns = calculate_multi_horizon_returns(prices, horizons)
    multi_horizon = MultiHorizonReturns(
        return_30d=horizon_returns.get(30),
        return_90d=horizon_returns.get(90),
        return_180d=horizon_returns.get(180),
        return_1y=horizon_returns.get(252),
    )

    # Moving averages
    mas = calculate_moving_averages(prices, (50, 200))
    sma_50 = mas.get(50)
    sma_200 = mas.get(200)

    # Price vs MA
    price_vs_sma50 = None
    if sma_50 is not None:
        price_vs_sma50 = "above" if current_price > sma_50 else "below"

    price_vs_sma200 = None
    if sma_200 is not None:
        price_vs_sma200 = "above" if current_price > sma_200 else "below"

    technical = TechnicalIndicators(
        sma_50=sma_50,
        sma_200=sma_200,
        price_vs_sma50=price_vs_sma50,
        price_vs_sma200=price_vs_sma200,
    )

    # Trend classification
    trend_score = calculate_trend_score(multi_horizon, technical)
    trend = classify_trend(trend_score)

    return StockAnalysis(
        ticker=ticker,
        current_price=current_price,
        multi_horizon=multi_horizon,
        trend=trend,
        technical=technical,
        trend_score=trend_score,
    )


def screen_bearish_stocks(
    stock_analyses: list[StockAnalysis],
    top_n: int | None = None,
) -> list[StockAnalysis]:
    """
    Screen for bearish stocks.

    Ranks by trend_score (most negative first), then by 90D return.
    Only returns BEARISH classified stocks.

    Args:
        stock_analyses: List of StockAnalysis objects
        top_n: Maximum number to return (None = all)

    Returns:
        Sorted list of bearish StockAnalysis objects
    """
    bearish = [s for s in stock_analyses if s.trend == Trend.BEARISH]

    # Sort: most negative score first, then most negative 90D return
    bearish.sort(
        key=lambda s: (
            s.trend_score,
            s.multi_horizon.return_90d if s.multi_horizon.return_90d is not None else 1.0,
        )
    )

    if top_n:
        return bearish[:top_n]
    return bearish


def calculate_annual_volatility(returns: pd.Series) -> float | None:
    """Calculate annualized volatility from daily returns."""
    if len(returns) < 2:
        return None
    daily_vol = returns.std()
    return float(daily_vol * np.sqrt(252))


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.045,
) -> float | None:
    """Calculate annualized Sharpe ratio."""
    if len(returns) < 2:
        return None
    annual_return = (1 + returns.mean()) ** 252 - 1
    annual_vol = calculate_annual_volatility(returns)
    if annual_vol is None or annual_vol == 0:
        return None
    return (annual_return - risk_free_rate) / annual_vol