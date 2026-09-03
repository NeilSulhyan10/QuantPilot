import pytest

from src.data.assets import (
    Asset,
    build_assets,
    validate_asset_selection,
)
from src.data.markets import Market


def test_build_us_assets():
    assets = build_assets(
        ["AAPL", "MSFT"],
        Market.US,
    )

    assert assets == [
        Asset(ticker="AAPL", name="AAPL"),
        Asset(ticker="MSFT", name="MSFT"),
    ]


def test_build_india_assets():
    assets = build_assets(
        ["RELIANCE", "TCS"],
        Market.INDIA,
    )

    assert assets == [
        Asset(ticker="RELIANCE.NS", name="RELIANCE.NS"),
        Asset(ticker="TCS.NS", name="TCS.NS"),
    ]


def test_validation_normalizes_tickers():
    result = validate_asset_selection(
        ["aapl", " msft "],
        Market.US,
    )

    assert result == ["AAPL", "MSFT"]


def test_validation_removes_duplicates():
    result = validate_asset_selection(
        ["AAPL", "aapl", "MSFT"],
        Market.US,
    )

    assert result == ["AAPL", "MSFT"]


def test_too_few_assets():
    with pytest.raises(ValueError, match="at least 2"):
        validate_asset_selection(
            ["AAPL"],
            Market.US,
        )


def test_too_many_assets():
    with pytest.raises(ValueError, match="at most 3"):
        validate_asset_selection(
            ["AAPL", "MSFT", "NVDA", "AMZN"],
            Market.US,
            max_assets=3,
        )


def test_invalid_limits():
    with pytest.raises(ValueError, match="max_assets"):
        validate_asset_selection(
            ["AAPL", "MSFT"],
            Market.US,
            min_assets=5,
            max_assets=3,
        )


def test_empty_selection():
    with pytest.raises(ValueError):
        validate_asset_selection(
            [],
            Market.US,
        )