import pytest

from src.data.markets import (
    Market,
    get_market_config,
    normalize_ticker,
    normalize_tickers,
)


def test_us_market_config():
    config = get_market_config(Market.US)

    assert config.name == "United States"
    assert config.currency == "USD"
    assert config.benchmark == "SPY"
    assert config.ticker_suffix == ""


def test_india_market_config():
    config = get_market_config(Market.INDIA)

    assert config.name == "India"
    assert config.currency == "INR"
    assert config.benchmark == "^NSEI"
    assert config.ticker_suffix == ".NS"


def test_market_accepts_string():
    assert get_market_config("us").currency == "USD"
    assert get_market_config("india").currency == "INR"


def test_invalid_market():
    with pytest.raises(ValueError, match="Unsupported market"):
        get_market_config("uk")


def test_us_ticker_normalization():
    assert normalize_ticker(" aapl ", Market.US) == "AAPL"
    assert normalize_ticker("AAPL.US", Market.US) == "AAPL"


def test_india_ticker_normalization():
    assert normalize_ticker("reliance", Market.INDIA) == "RELIANCE.NS"
    assert normalize_ticker("RELIANCE.NS", Market.INDIA) == "RELIANCE.NS"


def test_empty_ticker():
    with pytest.raises(ValueError, match="Ticker must not be empty"):
        normalize_ticker(" ", Market.US)


def test_non_string_ticker():
    with pytest.raises(TypeError, match="Ticker must be a string"):
        normalize_ticker(123, Market.US)


def test_duplicate_tickers_are_removed():
    assert normalize_tickers(
        ["AAPL", "aapl", " MSFT "],
        Market.US,
    ) == ["AAPL", "MSFT"]


def test_empty_ticker_list():
    with pytest.raises(ValueError, match="At least one ticker"):
        normalize_tickers([], Market.US)
