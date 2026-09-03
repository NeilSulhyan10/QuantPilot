import pandas as pd
import pytest

from src.data.market_data import MarketDataAdapter
from src.data.markets import Market


def make_valid_data():
    index = pd.date_range(
        "2025-01-01",
        periods=3,
        freq="D",
    )

    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [105.0, 106.0, 107.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [103.0, 104.0, 106.0],
            "Volume": [1000, 1100, 1200],
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# Market configuration
# ---------------------------------------------------------------------------


def test_adapter_us_market():
    adapter = MarketDataAdapter(Market.US)

    assert adapter.market == Market.US
    assert adapter.config.currency == "USD"
    assert adapter.config.benchmark == "SPY"


def test_adapter_india_market():
    adapter = MarketDataAdapter(Market.INDIA)

    assert adapter.market == Market.INDIA
    assert adapter.config.currency == "INR"
    assert adapter.config.benchmark == "^NSEI"


# ---------------------------------------------------------------------------
# Cached data
# ---------------------------------------------------------------------------


def test_load_cached_asset(tmp_path):
    data = make_valid_data()

    data.to_parquet(
        tmp_path / "AAPL.parquet"
    )

    adapter = MarketDataAdapter(
        Market.US,
        data_dir=tmp_path,
    )

    result = adapter.load_asset("AAPL")

    pd.testing.assert_frame_equal(
        result,
        data,
        check_freq=False,
    )


def test_cached_asset_is_normalized(tmp_path):
    data = make_valid_data()

    data.to_parquet(
        tmp_path / "RELIANCE.NS.parquet"
    )

    adapter = MarketDataAdapter(
        Market.INDIA,
        data_dir=tmp_path,
    )

    result = adapter.load_asset("reliance")

    pd.testing.assert_frame_equal(
        result,
        data,
        check_freq=False,
    )


def test_missing_cached_asset_without_download(tmp_path):
    adapter = MarketDataAdapter(
        Market.US,
        data_dir=tmp_path,
    )

    with pytest.raises(
        FileNotFoundError,
        match="No processed data found",
    ):
        adapter.load_asset(
            "AAPL",
            allow_download=False,
        )


# ---------------------------------------------------------------------------
# Date filtering
# ---------------------------------------------------------------------------


def test_date_filtering(tmp_path):
    data = make_valid_data()

    data.to_parquet(
        tmp_path / "AAPL.parquet"
    )

    adapter = MarketDataAdapter(
        Market.US,
        data_dir=tmp_path,
    )

    result = adapter.load_asset(
        "AAPL",
        start_date="2025-01-02",
        end_date="2025-01-02",
    )

    assert len(result) == 1
    assert result.index[0] == pd.Timestamp(
        "2025-01-02"
    )


def test_start_date_filtering(tmp_path):
    data = make_valid_data()

    data.to_parquet(
        tmp_path / "AAPL.parquet"
    )

    adapter = MarketDataAdapter(
        Market.US,
        data_dir=tmp_path,
    )

    result = adapter.load_asset(
        "AAPL",
        start_date="2025-01-02",
    )

    assert len(result) == 2
    assert result.index[0] == pd.Timestamp(
        "2025-01-02"
    )


def test_end_date_filtering(tmp_path):
    data = make_valid_data()

    data.to_parquet(
        tmp_path / "AAPL.parquet"
    )

    adapter = MarketDataAdapter(
        Market.US,
        data_dir=tmp_path,
    )

    result = adapter.load_asset(
        "AAPL",
        end_date="2025-01-02",
    )

    assert len(result) == 2
    assert result.index[-1] == pd.Timestamp(
        "2025-01-02"
    )


def test_date_filter_can_produce_empty_result(tmp_path):
    data = make_valid_data()

    data.to_parquet(
        tmp_path / "AAPL.parquet"
    )

    adapter = MarketDataAdapter(
        Market.US,
        data_dir=tmp_path,
    )

    with pytest.raises(
        ValueError,
        match="No valid data available",
    ):
        adapter.load_asset(
            "AAPL",
            start_date="2030-01-01",
        )


# ---------------------------------------------------------------------------
# Download fallback
# ---------------------------------------------------------------------------


def test_missing_cached_asset_downloads(
    monkeypatch,
    tmp_path,
):
    data = make_valid_data()

    def fake_download(
        ticker,
        start_date=None,
        end_date=None,
    ):
        assert ticker == "AAPL"
        return data

    monkeypatch.setattr(
        MarketDataAdapter,
        "_download",
        staticmethod(fake_download),
    )

    adapter = MarketDataAdapter(
        Market.US,
        data_dir=tmp_path,
    )

    result = adapter.load_asset("AAPL")

    pd.testing.assert_frame_equal(
        result,
        data,
        check_freq=False,
    )


def test_download_receives_date_range(
    monkeypatch,
    tmp_path,
):
    data = make_valid_data()

    calls = []

    def fake_download(
        ticker,
        start_date=None,
        end_date=None,
    ):
        calls.append(
            (ticker, start_date, end_date)
        )
        return data

    monkeypatch.setattr(
        MarketDataAdapter,
        "_download",
        staticmethod(fake_download),
    )

    adapter = MarketDataAdapter(
        Market.US,
        data_dir=tmp_path,
    )

    adapter.load_asset(
        "AAPL",
        start_date="2025-01-01",
        end_date="2025-01-03",
    )

    assert calls == [
        (
            "AAPL",
            "2025-01-01",
            "2025-01-03",
        )
    ]


# ---------------------------------------------------------------------------
# Download validation
# ---------------------------------------------------------------------------


def test_invalid_download_is_rejected(
    monkeypatch,
    tmp_path,
):
    data = make_valid_data()

    data.loc[
        data.index[0],
        "High",
    ] = 50.0

    monkeypatch.setattr(
        MarketDataAdapter,
        "_download",
        staticmethod(
            lambda ticker,
            start_date=None,
            end_date=None: data
        ),
    )

    adapter = MarketDataAdapter(
        Market.US,
        data_dir=tmp_path,
    )

    with pytest.raises(ValueError):
        adapter.load_asset("AAPL")


def test_empty_download_is_rejected(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        MarketDataAdapter,
        "_download",
        staticmethod(
            lambda ticker,
            start_date=None,
            end_date=None: pd.DataFrame()
        ),
    )

    adapter = MarketDataAdapter(
        Market.US,
        data_dir=tmp_path,
    )

    with pytest.raises(
        ValueError,
        match="No valid data available",
    ):
        adapter.load_asset("AAPL")


def test_download_missing_columns_is_rejected(
    monkeypatch,
    tmp_path,
):
    data = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [105.0],
            "Low": [99.0],
            "Close": [103.0],
        },
        index=pd.date_range(
            "2025-01-01",
            periods=1,
        ),
    )

    monkeypatch.setattr(
        MarketDataAdapter,
        "_download",
        staticmethod(
            lambda ticker,
            start_date=None,
            end_date=None: data
        ),
    )

    adapter = MarketDataAdapter(
        Market.US,
        data_dir=tmp_path,
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        adapter.load_asset("AAPL")


# ---------------------------------------------------------------------------
# Multiple assets
# ---------------------------------------------------------------------------


def test_load_multiple_assets(tmp_path):
    data = make_valid_data()

    data.to_parquet(
        tmp_path / "AAPL.parquet"
    )

    data.to_parquet(
        tmp_path / "MSFT.parquet"
    )

    adapter = MarketDataAdapter(
        Market.US,
        data_dir=tmp_path,
    )

    result = adapter.load_assets(
        ["AAPL", "MSFT", "AAPL"]
    )

    assert list(result.keys()) == [
        "AAPL",
        "MSFT",
    ]


def test_load_assets_normalizes_indian_tickers(
    monkeypatch,
    tmp_path,
):
    data = make_valid_data()

    calls = []

    def fake_download(
        ticker,
        start_date=None,
        end_date=None,
    ):
        calls.append(ticker)
        return data

    monkeypatch.setattr(
        MarketDataAdapter,
        "_download",
        staticmethod(fake_download),
    )

    adapter = MarketDataAdapter(
        Market.INDIA,
        data_dir=tmp_path,
    )

    result = adapter.load_assets(
        ["reliance", "TCS", "RELIANCE.NS"]
    )

    assert list(result.keys()) == [
        "RELIANCE.NS",
        "TCS.NS",
    ]

    assert calls == [
        "RELIANCE.NS",
        "TCS.NS",
    ]


def test_empty_asset_list():
    adapter = MarketDataAdapter(
        Market.US
    )

    with pytest.raises(
        ValueError,
        match="At least one ticker",
    ):
        adapter.load_assets([])


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def test_benchmark_uses_market_config(
    monkeypatch,
    tmp_path,
):
    data = make_valid_data()

    calls = []

    def fake_load_asset(
        self,
        ticker,
        allow_download=True,
        start_date=None,
        end_date=None,
    ):
        calls.append(ticker)
        return data

    monkeypatch.setattr(
        MarketDataAdapter,
        "load_asset",
        fake_load_asset,
    )

    adapter = MarketDataAdapter(
        Market.INDIA,
        data_dir=tmp_path,
    )

    result = adapter.load_benchmark()

    assert calls == ["^NSEI"]

    pd.testing.assert_frame_equal(
        result,
        data,
    )


def test_us_benchmark_is_spy(
    monkeypatch,
    tmp_path,
):
    data = make_valid_data()

    calls = []

    def fake_load_asset(
        self,
        ticker,
        allow_download=True,
        start_date=None,
        end_date=None,
    ):
        calls.append(ticker)
        return data

    monkeypatch.setattr(
        MarketDataAdapter,
        "load_asset",
        fake_load_asset,
    )

    adapter = MarketDataAdapter(
        Market.US,
        data_dir=tmp_path,
    )

    adapter.load_benchmark()

    assert calls == ["SPY"]


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def test_clean_removes_small_number_of_invalid_rows():
    data = make_valid_data()

    # Create exactly 101 rows.
    data = pd.concat(
        [data] * 34,
        ignore_index=True,
    ).iloc[:101].copy()

    data.index = pd.date_range(
        "2025-01-01",
        periods=len(data),
        freq="D",
    )

    # 1 / 101 < 1%
    data.loc[
        data.index[0],
        "High",
    ] = 50.0

    cleaned = MarketDataAdapter._clean_downloaded_data(
        data
    )

    assert len(cleaned) == 100


def test_clean_rejects_more_than_one_percent_invalid_rows():
    data = make_valid_data()

    # Create exactly 101 rows.
    data = pd.concat(
        [data] * 34,
        ignore_index=True,
    ).iloc[:101].copy()

    data.index = pd.date_range(
        "2025-01-01",
        periods=len(data),
        freq="D",
    )

    # 2 / 101 > 1%
    data.loc[
        data.index[0],
        "High",
    ] = 50.0

    data.loc[
        data.index[1],
        "High",
    ] = 50.0

    with pytest.raises(
        ValueError,
        match="More than 1%",
    ):
        MarketDataAdapter._clean_downloaded_data(
            data
        )


def test_clean_does_not_modify_valid_prices():
    data = make_valid_data()

    cleaned = MarketDataAdapter._clean_downloaded_data(
        data
    )

    pd.testing.assert_frame_equal(
        cleaned,
        data,
        check_freq=False,
    )