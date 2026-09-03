from pathlib import Path

import pandas as pd
import yfinance as yf

from .markets import Market, get_market_config, normalize_ticker
from .validation import normalize_columns, validate_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

MAX_INVALID_ROW_FRACTION = 0.01


class MarketDataAdapter:
    """
    Unified market-data interface for QuantPilot.

    Processed local Parquet files are preferred. Missing data can be
    downloaded from Yahoo Finance and validated before being returned.
    """

    REQUIRED_COLUMNS = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    def __init__(
        self,
        market: Market | str,
        data_dir: Path | None = None,
    ):
        self.market = Market(market)
        self.config = get_market_config(self.market)

        self.data_dir = (
            Path(data_dir)
            if data_dir is not None
            else PROCESSED_DATA_DIR
        )

    def load_asset(
        self,
        ticker: str,
        allow_download: bool = True,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Load one asset from local processed data or Yahoo Finance.

        Existing processed data is trusted because it has already passed
        QuantPilot's processing/validation pipeline.

        Newly downloaded data goes through normalization, conservative
        cleaning, and strict validation.
        """

        ticker = normalize_ticker(
            ticker,
            self.market,
        )

        cached = self._load_cached(ticker)

        if cached is not None:
            data = cached

            # Preserve existing V2 behavior: processed data is already
            # validated and should not be modified by the download cleaner.
            data = normalize_columns(data)

            missing = [
                column
                for column in self.REQUIRED_COLUMNS
                if column not in data.columns
            ]

            if missing:
                raise ValueError(
                    f"Cached data for {ticker} is missing "
                    f"required columns: {missing}"
                )

            data = data[self.REQUIRED_COLUMNS].copy()

        elif allow_download:
            data = self._download(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )

            data = self._clean_downloaded_data(data)

        else:
            raise FileNotFoundError(
                f"No processed data found for ticker: {ticker}"
            )

        data.index = pd.to_datetime(data.index)

        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        data = data.sort_index()

        if start_date is not None:
            data = data.loc[
                data.index >= pd.Timestamp(start_date)
            ]

        if end_date is not None:
            data = data.loc[
                data.index <= pd.Timestamp(end_date)
            ]

        if data.empty:
            raise ValueError(
                f"No valid data available for ticker: {ticker}"
            )

        return validate_data(data)[self.REQUIRED_COLUMNS].sort_index()

    def load_assets(
        self,
        tickers: list[str],
        allow_download: bool = True,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Load multiple assets using the same market configuration.
        """

        normalized = []

        for ticker in tickers:
            value = normalize_ticker(
                ticker,
                self.market,
            )

            if value not in normalized:
                normalized.append(value)

        if not normalized:
            raise ValueError(
                "At least one ticker is required."
            )

        return {
            ticker: self.load_asset(
                ticker=ticker,
                allow_download=allow_download,
                start_date=start_date,
                end_date=end_date,
            )
            for ticker in normalized
        }

    def load_benchmark(
        self,
        allow_download: bool = True,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Load the benchmark configured for the selected market.
        """

        return self.load_asset(
            ticker=self.config.benchmark,
            allow_download=allow_download,
            start_date=start_date,
            end_date=end_date,
        )

    def _load_cached(
        self,
        ticker: str,
    ) -> pd.DataFrame | None:
        """Load processed Parquet data if available."""

        path = self.data_dir / f"{ticker}.parquet"

        if not path.exists():
            return None

        return pd.read_parquet(path)

    @staticmethod
    def _download(
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Download daily OHLCV data from Yahoo Finance.
        """

        kwargs = {
            "interval": "1d",
            "auto_adjust": True,
            "progress": False,
        }

        if start_date is not None:
            kwargs["start"] = start_date

        if end_date is not None:
            kwargs["end"] = end_date

        if start_date is None and end_date is None:
            kwargs["period"] = "max"

        data = yf.download(
            ticker,
            **kwargs,
        )

        if data is None or data.empty:
            raise ValueError(
                f"No data downloaded for ticker: {ticker}"
            )

        data = normalize_columns(data)

        missing = [
            column
            for column in MarketDataAdapter.REQUIRED_COLUMNS
            if column not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Downloaded data for {ticker} is missing "
                f"columns: {missing}"
            )

        return data[
            MarketDataAdapter.REQUIRED_COLUMNS
        ].copy()

    @staticmethod
    def _clean_downloaded_data(
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Conservatively clean newly downloaded provider data.

        Price values are never modified. Invalid rows are removed.
        """
        if data is None or data.empty:
            raise ValueError(
                "No valid data available."
            )

        data = normalize_columns(data)

        missing = [
            column
            for column in MarketDataAdapter.REQUIRED_COLUMNS
            if column not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Data is missing required columns: {missing}"
            )

        data = data[
            MarketDataAdapter.REQUIRED_COLUMNS
        ].copy()

        data.index = pd.to_datetime(data.index)

        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        data = data.sort_index()

        # Remove rows with missing OHLCV values.
        data = data[
            data["Open"].notna()
            & data["High"].notna()
            & data["Low"].notna()
            & data["Close"].notna()
            & data["Volume"].notna()
        ]

        if data.empty:
            raise ValueError(
                "No valid data available."
            )

        valid = (
            (data["Open"] > 0)
            & (data["High"] > 0)
            & (data["Low"] > 0)
            & (data["Close"] > 0)
            & (data["Volume"] >= 0)
            & (data["High"] >= data["Low"])
            & (data["Open"] >= data["Low"])
            & (data["Open"] <= data["High"])
            & (data["Close"] >= data["Low"])
            & (data["Close"] <= data["High"])
        )

        invalid_count = int((~valid).sum())
        original_rows = len(data)

        if invalid_count:
            invalid_fraction = (
                invalid_count / original_rows
            )

            if invalid_fraction > MAX_INVALID_ROW_FRACTION:
                raise ValueError(
                    "More than 1% of downloaded data rows "
                    "are malformed."
                )

            data = data.loc[valid].copy()

        if data.empty:
            raise ValueError(
                "No valid data available."
            )

        return data