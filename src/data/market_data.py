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

            data = self._clean_downloaded_data(data, ticker=ticker,)

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
            kwargs["period"] = "10y"

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
        ticker: str = "unknown",
    ) -> pd.DataFrame:
        """
        Clean Yahoo Finance data conservatively.

        Provider rows containing missing values are discarded first.
        Structurally invalid OHLC rows are then identified and removed.
        Price values are never modified.

        The remaining data is passed through QuantPilot's strict
        validation layer.
        """

        if data is None or data.empty:
            raise ValueError(
                f"No valid data available for ticker: {ticker}"
            )

        data = normalize_columns(data.copy())

        required = MarketDataAdapter.REQUIRED_COLUMNS

        missing = [
            column
            for column in required
            if column not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Downloaded data for {ticker} is missing "
                f"required columns: {missing}"
            )

        data = data[required].copy()

        # Normalize index
        data.index = pd.to_datetime(data.index)

        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        data = data.sort_index()

        if data.empty:
            raise ValueError(
                f"No data available for ticker: {ticker}"
            )

        # ---------------------------------------------------------
        # 1. Remove rows containing missing provider values
        # ---------------------------------------------------------

        complete_mask = data.notna().all(axis=1)

        data = data.loc[complete_mask].copy()

        if data.empty:
            raise ValueError(
                f"No complete OHLCV rows available for ticker: {ticker}"
            )

        # ---------------------------------------------------------
        # 2. Identify malformed OHLC rows
        # ---------------------------------------------------------

        valid_mask = (
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

        malformed_count = int((~valid_mask).sum())
        total_complete_rows = len(data)

        # ---------------------------------------------------------
        # 3. Reject genuinely corrupted downloads
        # ---------------------------------------------------------

        malformed_fraction = (
            malformed_count / total_complete_rows
            if total_complete_rows > 0
            else 1.0
        )

        if malformed_fraction > MAX_INVALID_ROW_FRACTION:
            raise ValueError(
                f"More than 1% of downloaded data rows are malformed "
                f"for {ticker}: "
                f"{malformed_count:,}/{total_complete_rows:,} "
                f"({malformed_fraction:.2%})."
            )

        # ---------------------------------------------------------
        # 4. Remove the small number of malformed rows
        # ---------------------------------------------------------

        if malformed_count:
            data = data.loc[valid_mask].copy()

        if data.empty:
            raise ValueError(
                f"All downloaded rows for ticker {ticker} "
                f"failed OHLC validation."
            )

        # ---------------------------------------------------------
        # 5. Final strict QuantPilot validation
        # ---------------------------------------------------------

        data = validate_data(data)

        return data[required].sort_index()