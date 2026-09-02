from pathlib import Path

import pandas as pd
import pytest

from src.data.benchmark_loader import download_benchmark


def test_download_benchmark_rejects_empty_download(monkeypatch):
    def mock_download(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(
        "src.data.benchmark_loader.yf.download",
        mock_download,
    )

    with pytest.raises(ValueError):
        download_benchmark(
            ticker="SPY",
            start_date="2015-01-01",
        )