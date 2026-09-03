import pandas as pd
import pytest

from src.config import load_config

from src.goals.assets import (
    AssetSelection,
    build_selected_return_matrix,
    validate_asset_selection,
    validate_minimum_history,
    load_selected_assets,
)

AVAILABLE = ["AAPL", "MSFT", "NVDA", "AMZN"]


def test_asset_selection_normalizes_tickers():
    selection = AssetSelection(
        ("aapl", "MSFT")
    )

    assert selection.tickers == ("AAPL", "MSFT")


def test_asset_selection_rejects_empty_selection():
    with pytest.raises(ValueError):
        AssetSelection(())


def test_asset_selection_rejects_duplicates():
    with pytest.raises(ValueError):
        AssetSelection(("AAPL", "aapl"))


def test_validate_asset_selection():
    selected = validate_asset_selection(
        ["aapl", "MSFT"],
        AVAILABLE,
    )

    assert selected == ("AAPL", "MSFT")


def test_validate_asset_selection_rejects_unknown_ticker():
    with pytest.raises(ValueError, match="Unknown ticker"):
        validate_asset_selection(
            ["AAPL", "TSLA"],
            AVAILABLE,
        )


def test_validate_asset_selection_rejects_duplicates():
    with pytest.raises(ValueError, match="Duplicate"):
        validate_asset_selection(
            ["AAPL", "AAPL"],
            AVAILABLE,
        )


def test_build_selected_return_matrix():
    dates = pd.date_range(
        "2025-01-01",
        periods=5,
        freq="D",
    )

    asset_data = {
        "AAPL": pd.DataFrame(
            {"Close": [100, 101, 102, 103, 104]},
            index=dates,
        ),
        "MSFT": pd.DataFrame(
            {"Close": [200, 202, 204, 206, 208]},
            index=dates,
        ),
    }

    returns = build_selected_return_matrix(asset_data)

    assert list(returns.columns) == ["AAPL", "MSFT"]
    assert len(returns) == 4
    assert returns.index.is_monotonic_increasing
    assert not returns.isna().any().any()

def test_validate_minimum_history_passes():
    returns = pd.DataFrame(
        {
            "AAPL": [0.01] * 252,
            "MSFT": [0.02] * 252,
        }
    )

    validate_minimum_history(
        returns,
        minimum_observations=252,
    )


def test_validate_minimum_history_rejects_short_history():
    returns = pd.DataFrame(
        {
            "AAPL": [0.01] * 100,
            "MSFT": [0.02] * 100,
        }
    )

    with pytest.raises(ValueError, match="Insufficient"):
        validate_minimum_history(
            returns,
            minimum_observations=252,
        )


def test_validate_minimum_history_rejects_empty_data():
    returns = pd.DataFrame(
        columns=["AAPL", "MSFT"]
    )

    with pytest.raises(ValueError, match="No historical"):
        validate_minimum_history(returns)


def test_validate_minimum_history_rejects_invalid_requirement():
    returns = pd.DataFrame(
        {
            "AAPL": [0.01] * 10,
        }
    )

    with pytest.raises(ValueError):
        validate_minimum_history(
            returns,
            minimum_observations=0,
        )

def test_load_real_selected_assets():
    config = load_config()

    selected = ["AAPL", "MSFT", "NVDA"]

    assets = load_selected_assets(
        selected,
        config.universe.tickers,
    )

    assert set(assets) == set(selected)

    for ticker, data in assets.items():
        assert not data.empty
        assert "Close" in data.columns
        assert data.index.is_monotonic_increasing

def test_real_selected_assets_have_sufficient_history():
    config = load_config()

    selected = ["AAPL", "MSFT", "NVDA"]

    assets = load_selected_assets(
        selected,
        config.universe.tickers,
    )

    returns = build_selected_return_matrix(assets)

    validate_minimum_history(
        returns,
        minimum_observations=252,
    )