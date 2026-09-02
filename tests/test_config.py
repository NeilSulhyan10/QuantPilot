from pathlib import Path

import pytest

from src.config import load_config


def test_load_default_config():
    config = load_config()

    assert config.project_name == "QuantPilot"
    assert config.project_version == "1.0"

    assert len(config.universe.tickers) == 20
    assert len(set(config.universe.tickers)) == 20

    assert config.expected_returns.model == "v2"
    assert config.expected_returns.window == 60
    assert config.expected_returns.shrinkage_alpha == 0.5

    assert config.covariance.window == 60

    assert config.optimizer.risk_aversion == 10.0
    assert config.optimizer.max_weight == 0.10
    assert config.optimizer.long_only is True
    assert config.optimizer.fully_invested is True

    assert config.rebalancing.frequency == "monthly"

    assert config.costs.transaction_cost_bps == 10.0
    assert config.costs.slippage_bps == 5.0

    assert config.ml.enabled is True
    assert config.ml.model == "ridge"
    assert config.ml.min_train_size == 252
    assert config.ml.horizon == 21

    assert config.benchmarks.market == "SPY"


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config(Path("does_not_exist.yaml"))
