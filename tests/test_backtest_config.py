from src.backtesting.config import BacktestConfig


def test_default_config():
    config = BacktestConfig()

    assert config.estimation_window == 60
    assert config.risk_aversion == 10.0
    assert config.max_weight == 0.10
    assert config.transaction_cost_rate == 0.001
    assert config.slippage_rate == 0.0005


def test_custom_config():
    config = BacktestConfig(
        estimation_window=120,
        risk_aversion=5.0,
        max_weight=0.05,
        transaction_cost_rate=0.002,
        slippage_rate=0.001,
    )

    assert config.estimation_window == 120
    assert config.risk_aversion == 5.0
    assert config.max_weight == 0.05
    assert config.transaction_cost_rate == 0.002
    assert config.slippage_rate == 0.001


def test_backtest_config_from_quantpilot_config():
    from src.backtesting.config import BacktestConfig
    from src.config import load_config

    config = load_config()
    backtest_config = BacktestConfig.from_quantpilot_config(config)

    assert backtest_config.estimation_window == 60
    assert backtest_config.risk_aversion == 10.0
    assert backtest_config.max_weight == 0.10
    assert backtest_config.transaction_cost_rate == 0.001
    assert backtest_config.slippage_rate == 0.0005
