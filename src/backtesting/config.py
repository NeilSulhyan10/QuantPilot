from dataclasses import dataclass

from src.config import QuantPilotConfig, load_config


@dataclass(frozen=True)
class BacktestConfig:
    """
    Backtest configuration used by the portfolio backtesting engine.

    This class remains as a compatibility layer for the existing
    backtesting API. New code should prefer QuantPilotConfig.
    """

    estimation_window: int = 60
    risk_aversion: float = 10.0
    max_weight: float = 0.10
    transaction_cost_rate: float = 0.001
    slippage_rate: float = 0.0005

    @classmethod
    def from_quantpilot_config(
        cls,
        config: QuantPilotConfig | None = None,
    ) -> "BacktestConfig":
        """
        Build a BacktestConfig from the master QuantPilot configuration.
        """

        if config is None:
            config = load_config()

        return cls(
            estimation_window=config.expected_returns.window,
            risk_aversion=config.optimizer.risk_aversion,
            max_weight=config.optimizer.max_weight,
            transaction_cost_rate=(
                config.costs.transaction_cost_bps / 10_000
            ),
            slippage_rate=(
                config.costs.slippage_bps / 10_000
            ),
        )
