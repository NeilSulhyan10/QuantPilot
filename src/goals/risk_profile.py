"""Investor risk profiles for QuantPilot V2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RiskTolerance(str, Enum):
    """Supported investor risk-tolerance levels."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass(frozen=True)
class RiskProfile:
    """Quantitative representation of an investor risk profile."""

    tolerance: RiskTolerance
    max_annual_volatility: float


RISK_PROFILES: dict[RiskTolerance, RiskProfile] = {
    RiskTolerance.CONSERVATIVE: RiskProfile(
        tolerance=RiskTolerance.CONSERVATIVE,
        max_annual_volatility=0.12,
    ),
    RiskTolerance.MODERATE: RiskProfile(
        tolerance=RiskTolerance.MODERATE,
        max_annual_volatility=0.18,
    ),
    RiskTolerance.AGGRESSIVE: RiskProfile(
        tolerance=RiskTolerance.AGGRESSIVE,
        max_annual_volatility=0.25,
    ),
}


def get_risk_profile(
    tolerance: RiskTolerance | str,
) -> RiskProfile:
    """Return the quantitative profile for a risk-tolerance level."""

    if isinstance(tolerance, str):
        try:
            tolerance = RiskTolerance(tolerance.lower().strip())
        except ValueError as exc:
            valid = ", ".join(
                profile.value
                for profile in RiskTolerance
            )
            raise ValueError(
                f"Unknown risk tolerance '{tolerance}'. "
                f"Valid values: {valid}."
            ) from exc

    if not isinstance(tolerance, RiskTolerance):
        raise ValueError(
            "tolerance must be a RiskTolerance value or string."
        )

    return RISK_PROFILES[tolerance]