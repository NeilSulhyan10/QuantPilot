import pytest

from src.goals.risk_profile import (
    RiskTolerance,
    get_risk_profile,
)


def test_conservative_profile():
    profile = get_risk_profile(
        RiskTolerance.CONSERVATIVE
    )

    assert profile.tolerance == RiskTolerance.CONSERVATIVE
    assert profile.max_annual_volatility == pytest.approx(0.12)


def test_moderate_profile():
    profile = get_risk_profile("moderate")

    assert profile.tolerance == RiskTolerance.MODERATE
    assert profile.max_annual_volatility == pytest.approx(0.18)


def test_aggressive_profile():
    profile = get_risk_profile("AGGRESSIVE")

    assert profile.tolerance == RiskTolerance.AGGRESSIVE
    assert profile.max_annual_volatility == pytest.approx(0.25)


def test_risk_limits_increase_with_tolerance():
    conservative = get_risk_profile("conservative")
    moderate = get_risk_profile("moderate")
    aggressive = get_risk_profile("aggressive")

    assert (
        conservative.max_annual_volatility
        < moderate.max_annual_volatility
        < aggressive.max_annual_volatility
    )


def test_invalid_risk_tolerance():
    with pytest.raises(ValueError, match="Unknown risk tolerance"):
        get_risk_profile("extreme")


def test_invalid_risk_tolerance_type():
    with pytest.raises(ValueError):
        get_risk_profile(123)