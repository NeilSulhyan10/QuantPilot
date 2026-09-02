def calculate_slippage(
    turnover: float,
    slippage_rate: float = 0.0005,
) -> float:
    """
    Calculate portfolio-level slippage.

    Parameters
    ----------
    turnover : float
        One-way portfolio turnover as a decimal.
    slippage_rate : float
        Slippage as a fraction of traded value.
        0.0005 = 5 basis points = 0.05%.
    """

    if turnover < 0:
        raise ValueError("turnover cannot be negative.")

    if slippage_rate < 0:
        raise ValueError("slippage_rate cannot be negative.")

    return turnover * slippage_rate