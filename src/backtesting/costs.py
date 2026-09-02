def calculate_transaction_cost(
    turnover: float,
    cost_rate: float = 0.001,
) -> float:
    """
    Calculate portfolio-level transaction cost.

    Parameters
    ----------
    turnover : float
        One-way portfolio turnover as a decimal.
    cost_rate : float
        Transaction cost as a fraction of traded value.
        0.001 = 10 basis points = 0.10%.
    """

    if turnover < 0:
        raise ValueError("turnover cannot be negative.")

    if cost_rate < 0:
        raise ValueError("cost_rate cannot be negative.")

    return turnover * cost_rate