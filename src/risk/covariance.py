import numpy as np


def is_positive_semidefinite(
    covariance: np.ndarray,
    tolerance: float = 1e-10,
) -> bool:
    """
    Check whether a covariance matrix is positive semidefinite.
    """

    if covariance.ndim != 2:
        raise ValueError("Covariance matrix must be 2-dimensional.")

    rows, columns = covariance.shape

    if rows != columns:
        raise ValueError("Covariance matrix must be square.")

    if not np.isfinite(covariance).all():
        raise ValueError("Covariance matrix contains invalid values.")

    if not np.allclose(covariance, covariance.T, atol=tolerance):
        raise ValueError("Covariance matrix is not symmetric.")

    eigenvalues = np.linalg.eigvalsh(covariance)

    return eigenvalues.min() >= -tolerance