import numpy as np
import pandas as pd
import pytest

from src.evaluation.alignment import align_return_series


def test_align_return_series_uses_common_dates():
    first = pd.Series(
        [0.01, 0.02, 0.03],
        index=pd.to_datetime(
            ["2020-01-01", "2020-01-02", "2020-01-03"]
        ),
    )

    second = pd.Series(
        [0.04, 0.05, 0.06],
        index=pd.to_datetime(
            ["2020-01-02", "2020-01-03", "2020-01-04"]
        ),
    )

    aligned_first, aligned_second = align_return_series(
        first,
        second,
    )

    assert list(aligned_first.index) == list(
        pd.to_datetime(["2020-01-02", "2020-01-03"])
    )

    assert list(aligned_second.index) == list(
        pd.to_datetime(["2020-01-02", "2020-01-03"])
    )


def test_align_return_series_preserves_values():
    first = pd.Series(
        [0.01, 0.02],
        index=pd.to_datetime(
            ["2020-01-01", "2020-01-02"]
        ),
    )

    second = pd.Series(
        [0.03, 0.04],
        index=pd.to_datetime(
            ["2020-01-02", "2020-01-03"]
        ),
    )

    aligned_first, aligned_second = align_return_series(
        first,
        second,
    )

    assert np.isclose(aligned_first.iloc[0], 0.02)
    assert np.isclose(aligned_second.iloc[0], 0.03)


def test_align_return_series_sorts_dates():
    first = pd.Series(
        [0.02, 0.01],
        index=pd.to_datetime(
            ["2020-01-02", "2020-01-01"]
        ),
    )

    second = pd.Series(
        [0.04, 0.03],
        index=pd.to_datetime(
            ["2020-01-02", "2020-01-01"]
        ),
    )

    aligned_first, aligned_second = align_return_series(
        first,
        second,
    )

    assert aligned_first.index.is_monotonic_increasing
    assert aligned_second.index.is_monotonic_increasing


def test_align_return_series_rejects_empty_series():
    first = pd.Series(dtype=float)

    second = pd.Series(
        [0.01],
        index=pd.to_datetime(["2020-01-01"]),
    )

    with pytest.raises(ValueError):
        align_return_series(first, second)


def test_align_return_series_rejects_no_common_dates():
    first = pd.Series(
        [0.01],
        index=pd.to_datetime(["2020-01-01"]),
    )

    second = pd.Series(
        [0.02],
        index=pd.to_datetime(["2020-01-02"]),
    )

    with pytest.raises(ValueError):
        align_return_series(first, second)