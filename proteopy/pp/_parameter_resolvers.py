"""Shared resolvers for fractional detection thresholds."""

import math


def _resolve_min_fraction(min_fraction: float, n_items: int) -> int:
    """Return ``floor(min_fraction * n_items)`` as a detection count.

    Fractions must be between zero and one, inclusive, and the number
    of eligible items must be a nonnegative integer. A zero count
    permits membership even without a detection.
    """
    if isinstance(min_fraction, bool) or not isinstance(
        min_fraction, (int, float)
    ):
        raise TypeError("`min_fraction` must be a non-boolean number.")
    if not 0 <= min_fraction <= 1:
        raise ValueError(
            "`min_fraction` must be a finite number between 0 and 1."
        )
    if isinstance(n_items, bool) or not isinstance(n_items, int):
        raise TypeError("`n_items` must be a non-boolean int.")
    if n_items < 0:
        raise ValueError("`n_items` must be nonnegative.")

    return math.floor(min_fraction * n_items)
