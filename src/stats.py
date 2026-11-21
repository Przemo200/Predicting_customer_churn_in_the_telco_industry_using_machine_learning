"""
Statistical helpers used in phases 2 and 4
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def holm_adjust_pvalues(p_values) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)

    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0

    for rank, original_idx in enumerate(order):
        adjusted = (m - rank) * p_values[original_idx]
        running_max = max(running_max, adjusted)
        adjusted_sorted[rank] = min(running_max, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted

    return adjusted


def corrected_resampled_ttest(differences, n_train: int, n_test: int) -> tuple[float, float]:
    d = np.asarray(differences, dtype=float)
    k = len(d)
    variance = d.var(ddof=1)

    if variance == 0.0:
        if d.mean() == 0.0:
            return 0.0, 1.0
        return float(np.copysign(np.inf, d.mean())), 0.0

    t_stat = d.mean() / np.sqrt((1.0 / k + n_test / n_train) * variance)
    p_value = 2.0 * stats.t.sf(abs(t_stat), df=k - 1)

    return float(t_stat), float(p_value)
