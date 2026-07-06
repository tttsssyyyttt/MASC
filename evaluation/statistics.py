"""Statistical tests for comparing algorithms."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def paired_t_test(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Paired t-test between two groups.

    Returns:
        t_stat, p_value
    """
    from scipy import stats as sp_stats
    t_stat, p_value = sp_stats.ttest_rel(a, b)
    return t_stat, p_value


def wilcoxon_test(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Wilcoxon signed-rank test (non-parametric).

    Returns:
        stat, p_value
    """
    from scipy import stats as sp_stats
    stat, p_value = sp_stats.wilcoxon(a, b)
    return stat, p_value


def confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval for the mean.

    Returns:
        (lower, upper)
    """
    n = len(data)
    if n < 2:
        return (np.mean(data), np.mean(data))

    mean = np.mean(data)
    se = np.std(data, ddof=1) / np.sqrt(n)

    from scipy import stats as sp_stats
    h = se * sp_stats.t.ppf((1 + confidence) / 2, n - 1)
    return (mean - h, mean + h)


def effect_size_cohen_d(a: List[float], b: List[float]) -> float:
    """Compute Cohen's d effect size.

    Returns:
        d: effect size
    """
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(
        ((na - 1) * np.std(a, ddof=1) ** 2 + (nb - 1) * np.std(b, ddof=1) ** 2)
        / (na + nb - 2)
    )
    if pooled_std < 1e-10:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std
