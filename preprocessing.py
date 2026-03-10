"""Production history preprocessing for decline curve analysis.

Two functions upstream of fit_arps():
1. detect_decline_start() — finds the most recent significant peak
2. filter_outliers() — removes anomalous months via rolling median

Pure functions, no database dependency. All thresholds are parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeclineAnchor:
    peak_index: int
    peak_value: float
    all_peaks: list[int]
    production_trimmed: list[float]
    time_trimmed: list[int]


@dataclass(frozen=True)
class FilterResult:
    production_clean: list[float]
    time_clean: list[int]
    outlier_indices: list[int]
    outlier_count: int


def detect_decline_start(
    production: list[float],
    significance_threshold: float = 0.50,
    smoothing_window: int = 3,
    peak_merge_distance: int = 3,
) -> DeclineAnchor:
    """Find the most recent significant production peak.

    peak_merge_distance controls how close peaks can be before merging
    (independent of smoothing_window).
    """
    prod = np.array(production, dtype=float)
    prod = np.where(np.isnan(prod), 0.0, prod)

    if np.count_nonzero(prod) < 3:
        raise ValueError("Need at least 3 non-zero production months.")

    # Smooth
    if smoothing_window > 1 and len(prod) >= smoothing_window:
        half = smoothing_window // 2
        padded = np.pad(prod, half, mode="edge")
        kernel = np.ones(smoothing_window) / smoothing_window
        smoothed = np.convolve(padded, kernel, mode="valid")[:len(prod)]
    else:
        smoothed = prod.copy()

    # Find local maxima
    peaks = []
    n = len(smoothed)
    if n >= 2 and smoothed[0] >= smoothed[1] and prod[0] > 0:
        peaks.append(0)
    for i in range(1, n - 1):
        if smoothed[i] >= smoothed[i - 1] and smoothed[i] >= smoothed[i + 1] and prod[i] > 0:
            peaks.append(i)
    if n >= 2 and smoothed[-1] >= smoothed[-2] and prod[-1] > 0:
        peaks.append(n - 1)

    if not peaks:
        peaks = [int(np.argmax(prod))]

    # Merge nearby peaks using peak_merge_distance
    merged_peaks: list[int] = []
    group: list[int] = [peaks[0]]
    for i in range(1, len(peaks)):
        if peaks[i] - group[-1] <= peak_merge_distance:
            group.append(peaks[i])
        else:
            merged_peaks.append(max(group, key=lambda idx: prod[idx]))
            group = [peaks[i]]
    merged_peaks.append(max(group, key=lambda idx: prod[idx]))

    # Filter to significant peaks
    global_max = float(np.max(prod))
    threshold = global_max * significance_threshold
    significant_peaks = [p for p in merged_peaks if prod[p] >= threshold]
    if not significant_peaks:
        significant_peaks = [int(np.argmax(prod))]

    # Most recent significant peak
    peak_idx = significant_peaks[-1]
    trimmed = prod[peak_idx:].tolist()
    time_trimmed = list(range(1, len(trimmed) + 1))

    return DeclineAnchor(
        peak_index=peak_idx,
        peak_value=float(prod[peak_idx]),
        all_peaks=peaks,
        production_trimmed=trimmed,
        time_trimmed=time_trimmed,
    )


def filter_outliers(
    production: list[float],
    time_indices: list[int] | None = None,
    window: int = 3,
    threshold: float = 0.30,
    min_clean_months: int = 3,
) -> FilterResult:
    """Remove anomalous production months using rolling median.

    Raises ValueError if fewer than min_clean_months remain after filtering.
    """
    prod = np.array(production, dtype=float)
    t_idx = list(time_indices) if time_indices is not None else list(range(1, len(prod) + 1))

    if len(prod) < window:
        result = FilterResult(
            production_clean=prod.tolist(),
            time_clean=t_idx,
            outlier_indices=[],
            outlier_count=0,
        )
        if len(result.production_clean) < min_clean_months:
            raise ValueError(
                f"Only {len(result.production_clean)} clean months, need {min_clean_months}."
            )
        return result

    # Rolling median
    half_w = window // 2
    rolling_median = np.full(len(prod), np.nan)
    for i in range(len(prod)):
        start = max(0, i - half_w)
        end = min(len(prod), i + half_w + 1)
        nonzero = prod[start:end][prod[start:end] > 0]
        if len(nonzero) > 0:
            rolling_median[i] = np.median(nonzero)

    # Flag outliers
    outlier_mask = np.zeros(len(prod), dtype=bool)
    for i in range(len(prod)):
        if np.isnan(rolling_median[i]) or rolling_median[i] <= 0:
            continue
        if np.isnan(prod[i]) or prod[i] < threshold * rolling_median[i]:
            outlier_mask[i] = True

    outlier_indices = list(np.where(outlier_mask)[0])
    clean_mask = ~outlier_mask
    production_clean = prod[clean_mask].tolist()
    time_clean = [t_idx[i] for i in range(len(t_idx)) if clean_mask[i]]

    if len(production_clean) < min_clean_months:
        raise ValueError(
            f"Only {len(production_clean)} clean months after filtering, need {min_clean_months}."
        )

    return FilterResult(
        production_clean=production_clean,
        time_clean=time_clean,
        outlier_indices=outlier_indices,
        outlier_count=int(np.sum(outlier_mask)),
    )
