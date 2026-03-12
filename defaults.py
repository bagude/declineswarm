"""Shared baseline parameters for well atoms."""

BASELINE_PARAMS = {
    "version": 1,
    "description": "Baseline -- global defaults",
    "preprocessing": {
        "significance_threshold": 0.50,
        "smoothing_window": 3,
        "outlier_window": 3,
        "outlier_threshold": 0.30,
        "peak_merge_distance": 3,
        "min_clean_months": 3,
    },
    "fitting": {
        "d_min": 0.05,
        "di_initial": 0.50,
        "b_initial": 0.50,
        "qi_guess_strategy": "first",
        "qi_multiplier_upper": 5.0,
        "di_upper_bound": 0.99,
        "b_upper_bound": 2.0,
    },
}
