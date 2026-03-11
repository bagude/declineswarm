"""Standalone evaluation — fit decline and score a single well atom.

No database dependency. Reads production.csv, applies params.json settings.
"""

from __future__ import annotations

import csv
from pathlib import Path

from arps import create_model, fit_arps, generate_forecast
from preprocessing import detect_decline_start, filter_outliers


def read_production(csv_path: str | Path) -> tuple[list[str], list[float], list[float], list[float]]:
    """Read production.csv → (dates, oil, gas, water)."""
    dates, oil, gas, water = [], [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dates.append(row["date"])
            oil.append(float(row["oil_bbl"]))
            gas.append(float(row["gas_mcf"]))
            water.append(float(row["water_bbl"]))
    return dates, oil, gas, water


def preprocess_and_fit(oil: list[float], params: dict) -> tuple:
    """Run the full preprocessing + fitting pipeline on oil production data.

    Returns (anchor, filtered, fit_result) or raises ValueError on failure.
    """
    pre = params.get("preprocessing", {})
    fit_params = params.get("fitting", {})

    anchor = detect_decline_start(
        oil,
        significance_threshold=pre.get("significance_threshold", 0.50),
        smoothing_window=pre.get("smoothing_window", 3),
        peak_merge_distance=pre.get("peak_merge_distance", 3),
    )

    filtered = filter_outliers(
        anchor.production_trimmed,
        anchor.time_trimmed,
        window=pre.get("outlier_window", 3),
        threshold=pre.get("outlier_threshold", 0.30),
        min_clean_months=pre.get("min_clean_months", 3),
    )

    fit_result = fit_arps(
        filtered.production_clean,
        filtered.time_clean,
        model_type="hyperbolic",
        d_min=fit_params.get("d_min", 0.05),
        di_initial=fit_params.get("di_initial", 0.50),
        b_initial=fit_params.get("b_initial", 0.50),
        qi_guess_strategy=fit_params.get("qi_guess_strategy", "first"),
        qi_multiplier_upper=fit_params.get("qi_multiplier_upper", 5.0),
        di_upper_bound=fit_params.get("di_upper_bound", 0.99),
        b_upper_bound=fit_params.get("b_upper_bound", 2.0),
    )

    return anchor, filtered, fit_result


def fit_and_score(production_csv: str | Path, params: dict) -> dict:
    """Fit decline to a well's production and return scores.

    Parameters
    ----------
    production_csv : path to the well's production.csv
    params : dict matching params.json structure (preprocessing + fitting sections)

    Returns
    -------
    dict with fit results, quality metrics, and metadata.
    """
    dates, oil, gas, water = read_production(production_csv)

    if len(oil) < 3:
        return {"error": "insufficient_history", "months": len(oil)}

    anchor, filtered, fit_result = preprocess_and_fit(oil, params)

    fit_months = len(filtered.production_clean)
    total_months = len(oil)
    outlier_rejection_rate = filtered.outlier_count / max(len(anchor.production_trimmed), 1)

    # Compute gas qi from peak
    peak_idx = anchor.peak_index
    qi_gas = gas[peak_idx] if peak_idx < len(gas) else 0.0

    return {
        "model": fit_result["decline_model"],
        "qi": fit_result["qi"],
        "di": fit_result["di"],
        "b": fit_result["b"],
        "d_min": fit_result["d_min"],
        "qi_gas": qi_gas,
        "r2_insample": fit_result["r_squared"],
        "convergence": fit_result["convergence"],
        "fit_months": fit_months,
        "total_months": total_months,
        "peak_index": peak_idx,
        "peak_value": anchor.peak_value,
        "outlier_rejection_rate": round(outlier_rejection_rate, 4),
    }
