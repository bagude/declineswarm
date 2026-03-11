"""Hindcast evaluator for one well atom.

Usage:
    python run_eval.py <state> <api_number>

Reads wells/{state}/{api}/params.json and production.csv.
Fits decline on months 1..N-12, forecasts months N-11..N,
computes MAPE between forecast and actual.
Outputs JSON to stdout.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from arps import create_model, generate_forecast
from evaluation import fit_and_score, preprocess_and_fit, read_production


def hindcast(state: str, api: str) -> dict:
    """Run hindcast evaluation for one well."""
    well_dir = Path("wells") / state / api
    production_csv = well_dir / "production.csv"
    params_file = well_dir / "params.json"

    if not production_csv.exists():
        return {"api": api, "error": f"production.csv not found in {well_dir}"}
    if not params_file.exists():
        return {"api": api, "error": f"params.json not found in {well_dir}"}

    params = json.loads(params_file.read_text())

    # Read full production
    dates, oil, gas, water = read_production(production_csv)
    total_months = len(oil)

    # Minimum 18 months for hindcast
    if total_months < 18:
        return {"api": api, "error": "insufficient_history", "months": total_months}

    holdout = 12
    train_months = total_months - holdout

    oil_train = oil[:train_months]
    oil_holdout = oil[train_months:]

    # Preprocess and fit on training data
    try:
        anchor, filtered, fit_result = preprocess_and_fit(oil_train, params)
    except ValueError as e:
        return {"api": api, "error": f"preprocess_fit: {e}"}

    # Build forecast model and predict holdout period
    model = create_model(
        decline_model=fit_result["decline_model"],
        qi_oil=fit_result["qi"],
        di=fit_result["di"],
        b=fit_result["b"],
        d_min=fit_result["d_min"],
    )

    # Forecast enough months to cover training + holdout from decline start
    months_from_peak = train_months - anchor.peak_index + holdout
    forecast = generate_forecast(model, months=max(months_from_peak + 12, 120))

    # The holdout starts at month (train_months - peak_index + 1) in forecast time
    holdout_start_idx = train_months - anchor.peak_index  # 0-based in forecast
    forecast_holdout = []
    for i in range(holdout):
        fc_idx = holdout_start_idx + i
        if fc_idx < len(forecast):
            forecast_holdout.append(forecast[fc_idx].oil_bbl)
        else:
            forecast_holdout.append(0.0)

    # Compute MAPE on holdout (skip zero actuals)
    actual = np.array(oil_holdout[:holdout], dtype=float)
    predicted = np.array(forecast_holdout[:len(actual)], dtype=float)

    nonzero_mask = actual > 0
    if np.sum(nonzero_mask) == 0:
        return {"api": api, "error": "holdout_all_zeros"}

    actual_nz = actual[nonzero_mask]
    predicted_nz = predicted[nonzero_mask]
    mape = float(np.mean(np.abs((actual_nz - predicted_nz) / actual_nz)))

    # R² on holdout
    ss_res = np.sum((actual_nz - predicted_nz) ** 2)
    ss_tot = np.sum((actual_nz - np.mean(actual_nz)) ** 2)
    if ss_tot > 0:
        r2_outsample = float(1.0 - ss_res / ss_tot)
    else:
        r2_outsample = 1.0 if ss_res == 0 else 0.0

    fit_months = len(filtered.production_clean)
    outlier_rejection_rate = filtered.outlier_count / max(len(anchor.production_trimmed), 1)

    # Full fit on all data for fit.json (in-sample R²)
    full_fit = fit_and_score(production_csv, params)
    r2_insample = full_fit.get("r2_insample", fit_result["r_squared"])

    return {
        "api": api,
        "param_version": params.get("version", 1),
        "hindcast_mape": round(mape, 4),
        "r2_outsample": round(r2_outsample, 4),
        "r2_insample": round(r2_insample, 4),
        "fit_months": fit_months,
        "holdout_months": holdout,
        "convergence": fit_result["convergence"],
        "outlier_rejection_rate": round(outlier_rejection_rate, 4),
        "_full_fit": full_fit,
    }


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: python run_eval.py <state> <api_number>"}))
        sys.exit(1)

    state = sys.argv[1]
    api = sys.argv[2]
    result = hindcast(state, api)

    # Write fit.json and update hindcast.json
    well_dir = Path("wells") / state / api
    if "error" not in result:
        params = json.loads((well_dir / "params.json").read_text())
        full_fit = result.pop("_full_fit")

        fit_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "param_version": params.get("version", 1),
            "model": full_fit["model"],
            "qi": round(full_fit["qi"], 2),
            "di": round(full_fit["di"], 4),
            "b": round(full_fit["b"], 4),
            "d_min": full_fit["d_min"],
            "r2_insample": round(full_fit["r2_insample"], 4),
            "hindcast_mape": result["hindcast_mape"],
            "convergence": full_fit["convergence"],
        }
        (well_dir / "fit.json").write_text(json.dumps(fit_data, indent=2))

        # Append to hindcast.json
        hindcast_file = well_dir / "hindcast.json"
        history = json.loads(hindcast_file.read_text()) if hindcast_file.exists() else []
        history.append({
            "version": params.get("version", 1),
            "mape": result["hindcast_mape"],
            "r2_outsample": result["r2_outsample"],
        })
        hindcast_file.write_text(json.dumps(history, indent=2))
    else:
        result.pop("_full_fit", None)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
