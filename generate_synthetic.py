"""Generate synthetic production wells with known ground truth parameters.

Creates wells compatible with the existing declineswarm pipeline. Each well
has a known base regime, optional operational events, noise layer, and
ground truth metadata for validation.

Usage:
    python generate_synthetic.py                        # Easy tier (SYN-01..SYN-05)
    python generate_synthetic.py --wells SYN-01 SYN-07  # Specific wells
    python generate_synthetic.py --tier easy             # By difficulty
    python generate_synthetic.py --all                   # All 20 wells
    python generate_synthetic.py --seed 42               # Reproducible
    python generate_synthetic.py --validate              # Validate existing
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from petbox import dca

from defaults import BASELINE_PARAMS

# ---------------------------------------------------------------------------
# Section D: Portfolio -- all 20 well configurations
# ---------------------------------------------------------------------------

REGIMES = {
    "R1": {
        "name": "Permian tight oil",
        "b": (0.8, 1.2),
        "Di": (0.70, 0.90),
        "d_min": (0.07, 0.09),
        "qi_oil": (5000, 25000),
        "qi_gas": None,
        "gor": (1.0, 5.0),
        "wor": (0.5, 3.0),
        "is_gas": False,
    },
    "R2": {
        "name": "Eagle Ford moderate",
        "b": (0.6, 0.9),
        "Di": (0.55, 0.75),
        "d_min": (0.05, 0.07),
        "qi_oil": (5000, 25000),
        "qi_gas": None,
        "gor": (1.0, 5.0),
        "wor": (0.5, 3.0),
        "is_gas": False,
    },
    "R3": {
        "name": "Shale gas Marcellus",
        "b": (0.4, 0.7),
        "Di": (0.60, 0.85),
        "d_min": (0.04, 0.06),
        "qi_oil": (500, 3000),
        "qi_gas": (50000, 200000),
        "gor": (20.0, 100.0),
        "wor": (0.5, 3.0),
        "is_gas": True,
    },
    "R4": {
        "name": "Bakken mature",
        "b": (1.2, 1.6),
        "Di": (0.50, 0.70),
        "d_min": (0.08, 0.10),
        "qi_oil": (5000, 25000),
        "qi_gas": None,
        "gor": (1.0, 5.0),
        "wor": (1.0, 5.0),
        "is_gas": False,
    },
    "R5": {
        "name": "Conventional vertical",
        "b": (0.2, 0.5),
        "Di": (0.10, 0.25),
        "d_min": (0.03, 0.05),
        "qi_oil": (5000, 25000),
        "qi_gas": None,
        "gor": (1.0, 5.0),
        "wor": (1.0, 5.0),
        "is_gas": False,
    },
    "R6": {
        "name": "Post-refrac",
        "b": (1.0, 1.4),
        "Di": (0.60, 0.80),
        "d_min": (0.06, 0.08),
        "qi_oil": (5000, 25000),
        "qi_gas": None,
        "gor": (1.0, 5.0),
        "wor": (0.5, 3.0),
        "is_gas": False,
    },
}

WELL_PORTFOLIO = [
    # --- Easy tier ---
    {
        "well_id": "SYN-01",
        "regime": "R1",
        "events": [],
        "tier": "easy",
        "noise_sigma": 0.05,
        "months": 72,
        "description": "Permian tight oil, no events, clean signal",
    },
    {
        "well_id": "SYN-02",
        "regime": "R2",
        "events": [],
        "tier": "easy",
        "noise_sigma": 0.05,
        "months": 72,
        "description": "Eagle Ford moderate, no events, clean signal",
    },
    {
        "well_id": "SYN-03",
        "regime": "R5",
        "events": [],
        "tier": "easy",
        "noise_sigma": 0.05,
        "months": 72,
        "description": "Conventional vertical, no events, validates d_min lower bound",
    },
    {
        "well_id": "SYN-04",
        "regime": "R1",
        "events": [
            {"function": "apply_shut_in", "params": {"start_month": 8, "duration_months": 2}},
        ],
        "tier": "easy",
        "noise_sigma": 0.05,
        "months": 72,
        "description": "Permian tight oil, 1 voluntary shut-in (month 8, 6 weeks)",
    },
    {
        "well_id": "SYN-05",
        "regime": "R2",
        "events": [
            {"function": "apply_partial_month", "params": {"month": 5, "days_produced": 15}},
        ],
        "tier": "easy",
        "noise_sigma": 0.05,
        "months": 72,
        "description": "Eagle Ford, 1 partial production month (month 5)",
    },
    # --- Medium tier ---
    {
        "well_id": "SYN-06",
        "regime": "R1",
        "events": [
            {"function": "apply_shut_in", "params": {"start_month": 10, "duration_months": 2}},
            {"function": "apply_workover_bump", "params": {"month": 20, "amplitude_fraction": 0.3, "decay_months": 4}},
        ],
        "tier": "medium",
        "noise_sigma": 0.08,
        "months": 72,
        "description": "Permian tight oil, shut-in + workover bump, moderate noise",
    },
    {
        "well_id": "SYN-07",
        "regime": "R3",
        "events": [],
        "tier": "medium",
        "noise_sigma": 0.08,
        "months": 72,
        "description": "Shale gas Marcellus, no events, gas-dominant well",
    },
    {
        "well_id": "SYN-08",
        "regime": "R4",
        "events": [
            {"function": "apply_esp_installation", "params": {"month": 15, "rate_jump_fraction": 0.4}},
        ],
        "tier": "medium",
        "noise_sigma": 0.08,
        "months": 72,
        "description": "Bakken mature, ESP installation at month 15",
    },
    {
        "well_id": "SYN-09",
        "regime": "R2",
        "events": [
            {"function": "apply_pipeline_curtailment", "params": {"start_month": 12, "duration_months": 3, "curtailment_fraction": 0.5}},
        ],
        "tier": "medium",
        "noise_sigma": 0.08,
        "months": 72,
        "description": "Eagle Ford, pipeline curtailment months 12-14",
    },
    {
        "well_id": "SYN-10",
        "regime": "R6",
        "events": [
            {"function": "apply_partial_month", "params": {"month": 6, "days_produced": 12}},
            {"function": "apply_shut_in", "params": {"start_month": 18, "duration_months": 1}},
        ],
        "tier": "medium",
        "noise_sigma": 0.08,
        "months": 72,
        "description": "Post-refrac, partial month + shut-in, moderate noise",
    },
    # --- Hard tier ---
    {
        "well_id": "SYN-11",
        "regime": "R1",
        "events": [
            {"function": "apply_frac_hit", "params": {"month": 14, "severity": 0.4, "recovery_months": 6}},
        ],
        "tier": "hard",
        "noise_sigma": 0.12,
        "months": 84,
        "description": "Permian tight oil, frac hit at month 14, high noise",
    },
    {
        "well_id": "SYN-12",
        "regime": "R4",
        "events": [
            {"function": "apply_esp_installation", "params": {"month": 10, "rate_jump_fraction": 0.5}},
            {"function": "apply_esp_failure", "params": {"month": 24, "repair_months": 2}},
        ],
        "tier": "hard",
        "noise_sigma": 0.12,
        "months": 84,
        "description": "Bakken, ESP install then failure, high noise",
    },
    {
        "well_id": "SYN-13",
        "regime": "R3",
        "events": [
            {"function": "apply_liquid_loading", "params": {"start_month": 20, "steepening_factor": 1.5}},
        ],
        "tier": "hard",
        "noise_sigma": 0.12,
        "months": 84,
        "description": "Marcellus gas, liquid loading onset at month 20",
    },
    {
        "well_id": "SYN-14",
        "regime": "R2",
        "events": [
            {"function": "apply_shut_in", "params": {"start_month": 8, "duration_months": 2}},
            {"function": "apply_frac_hit", "params": {"month": 22, "severity": 0.3, "recovery_months": 5}},
            {"function": "apply_workover_bump", "params": {"month": 35, "amplitude_fraction": 0.25, "decay_months": 3}},
        ],
        "tier": "hard",
        "noise_sigma": 0.12,
        "months": 84,
        "description": "Eagle Ford, triple event chain: shut-in + frac hit + workover",
    },
    {
        "well_id": "SYN-15",
        "regime": "R5",
        "events": [
            {"function": "apply_pressure_plateau", "params": {"start_month": 1, "duration_months": 8, "plateau_fraction": 0.9}},
        ],
        "tier": "hard",
        "noise_sigma": 0.12,
        "months": 84,
        "description": "Conventional, early pressure plateau delays decline, high noise",
    },
    # --- Adversarial tier ---
    {
        "well_id": "SYN-16",
        "regime": "R1",
        "events": [
            {"function": "apply_frac_hit", "params": {"month": 10, "severity": 0.5, "recovery_months": 8}},
            {"function": "apply_refrac", "params": {"month": 30, "new_qi_multiplier": 0.6, "new_Di": 0.75, "new_b": 1.0}},
        ],
        "tier": "adversarial",
        "noise_sigma": 0.15,
        "months": 84,
        "description": "Permian, frac hit + refrac, adversarial noise",
    },
    {
        "well_id": "SYN-17",
        "regime": "R3",
        "events": [
            {"function": "apply_liquid_loading", "params": {"start_month": 15, "steepening_factor": 2.0}},
            {"function": "apply_gor_breakout", "params": {"start_month": 25, "oil_decline_factor": 1.3, "gas_boost_factor": 1.2}},
        ],
        "tier": "adversarial",
        "noise_sigma": 0.15,
        "months": 84,
        "description": "Marcellus, liquid loading + GOR breakout, adversarial noise",
    },
    {
        "well_id": "SYN-18",
        "regime": "R4",
        "events": [
            {"function": "apply_shut_in", "params": {"start_month": 6, "duration_months": 3}},
            {"function": "apply_allocation_noise", "params": {"months": list(range(20, 30)), "magnitude": 0.3}},
            {"function": "apply_esp_installation", "params": {"month": 35, "rate_jump_fraction": 0.5}},
            {"function": "apply_esp_failure", "params": {"month": 50, "repair_months": 2}},
        ],
        "tier": "adversarial",
        "noise_sigma": 0.15,
        "months": 84,
        "description": "Bakken, 4-event chain: shut-in + allocation noise + ESP install + ESP failure",
    },
    {
        "well_id": "SYN-19",
        "regime": "R6",
        "events": [
            {"function": "apply_refrac", "params": {"month": 20, "new_qi_multiplier": 0.7, "new_Di": 0.70, "new_b": 1.1}},
            {"function": "apply_frac_hit", "params": {"month": 35, "severity": 0.35, "recovery_months": 6}},
        ],
        "tier": "adversarial",
        "noise_sigma": 0.15,
        "months": 84,
        "description": "Post-refrac, refrac + frac hit, adversarial noise",
    },
    {
        "well_id": "SYN-20",
        "regime": "R2",
        "events": [
            {"function": "apply_pipeline_curtailment", "params": {"start_month": 5, "duration_months": 4, "curtailment_fraction": 0.6}},
            {"function": "apply_shut_in", "params": {"start_month": 15, "duration_months": 2}},
            {"function": "apply_workover_bump", "params": {"month": 25, "amplitude_fraction": 0.35, "decay_months": 5}},
            {"function": "apply_frac_hit", "params": {"month": 40, "severity": 0.4, "recovery_months": 7}},
            {"function": "apply_allocation_noise", "params": {"months": list(range(50, 60)), "magnitude": 0.25}},
        ],
        "tier": "adversarial",
        "noise_sigma": 0.15,
        "months": 84,
        "description": "Eagle Ford, 5-event gauntlet, adversarial noise",
    },
]


# ---------------------------------------------------------------------------
# Section A: Base curve generation
# ---------------------------------------------------------------------------

def sample_regime_params(regime_key: str) -> dict:
    """Sample ground truth DCA parameters from a regime's ranges."""
    reg = REGIMES[regime_key]
    b = random.uniform(*reg["b"])
    Di = random.uniform(*reg["Di"])
    d_min = random.uniform(*reg["d_min"])
    qi_oil = random.uniform(*reg["qi_oil"])

    if reg["is_gas"] and reg["qi_gas"] is not None:
        qi_gas = random.uniform(*reg["qi_gas"])
    else:
        qi_gas = None

    gor = random.uniform(*reg["gor"])
    wor = random.uniform(*reg["wor"])

    return {
        "b": b,
        "Di": Di,
        "d_min": d_min,
        "qi_oil": qi_oil,
        "qi_gas": qi_gas,
        "gor": gor,
        "wor": wor,
    }


def generate_base_curve(params: dict, months: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate base oil/gas/water production arrays from ground truth params.

    Returns (oil, gas, water) as numpy arrays of monthly volumes.
    """
    qi_oil_monthly = params["qi_oil"]
    qi_oil_daily = qi_oil_monthly / 30.4375
    Di = params["Di"]
    b = params["b"]
    d_min = params["d_min"]

    model = dca.MH(qi=qi_oil_daily, Di=Di, bi=b, Dterm=d_min)
    days_per_month = 30.4375
    t = np.arange(1, months + 1) * days_per_month
    oil = model.monthly_vol(t)

    # Gas from GOR (or independent gas model for gas wells)
    if params["qi_gas"] is not None:
        qi_gas_daily = params["qi_gas"] / 30.4375
        gas_model = dca.MH(qi=qi_gas_daily, Di=Di, bi=b, Dterm=d_min)
        gas = gas_model.monthly_vol(t)
    else:
        gas = oil * params["gor"]

    # Water from WOR with ~15% annual increase
    base_wor = params["wor"]
    month_indices = np.arange(months)
    wor_over_time = base_wor * (1.0 + 0.15 * month_indices / 12.0)
    water = oil * wor_over_time

    return oil, gas, water


# ---------------------------------------------------------------------------
# Section F: Apply events (imports from synthetic_events.py)
# ---------------------------------------------------------------------------

def apply_events(oil: np.ndarray, gas: np.ndarray, water: np.ndarray,
                 events: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a list of operational events to production arrays.

    Each event is a dict with 'function' (name) and 'params' (kwargs).
    Events are imported from synthetic_events.py.
    """
    if not events:
        return oil, gas, water

    from synthetic_events import (
        apply_shut_in,
        apply_frac_hit,
        apply_workover_bump,
        apply_pipeline_curtailment,
        apply_refrac,
        apply_partial_month,
        apply_allocation_noise,
        apply_esp_installation,
        apply_esp_failure,
        apply_liquid_loading,
        apply_pressure_plateau,
        apply_gor_breakout,
    )

    event_funcs = {
        "apply_shut_in": apply_shut_in,
        "apply_frac_hit": apply_frac_hit,
        "apply_workover_bump": apply_workover_bump,
        "apply_pipeline_curtailment": apply_pipeline_curtailment,
        "apply_refrac": apply_refrac,
        "apply_partial_month": apply_partial_month,
        "apply_allocation_noise": apply_allocation_noise,
        "apply_esp_installation": apply_esp_installation,
        "apply_esp_failure": apply_esp_failure,
        "apply_liquid_loading": apply_liquid_loading,
        "apply_pressure_plateau": apply_pressure_plateau,
        "apply_gor_breakout": apply_gor_breakout,
    }

    for event in events:
        func_name = event["function"]
        params = event.get("params", {})
        if func_name not in event_funcs:
            raise ValueError(f"Unknown event function: {func_name}")
        oil, gas, water = event_funcs[func_name](oil, gas, water, **params)

    return oil, gas, water


# ---------------------------------------------------------------------------
# Section B: Noise layer
# ---------------------------------------------------------------------------

def apply_noise(oil: np.ndarray, gas: np.ndarray, water: np.ndarray,
                noise_sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply log-normal multiplicative noise and partial-month artifacts.

    Noise is applied LAST, after all events.
    """
    n = len(oil)

    # Log-normal multiplicative noise (correlated across streams for realism)
    noise_oil = np.random.lognormal(mean=0, sigma=noise_sigma, size=n)
    noise_gas = np.random.lognormal(mean=0, sigma=noise_sigma, size=n)
    noise_water = np.random.lognormal(mean=0, sigma=noise_sigma, size=n)

    oil = oil * noise_oil
    gas = gas * noise_gas
    water = water * noise_water

    # Partial-month artifacts: 10% probability per month
    for i in range(n):
        if random.random() < 0.10:
            days_produced = random.randint(10, 25)
            scale = days_produced / 30.0
            oil[i] *= scale
            gas[i] *= scale
            water[i] *= scale

    # Ensure non-negative
    oil = np.maximum(oil, 0.0)
    gas = np.maximum(gas, 0.0)
    water = np.maximum(water, 0.0)

    return oil, gas, water


# ---------------------------------------------------------------------------
# Section C: Validation checks
# ---------------------------------------------------------------------------

def validate_well(oil: np.ndarray, gas: np.ndarray, water: np.ndarray,
                  well_config: dict, ground_truth: dict) -> list[str]:
    """Validate a generated well. Returns list of warning/error strings (empty = pass)."""
    issues = []
    months = len(oil)

    # 1. Non-negative production
    if np.any(oil < 0):
        issues.append("FAIL: negative oil production detected")
    if np.any(gas < 0):
        issues.append("FAIL: negative gas production detected")
    if np.any(water < 0):
        issues.append("FAIL: negative water production detected")

    # 2. No month > 3x prior without documented event
    event_months = set()
    for ev in well_config.get("events", []):
        p = ev.get("params", {})
        if "month" in p:
            event_months.add(p["month"])
        if "start_month" in p:
            duration = p.get("duration_months", 2)
            for m in range(p["start_month"], p["start_month"] + duration + 2):
                event_months.add(m)

    for i in range(1, months):
        if oil[i - 1] > 0 and oil[i] > 3 * oil[i - 1]:
            # Check if this is near a documented event
            # Month indices in events are 1-based; array index i corresponds to month i+1
            month_num = i + 1
            nearby = any(abs(month_num - em) <= 2 for em in event_months)
            if not nearby:
                issues.append(
                    f"WARN: month {month_num} oil ({oil[i]:.0f}) > 3x prior "
                    f"({oil[i-1]:.0f}) without documented event"
                )

    # 3. Last 12 months (holdout) contain no injected events
    holdout_start_month = months - 12 + 1  # 1-based month number
    for ev in well_config.get("events", []):
        p = ev.get("params", {})
        ev_month = p.get("month") or p.get("start_month")
        if ev_month is not None:
            duration = p.get("duration_months", 0)
            ev_end = ev_month + duration
            if ev_end >= holdout_start_month:
                issues.append(
                    f"FAIL: event '{ev['function']}' at month {ev_month} "
                    f"overlaps holdout period (months {holdout_start_month}-{months})"
                )

    # 4 & 5. MAPE checks (only for easy tier, requires pipeline)
    if well_config.get("tier") == "easy":
        try:
            mape = _compute_ground_truth_mape(oil)
            if mape is not None and mape > 0.15:
                issues.append(
                    f"WARN: easy well ground truth MAPE {mape:.4f} > 0.15 threshold"
                )
        except Exception as e:
            issues.append(f"WARN: could not compute ground truth MAPE: {e}")

    return issues


def _compute_ground_truth_mape(oil: np.ndarray) -> float | None:
    """Compute MAPE using default params on generated oil data.

    Fits on months 1..N-12, scores on last 12 months.
    Returns None if fitting fails.
    """
    try:
        from evaluation import preprocess_and_fit
        from arps import create_model, generate_forecast
    except ImportError:
        return None

    total = len(oil)
    if total < 18:
        return None

    holdout = 12
    train_months = total - holdout
    oil_train = oil[:train_months].tolist()
    oil_holdout = oil[train_months:]

    params = copy.deepcopy(BASELINE_PARAMS)

    try:
        anchor, filtered, fit_result = preprocess_and_fit(oil_train, params)
    except (ValueError, Exception):
        return None

    model = create_model(
        decline_model=fit_result["decline_model"],
        qi_oil=fit_result["qi"],
        di=fit_result["di"],
        b=fit_result["b"],
        d_min=fit_result["d_min"],
    )

    months_from_peak = train_months - anchor.peak_index + holdout
    forecast = generate_forecast(model, months=max(months_from_peak + 12, 120))

    holdout_start_idx = train_months - anchor.peak_index
    forecast_holdout = []
    for i in range(holdout):
        fc_idx = holdout_start_idx + i
        if fc_idx < len(forecast):
            forecast_holdout.append(forecast[fc_idx].oil_bbl)
        else:
            forecast_holdout.append(0.0)

    actual = np.array(oil_holdout[:holdout], dtype=float)
    predicted = np.array(forecast_holdout[:len(actual)], dtype=float)

    nonzero = actual > 0
    if np.sum(nonzero) == 0:
        return None

    mape = float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])))
    return mape


# ---------------------------------------------------------------------------
# Section E: Output -- write well atom files
# ---------------------------------------------------------------------------

def generate_dates(months: int, start_year: int = 2020, start_month: int = 1) -> list[str]:
    """Generate YYYY-MM date strings for N months starting from given date."""
    dates = []
    y, m = start_year, start_month
    for _ in range(months):
        dates.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return dates


def write_well(well_id: str, oil: np.ndarray, gas: np.ndarray, water: np.ndarray,
               ground_truth: dict, well_config: dict) -> Path:
    """Write a synthetic well atom to wells/SYNTHETIC/{well_id}/."""
    well_dir = Path("wells") / "SYNTHETIC" / well_id
    well_dir.mkdir(parents=True, exist_ok=True)

    months = len(oil)
    dates = generate_dates(months)

    # production.csv
    csv_path = well_dir / "production.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "oil_bbl", "gas_mcf", "water_bbl"])
        for i in range(months):
            writer.writerow([
                dates[i],
                round(float(oil[i]), 1),
                round(float(gas[i]), 1),
                round(float(water[i]), 1),
            ])

    # params.json -- baseline defaults
    params = copy.deepcopy(BASELINE_PARAMS)
    params_path = well_dir / "params.json"
    params_path.write_text(json.dumps(params, indent=2))

    # well_metadata.json -- ground truth (not exposed to agent)
    metadata = {
        "well_id": well_id,
        "regime": well_config["regime"],
        "regime_name": REGIMES[well_config["regime"]]["name"],
        "tier": well_config["tier"],
        "description": well_config["description"],
        "ground_truth": {
            "qi_oil": ground_truth["qi_oil"],
            "qi_gas": ground_truth["qi_gas"],
            "Di": ground_truth["Di"],
            "b": ground_truth["b"],
            "d_min": ground_truth["d_min"],
            "gor": ground_truth["gor"],
            "wor": ground_truth["wor"],
        },
        "noise_sigma": well_config["noise_sigma"],
        "events": well_config["events"],
        "months": months,
    }
    meta_path = well_dir / "well_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    return well_dir


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

def generate_well(well_config: dict, base_seed: int, well_index: int) -> tuple[Path, list[str]]:
    """Generate a single synthetic well.

    Sets random seed = base_seed + well_index for reproducibility.
    Returns (well_dir, validation_issues).
    """
    seed = base_seed + well_index
    random.seed(seed)
    np.random.seed(seed)

    regime_key = well_config["regime"]
    months = well_config["months"]

    # Sample ground truth parameters
    gt_params = sample_regime_params(regime_key)

    # Generate base curve
    oil, gas, water = generate_base_curve(gt_params, months)

    # Apply events
    oil, gas, water = apply_events(oil, gas, water, well_config["events"])

    # Apply noise (last)
    oil, gas, water = apply_noise(oil, gas, water, well_config["noise_sigma"])

    # Validate
    issues = validate_well(oil, gas, water, well_config, gt_params)

    # Write output
    well_dir = write_well(well_config["well_id"], oil, gas, water, gt_params, well_config)

    return well_dir, issues


# ---------------------------------------------------------------------------
# Validation of existing wells
# ---------------------------------------------------------------------------

def validate_existing_wells() -> None:
    """Validate all existing synthetic wells in wells/SYNTHETIC/."""
    synth_dir = Path("wells") / "SYNTHETIC"
    if not synth_dir.exists():
        print("No synthetic wells found.")
        return

    well_dirs = sorted(d for d in synth_dir.iterdir() if d.is_dir())
    if not well_dirs:
        print("No synthetic wells found.")
        return

    for wd in well_dirs:
        well_id = wd.name
        csv_path = wd / "production.csv"
        meta_path = wd / "well_metadata.json"

        if not csv_path.exists():
            print(f"  {well_id}: MISSING production.csv")
            continue
        if not meta_path.exists():
            print(f"  {well_id}: MISSING well_metadata.json")
            continue

        # Read production
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            oil, gas, water = [], [], []
            for row in reader:
                oil.append(float(row["oil_bbl"]))
                gas.append(float(row["gas_mcf"]))
                water.append(float(row["water_bbl"]))

        oil = np.array(oil)
        gas = np.array(gas)
        water = np.array(water)

        metadata = json.loads(meta_path.read_text())

        # Find matching portfolio config
        config = None
        for wc in WELL_PORTFOLIO:
            if wc["well_id"] == well_id:
                config = wc
                break

        if config is None:
            print(f"  {well_id}: not in portfolio definition, skipping validation")
            continue

        issues = validate_well(oil, gas, water, config, metadata.get("ground_truth", {}))
        if issues:
            print(f"  {well_id}: {len(issues)} issue(s)")
            for issue in issues:
                print(f"    {issue}")
        else:
            print(f"  {well_id}: OK ({len(oil)} months, tier={config['tier']})")


# ---------------------------------------------------------------------------
# Section G: CLI
# ---------------------------------------------------------------------------

def get_wells_by_tier(tier: str) -> list[dict]:
    """Return well configs matching a difficulty tier."""
    return [w for w in WELL_PORTFOLIO if w["tier"] == tier]


def get_wells_by_ids(well_ids: list[str]) -> list[dict]:
    """Return well configs matching specific IDs."""
    id_set = set(well_ids)
    found = [w for w in WELL_PORTFOLIO if w["well_id"] in id_set]
    missing = id_set - {w["well_id"] for w in found}
    if missing:
        print(f"WARNING: unknown well IDs: {sorted(missing)}", file=sys.stderr)
    return found


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic production wells with known ground truth parameters."
    )
    parser.add_argument(
        "--wells", nargs="+", metavar="ID",
        help="Specific well IDs to generate (e.g., SYN-01 SYN-07)",
    )
    parser.add_argument(
        "--tier", choices=["easy", "medium", "hard", "adversarial"],
        help="Generate all wells of a difficulty tier",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Generate all 20 wells",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate existing generated wells instead of generating",
    )

    args = parser.parse_args()

    if args.validate:
        validate_existing_wells()
        return

    # Determine which wells to generate
    if args.all:
        wells_to_gen = WELL_PORTFOLIO
    elif args.wells:
        wells_to_gen = get_wells_by_ids(args.wells)
    elif args.tier:
        wells_to_gen = get_wells_by_tier(args.tier)
    else:
        # Default: easy tier
        wells_to_gen = get_wells_by_tier("easy")

    if not wells_to_gen:
        print("No wells to generate.")
        return

    print(f"Generating {len(wells_to_gen)} synthetic well(s) with seed={args.seed}...")

    # Build index map for reproducible per-well seeding
    id_to_index = {w["well_id"]: i for i, w in enumerate(WELL_PORTFOLIO)}

    all_issues = {}
    for wc in wells_to_gen:
        well_index = id_to_index[wc["well_id"]]
        well_dir, issues = generate_well(wc, args.seed, well_index)
        all_issues[wc["well_id"]] = issues
        status = "OK" if not issues else f"{len(issues)} issue(s)"
        print(f"  {wc['well_id']}: {well_dir} [{status}]")

    # Summary
    total_issues = sum(len(v) for v in all_issues.values())
    if total_issues > 0:
        print(f"\n{total_issues} validation issue(s) found:")
        for wid, issues in all_issues.items():
            for issue in issues:
                print(f"  {wid}: {issue}")
    else:
        print(f"\nAll {len(wells_to_gen)} well(s) generated and validated successfully.")


if __name__ == "__main__":
    main()
