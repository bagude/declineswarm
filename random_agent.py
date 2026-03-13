"""Memoryless random search agent for decline-curve parameter tuning.

Mirrors the Claude agent loop (program.md) but replaces LLM reasoning
with uniform random draws. Does NOT read trace.jsonl for decisions.

Usage:
    python random_agent.py NM 30-025-50364
    python random_agent.py NM 30-025-50364 --iterations 20
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Parameter search space (from program.md)
# di_upper_bound is EXCLUDED — hardcoded at 0.99
# ---------------------------------------------------------------------------

CONTINUOUS_PARAMS = {
    "significance_threshold": (0.30, 0.75),
    "smoothing_window": None,       # discrete — handled below
    "outlier_window": None,          # discrete
    "outlier_threshold": (0.15, 0.50),
    "peak_merge_distance": None,     # discrete
    "d_min": (0.03, 0.10),
    "di_initial": (0.30, 1.00),
    "b_initial": (0.30, 1.20),
    "qi_multiplier_upper": (3.0, 10.0),
    "b_upper_bound": (1.0, 3.0),
}

DISCRETE_PARAMS = {
    "smoothing_window": [2, 3, 4, 5, 6],
    "outlier_window": [3, 4, 5, 6, 7],
    "peak_merge_distance": [2, 3, 4, 5, 6, 7, 8],
}

CATEGORICAL_PARAMS = {
    "qi_guess_strategy": ["first", "max3", "peak_value"],
}

# Which block each parameter lives in
PREPROCESSING_PARAMS = {
    "significance_threshold", "smoothing_window", "outlier_window",
    "outlier_threshold", "peak_merge_distance",
}
FITTING_PARAMS = {
    "d_min", "di_initial", "b_initial", "qi_guess_strategy",
    "qi_multiplier_upper", "b_upper_bound",
}

ALL_TUNABLE = sorted(PREPROCESSING_PARAMS | FITTING_PARAMS)


def sample_param(name: str):
    """Sample a random value for the given parameter."""
    if name in CATEGORICAL_PARAMS:
        return random.choice(CATEGORICAL_PARAMS[name])
    if name in DISCRETE_PARAMS:
        return random.choice(DISCRETE_PARAMS[name])
    lo, hi = CONTINUOUS_PARAMS[name]
    return round(random.uniform(lo, hi), 4)


def read_params(well_dir: Path) -> dict:
    """Read params.json from well directory."""
    return json.loads((well_dir / "params.json").read_text())


def write_params(well_dir: Path, params: dict) -> None:
    """Write params.json to well directory."""
    (well_dir / "params.json").write_text(json.dumps(params, indent=2))


def run_eval(state: str, api: str) -> dict:
    """Run run_eval.py and parse JSON output."""
    result = subprocess.run(
        [sys.executable, "run_eval.py", state, api],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        # Try to parse stderr or stdout for error info
        try:
            return json.loads(result.stdout)
        except (json.JSONDecodeError, ValueError):
            return {"error": f"run_eval failed: {result.stderr.strip()}"}
    try:
        return json.loads(result.stdout)
    except (json.JSONDecodeError, ValueError):
        return {"error": f"run_eval bad output: {result.stdout[:200]}"}


def git_commit(state: str, api: str, version: int, description: str) -> bool:
    """Stage well files and commit. Returns True on success."""
    well_path = f"wells/{state}/{api}"
    files = [
        f"{well_path}/params.json",
        f"{well_path}/fit.json",
        f"{well_path}/trace.jsonl",
        f"{well_path}/hindcast.json",
    ]
    add_result = subprocess.run(
        ["git", "add"] + files,
        capture_output=True, text=True,
    )
    if add_result.returncode != 0:
        print(f"  git add failed: {add_result.stderr.strip()}")
        return False

    msg = f"{api} v{version}: {description}"
    commit_result = subprocess.run(
        ["git", "commit", "-m", msg],
        capture_output=True, text=True,
    )
    if commit_result.returncode != 0:
        print(f"  git commit failed: {commit_result.stderr.strip()}")
        return False
    return True


def git_revert_params(state: str, api: str) -> None:
    """Revert only params.json (not trace/hindcast/fit)."""
    subprocess.run(
        ["git", "checkout", "--", f"wells/{state}/{api}/params.json"],
        capture_output=True, text=True,
    )


def run_random_agent(state: str, api: str, iterations: int = 50) -> str:
    """Run random search on one well. Returns status string."""
    well_dir = Path("wells") / state / api

    if not well_dir.exists():
        return f"ERROR: well directory {well_dir} does not exist"

    # Run baseline evaluation first (mirrors Claude agent program.md step 3)
    print(f"  Evaluating baseline...")
    baseline_result = run_eval(state, api)
    if "error" in baseline_result:
        return f"ERROR: baseline eval failed: {baseline_result['error']}"
    best_mape = baseline_result.get("hindcast_mape")
    if best_mape is None:
        return f"ERROR: baseline eval returned no MAPE"
    print(f"  Baseline MAPE={best_mape:.4f}")

    consecutive_non_improvements = 0
    total_improvements = 0

    for i in range(1, iterations + 1):
        # Read params fresh each iteration (may have been reverted)
        params = read_params(well_dir)
        current_version = params.get("version", 1)
        new_version = current_version + 1

        # Pick one random parameter and sample a value
        param_name = random.choice(ALL_TUNABLE)
        new_value = sample_param(param_name)

        # Write updated params.json
        block = "preprocessing" if param_name in PREPROCESSING_PARAMS else "fitting"
        params[block][param_name] = new_value
        params["version"] = new_version
        params["description"] = f"random: {param_name}={new_value}"
        write_params(well_dir, params)

        # Run evaluation
        result = run_eval(state, api)

        if "error" in result:
            print(f"  [{i}/{iterations}] v{new_version} {param_name}={new_value} -> ERROR: {result['error']}")
            git_revert_params(state, api)
            consecutive_non_improvements += 1
            if consecutive_non_improvements >= 10:
                print(f"  Stopping: 10 consecutive non-improvements")
                break
            continue

        mape = result.get("hindcast_mape")
        if mape is None:
            print(f"  [{i}/{iterations}] v{new_version} {param_name}={new_value} -> no MAPE in output")
            git_revert_params(state, api)
            consecutive_non_improvements += 1
            if consecutive_non_improvements >= 10:
                print(f"  Stopping: 10 consecutive non-improvements")
                break
            continue

        improved = mape < best_mape

        if improved:
            prev_mape = best_mape
            best_mape = mape
            description = f"random: {param_name}={new_value}"
            git_commit(state, api, new_version, description)
            total_improvements += 1
            consecutive_non_improvements = 0
            print(f"  [{i}/{iterations}] v{new_version} {param_name}={new_value} -> MAPE={mape:.4f} (improved from {prev_mape:.4f})")
        else:
            git_revert_params(state, api)
            consecutive_non_improvements += 1
            print(f"  [{i}/{iterations}] v{new_version} {param_name}={new_value} -> MAPE={mape:.4f} (no improvement, best={best_mape:.4f}, streak={consecutive_non_improvements})")
            if consecutive_non_improvements >= 10:
                print(f"  Stopping: 10 consecutive non-improvements")
                break

    status = (
        f"{api}: {total_improvements} improvements in {i} iterations, "
        f"best MAPE={best_mape:.4f}" if best_mape is not None
        else f"{api}: no successful evaluations in {i} iterations"
    )
    print(status)
    return status


def main():
    parser = argparse.ArgumentParser(
        description="Memoryless random search agent for decline-curve tuning"
    )
    parser.add_argument("state", help="State abbreviation (e.g. NM)")
    parser.add_argument("api", help="API number (e.g. 30-025-50364)")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Maximum iterations (default: 50)")
    args = parser.parse_args()

    run_random_agent(args.state, args.api, args.iterations)


if __name__ == "__main__":
    main()
