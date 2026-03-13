"""Ablation experiment: Claude agent vs Random agent on decline-curve tuning.

Usage:
    python run_ablation.py NM                          # all wells, 3 reps, 50 iterations
    python run_ablation.py NM --reps 1                 # 1 rep (pipeline verification)
    python run_ablation.py NM --iterations 20          # fewer iterations per run
    python run_ablation.py NM --condition random       # run only random condition
    python run_ablation.py NM --condition claude        # run only Claude condition
    python run_ablation.py NM --wells 30-025-50364     # specific well(s)
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from run_swarm import reset_atom
from random_agent import run_random_agent

WELLS_NM = [
    "30-025-50360",
    "30-025-50362",
    "30-025-50363",
    "30-025-50364",
    "30-025-50366",
]

RESULT_FILES = ["trace.jsonl", "hindcast.json", "fit.json", "params.json"]


def run_claude_agent(state: str, api: str, iterations: int) -> dict:
    """Run Claude agent on a single well. Returns result dict."""
    program = (Path("program.md").read_text(encoding="utf-8")
               .replace("{api}", api)
               .replace("{state}", state))

    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

    t0 = time.time()
    result = subprocess.run(
        [
            "claude",
            "--print",
            "--dangerously-skip-permissions",
            "--max-turns", str(iterations),
            program,
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(Path(".").resolve()),
    )
    elapsed = time.time() - t0

    return {
        "returncode": result.returncode,
        "elapsed_s": round(elapsed, 1),
        "error": (result.stderr or "").strip()[:300] if result.returncode != 0 else None,
    }


def run_random_condition(state: str, api: str, iterations: int) -> dict:
    """Run random agent on a single well. Returns result dict."""
    t0 = time.time()
    try:
        result = run_random_agent(state, api, iterations=iterations)
        elapsed = time.time() - t0
        return {
            "returncode": 0,
            "elapsed_s": round(elapsed, 1),
            "error": None,
            "result": result,
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "returncode": 1,
            "elapsed_s": round(elapsed, 1),
            "error": str(e)[:300],
        }


def copy_results(state: str, api: str, dest_dir: Path):
    """Copy well result files to the ablation results directory."""
    well_dir = Path("wells") / state / api
    os.makedirs(dest_dir, exist_ok=True)

    for filename in RESULT_FILES:
        src = well_dir / filename
        if src.exists():
            shutil.copy2(src, dest_dir / filename)


def main():
    parser = argparse.ArgumentParser(
        description="Ablation experiment: Claude agent vs Random agent"
    )
    parser.add_argument("state", help="State subdirectory (NM, TX, etc.)")
    parser.add_argument("--reps", type=int, default=3,
                        help="Replications per condition per well (default: 3)")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Iterations per run (default: 50)")
    parser.add_argument("--condition", choices=["claude", "random"],
                        default=None,
                        help="Run only one condition (default: both)")
    parser.add_argument("--wells", nargs="*", default=None,
                        help="Specific well APIs (default: all NM wells)")
    args = parser.parse_args()

    # Determine wells
    if args.wells:
        apis = args.wells
    else:
        state_dir = Path("wells") / args.state
        if not state_dir.exists():
            print(f"Error: {state_dir} does not exist")
            return
        apis = sorted(p.name for p in state_dir.iterdir() if p.is_dir())

    # Determine conditions
    if args.condition:
        conditions = [args.condition]
    else:
        conditions = ["claude", "random"]

    total_runs = len(conditions) * len(apis) * args.reps
    print(f"Ablation experiment: {args.state}")
    print(f"  Conditions: {conditions}")
    print(f"  Wells: {len(apis)}")
    print(f"  Reps: {args.reps}")
    print(f"  Iterations per run: {args.iterations}")
    print(f"  Total runs: {total_runs}")
    print()

    results_root = Path("ablation_results")
    completed = 0
    errors = []

    for condition in conditions:
        for rep in range(1, args.reps + 1):
            for api in apis:
                run_label = f"{condition} rep{rep} on {api}"
                print(f"Running {run_label}...")

                # 1. Reset well to baseline and commit so git checkout works
                try:
                    reset_atom(args.state, api)
                    well_path = f"wells/{args.state}/{api}"
                    subprocess.run(
                        ["git", "add",
                         f"{well_path}/params.json",
                         f"{well_path}/trace.jsonl",
                         f"{well_path}/hindcast.json"],
                        capture_output=True, text=True,
                    )
                    subprocess.run(
                        ["git", "commit", "-m",
                         f"{api}: reset to baseline for {condition} rep{rep}"],
                        capture_output=True, text=True,
                    )
                except Exception as e:
                    msg = f"{run_label}: reset failed -- {e}"
                    print(f"  ERROR: {msg}")
                    errors.append(msg)
                    continue

                # 2. Run the agent
                if condition == "claude":
                    result = run_claude_agent(args.state, api, args.iterations)
                else:
                    result = run_random_condition(args.state, api, args.iterations)

                # 3. Copy results
                dest_dir = results_root / condition / f"rep{rep}" / api
                try:
                    copy_results(args.state, api, dest_dir)
                except Exception as e:
                    msg = f"{run_label}: copy failed — {e}"
                    print(f"  ERROR: {msg}")
                    errors.append(msg)
                    continue

                completed += 1
                status = "ok" if result["returncode"] == 0 else f"FAILED (rc={result['returncode']})"
                print(f"  {status} ({result['elapsed_s']}s)")

                if result.get("error"):
                    msg = f"{run_label}: {result['error']}"
                    errors.append(msg)
                    print(f"  Error detail: {result['error']}")

    # Summary
    print()
    print("=" * 60)
    print(f"Ablation complete: {completed}/{total_runs} runs finished")
    if errors:
        print(f"Errors ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")
    else:
        print("No errors.")
    print(f"Results saved to: {results_root.resolve()}")


if __name__ == "__main__":
    main()
