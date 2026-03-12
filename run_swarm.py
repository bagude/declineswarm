"""Spawn one agent per well atom.

Usage:
    python run_swarm.py TX                        # all TX wells, 4 workers
    python run_swarm.py NM --max-workers 8        # all NM wells, 8 workers
    python run_swarm.py TX --wells 42-033-32691 42-001-32003
    python run_swarm.py NM --iterations 20
"""

import argparse
import json
import os
import subprocess
import concurrent.futures
from pathlib import Path

from defaults import BASELINE_PARAMS


def reset_atom(state: str, api: str):
    """Reset a well atom to blank slate before the agent starts."""
    well_dir = Path("wells") / state / api

    # Preserve well-specific metadata from existing params
    params_file = well_dir / "params.json"
    meta = {}
    if params_file.exists():
        old = json.loads(params_file.read_text(encoding="utf-8"))
        for key in ("state", "vintage", "well_type"):
            if key in old:
                meta[key] = old[key]

    # Reset params to baseline with preserved metadata
    params = {**BASELINE_PARAMS, **meta}
    params_file.write_text(json.dumps(params, indent=2))

    # Clear trace, hindcast, and fit
    (well_dir / "trace.jsonl").write_text("")
    (well_dir / "hindcast.json").write_text("[]")
    fit_file = well_dir / "fit.json"
    if fit_file.exists():
        fit_file.unlink()


def run_agent_on_atom(state: str, api: str, n_iterations: int = 50) -> str:
    """Spawn a Claude Code instance targeting one well."""
    reset_atom(state, api)

    program = (Path("program.md").read_text(encoding="utf-8")
               .replace("{api}", api)
               .replace("{state}", state))

    # Clean environment so nested claude sessions are allowed
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

    result = subprocess.run(
        [
            "claude",
            "--print",
            "--dangerously-skip-permissions",
            "--max-turns", str(n_iterations),
            program,
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(Path(".").resolve()),
    )

    if result.returncode != 0:
        err = (result.stderr or result.stdout or "no output").strip()[:300]
        return f"{state}/{api}: exit={result.returncode} — {err}"
    return f"{state}/{api}: ok"


def main():
    parser = argparse.ArgumentParser(description="Run DCA optimization swarm")
    parser.add_argument("state", help="State subdirectory (TX, NM, etc.)")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Parallel agent count (default: 4)")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Max LLM turns per well (default: 50). "
                             "Each optimization iteration costs ~4-5 turns, "
                             "so 50 turns ≈ 10 experiments.")
    parser.add_argument("--wells", nargs="*", default=None,
                        help="Specific well APIs to run (default: all)")
    args = parser.parse_args()

    state_dir = Path("wells") / args.state
    if not state_dir.exists():
        print(f"Error: {state_dir} does not exist")
        return

    if args.wells:
        apis = args.wells
    else:
        apis = sorted(p.name for p in state_dir.iterdir() if p.is_dir())

    print(f"Swarm: {args.state} — {len(apis)} wells, {args.max_workers} workers, {args.iterations} iterations each")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(run_agent_on_atom, args.state, api, args.iterations): api
            for api in apis
        }
        for future in concurrent.futures.as_completed(futures):
            api = futures[future]
            try:
                result = future.result()
                print(f"  {result}")
            except Exception as e:
                print(f"  {api}: ERROR — {e}")

    print("Swarm complete.")


if __name__ == "__main__":
    main()
