"""Spawn one agent per well atom.

Usage:
    python run_swarm.py                    # all wells, 4 workers
    python run_swarm.py --max-workers 1    # single well at a time
    python run_swarm.py --wells 42-033-32691 42-001-32003
    python run_swarm.py --iterations 20    # limit iterations per well
"""

import argparse
import subprocess
import concurrent.futures
from pathlib import Path


def run_agent_on_atom(api: str, n_iterations: int = 50) -> str:
    """Spawn a Claude Code instance targeting one well."""
    program = Path("program.md").read_text().replace("{api}", api)

    # Count current version to set meaningful iteration limit
    result = subprocess.run(
        [
            "claude",
            "--print",
            "--directory", str(Path(".")),
            "--prompt", program,
            "--max-turns", str(n_iterations),
        ],
        capture_output=True,
        text=True,
    )

    status = "ok" if result.returncode == 0 else f"exit={result.returncode}"
    return f"{api}: {status}"


def main():
    parser = argparse.ArgumentParser(description="Run DCA optimization swarm")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Parallel agent count (default: 4)")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Max iterations per well (default: 50)")
    parser.add_argument("--wells", nargs="*", default=None,
                        help="Specific well APIs to run (default: all)")
    args = parser.parse_args()

    if args.wells:
        apis = args.wells
    else:
        apis = sorted(p.name for p in Path("wells").iterdir() if p.is_dir())

    print(f"Swarm: {len(apis)} wells, {args.max_workers} workers, {args.iterations} iterations each")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(run_agent_on_atom, api, args.iterations): api
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
