"""Read swarm results — portfolio-level summary of what the swarm discovered.

Usage:
    python read_results.py
"""

import json
import statistics
from pathlib import Path


def main():
    results = []
    errors = []

    for well_dir in sorted(Path("wells").iterdir()):
        if not well_dir.is_dir():
            continue

        fit_file = well_dir / "fit.json"
        params_file = well_dir / "params.json"

        if not params_file.exists():
            continue

        params = json.loads(params_file.read_text())
        api = well_dir.name
        well_type = params.get("well_type", "unknown")

        if not fit_file.exists():
            errors.append(f"{api} ({well_type}): no fit.json")
            continue

        fit = json.loads(fit_file.read_text())
        results.append({
            "api": api,
            "well_type": well_type,
            "param_version": params.get("version", 1),
            "hindcast_mape": fit.get("hindcast_mape"),
            "r2_insample": fit.get("r2_insample"),
            "d_min": params["fitting"]["d_min"],
            "di": fit.get("di"),
            "b": fit.get("b"),
            "model": fit.get("model"),
            "convergence": fit.get("convergence"),
        })

    if not results:
        print("No results found. Run the swarm first (python run_swarm.py).")
        return

    # Summary by well type
    print("=== Portfolio Summary ===\n")
    for well_type in sorted(set(r["well_type"] for r in results)):
        subset = [r for r in results if r["well_type"] == well_type]
        d_mins = [r["d_min"] for r in subset]
        mapes = [r["hindcast_mape"] for r in subset if r["hindcast_mape"] is not None]
        r2s = [r["r2_insample"] for r in subset if r["r2_insample"] is not None]
        versions = [r["param_version"] for r in subset]

        print(f"{well_type} ({len(subset)} wells):")
        if d_mins:
            print(f"  d_min   — mean={statistics.mean(d_mins):.4f}, "
                  f"min={min(d_mins):.4f}, max={max(d_mins):.4f}")
        if mapes:
            print(f"  MAPE    — mean={statistics.mean(mapes):.4f}, "
                  f"min={min(mapes):.4f}, max={max(mapes):.4f}")
        if r2s:
            print(f"  R² in   — mean={statistics.mean(r2s):.4f}, "
                  f"min={min(r2s):.4f}, max={max(r2s):.4f}")
        print(f"  versions— mean={statistics.mean(versions):.1f}, max={max(versions)}")
        print()

    # Overall
    all_mapes = [r["hindcast_mape"] for r in results if r["hindcast_mape"] is not None]
    print(f"=== Overall: {len(results)} wells ===")
    if all_mapes:
        print(f"  MAPE mean={statistics.mean(all_mapes):.4f}, "
              f"median={statistics.median(all_mapes):.4f}")
    converged = sum(1 for r in results if r["convergence"])
    print(f"  Convergence: {converged}/{len(results)}")

    # Wells without results
    if errors:
        print(f"\n=== {len(errors)} wells without fit results ===")
        for e in errors:
            print(f"  {e}")


if __name__ == "__main__":
    main()
