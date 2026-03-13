"""Ablation experiment analysis -- does Claude outperform random search?

Reads results from ablation_results/{claude,random}/rep{N}/{well}/ and produces:
  - ablation_results/summary.csv    (flat per-well, per-condition, per-rep metrics)
  - ablation_results/boxplot_mape.png   (final MAPE distributions)
  - ablation_results/convergence.png    (MAPE vs iteration curves)
  - stdout: statistical test results and interpretation

Usage:
    python analyze_ablation.py
    python analyze_ablation.py --results-dir path/to/results
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trace(trace_path):
    """Load trace.jsonl, return list of dicts."""
    entries = []
    try:
        with open(trace_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return entries


def load_hindcast(hindcast_path):
    """Load hindcast.json, return list of dicts."""
    try:
        with open(hindcast_path, "r") as f:
            return json.loads(f.read())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def load_fit(fit_path):
    """Load fit.json, return dict."""
    try:
        with open(fit_path, "r") as f:
            return json.loads(f.read())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(trace, hindcast):
    """Compute the four key metrics from trace and hindcast data.

    Returns dict with: final_mape, convergence_iter, improvement_rate, dead_end_rate
    or None if insufficient data.
    """
    if not trace and not hindcast:
        return None

    # Final MAPE: minimum mape from hindcast (or trace as fallback)
    final_mape = None
    if hindcast:
        mapes = [h["mape"] for h in hindcast if "mape" in h and h["mape"] is not None]
        if mapes:
            final_mape = min(mapes)

    if final_mape is None and trace:
        mapes = [t["mape"] for t in trace if "mape" in t and t["mape"] is not None]
        if mapes:
            final_mape = min(mapes)

    if final_mape is None:
        return None

    # Convergence iteration: version at which final best MAPE was first reached
    convergence_iter = None
    if trace:
        for entry in trace:
            mape_val = entry.get("mape")
            if mape_val is not None and abs(mape_val - final_mape) < 1e-10:
                convergence_iter = entry.get("version", 1)
                break
    if convergence_iter is None and hindcast:
        for entry in hindcast:
            mape_val = entry.get("mape")
            if mape_val is not None and abs(mape_val - final_mape) < 1e-10:
                convergence_iter = entry.get("version", 1)
                break

    # MAPE improvement rate: (baseline_mape - final_mape) / n_iterations
    improvement_rate = None
    if trace:
        n_iterations = len(trace)
        # Baseline mape is the first entry's mape (version 1)
        baseline_mape = None
        for entry in trace:
            if entry.get("version", 0) == 1 and "mape" in entry:
                baseline_mape = entry["mape"]
                break
        if baseline_mape is None and hindcast:
            for entry in hindcast:
                if entry.get("version", 0) == 1 and "mape" in entry:
                    baseline_mape = entry["mape"]
                    break
        if baseline_mape is None and trace:
            # Fall back to first trace entry
            baseline_mape = trace[0].get("mape")
        if baseline_mape is not None and n_iterations > 0:
            improvement_rate = (baseline_mape - final_mape) / n_iterations

    # Dead-end rate: fraction of iterations that did not improve
    dead_end_rate = None
    if trace:
        total = len(trace)
        not_improved = sum(1 for t in trace if t.get("improved") is False)
        if total > 0:
            dead_end_rate = not_improved / total

    n_iterations = len(trace) if trace else (len(hindcast) if hindcast else 0)

    return {
        "final_mape": final_mape,
        "convergence_iter": convergence_iter,
        "improvement_rate": improvement_rate,
        "dead_end_rate": dead_end_rate,
        "n_iterations": n_iterations,
        "baseline_mape": baseline_mape if trace and baseline_mape is not None else None,
    }


def build_mape_curve(trace):
    """Build running-minimum MAPE curve from trace entries.

    Returns (versions, running_min_mapes) as lists.
    """
    if not trace:
        return [], []

    versions = []
    running_min = []
    best_so_far = float("inf")
    for entry in trace:
        v = entry.get("version", len(versions) + 1)
        m = entry.get("mape")
        if m is not None:
            best_so_far = min(best_so_far, m)
            versions.append(v)
            running_min.append(best_so_far)
    return versions, running_min


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_results(results_dir):
    """Scan results_dir/{condition}/rep{N}/{well}/ and return list of row dicts."""
    results_dir = Path(results_dir)
    rows = []

    for condition in ("claude", "random"):
        condition_dir = results_dir / condition
        if not condition_dir.is_dir():
            print(f"Warning: {condition_dir} not found, skipping.")
            continue

        for rep_dir in sorted(condition_dir.iterdir()):
            if not rep_dir.is_dir():
                continue
            rep_name = rep_dir.name  # e.g. "rep1"

            for well_dir in sorted(rep_dir.iterdir()):
                if not well_dir.is_dir():
                    continue
                well = well_dir.name

                trace = load_trace(well_dir / "trace.jsonl")
                hindcast = load_hindcast(well_dir / "hindcast.json")
                fit = load_fit(well_dir / "fit.json")

                metrics = compute_metrics(trace, hindcast)
                if metrics is None:
                    print(f"Warning: no usable data for {condition}/{rep_name}/{well}, skipping.")
                    continue

                rows.append({
                    "condition": condition,
                    "rep": rep_name,
                    "well": well,
                    "final_mape": metrics["final_mape"],
                    "convergence_iter": metrics["convergence_iter"],
                    "improvement_rate": metrics["improvement_rate"],
                    "dead_end_rate": metrics["dead_end_rate"],
                    "n_iterations": metrics["n_iterations"],
                    "baseline_mape": metrics["baseline_mape"],
                    "r2_insample": fit.get("r2_insample"),
                    "r2_outsample": fit.get("r2_outsample"),
                })

    return rows


def collect_curves(results_dir):
    """Collect MAPE-vs-iteration curves grouped by condition.

    Returns dict: condition -> list of (versions, running_min_mapes) tuples.
    """
    results_dir = Path(results_dir)
    curves = {"claude": [], "random": []}

    for condition in ("claude", "random"):
        condition_dir = results_dir / condition
        if not condition_dir.is_dir():
            continue
        for rep_dir in sorted(condition_dir.iterdir()):
            if not rep_dir.is_dir():
                continue
            for well_dir in sorted(rep_dir.iterdir()):
                if not well_dir.is_dir():
                    continue
                trace = load_trace(well_dir / "trace.jsonl")
                vs, ms = build_mape_curve(trace)
                if vs:
                    curves[condition].append((vs, ms))

    return curves


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def write_summary_csv(rows, output_path):
    """Write rows to a flat CSV."""
    if not rows:
        print("No data to write to summary CSV.")
        return

    fieldnames = [
        "condition", "rep", "well", "final_mape", "convergence_iter",
        "improvement_rate", "dead_end_rate", "n_iterations", "baseline_mape",
        "r2_insample", "r2_outsample",
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Summary CSV written to {output_path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_boxplot(rows, output_path):
    """Boxplot of final MAPE by condition."""
    claude_mapes = [r["final_mape"] for r in rows if r["condition"] == "claude"]
    random_mapes = [r["final_mape"] for r in rows if r["condition"] == "random"]

    if not claude_mapes and not random_mapes:
        print("No data for boxplot.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    data = []
    labels = []
    colors = []
    if claude_mapes:
        data.append(claude_mapes)
        labels.append(f"Claude (n={len(claude_mapes)})")
        colors.append("#4C72B0")
    if random_mapes:
        data.append(random_mapes)
        labels.append(f"Random (n={len(random_mapes)})")
        colors.append("#DD8452")

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual points
    for i, d in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.5, s=20, color="black", zorder=3)

    ax.set_ylabel("Final Hindcast MAPE")
    ax.set_title("Ablation: Final MAPE by Condition")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Boxplot saved to {output_path}")


def plot_convergence(curves, output_path):
    """Convergence plot: running-min MAPE vs iteration, averaged across runs."""
    fig, ax = plt.subplots(figsize=(8, 5))

    condition_styles = {
        "claude": {"color": "#4C72B0", "label": "Claude"},
        "random": {"color": "#DD8452", "label": "Random"},
    }

    for condition, curve_list in curves.items():
        if not curve_list:
            continue

        style = condition_styles[condition]

        # Find max iteration across all curves for this condition
        max_iter = max(max(vs) for vs, _ in curve_list)

        # Interpolate all curves onto a common iteration axis
        iters = np.arange(1, max_iter + 1)
        interpolated = []

        for vs, ms in curve_list:
            if len(vs) < 2:
                # Single point: extend as constant
                interp_vals = np.full_like(iters, ms[0], dtype=float)
            else:
                interp_vals = np.interp(iters, vs, ms)
            interpolated.append(interp_vals)

        interpolated = np.array(interpolated)
        mean_curve = np.mean(interpolated, axis=0)
        std_curve = np.std(interpolated, axis=0)

        ax.plot(iters, mean_curve, color=style["color"], label=style["label"], linewidth=2)
        ax.fill_between(
            iters,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=style["color"],
            alpha=0.15,
        )

    ax.set_xlabel("Iteration (version)")
    ax.set_ylabel("Best MAPE so far (running minimum)")
    ax.set_title("Convergence: MAPE vs Iteration")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Convergence plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_statistical_tests(rows):
    """Run Wilcoxon signed-rank and Mann-Whitney U tests. Print results."""
    claude_rows = [r for r in rows if r["condition"] == "claude"]
    random_rows = [r for r in rows if r["condition"] == "random"]

    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    print(f"\nSample sizes: Claude={len(claude_rows)}, Random={len(random_rows)}")

    # Descriptive stats
    for label, subset in [("Claude", claude_rows), ("Random", random_rows)]:
        if not subset:
            print(f"\n{label}: no data")
            continue
        mapes = [r["final_mape"] for r in subset]
        conv = [r["convergence_iter"] for r in subset if r["convergence_iter"] is not None]
        dead = [r["dead_end_rate"] for r in subset if r["dead_end_rate"] is not None]
        impr = [r["improvement_rate"] for r in subset if r["improvement_rate"] is not None]

        print(f"\n{label}:")
        print(f"  Final MAPE     -- mean={np.mean(mapes):.4f}, median={np.median(mapes):.4f}, "
              f"std={np.std(mapes):.4f}")
        if conv:
            print(f"  Convergence    -- mean={np.mean(conv):.1f}, median={np.median(conv):.1f}")
        if dead:
            print(f"  Dead-end rate  -- mean={np.mean(dead):.3f}, median={np.median(dead):.3f}")
        if impr:
            print(f"  Improvement/it -- mean={np.mean(impr):.6f}, median={np.median(impr):.6f}")

    # ---- Paired Wilcoxon signed-rank on final MAPE ----
    print("\n" + "-" * 60)
    print("TEST 1: Wilcoxon signed-rank test on paired final MAPEs")
    print("-" * 60)

    # Build paired data: match by (well, rep)
    claude_map = {(r["well"], r["rep"]): r["final_mape"] for r in claude_rows}
    random_map = {(r["well"], r["rep"]): r["final_mape"] for r in random_rows}
    paired_keys = sorted(set(claude_map.keys()) & set(random_map.keys()))

    if len(paired_keys) < 6:
        print(f"Only {len(paired_keys)} paired observations (need >= 6 for Wilcoxon).")
        print("Cannot run Wilcoxon signed-rank test with so few pairs.")
        wilcoxon_p = None
        effect_size_r = None
    else:
        claude_paired = np.array([claude_map[k] for k in paired_keys])
        random_paired = np.array([random_map[k] for k in paired_keys])
        diffs = claude_paired - random_paired

        # Check if all differences are zero
        if np.all(diffs == 0):
            print("All paired differences are zero -- no test possible.")
            wilcoxon_p = None
            effect_size_r = None
        else:
            try:
                w_stat, w_p = stats.wilcoxon(claude_paired, random_paired, alternative="two-sided")
                n_pairs = len(paired_keys)
                # Rank-biserial correlation: r = 1 - (2*W) / (n*(n+1)/2)
                effect_size_r = 1.0 - (2.0 * w_stat) / (n_pairs * (n_pairs + 1) / 2.0)

                print(f"Paired observations: {n_pairs}")
                print(f"Mean diff (Claude - Random): {np.mean(diffs):.6f}")
                print(f"Median diff: {np.median(diffs):.6f}")
                print(f"Wilcoxon W = {w_stat:.1f}, p = {w_p:.4f}")
                print(f"Rank-biserial r = {effect_size_r:.4f}")
                if w_p < 0.05:
                    direction = "lower" if np.median(diffs) < 0 else "higher"
                    print(f"Result: SIGNIFICANT (p < 0.05). Claude MAPEs are {direction}.")
                else:
                    print("Result: NOT significant (p >= 0.05).")

                wilcoxon_p = w_p
            except ValueError as e:
                print(f"Wilcoxon test failed: {e}")
                wilcoxon_p = None
                effect_size_r = None

    # ---- Mann-Whitney U on convergence iterations ----
    print("\n" + "-" * 60)
    print("TEST 2: Mann-Whitney U test on convergence iterations")
    print("-" * 60)

    claude_conv = [r["convergence_iter"] for r in claude_rows if r["convergence_iter"] is not None]
    random_conv = [r["convergence_iter"] for r in random_rows if r["convergence_iter"] is not None]

    mw_p = None
    if len(claude_conv) < 3 or len(random_conv) < 3:
        print(f"Claude n={len(claude_conv)}, Random n={len(random_conv)} -- too few for Mann-Whitney.")
    else:
        try:
            u_stat, mw_p = stats.mannwhitneyu(claude_conv, random_conv, alternative="two-sided")
            print(f"Claude convergence: median={np.median(claude_conv):.1f}, "
                  f"mean={np.mean(claude_conv):.1f}")
            print(f"Random convergence: median={np.median(random_conv):.1f}, "
                  f"mean={np.mean(random_conv):.1f}")
            print(f"Mann-Whitney U = {u_stat:.1f}, p = {mw_p:.4f}")
            if mw_p < 0.05:
                faster = "Claude" if np.median(claude_conv) < np.median(random_conv) else "Random"
                print(f"Result: SIGNIFICANT (p < 0.05). {faster} converges faster.")
            else:
                print("Result: NOT significant (p >= 0.05).")
        except ValueError as e:
            print(f"Mann-Whitney test failed: {e}")

    # ---- Interpretation ----
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    print()
    print("Reference table:")
    print("  Claude sig. better + faster convergence  -> LLM reasoning is contributing")
    print("  Claude sig. better + similar convergence  -> LLM finds better params, not faster")
    print("  No significant difference                 -> Loop structure drives improvement, not reasoning")
    print("  Random sig. better                        -> Agent is overcorrecting or getting stuck")
    print()

    if wilcoxon_p is not None and mw_p is not None:
        claude_better = wilcoxon_p < 0.05 and np.median(diffs) < 0
        random_better = wilcoxon_p < 0.05 and np.median(diffs) > 0
        faster_conv = mw_p < 0.05 and np.median(claude_conv) < np.median(random_conv)

        if claude_better and faster_conv:
            verdict = "LLM reasoning IS contributing -- Claude achieves better MAPE and converges faster."
        elif claude_better and not faster_conv:
            verdict = "LLM finds better parameters but not faster -- reasoning helps quality, not speed."
        elif random_better:
            verdict = "Random search outperforms Claude -- agent may be overcorrecting or getting stuck."
        else:
            verdict = "No significant difference -- the optimization loop structure drives improvement, not LLM reasoning."
    elif wilcoxon_p is not None:
        claude_better = wilcoxon_p < 0.05 and np.median(diffs) < 0
        random_better = wilcoxon_p < 0.05 and np.median(diffs) > 0
        if claude_better:
            verdict = "Claude achieves significantly better MAPE (convergence test inconclusive)."
        elif random_better:
            verdict = "Random search outperforms Claude on MAPE (convergence test inconclusive)."
        else:
            verdict = "No significant difference in MAPE -- loop structure likely drives improvement, not reasoning."
    else:
        verdict = "Insufficient data for statistical conclusions. Collect more runs."

    print(f"VERDICT: {verdict}")

    # Power warning
    n_total = len(paired_keys) if 'paired_keys' in dir() else 0
    if n_total > 0 and n_total < 20:
        print(f"\nWARNING: Only {n_total} paired observations. "
              f"Statistical power is low -- interpret with caution.")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze ablation experiment: Claude vs random search for decline-curve tuning."
    )
    parser.add_argument(
        "--results-dir",
        default="ablation_results",
        help="Path to results directory (default: ablation_results/)",
    )
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    if not results_dir.is_dir():
        print(f"Error: results directory '{results_dir}' not found.")
        print("Run the ablation experiment first, or specify --results-dir.")
        sys.exit(1)

    # Collect data
    print("Collecting results...")
    rows = collect_results(results_dir)
    if not rows:
        print("No valid results found. Exiting.")
        sys.exit(1)

    print(f"Found {len(rows)} valid (condition, rep, well) results.\n")

    # Write summary CSV
    write_summary_csv(rows, results_dir / "summary.csv")

    # Collect convergence curves
    curves = collect_curves(results_dir)

    # Generate plots
    plot_boxplot(rows, results_dir / "boxplot_mape.png")
    plot_convergence(curves, results_dir / "convergence.png")

    # Statistical analysis
    run_statistical_tests(rows)


if __name__ == "__main__":
    main()
