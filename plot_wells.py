"""Generate per-well hindcast figures in journal-ready style.

Usage:
    python plot_wells.py TX                              # all TX wells
    python plot_wells.py NM                              # all NM wells
    python plot_wells.py TX 42-001-32769                 # single well
    python plot_wells.py TX 42-001-32769 42-003-00377    # multiple wells

Output: plots/{state}/{api}.png
"""

import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import textwrap

from arps import create_model, generate_forecast
from preprocessing import detect_decline_start, filter_outliers
from arps import fit_arps


# Journal style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "grid.linewidth": 0.3,
    "lines.linewidth": 1.0,
})

BASELINE_PARAMS = {
    "version": 1,
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
        "di_upper_bound": 5.0,
        "b_upper_bound": 2.0,
    },
}

PRE_KEYS = {
    "significance_threshold", "smoothing_window", "outlier_window",
    "outlier_threshold", "peak_merge_distance", "min_clean_months",
}
FIT_KEYS = {
    "d_min", "di_initial", "b_initial", "qi_guess_strategy",
    "qi_multiplier_upper", "di_upper_bound", "b_upper_bound",
}

# Distinct blue ramp for kept versions (light → dark)
KEPT_COLORS = [
    "#aec6cf", "#7fb3d3", "#5299c6", "#2b7bba",
    "#1a5e99", "#0e4478", "#08306b", "#041a3d",
]
# Rejected versions: red shades
REJECTED_COLOR = "#d94040"
# Best/final version: dark navy, always most prominent
BEST_COLOR = "#08306b"


def sanitize_text(s):
    """Replace em-dashes and other problematic unicode with ASCII equivalents."""
    if not isinstance(s, str):
        return str(s)
    return (s
            .replace("\u2014", "--")   # em-dash
            .replace("\u2013", "-")    # en-dash
            .replace("\u2018", "'")    # left single quote
            .replace("\u2019", "'")    # right single quote
            .replace("\u201c", '"')    # left double quote
            .replace("\u201d", '"')    # right double quote
            .replace("\u2026", "...")  # ellipsis
            )


def read_production(csv_path):
    dates, oil = [], []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dates.append(row["date"])
            oil.append(float(row["oil_bbl"]))
    return dates, oil


def read_trace(trace_path):
    entries = []
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def is_improved(entry):
    """Handle multiple trace formats."""
    if "improved" in entry:
        return entry["improved"] is True
    if "result" in entry:
        return entry["result"] == "improved"
    return False


def get_mape(entry):
    """Extract MAPE from various trace formats."""
    if "mape_after" in entry:
        return entry["mape_after"]
    if "hindcast_mape" in entry:
        return entry["hindcast_mape"]
    return None


def get_note(entry):
    return sanitize_text(entry.get("note", entry.get("observation", "")))


def get_param_change(entry):
    """Extract what parameter was changed and to what value."""
    if "parameter" in entry:
        param = entry["parameter"]
        val = entry.get("to", entry.get("value", "?"))
        return param, val

    for key in PRE_KEYS | FIT_KEYS:
        if key in entry and key not in ("version", "improved", "note", "observation",
                                         "mape_before", "mape_after", "result",
                                         "hindcast_mape", "r2_outsample", "parameter",
                                         "from", "to", "value", "param"):
            return key, entry[key]

    if "param" in entry:
        return entry["param"], entry.get("to", entry.get("value", "?"))

    return None, None


def reconstruct_all_versions(trace_entries):
    """Reconstruct params for EVERY version (improved and rejected).

    Returns list of (version, params_dict, improved_bool, mape, note).
    """
    committed = deepcopy(BASELINE_PARAMS)
    all_versions = []

    v1_mape = None
    v1_note = "Baseline global defaults"
    for e in trace_entries:
        if e.get("version") == 1:
            v1_mape = get_mape(e)
            v1_note = get_note(e) or v1_note
            break
    all_versions.append((1, deepcopy(committed), True, v1_mape, v1_note))

    for entry in trace_entries:
        ver = entry.get("version", "?")
        if ver == 1:
            continue

        improved = is_improved(entry)
        mape = get_mape(entry)
        note = get_note(entry)

        trial = deepcopy(committed)
        trial["version"] = ver

        param, val = get_param_change(entry)
        if param and val is not None and param != "baseline":
            if isinstance(val, str) and param != "qi_guess_strategy":
                pass
            elif param in PRE_KEYS:
                trial["preprocessing"][param] = val
            elif param in FIT_KEYS:
                trial["fitting"][param] = val

        all_versions.append((ver, deepcopy(trial), improved, mape, note))

        if improved:
            committed = deepcopy(trial)

    return all_versions


def generate_version_forecast(params, oil_train, holdout_months):
    """Fit and forecast for a given param set."""
    pre = params.get("preprocessing", {})
    fit_params = params.get("fitting", {})

    try:
        anchor = detect_decline_start(
            oil_train,
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
            di_upper_bound=fit_params.get("di_upper_bound", 5.0),
            b_upper_bound=fit_params.get("b_upper_bound", 2.0),
        )
    except (ValueError, RuntimeError):
        return None, None

    model = create_model(
        decline_model=fit_result["decline_model"],
        qi_oil=fit_result["qi"],
        di=fit_result["di"],
        b=fit_result["b"],
        d_min=fit_result["d_min"],
    )

    total_needed = len(oil_train) - anchor.peak_index + holdout_months + 12
    fc = generate_forecast(model, months=max(total_needed, 120))

    forecast_oil = []
    for i in range(len(oil_train) + holdout_months):
        fc_idx = i - anchor.peak_index
        if 0 <= fc_idx < len(fc):
            forecast_oil.append(fc[fc_idx].oil_bbl)
        else:
            forecast_oil.append(None)

    return forecast_oil, anchor.peak_index


def _fmt_mape(mape):
    """Safely format a MAPE value that may be numeric or string."""
    if mape is None:
        return "N/A"
    try:
        return f"{float(mape):.4f}"
    except (ValueError, TypeError):
        return str(mape)


def plot_well(state, api, output_dir="plots"):
    well_dir = Path("wells") / state / api
    if not well_dir.exists():
        print(f"  {state}/{api}: directory not found, skipping")
        return

    production_csv = well_dir / "production.csv"
    trace_path = well_dir / "trace.jsonl"
    hindcast_path = well_dir / "hindcast.json"
    params_path = well_dir / "params.json"

    if not production_csv.exists():
        print(f"  {state}/{api}: no production.csv, skipping")
        return

    dates, oil = read_production(production_csv)
    total_months = len(oil)

    if total_months < 18:
        print(f"  {state}/{api}: only {total_months} months, skipping")
        return

    holdout = 12
    train_months = total_months - holdout
    oil_train = oil[:train_months]
    oil_holdout = oil[train_months:]

    trace_entries = read_trace(trace_path) if trace_path.exists() else []
    hindcast = json.loads(hindcast_path.read_text(encoding="utf-8")) if hindcast_path.exists() else []
    params = json.loads(params_path.read_text(encoding="utf-8")) if params_path.exists() else {}

    all_versions = reconstruct_all_versions(trace_entries)

    # Separate kept vs rejected for color assignment
    kept_indices = [i for i, (_, _, imp, _, _) in enumerate(all_versions) if imp]
    rejected_indices = [i for i, (_, _, imp, _, _) in enumerate(all_versions) if not imp]

    # --- Figure layout: taller to give text panel room ---
    n_trace = len(all_versions)
    fig_height = max(6.5, 5.5 + n_trace * 0.12)
    fig = plt.figure(figsize=(13, fig_height), facecolor="white")
    gs = gridspec.GridSpec(
        1, 2, width_ratios=[3, 2], wspace=0.03,
        left=0.06, right=0.97, top=0.92, bottom=0.10,
    )
    ax = fig.add_subplot(gs[0])
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis("off")

    month_indices = np.arange(total_months)

    tick_step = max(1, total_months // 8)
    tick_positions = list(range(0, total_months, tick_step))
    tick_labels = [dates[i] for i in tick_positions]

    # --- Production data ---
    ax.plot(
        month_indices[:train_months], oil[:train_months],
        color="#2ca02c", linewidth=1.3, zorder=10, label="Observed (training)",
    )

    holdout_x = month_indices[train_months - 1:]
    holdout_y = [oil[train_months - 1]] + oil_holdout
    ax.plot(
        holdout_x, holdout_y,
        color="#7f7f7f", linewidth=1.3, linestyle="--", zorder=10,
        label="Observed (holdout)",
    )

    ax.axvline(
        x=train_months - 0.5, color="#aaaaaa", linewidth=0.6,
        linestyle=":", zorder=2,
    )

    # --- Forecast lines for ALL versions from trace ---
    # Assign colors: kept versions get sequential blues, rejected get red
    kept_count = 0
    best_ver = params.get("version")
    forecast_data = []  # (ver, fc_x, fc_y, color, ls, lw, alpha, improved, mape)

    # Track which versions we've plotted so we can detect if best is missing
    plotted_versions = set()

    for idx, (ver, vparams, improved, mape, note) in enumerate(all_versions):
        # Skip the best version here — we plot it separately below
        if ver == best_ver:
            plotted_versions.add(ver)
            continue

        fc_oil, peak_idx = generate_version_forecast(vparams, oil_train, holdout)
        if fc_oil is None:
            continue

        plotted_versions.add(ver)

        if improved:
            ci = min(kept_count, len(KEPT_COLORS) - 2)  # reserve darkest for best
            color = KEPT_COLORS[ci]
            kept_count += 1
            ls = "-"
            lw = 1.0
            alpha = 0.7
        else:
            color = REJECTED_COLOR
            ls = ":"
            lw = 0.7
            alpha = 0.4

        fc_x, fc_y = [], []
        for j in range(total_months):
            if fc_oil[j] is not None:
                fc_x.append(j)
                fc_y.append(fc_oil[j])

        forecast_data.append((ver, fc_x, fc_y, color, ls, lw, alpha, improved, mape))

    # --- Always plot the BEST version from current params.json ---
    # This is independent of trace reconstruction — uses the actual stored params
    best_fc_oil, best_peak = generate_version_forecast(params, oil_train, holdout)
    best_mape_val = hindcast[-1]["mape"] if hindcast else None
    best_fc_x, best_fc_y = [], []
    if best_fc_oil is not None:
        for j in range(total_months):
            if best_fc_oil[j] is not None:
                best_fc_x.append(j)
                best_fc_y.append(best_fc_oil[j])

    # Plot: rejected first (back), then kept iterations, then best on top
    for ver, fc_x, fc_y, color, ls, lw, alpha, improved, mape in forecast_data:
        if not improved:
            ax.plot(fc_x, fc_y, color=color, linewidth=lw, alpha=alpha,
                    linestyle=ls, zorder=3)
    for ver, fc_x, fc_y, color, ls, lw, alpha, improved, mape in forecast_data:
        if improved:
            ax.plot(fc_x, fc_y, color=color, linewidth=lw, alpha=alpha,
                    linestyle=ls, zorder=5)

    # Best version: always on top, always prominent
    if best_fc_x:
        ax.plot(best_fc_x, best_fc_y, color=BEST_COLOR, linewidth=2.5,
                alpha=1.0, linestyle="-", zorder=8)

    # --- Version labels directly on curves ---
    for ver, fc_x, fc_y, color, ls, lw, alpha, improved, mape in forecast_data:
        if not fc_x:
            continue
        lx, ly = fc_x[-1], fc_y[-1]
        ax.annotate(
            f"v{ver}",
            xy=(lx, ly),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=5.5,
            fontweight="normal",
            color=color,
            alpha=min(alpha + 0.2, 1.0),
            va="center",
            ha="left",
            zorder=12,
        )

    # Best version label — always present, always bold
    if best_fc_x:
        ax.annotate(
            f"v{best_ver}",
            xy=(best_fc_x[-1], best_fc_y[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color=BEST_COLOR,
            alpha=1.0,
            va="center",
            ha="left",
            zorder=13,
        )

    # --- Axis styling ---
    well_type = params.get("well_type", "unknown")
    final_ver = params.get("version", "?")
    best_mape = hindcast[-1]["mape"] if hindcast else "N/A"

    ax.set_title(
        f"{api}  ({well_type})",
        fontweight="bold", fontsize=11, pad=8,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Oil production (bbl/month)")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.grid(True, axis="both", color="#e0e0e0", linewidth=0.3, zorder=0)
    ax.set_xlim(-1, total_months + 3)  # extra room for version labels
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Legend: compact, shows encoding scheme ---
    legend_handles = [
        Line2D([0], [0], color="#2ca02c", linewidth=1.3, label="Observed (training)"),
        Line2D([0], [0], color="#7f7f7f", linewidth=1.3, linestyle="--",
               label="Observed (holdout)"),
        Line2D([0], [0], color=KEPT_COLORS[0], linewidth=1.0, linestyle="-",
               label="Kept iterations (solid, blue)"),
        Line2D([0], [0], color=REJECTED_COLOR, linewidth=0.7, linestyle=":",
               alpha=0.6, label="Rejected iterations (dotted, red)"),
        Line2D([0], [0], color=BEST_COLOR, linewidth=2.5, linestyle="-",
               label=f"Best: v{final_ver}, MAPE={_fmt_mape(best_mape)}"),
    ]

    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9,
              edgecolor="#cccccc", fancybox=False)

    # --- Right panel: trace table + full reasoning ---
    lines = []
    lines.append(f"{'v':>3s}  {'MAPE':>7s}  {'':>1s}  Parameter change")
    lines.append("-" * 55)

    for ver, vparams, improved, mape, note in all_versions:
        marker = "+" if improved else "x"
        mape_str = _fmt_mape(mape).rjust(7)

        param_change = ""
        if ver == 1:
            param_change = "baseline defaults"
        else:
            for e in trace_entries:
                if e.get("version") == ver:
                    p, val = get_param_change(e)
                    if p and p != "baseline":
                        param_change = f"{p}={val}"
                    break

        lines.append(f" {ver:>2}   {mape_str}  {marker}  {param_change}")

    lines.append("-" * 55)
    lines.append(f"Best: v{final_ver}, MAPE={_fmt_mape(best_mape)}")
    lines.append("")

    # Full reasoning for improved iterations
    improved_versions = [(v, n) for v, _, imp, _, n in all_versions
                         if imp and v != 1 and n]
    if improved_versions:
        lines.append("Kept iterations:")
        lines.append("")
        for ver, note in improved_versions:
            wrapped = textwrap.wrap(note, width=58)
            lines.append(f"  v{ver}: {wrapped[0]}" if wrapped else f"  v{ver}:")
            for wl in wrapped[1:]:
                lines.append(f"      {wl}")
            lines.append("")

    # Full reasoning for rejected iterations
    rejected_versions = [(v, n) for v, _, imp, _, n in all_versions
                         if not imp and v != 1 and n]
    if rejected_versions:
        lines.append("Rejected iterations:")
        lines.append("")
        for ver, note in rejected_versions:
            wrapped = textwrap.wrap(note, width=58)
            lines.append(f"  v{ver}: {wrapped[0]}" if wrapped else f"  v{ver}:")
            for wl in wrapped[1:]:
                lines.append(f"      {wl}")
            lines.append("")

    trace_text = "\n".join(lines)

    ax_text.text(
        0.02, 0.98, trace_text,
        transform=ax_text.transAxes,
        fontsize=6, fontfamily="monospace",
        verticalalignment="top",
        linespacing=1.2,
        color="#333333",
    )

    ax_text.set_title(
        f"Optimization trace  |  Best MAPE: {_fmt_mape(best_mape)}",
        fontsize=9, fontweight="bold", pad=8,
    )

    out_dir = Path(output_dir) / state
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{api}.png"
    fig.savefig(out_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  {state}/{api}: saved {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_wells.py <state> [api ...]")
        print("  python plot_wells.py TX")
        print("  python plot_wells.py NM 30-025-50075")
        sys.exit(1)

    state = sys.argv[1]
    if len(sys.argv) > 2:
        apis = sys.argv[2:]
    else:
        state_dir = Path("wells") / state
        if not state_dir.exists():
            print(f"Error: {state_dir} does not exist")
            sys.exit(1)
        apis = sorted(p.name for p in state_dir.iterdir() if p.is_dir())

    print(f"Generating figures for {state} -- {len(apis)} wells...")
    for api in apis:
        plot_well(state, api)
    print("Done.")


if __name__ == "__main__":
    main()
