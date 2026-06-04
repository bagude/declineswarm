#!/usr/bin/env python
"""Runnable example for the segmented_decline package.

Generates three synthetic wells (single-segment power law, two-segment power
law, shifted-clock power law), adds multiplicative lognormal noise, fits each
with the segmented decline fitter, prints a summary, and saves diagnostic plots.

Usage:
    python scripts/run_segmented_decline.py
    python scripts/run_segmented_decline.py --output-dir /tmp/seg_out --model shifted_power
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running directly from the repo without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from segmented_decline import (  # noqa: E402
    SegmentedDeclineConfig,
    fit_segmented_decline,
    summarize_fit,
    segment_table,
)
from segmented_decline.examples import generate_example_wells  # noqa: E402
from segmented_decline.plotting import summary_figure  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=os.path.join("outputs", "segmented_decline"),
                        help="Where to write plots (created if missing).")
    parser.add_argument("--model", default="shifted_power",
                        choices=["global_power", "shifted_power", "restarted_hyperbolic", "exponential"])
    parser.add_argument("--criterion", default="bic", choices=["bic", "aic"])
    parser.add_argument("--max-segments", type=int, default=4)
    parser.add_argument("--min-segment-length", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = SegmentedDeclineConfig(
        model_type=args.model,
        criterion=args.criterion,
        max_segments=args.max_segments,
        min_segment_length=args.min_segment_length,
    )

    wells = generate_example_wells(seed=args.seed)
    for well in wells:
        fit = fit_segmented_decline(well.t, well.q, config, well_id=well.well_id)
        summary = summarize_fit(fit)

        print("=" * 70)
        print(f"well: {well.well_id}    (truth: {well.truth})")
        print(f"  selected K = {summary['selected_n_segments']}  "
              f"model = {summary['selected_model_type']}  criterion = {summary['criterion']}")
        print(f"  breakpoints_t = {[round(b, 1) for b in summary['breakpoints_t']]}")
        print(f"  RSS = {summary['rss']:.4f}   AIC = {summary['aic']:.2f}   BIC = {summary['bic']:.2f}")
        print("  per-K trace:")
        for r in fit.k_results:
            print(f"    K={r['k']}  RSS={r['rss']:.4f}  AIC={r['aic']:.2f}  BIC={r['bic']:.2f}")
        print("  segments:")
        print(segment_table(fit).to_string(index=False))

        fig = summary_figure(well.t, well.q, fit)
        out_path = os.path.join(args.output_dir, f"{well.well_id}.png")
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        print(f"  saved plot -> {out_path}")

    print("=" * 70)
    print(f"Done. Plots in: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
