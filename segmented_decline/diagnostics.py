"""Summaries and batch fitting for segmented decline fits."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import SegmentedDeclineConfig, SegmentedDeclineFit, SegmentFit
from .dp import fit_segmented_decline


def _segment_arps(seg: SegmentFit):
    """Return (alpha, c, b_equiv, D_equiv) for a segment, or Nones if N/A."""
    p = seg.params
    alpha = p.get("alpha")
    c = p.get("c")
    b_eq = p.get("b_equiv")
    D_eq = p.get("D_equiv")
    # D_equiv only valid when the offset (T + c) is positive.
    if D_eq is not None and not np.isfinite(D_eq):
        D_eq = None
    return alpha, c, b_eq, D_eq


def segment_table(fit: SegmentedDeclineFit) -> pd.DataFrame:
    """One row per segment with start/end, params, and Arps-equivalent values."""
    rows = []
    for j, seg in enumerate(fit.segments):
        alpha, c, b_eq, D_eq = _segment_arps(seg)
        rows.append({
            "well_id": fit.well_id,
            "segment": j,
            "model_type": seg.model_type,
            "start_idx": seg.start_idx,
            "end_idx": seg.end_idx,
            "segment_start_t": seg.start_t,
            "segment_end_t": seg.end_t,
            "segment_n_points": seg.n,
            "alpha": alpha,
            "c": c,
            "equivalent_b": b_eq,
            "equivalent_D": D_eq,
            "M": seg.params.get("M"),
            "segment_rmse_log": seg.rmse,
        })
    return pd.DataFrame(rows)


def summarize_fit(fit: SegmentedDeclineFit) -> dict:
    """Flat well-level summary dict matching the requested output schema."""
    segs = fit.segments
    arps = [_segment_arps(s) for s in segs]
    return {
        "well_id": fit.well_id,
        "selected_model_type": fit.model_type,
        "selected_n_segments": fit.n_segments,
        "criterion": fit.criterion,
        "rss": fit.rss,
        "aic": fit.aic,
        "bic": fit.bic,
        "breakpoints_t": list(fit.breakpoints_t),
        "segment_start_t": [s.start_t for s in segs],
        "segment_end_t": [s.end_t for s in segs],
        "alpha": [a[0] for a in arps],
        "c": [a[1] for a in arps],
        "equivalent_b": [a[2] for a in arps],
        "equivalent_D": [a[3] for a in arps],
        "segment_rmse_log": [s.rmse for s in segs],
        "segment_n_points": [s.n for s in segs],
    }


def fit_many_wells(
    df: pd.DataFrame,
    well_col: str,
    t_col: str,
    q_col: str,
    config: Optional[SegmentedDeclineConfig] = None,
) -> tuple[pd.DataFrame, dict[str, SegmentedDeclineFit]]:
    """Fit every well in a long DataFrame.

    Expects already-clean columns (see prepare_production_data). Returns a
    one-row-per-well summary DataFrame plus a dict of detailed fit objects keyed
    by well id. Wells that fail to fit are skipped with their error recorded.
    """
    config = config or SegmentedDeclineConfig()
    summaries = []
    fits: dict[str, SegmentedDeclineFit] = {}

    for well_id, g in df.groupby(well_col, sort=False):
        g = g.sort_values(t_col)
        t = g[t_col].to_numpy(dtype=float)
        q = g[q_col].to_numpy(dtype=float)
        try:
            fit = fit_segmented_decline(t, q, config, well_id=str(well_id))
        except ValueError as exc:
            summaries.append({"well_id": str(well_id), "error": str(exc)})
            continue
        fits[str(well_id)] = fit
        summaries.append(summarize_fit(fit))

    return pd.DataFrame(summaries), fits
