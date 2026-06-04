"""Production-data preparation for segmented decline analysis.

Turns a raw production DataFrame into a clean (well_id, t, q[, date]) frame that
the fitter can consume. Works on *rate*, not raw volume; rate can be supplied
directly or computed from volume / days_produced.

Because the power-law form uses log(t / T_j), both t and T_j must be positive.
We therefore default to a one-based month index (t = 1, 2, 3, ...).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

DAYS_PER_MONTH = 30.4375


def prepare_production_data(
    df: pd.DataFrame,
    *,
    well_col: Optional[str] = None,
    date_col: Optional[str] = None,
    month_index_col: Optional[str] = None,
    rate_col: Optional[str] = None,
    volume_col: Optional[str] = None,
    days_col: Optional[str] = None,
    normalize_monthly: bool = False,
    one_based: bool = True,
    min_rate: float = 1e-6,
    drop_invalid: bool = True,
) -> pd.DataFrame:
    """Clean and standardize a production DataFrame.

    Rate source (in priority order):
      1. ``rate_col`` — use directly.
      2. ``volume_col`` / ``days_col`` — rate = volume / days_produced, optionally
         normalized to a 30.4375-day month when ``normalize_monthly`` is True.

    Time index:
      * If ``month_index_col`` is given it is used as t.
      * Otherwise a per-well sequential index is built (one-based by default so
        that log(t) is well defined).

    Invalid rows (null / zero / negative / infinite rate, or non-positive
    days_produced) are dropped when ``drop_invalid`` is True, otherwise flagged
    in a boolean ``valid`` column.

    Returns a DataFrame with columns: well_id, t, q, valid, and date if available.
    """
    if rate_col is None and (volume_col is None or days_col is None):
        raise ValueError("Provide either rate_col or both volume_col and days_col.")

    out = df.copy()

    # --- well id ---
    if well_col is not None:
        out["well_id"] = out[well_col].astype(str)
    else:
        out["well_id"] = "well_0"

    # --- date ---
    has_date = date_col is not None
    if has_date:
        out["date"] = pd.to_datetime(out[date_col], errors="coerce")

    # --- rate ---
    if rate_col is not None:
        q = pd.to_numeric(out[rate_col], errors="coerce").astype(float)
    else:
        vol = pd.to_numeric(out[volume_col], errors="coerce").astype(float)
        days = pd.to_numeric(out[days_col], errors="coerce").astype(float)
        days = days.where(days > 0, np.nan)  # non-positive days -> invalid rate
        q = vol / days
        if normalize_monthly:
            q = q * DAYS_PER_MONTH
    out["q"] = q.to_numpy()

    # --- sort by well then time ---
    sort_keys = ["well_id"]
    if month_index_col is not None:
        sort_keys.append(month_index_col)
    elif has_date:
        sort_keys.append("date")
    out = out.sort_values(sort_keys, kind="stable").reset_index(drop=True)

    # --- time index (one-based per well) ---
    if month_index_col is not None:
        out["t"] = pd.to_numeric(out[month_index_col], errors="coerce").astype(float)
    else:
        base = 1 if one_based else 0
        out["t"] = out.groupby("well_id").cumcount().to_numpy(dtype=float) + base

    # --- validity ---
    qv = out["q"].to_numpy()
    valid = np.isfinite(qv) & (qv > min_rate) & np.isfinite(out["t"].to_numpy())
    out["valid"] = valid

    cols = ["well_id", "t", "q", "valid"]
    if has_date:
        cols.insert(2, "date")
    out = out[cols]

    if drop_invalid:
        out = out[out["valid"]].reset_index(drop=True)

    return out
