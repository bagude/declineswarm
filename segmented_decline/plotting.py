"""Plotting helpers for segmented decline fits.

All functions accept an optional matplotlib Axes and return it, so they compose
into multi-panel figures. They never call plt.show(); the caller decides whether
to display or save.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402

from .config import SegmentedDeclineFit


def _new_ax(ax):
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    return ax


def plot_segmented_fit(t, q, fit: SegmentedDeclineFit, ax=None, logy: bool = True):
    """Observed rates (points) with the fitted segmented curve and breakpoints."""
    ax = _new_ax(ax)
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    ax.scatter(t, q, s=18, color="0.4", alpha=0.7, label="observed", zorder=2)

    colors = plt.cm.viridis(np.linspace(0, 0.85, len(fit.segments)))
    for seg, color in zip(fit.segments, colors):
        sl = slice(seg.start_idx, seg.end_idx)
        ax.plot(t[sl], fit.q_hat[sl], color=color, lw=2.2,
                label=f"seg {seg.start_idx}-{seg.end_idx}", zorder=3)

    for bt in fit.breakpoints_t:
        ax.axvline(bt, color="crimson", ls="--", lw=1.2, alpha=0.7)

    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("t (month index)")
    ax.set_ylabel("rate q")
    title = f"{fit.well_id or 'well'} — K={fit.n_segments} ({fit.model_type}, {fit.criterion})"
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    return ax


def plot_log_fit(t, q, fit: SegmentedDeclineFit, ax=None):
    """log(q) observed vs fitted."""
    ax = _new_ax(ax)
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    ax.scatter(t, np.log(q), s=18, color="0.4", alpha=0.7, label="log observed")
    ax.plot(t, np.log(fit.q_hat), color="C0", lw=2, label="log fitted")
    for bt in fit.breakpoints_t:
        ax.axvline(bt, color="crimson", ls="--", lw=1.2, alpha=0.7)
    ax.set_xlabel("t (month index)")
    ax.set_ylabel("log q")
    ax.set_title("log-rate fit")
    ax.legend(fontsize=8)
    return ax


def plot_residuals(t, fit: SegmentedDeclineFit, ax=None):
    """Log residuals vs time."""
    ax = _new_ax(ax)
    t = np.asarray(t, dtype=float)
    ax.axhline(0, color="0.6", lw=1)
    ax.scatter(t, fit.residuals_log, s=18, color="C3", alpha=0.8)
    for bt in fit.breakpoints_t:
        ax.axvline(bt, color="crimson", ls="--", lw=1.2, alpha=0.7)
    ax.set_xlabel("t (month index)")
    ax.set_ylabel("log residual")
    ax.set_title("residuals (log space)")
    return ax


def plot_model_selection(k_results: Sequence[dict], criterion: str = "bic", ax=None):
    """AIC and BIC versus number of segments K."""
    ax = _new_ax(ax)
    ks = [r["k"] for r in k_results]
    ax.plot(ks, [r["aic"] for r in k_results], "o-", label="AIC")
    ax.plot(ks, [r["bic"] for r in k_results], "s-", label="BIC")
    # mark the chosen K by the configured criterion
    key = criterion
    best = min(k_results, key=lambda r: r[key])
    ax.axvline(best["k"], color="green", ls=":", lw=1.5,
               label=f"selected K={best['k']} ({criterion.upper()})")
    ax.set_xlabel("number of segments K")
    ax.set_ylabel("information criterion")
    ax.set_title("model selection")
    ax.set_xticks(ks)
    ax.legend(fontsize=8)
    return ax


def plot_breakpoints(t, q, fit: SegmentedDeclineFit, ax=None):
    """Observed data with breakpoint locations highlighted."""
    ax = _new_ax(ax)
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    ax.plot(t, q, color="0.5", lw=1, marker="o", ms=3, label="observed")
    for bi, bt in zip(fit.breakpoints_idx, fit.breakpoints_t):
        ax.axvline(bt, color="crimson", ls="--", lw=1.5)
        ax.annotate(f"t={bt:g}", xy=(bt, q[bi]), xytext=(5, 5),
                    textcoords="offset points", color="crimson", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("t (month index)")
    ax.set_ylabel("rate q")
    ax.set_title("breakpoints")
    return ax


def summary_figure(t, q, fit: SegmentedDeclineFit):
    """Convenience 2x2 panel: fit, log-fit, residuals, model selection."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plot_segmented_fit(t, q, fit, ax=axes[0, 0])
    plot_log_fit(t, q, fit, ax=axes[0, 1])
    plot_residuals(t, fit, ax=axes[1, 0])
    if fit.k_results:
        plot_model_selection(fit.k_results, fit.criterion, ax=axes[1, 1])
    fig.tight_layout()
    return fig
