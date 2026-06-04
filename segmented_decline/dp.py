"""Dynamic-programming segmentation and model selection over K.

Segments are half-open index intervals [start, end). A K-segment model is a
chain 0 = s_0 < s_1 < ... < s_K = n; the breakpoints are s_1..s_{K-1}. The DP
recurrence

    DP[k, end] = min over start { DP[k-1, start] + cost[start, end] }

finds, for each k, the segmentation with the smallest total squared-log-error.
We then score each K with AIC/BIC and keep the best.
"""

from __future__ import annotations

import numpy as np

from .config import SegmentedDeclineConfig, SegmentFit, SegmentedDeclineFit
from .fitting import build_cost_matrix, fit_segment


def information_criteria(rss: float, n: int, p: int) -> tuple[float, float]:
    """Gaussian AIC/BIC from a residual sum of squares.

        AIC = n*log(RSS/n) + 2p
        BIC = n*log(RSS/n) + p*log(n)
    """
    rss = max(rss, 1e-300)
    ll_term = n * np.log(rss / n)
    aic = ll_term + 2 * p
    bic = ll_term + p * np.log(n)
    return float(aic), float(bic)


def param_count(k: int, config: SegmentedDeclineConfig) -> int:
    """Number of fitted parameters for a K-segment model.

        p = K * params_per_segment + (K - 1) breakpoints

    Each segment's level/anchor is already included in params_per_segment, so
    enabling jumps (which only frees that level rather than fixing it to the
    observed start) does not add extra parameters here.
    """
    return k * config.params_per_segment + (k - 1)


def solve_best_segmentation(cost_matrix, segment_fit_lookup, n, k,
                            config: SegmentedDeclineConfig):
    """Best K-segment fit by dynamic programming.

    Returns (total_rss, [SegmentFit, ...]) for the optimal segmentation into
    exactly ``k`` segments, or None if no feasible segmentation exists (e.g. not
    enough rows for k segments of min length).
    """
    L = config.min_segment_length
    INF = np.inf

    # dp[kk][end] = best total RSS covering [0, end) with kk segments.
    dp = np.full((k + 1, n + 1), INF)
    back = np.full((k + 1, n + 1), -1, dtype=int)

    for end in range(L, n + 1):
        dp[1][end] = cost_matrix[0, end]
        back[1][end] = 0

    for kk in range(2, k + 1):
        # earliest end that can hold kk segments of length >= L
        for end in range(kk * L, n + 1):
            best = INF
            best_start = -1
            for start in range((kk - 1) * L, end - L + 1):
                prev = dp[kk - 1][start]
                if prev == INF:
                    continue
                c = cost_matrix[start, end]
                if c == INF:
                    continue
                total = prev + c
                if total < best:
                    best = total
                    best_start = start
            dp[kk][end] = best
            back[kk][end] = best_start

    if dp[k][n] == INF:
        return None

    # Backtrack to recover segment boundaries.
    bounds = [n]
    end = n
    for kk in range(k, 0, -1):
        start = back[kk][end]
        bounds.append(start)
        end = start
    bounds.reverse()  # [0, s_1, ..., n]

    segments = [segment_fit_lookup[(bounds[i], bounds[i + 1])] for i in range(k)]
    return float(dp[k][n]), segments


def _assemble(segments: list[SegmentFit], n: int):
    """Build the full q_hat and log-residual arrays from per-segment fits."""
    q_hat = np.empty(n)
    resid = np.empty(n)
    for seg in segments:
        Q = seg.params["Q"]
        seg_q_hat = Q * np.exp(seg.y_hat)
        q_hat[seg.start_idx:seg.end_idx] = seg_q_hat
        resid[seg.start_idx:seg.end_idx] = seg.y_obs - seg.y_hat
    return q_hat, resid


def fit_segmented_decline(t, q, config: SegmentedDeclineConfig | None = None,
                          well_id: str | None = None) -> SegmentedDeclineFit:
    """Fit a segmented decline curve, selecting the number of segments by IC.

    Tries K = 1..max_segments, scores each with the configured criterion
    (BIC by default), and returns the best SegmentedDeclineFit.
    """
    config = config or SegmentedDeclineConfig()
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    n = len(t)
    if n < config.min_segment_length:
        raise ValueError(
            f"Need at least min_segment_length={config.min_segment_length} points, got {n}."
        )

    cost, lookup = build_cost_matrix(t, q, config)

    k_results: list[dict] = []
    best = None  # (criterion_value, K, rss, segments, aic, bic)

    max_k = min(config.max_segments, n // config.min_segment_length)
    for k in range(1, max_k + 1):
        sol = solve_best_segmentation(cost, lookup, n, k, config)
        if sol is None:
            continue
        rss, segments = sol
        p = param_count(k, config)
        aic, bic = information_criteria(rss, n, p)
        crit_val = bic if config.criterion == "bic" else aic
        k_results.append({
            "k": k, "rss": rss, "aic": aic, "bic": bic,
            "p": p, "criterion_value": crit_val,
            "breakpoints_idx": [s.start_idx for s in segments[1:]],
        })
        if best is None or crit_val < best[0]:
            best = (crit_val, k, rss, segments, aic, bic)

    if best is None:
        raise ValueError("No feasible segmentation found for the given config.")

    _, k_best, rss, segments, aic, bic = best

    # Optional robust refinement of the *selected* segments only.
    if config.robust:
        segments = [fit_segment(t, q, s.start_idx, s.end_idx, config) for s in segments]
        rss = float(sum(s.rss for s in segments))
        p = param_count(k_best, config)
        aic, bic = information_criteria(rss, n, p)

    q_hat, resid = _assemble(segments, n)
    breakpoints_idx = [s.start_idx for s in segments[1:]]
    breakpoints_t = [float(t[i]) for i in breakpoints_idx]

    return SegmentedDeclineFit(
        well_id=well_id,
        model_type=config.model_type,
        criterion=config.criterion,
        n_segments=k_best,
        breakpoints_idx=breakpoints_idx,
        breakpoints_t=breakpoints_t,
        segments=segments,
        rss=rss,
        aic=aic,
        bic=bic,
        q_hat=q_hat,
        residuals_log=resid,
        config=dict(config.__dict__),
        k_results=k_results,
    )
