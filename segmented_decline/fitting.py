"""Single-segment fitting and the segment cost matrix.

Every fit works in log-rate space. For a segment anchored at (T, Q) we define

    r_i = q_i / Q        y_i = log(r_i)

and fit a model to y. Power-law models are linear in the exponent, so alpha is
solved in closed form; the shifted-clock offset c is the only nonlinear knob and
is found with a 1-D optimizer. Residuals are log-rate residuals, so the cost is
naturally a percentage (multiplicative) error.

The cost stored in the matrix is always the plain squared-log-error RSS, even
when ``config.robust`` is set — robustness only changes how parameters are
refined, not how segments are compared during model selection.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from .config import SegmentFit, SegmentedDeclineConfig
from .models import shifted_power_to_hyperbolic

# Smallest allowed value for the shifted-clock offset (T + c).
_OFFSET_MIN = 1e-3
_OFFSET_MAX = 1e6


def _loglinear_through(x: np.ndarray, y: np.ndarray, allow_jumps: bool,
                       min_alpha: float, max_alpha: float) -> tuple[float, float, np.ndarray, float]:
    """Fit y ~ beta0 - alpha * x in closed form.

    Without jumps the intercept beta0 is forced to 0 (the segment is anchored at
    the observed start rate). With jumps beta0 is free and exp(beta0) is the jump
    multiplier M. alpha is clamped to [min_alpha, max_alpha]; if clamping kicks in
    the intercept is re-solved for the fixed alpha.

    Returns (alpha, beta0, y_hat, rss).
    """
    if allow_jumps:
        A = np.column_stack([np.ones_like(x), -x])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        beta0, alpha = float(coef[0]), float(coef[1])
    else:
        xtx = float(x @ x)
        alpha = -float(x @ y) / xtx if xtx > 0 else min_alpha
        beta0 = 0.0

    clamped = min(max(alpha, min_alpha), max_alpha)
    if clamped != alpha:
        alpha = clamped
        beta0 = float(np.mean(y + alpha * x)) if allow_jumps else 0.0

    y_hat = beta0 - alpha * x
    rss = float(np.sum((y - y_hat) ** 2))
    return alpha, beta0, y_hat, rss


def _slice(t: np.ndarray, q: np.ndarray, start_idx: int, end_idx: int):
    ts = np.asarray(t[start_idx:end_idx], dtype=float)
    qs = np.asarray(q[start_idx:end_idx], dtype=float)
    T = float(ts[0])
    Q = float(qs[0])
    y = np.log(qs / Q)
    return ts, qs, T, Q, y


def _make_fit(start_idx, end_idx, ts, T, Q, model_type, params, y, y_hat, rss) -> SegmentFit:
    n = len(ts)
    rmse = float(np.sqrt(rss / n)) if n > 0 else 0.0
    return SegmentFit(
        start_idx=int(start_idx),
        end_idx=int(end_idx),
        start_t=float(T),
        end_t=float(ts[-1]),
        model_type=model_type,
        params=params,
        rss=float(rss),
        rmse=rmse,
        n=int(n),
        y_obs=y.copy(),
        y_hat=y_hat.copy(),
    )


def fit_global_power_segment(t, q, start_idx, end_idx, config: SegmentedDeclineConfig) -> SegmentFit:
    """Fit r(t) = (t/T)^(-alpha). alpha solved by least squares through origin."""
    ts, qs, T, Q, y = _slice(t, q, start_idx, end_idx)
    x = np.log(ts / T)

    alpha, beta0, y_hat, rss = _loglinear_through(
        x, y, config.allow_jumps, config.min_alpha, config.max_alpha
    )

    if config.robust:
        alpha, beta0, y_hat, rss = _refine_robust(
            x, y, alpha, beta0, config
        )

    b_eq, D_eq = shifted_power_to_hyperbolic(alpha, 0.0, T)
    params = {
        "alpha": alpha, "c": 0.0, "T": T, "Q": Q,
        "M": float(np.exp(beta0)), "b_equiv": b_eq, "D_equiv": D_eq,
    }
    return _make_fit(start_idx, end_idx, ts, T, Q, "global_power", params, y, y_hat, rss)


def fit_shifted_power_segment(t, q, start_idx, end_idx, config: SegmentedDeclineConfig) -> SegmentFit:
    """Fit r(t) = ((t+c)/(T+c))^(-alpha).

    For a fixed offset = T + c the exponent is linear, so alpha is closed-form;
    a 1-D bounded search over log(offset) finds c.
    """
    ts, qs, T, Q, y = _slice(t, q, start_idx, end_idx)

    def rss_at_offset(log_offset: float) -> float:
        offset = np.exp(log_offset)
        # x_i = log(t_i + c) - log(T + c) = log(t_i - T + offset) - log(offset)
        x = np.log(ts - T + offset) - np.log(offset)
        _, _, _, rss = _loglinear_through(
            x, y, config.allow_jumps, config.min_alpha, config.max_alpha
        )
        return rss

    res = minimize_scalar(
        rss_at_offset,
        bounds=(np.log(_OFFSET_MIN), np.log(_OFFSET_MAX)),
        method="bounded",
    )
    offset = float(np.exp(res.x))
    c = offset - T
    x = np.log(ts - T + offset) - np.log(offset)
    alpha, beta0, y_hat, rss = _loglinear_through(
        x, y, config.allow_jumps, config.min_alpha, config.max_alpha
    )

    if config.robust:
        alpha, beta0, offset, y_hat, rss = _refine_robust_shifted(
            ts, T, y, alpha, beta0, offset, config
        )
        c = offset - T

    b_eq, D_eq = shifted_power_to_hyperbolic(alpha, c, T)
    params = {
        "alpha": alpha, "c": c, "offset": offset, "T": T, "Q": Q,
        "M": float(np.exp(beta0)), "b_equiv": b_eq, "D_equiv": D_eq,
    }
    return _make_fit(start_idx, end_idx, ts, T, Q, "shifted_power", params, y, y_hat, rss)


def fit_restarted_hyperbolic_segment(t, q, start_idx, end_idx, config) -> SegmentFit:
    """Fit the restarted hyperbolic via its equivalent shifted-clock power law."""
    fit = fit_shifted_power_segment(t, q, start_idx, end_idx, config)
    p = fit.params
    fit.model_type = "restarted_hyperbolic"
    # Promote the Arps-style params to the primary keys.
    fit.params = {
        "b": p["b_equiv"], "D": p["D_equiv"],
        "alpha": p["alpha"], "c": p["c"], "T": p["T"], "Q": p["Q"],
        "M": p["M"], "b_equiv": p["b_equiv"], "D_equiv": p["D_equiv"],
    }
    return fit


def fit_exponential_segment(t, q, start_idx, end_idx, config: SegmentedDeclineConfig) -> SegmentFit:
    """Fit r(t) = exp(-D*(t-T)). D solved by least squares through the origin."""
    ts, qs, T, Q, y = _slice(t, q, start_idx, end_idx)
    u = ts - T

    if config.allow_jumps:
        A = np.column_stack([np.ones_like(u), -u])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        beta0, D = float(coef[0]), float(coef[1])
    else:
        utu = float(u @ u)
        D = -float(u @ y) / utu if utu > 0 else 0.0
        beta0 = 0.0
    D = max(D, 1e-8)
    if config.allow_jumps:
        # re-solve intercept for the (possibly clamped) D
        beta0 = float(np.mean(y + D * u))
    y_hat = beta0 - D * u
    rss = float(np.sum((y - y_hat) ** 2))

    params = {"D": D, "T": T, "Q": Q, "M": float(np.exp(beta0))}
    return _make_fit(start_idx, end_idx, ts, T, Q, "exponential", params, y, y_hat, rss)


# --- robust (soft_l1) refinement -------------------------------------------

def _refine_robust(x, y, alpha0, beta0_0, config):
    """Robust refine for fixed-x power models (global power)."""
    if config.allow_jumps:
        def resid(p):
            beta0, alpha = p
            return y - (beta0 - alpha * x)
        p0 = [beta0_0, alpha0]
        lb = [-np.inf, config.min_alpha]
        ub = [np.inf, config.max_alpha]
    else:
        def resid(p):
            alpha = p[0]
            return y - (-alpha * x)
        p0 = [alpha0]
        lb = [config.min_alpha]
        ub = [config.max_alpha]
    sol = least_squares(resid, p0, loss="soft_l1", bounds=(lb, ub))
    if config.allow_jumps:
        beta0, alpha = float(sol.x[0]), float(sol.x[1])
    else:
        alpha, beta0 = float(sol.x[0]), 0.0
    y_hat = beta0 - alpha * x
    rss = float(np.sum((y - y_hat) ** 2))
    return alpha, beta0, y_hat, rss


def _refine_robust_shifted(ts, T, y, alpha0, beta0_0, offset0, config):
    """Robust refine for the shifted-clock power law (params: log_offset, alpha[, beta0])."""
    log_off0 = np.log(max(offset0, _OFFSET_MIN))

    def x_of(log_offset):
        offset = np.exp(log_offset)
        return np.log(ts - T + offset) - np.log(offset)

    if config.allow_jumps:
        def resid(p):
            log_offset, alpha, beta0 = p
            return y - (beta0 - alpha * x_of(log_offset))
        p0 = [log_off0, alpha0, beta0_0]
        lb = [np.log(_OFFSET_MIN), config.min_alpha, -np.inf]
        ub = [np.log(_OFFSET_MAX), config.max_alpha, np.inf]
    else:
        def resid(p):
            log_offset, alpha = p
            return y - (-alpha * x_of(log_offset))
        p0 = [log_off0, alpha0]
        lb = [np.log(_OFFSET_MIN), config.min_alpha]
        ub = [np.log(_OFFSET_MAX), config.max_alpha]

    sol = least_squares(resid, p0, loss="soft_l1", bounds=(lb, ub))
    if config.allow_jumps:
        log_offset, alpha, beta0 = float(sol.x[0]), float(sol.x[1]), float(sol.x[2])
    else:
        log_offset, alpha, beta0 = float(sol.x[0]), float(sol.x[1]), 0.0
    offset = float(np.exp(log_offset))
    y_hat = beta0 - alpha * (np.log(ts - T + offset) - np.log(offset))
    rss = float(np.sum((y - y_hat) ** 2))
    return alpha, beta0, offset, y_hat, rss


# --- dispatch + cost matrix -------------------------------------------------

_FITTERS = {
    "global_power": fit_global_power_segment,
    "shifted_power": fit_shifted_power_segment,
    "restarted_hyperbolic": fit_restarted_hyperbolic_segment,
    "exponential": fit_exponential_segment,
}


def fit_segment(t, q, start_idx, end_idx, config: SegmentedDeclineConfig) -> SegmentFit:
    """Fit a single segment over [start_idx, end_idx) using config.model_type."""
    return _FITTERS[config.model_type](t, q, start_idx, end_idx, config)


def build_cost_matrix(t, q, config: SegmentedDeclineConfig):
    """Fit every candidate segment and tabulate its squared-log-error RSS.

    Returns (cost, lookup) where cost is an (n+1, n+1) array such that
    cost[a, b] is the RSS of the segment covering rows [a, b) (b exclusive), and
    lookup[(a, b)] is the corresponding SegmentFit. Entries shorter than
    ``min_segment_length`` are left as np.inf / absent.
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    n = len(t)
    cost = np.full((n + 1, n + 1), np.inf)
    lookup: dict[tuple[int, int], SegmentFit] = {}

    # Cost matrix never uses robust loss (kept fast + comparable across K).
    fast_cfg = config
    if config.robust:
        fast_cfg = SegmentedDeclineConfig(**{**config.__dict__, "robust": False})

    L = config.min_segment_length
    for a in range(n):
        for b in range(a + L, n + 1):
            fit = fit_segment(t, q, a, b, fast_cfg)
            cost[a, b] = fit.rss
            lookup[(a, b)] = fit
    return cost, lookup
