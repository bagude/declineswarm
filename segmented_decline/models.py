"""Segment ratio models for decline-curve analysis.

Every segment model is expressed as a *ratio* curve r_j(t) anchored at the
segment start T_j so that r_j(T_j) = 1. The segment rate is then

    q_hat(t) = Q_j * r_j(t)

where Q_j is the segment-start rate. Working in ratio form keeps the global
time clock intact: we re-anchor the ratio at the segment start without
pretending the well physically restarted.

All functions are pure and operate on numpy arrays.
"""

from __future__ import annotations

import numpy as np


def global_power_ratio(t: np.ndarray, T: float, alpha: float) -> np.ndarray:
    """Global-clock power law.

        r(t) = (t / T)^(-alpha)

    Equivalent to the shifted-clock model with c = 0. Requires t > 0 and T > 0.
    """
    t = np.asarray(t, dtype=float)
    return (t / T) ** (-alpha)


def shifted_power_ratio(t: np.ndarray, T: float, alpha: float, c: float) -> np.ndarray:
    """Shifted-clock power law.

        r(t) = ((t + c) / (T + c))^(-alpha)

    The shift c re-anchors the ratio while preserving the global clock.
    Requires T + c > 0 and t + c > 0.
    """
    t = np.asarray(t, dtype=float)
    return ((t + c) / (T + c)) ** (-alpha)


def restarted_hyperbolic_ratio(t: np.ndarray, T: float, D: float, b: float) -> np.ndarray:
    """Restarted Arps hyperbolic segment.

        r(t) = [1 + b * D * (t - T)]^(-1 / b)

    Mathematically equivalent to a shifted-clock power law via
    alpha = 1/b and c = 1/(b*D) - T. Requires b > 0, D > 0, t >= T.
    """
    t = np.asarray(t, dtype=float)
    return (1.0 + b * D * (t - T)) ** (-1.0 / b)


def exponential_ratio(t: np.ndarray, T: float, D: float) -> np.ndarray:
    """Exponential segment.

        r(t) = exp(-D * (t - T))

    Useful as a fallback / comparison model. Requires D > 0, t >= T.
    """
    t = np.asarray(t, dtype=float)
    return np.exp(-D * (t - T))


def shifted_power_to_hyperbolic(alpha: float, c: float, T: float) -> tuple[float, float]:
    """Map shifted-clock power-law params to equivalent Arps (b, D).

        b = 1 / alpha
        D = 1 / (b * (T + c))

    Returns (b, D). D is only meaningful when T + c > 0.
    """
    b = 1.0 / alpha
    D = 1.0 / (b * (T + c))
    return b, D


def hyperbolic_to_shifted_power(D: float, b: float, T: float) -> tuple[float, float]:
    """Map Arps (D, b) to equivalent shifted-clock power-law params.

        alpha = 1 / b
        c = 1 / (b * D) - T

    Returns (alpha, c).
    """
    alpha = 1.0 / b
    c = 1.0 / (b * D) - T
    return alpha, c
