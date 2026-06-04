"""Synthetic well generators for testing and demonstration.

Production error is closer to multiplicative than additive, so noise is applied
as q_obs = q_true * exp(N(0, sigma)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import global_power_ratio, shifted_power_ratio


@dataclass
class SyntheticWell:
    well_id: str
    t: np.ndarray
    q: np.ndarray            # noisy observed rate
    q_true: np.ndarray       # noiseless rate
    truth: dict              # generating parameters


def add_lognormal_noise(q_true: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """q_obs = q_true * exp(N(0, sigma))."""
    return q_true * np.exp(rng.normal(0.0, sigma, size=len(q_true)))


def make_single_segment_power(
    n: int = 60, qi: float = 1000.0, alpha: float = 0.8,
    sigma: float = 0.05, seed: int = 0, well_id: str = "single_power",
) -> SyntheticWell:
    """q(t) = qi * (t / 1)^(-alpha), one-based t."""
    rng = np.random.default_rng(seed)
    t = np.arange(1, n + 1, dtype=float)
    q_true = qi * global_power_ratio(t, T=1.0, alpha=alpha)
    q = add_lognormal_noise(q_true, sigma, rng)
    return SyntheticWell(well_id, t, q, q_true,
                         {"kind": "global_power", "qi": qi, "alpha": alpha})


def make_two_segment_power(
    n: int = 72, break_month: int = 36, qi: float = 1000.0,
    alpha1: float = 0.5, alpha2: float = 1.4,
    sigma: float = 0.05, seed: int = 1, well_id: str = "two_segment",
) -> SyntheticWell:
    """Continuous two-segment global power law with a breakpoint at break_month."""
    rng = np.random.default_rng(seed)
    t = np.arange(1, n + 1, dtype=float)
    T1 = 1.0
    Tb = float(break_month)
    q_true = np.empty(n)
    mask1 = t < Tb
    q_true[mask1] = qi * global_power_ratio(t[mask1], T=T1, alpha=alpha1)
    q_break = qi * global_power_ratio(np.array([Tb]), T=T1, alpha=alpha1)[0]
    q_true[~mask1] = q_break * global_power_ratio(t[~mask1], T=Tb, alpha=alpha2)
    q = add_lognormal_noise(q_true, sigma, rng)
    return SyntheticWell(well_id, t, q, q_true,
                         {"kind": "two_segment_power", "qi": qi,
                          "alpha1": alpha1, "alpha2": alpha2, "break_month": break_month})


def make_shifted_power(
    n: int = 60, qi: float = 1000.0, alpha: float = 1.2, c: float = 20.0,
    sigma: float = 0.05, seed: int = 2, well_id: str = "shifted_power",
) -> SyntheticWell:
    """q(t) = qi * ((t + c)/(1 + c))^(-alpha), one-based t."""
    rng = np.random.default_rng(seed)
    t = np.arange(1, n + 1, dtype=float)
    q_true = qi * shifted_power_ratio(t, T=1.0, alpha=alpha, c=c)
    q = add_lognormal_noise(q_true, sigma, rng)
    return SyntheticWell(well_id, t, q, q_true,
                         {"kind": "shifted_power", "qi": qi, "alpha": alpha, "c": c})


def generate_example_wells(seed: int = 7) -> list[SyntheticWell]:
    """Three demo wells: single power, two-segment power, shifted-clock power."""
    return [
        make_single_segment_power(seed=seed, sigma=0.06),
        make_two_segment_power(seed=seed + 1, sigma=0.06),
        make_shifted_power(seed=seed + 2, sigma=0.06),
    ]
