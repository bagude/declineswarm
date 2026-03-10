"""Parameterized DCA wrapper around petbox-dca.

All fitting parameters are explicit arguments — nothing hardcoded.
Pure functions, no database dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from petbox import dca


DeclineModelType = Literal["exponential", "hyperbolic", "harmonic"]


@dataclass(frozen=True)
class ForecastMonth:
    month_index: int
    oil_bbl: float
    gas_mcf: float
    cumulative_oil_bbl: float
    cumulative_gas_mcf: float


def create_model(
    decline_model: DeclineModelType,
    qi_oil: float,
    di: float,
    b: float = 0.0,
    d_min: float = 0.05,
    qi_gas: float = 0.0,
) -> dict:
    """Create a petbox-dca model from Arps parameters.

    qi_oil/qi_gas in BBL/month or MCF/month. di is nominal annual.
    """
    if decline_model == "exponential":
        b = 0.0
    if decline_model == "harmonic":
        b = 1.0

    qi_oil_daily = qi_oil / 30.4375
    qi_gas_daily = qi_gas / 30.4375

    oil_model = dca.MH(qi=qi_oil_daily, Di=di, bi=b, Dterm=d_min)
    gas_model = dca.MH(qi=qi_gas_daily, Di=di, bi=b, Dterm=d_min) if qi_gas > 0 else None

    return {
        "oil_model": oil_model,
        "gas_model": gas_model,
        "params": {
            "decline_model": decline_model,
            "qi_oil": qi_oil,
            "qi_gas": qi_gas,
            "di": di,
            "b": b,
            "d_min": d_min,
        },
    }


def generate_forecast(model: dict, months: int = 600) -> list[ForecastMonth]:
    """Generate monthly production forecast."""
    oil_model = model["oil_model"]
    gas_model = model["gas_model"]

    days_per_month = 30.4375
    t = np.arange(1, months + 1) * days_per_month

    oil_monthly = oil_model.monthly_vol(t)
    oil_cumulative = np.cumsum(oil_monthly)

    if gas_model is not None:
        gas_monthly = gas_model.monthly_vol(t)
        gas_cumulative = np.cumsum(gas_monthly)
    else:
        gas_monthly = np.zeros(len(oil_monthly))
        gas_cumulative = np.zeros(len(oil_monthly))

    return [
        ForecastMonth(
            month_index=i + 1,
            oil_bbl=float(oil_monthly[i]),
            gas_mcf=float(gas_monthly[i]),
            cumulative_oil_bbl=float(oil_cumulative[i]),
            cumulative_gas_mcf=float(gas_cumulative[i]),
        )
        for i in range(len(oil_monthly))
    ]


def fit_arps(
    production: list[float],
    time_months: list[int] | None = None,
    model_type: DeclineModelType = "hyperbolic",
    *,
    d_min: float = 0.05,
    di_initial: float = 0.50,
    b_initial: float = 0.50,
    qi_guess_strategy: str = "first",
    qi_multiplier_upper: float = 5.0,
    di_upper_bound: float = 5.0,
    b_upper_bound: float = 2.0,
) -> dict:
    """Fit Arps decline parameters to monthly production.

    All tuning knobs are explicit parameters matching params.json/fitting.
    """
    from scipy.optimize import curve_fit

    prod = np.array(production, dtype=float)

    if time_months is None:
        t_months = np.arange(1, len(prod) + 1)
    else:
        t_months = np.array(time_months)

    mask = (prod > 0) & ~np.isnan(prod)
    prod_clean = prod[mask]
    t_clean = t_months[mask]

    if len(prod_clean) < 3:
        raise ValueError("Need at least 3 non-zero production months to fit.")

    prod_daily = prod_clean / 30.4375
    t_days = t_clean * 30.4375

    # qi guess based on strategy
    if qi_guess_strategy == "max3":
        qi_guess = float(np.max(prod_daily[:3])) if len(prod_daily) >= 3 else float(prod_daily[0])
    elif qi_guess_strategy == "peak_value":
        qi_guess = float(np.max(prod_daily))
    else:  # "first"
        qi_guess = float(prod_daily[0])

    convergence = True

    if model_type == "exponential":
        def _rate_func(t, qi, di_param):
            m = dca.MH(qi=qi, Di=di_param, bi=0.0, Dterm=0.0)
            return m.rate(t)

        try:
            popt, _ = curve_fit(
                _rate_func, t_days, prod_daily,
                p0=[qi_guess, di_initial],
                bounds=([0.001, 0.001], [qi_guess * qi_multiplier_upper, di_upper_bound]),
                maxfev=5000,
            )
        except RuntimeError:
            convergence = False
            popt = [qi_guess, di_initial]

        qi_fit, di_fit = popt
        b_fit = 0.0

    elif model_type == "harmonic":
        def _rate_func(t, qi, di_param):
            m = dca.MH(qi=qi, Di=di_param, bi=1.0, Dterm=d_min)
            return m.rate(t)

        try:
            popt, _ = curve_fit(
                _rate_func, t_days, prod_daily,
                p0=[qi_guess, di_initial],
                bounds=([0.001, 0.001], [qi_guess * qi_multiplier_upper, di_upper_bound]),
                maxfev=5000,
            )
        except RuntimeError:
            convergence = False
            popt = [qi_guess, di_initial]

        qi_fit, di_fit = popt
        b_fit = 1.0

    else:  # hyperbolic
        def _rate_func(t, qi, di_param, b_param):
            m = dca.MH(qi=qi, Di=di_param, bi=b_param, Dterm=d_min)
            return m.rate(t)

        try:
            popt, _ = curve_fit(
                _rate_func, t_days, prod_daily,
                p0=[qi_guess, di_initial, b_initial],
                bounds=(
                    [0.001, 0.001, 0.001],
                    [qi_guess * qi_multiplier_upper, di_upper_bound, b_upper_bound],
                ),
                maxfev=5000,
            )
        except RuntimeError:
            convergence = False
            popt = [qi_guess, di_initial, b_initial]

        qi_fit, di_fit, b_fit = popt

    qi_monthly = qi_fit * 30.4375

    # Fit quality
    model = dca.MH(qi=qi_fit, Di=di_fit, bi=b_fit, Dterm=d_min)
    predicted = model.rate(t_days)
    residuals = prod_daily - predicted
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((prod_daily - np.mean(prod_daily)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "decline_model": model_type,
        "qi": qi_monthly,
        "di": float(di_fit),
        "b": float(b_fit),
        "d_min": d_min,
        "r_squared": float(r_squared),
        "convergence": convergence,
    }
