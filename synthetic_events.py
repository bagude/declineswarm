"""Operational event injection functions for synthetic production data.

Each function takes numpy arrays of oil, gas, and water production (monthly volumes)
and returns modified arrays with the event applied. All functions are pure -- they
do not modify input arrays.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp_nonneg(*arrays):
    """Return copies clamped to non-negative values."""
    return tuple(np.maximum(a, 0.0) for a in arrays)


def _sigmoid_ramp(length: int) -> np.ndarray:
    """Smooth 0-to-1 ramp over *length* steps using a logistic sigmoid."""
    if length <= 0:
        return np.array([])
    if length == 1:
        return np.array([1.0])
    x = np.linspace(-4, 4, length)
    s = 1.0 / (1.0 + np.exp(-x))
    # Normalise so first is ~0 and last is ~1
    s = (s - s[0]) / (s[-1] - s[0])
    return s


def _linear_ramp(length: int) -> np.ndarray:
    """Linear 0-to-1 ramp over *length* steps."""
    if length <= 0:
        return np.array([])
    if length == 1:
        return np.array([1.0])
    return np.linspace(0.0, 1.0, length)


# ---------------------------------------------------------------------------
# 1. Shut-in  (spec 3.1)
# ---------------------------------------------------------------------------

def apply_shut_in(oil, gas, water, start_month, duration_months, recovery_fraction=0.90):
    """Voluntary shut-in: zeros production, recovers at recovery_fraction of projected trend.

    - Months start_month to start_month+duration_months-1: production = 0
    - Recovery: first month back = recovery_fraction * projected rate
    - Ramp back to full projected rate over 3 months
    - Water often spikes 1.5-2x on first month of return
    """
    oil = oil.copy()
    gas = gas.copy()
    water = water.copy()
    n = len(oil)

    if start_month >= n:
        return _clamp_nonneg(oil, gas, water)

    end_shutin = min(start_month + duration_months, n)

    # Zero out the shut-in window
    oil[start_month:end_shutin] = 0.0
    gas[start_month:end_shutin] = 0.0
    water[start_month:end_shutin] = 0.0

    # Recovery ramp (3 months from recovery_fraction back to 1.0)
    ramp_months = 3
    recovery_start = end_shutin
    recovery_end = min(recovery_start + ramp_months, n)

    if recovery_start < n:
        ramp_len = recovery_end - recovery_start
        ramp = np.linspace(recovery_fraction, 1.0, ramp_len)

        oil[recovery_start:recovery_end] *= ramp
        gas[recovery_start:recovery_end] *= ramp

        # Water spike on first month back: 1.8x, then decays
        water_spike = np.linspace(1.8, 1.0, ramp_len)
        water[recovery_start:recovery_end] *= water_spike * ramp

    return _clamp_nonneg(oil, gas, water)


# ---------------------------------------------------------------------------
# 2. Frac Hit  (spec 3.2)
# ---------------------------------------------------------------------------

def apply_frac_hit(oil, gas, water, month, severity, recovery_months, water_spike_multiplier=3.0):
    """Frac hit (well interference): production drop with partial recovery.

    severity: fraction drop (0.1 = 10% drop, 0.5 = 50% drop)
    - At hit month: oil drops by severity fraction
    - Recovery over recovery_months back to (1 - severity*0.2) of projected rate
      (i.e., severe hits have permanent component: 20% of drop is permanent)
    - Water spikes by water_spike_multiplier at hit month, decays over 2-3 months
    - Gas drops proportionally to oil
    """
    oil = oil.copy()
    gas = gas.copy()
    water = water.copy()
    n = len(oil)

    if month >= n:
        return _clamp_nonneg(oil, gas, water)

    # Permanent damage fraction: 20% of the severity is permanent
    permanent_loss = severity * 0.2
    recovery_target = 1.0 - permanent_loss

    # At hit month: immediate drop
    drop_factor = 1.0 - severity
    oil[month] *= drop_factor
    gas[month] *= drop_factor

    # Recovery ramp from drop_factor to recovery_target over recovery_months
    rec_start = month + 1
    rec_end = min(rec_start + recovery_months, n)

    if rec_start < n and recovery_months > 0:
        ramp_len = rec_end - rec_start
        ramp = np.linspace(drop_factor, recovery_target, ramp_len)
        oil[rec_start:rec_end] *= ramp
        gas[rec_start:rec_end] *= ramp

    # After recovery: permanent reduction
    after_rec = rec_end
    if after_rec < n:
        oil[after_rec:] *= recovery_target
        gas[after_rec:] *= recovery_target

    # Water spike at hit month, decaying over 3 months
    water_decay_months = min(3, n - month)
    for i in range(water_decay_months):
        idx = month + i
        if idx < n:
            spike = water_spike_multiplier - (water_spike_multiplier - 1.0) * (i / max(water_decay_months - 1, 1))
            water[idx] *= spike

    return _clamp_nonneg(oil, gas, water)


# ---------------------------------------------------------------------------
# 3. Workover Bump  (spec 3.3)
# ---------------------------------------------------------------------------

def apply_workover_bump(oil, gas, water, month, amplitude_fraction, decay_months):
    """Workover/stimulation: temporary production increase above trend.

    amplitude_fraction: e.g., 0.25 means 25% production increase at peak
    - Gaussian-shaped bump centered at month
    - Peak at month, decays with sigma = decay_months/2
    - Affects oil and gas proportionally
    - Water slightly elevated (0.5x the oil amplitude)
    """
    oil = oil.copy()
    gas = gas.copy()
    water = water.copy()
    n = len(oil)

    if month >= n:
        return _clamp_nonneg(oil, gas, water)

    sigma = max(decay_months / 2.0, 0.5)
    t = np.arange(n, dtype=float)
    bump = amplitude_fraction * np.exp(-0.5 * ((t - month) / sigma) ** 2)

    oil *= (1.0 + bump)
    gas *= (1.0 + bump)
    water *= (1.0 + bump * 0.5)

    return _clamp_nonneg(oil, gas, water)


# ---------------------------------------------------------------------------
# 4. Pipeline Curtailment  (spec 3.1)
# ---------------------------------------------------------------------------

def apply_pipeline_curtailment(oil, gas, water, start_month, duration_months, curtailment_fraction):
    """Pipeline curtailment: soft decline to fraction of rate, not full zero.

    curtailment_fraction: e.g., 0.4 means production drops to 40% of rate
    - Months start_month to start_month+duration_months-1: multiply all streams by curtailment_fraction
    - Smooth transition in (1 month ramp) and out (1 month ramp)
    """
    oil = oil.copy()
    gas = gas.copy()
    water = water.copy()
    n = len(oil)

    if start_month >= n:
        return _clamp_nonneg(oil, gas, water)

    end_month = min(start_month + duration_months, n)

    # Build a multiplier array: starts at 1.0, ramps down over 1 month to
    # curtailment_fraction, holds, then ramps back up over 1 month.
    multiplier = np.ones(n)

    # Ramp-in: the month before the curtailment window or the first month
    ramp_in_idx = start_month
    if ramp_in_idx < n:
        multiplier[ramp_in_idx] = (1.0 + curtailment_fraction) / 2.0  # midpoint

    # Full curtailment window (after ramp-in)
    full_start = min(start_month + 1, n)
    full_end = min(end_month, n)
    if full_start < full_end:
        multiplier[full_start:full_end] = curtailment_fraction

    # Ramp-out: the month after the curtailment window
    if end_month < n:
        multiplier[end_month] = (1.0 + curtailment_fraction) / 2.0

    oil *= multiplier
    gas *= multiplier
    water *= multiplier

    return _clamp_nonneg(oil, gas, water)


# ---------------------------------------------------------------------------
# 5. Refrac / Restimulation  (spec 3.3)
# ---------------------------------------------------------------------------

def apply_refrac(oil, gas, water, month, new_qi_multiplier, new_Di, new_b):
    """Refrac: production jumps and new decline phase begins.

    new_qi_multiplier: multiplier on the current rate at refrac month (e.g., 2.0 = doubles)
    - At refrac month: oil jumps to current_rate * new_qi_multiplier
    - From refrac month onward: new decline with new_Di, new_b applied
    - Uses petbox-dca MH model for the new decline segment
    - Gas follows oil with existing GOR
    - Water increases proportionally
    """
    from petbox import dca

    oil = oil.copy()
    gas = gas.copy()
    water = water.copy()
    n = len(oil)

    if month >= n:
        return _clamp_nonneg(oil, gas, water)

    # Current rates at refrac month
    oil_at_refrac = oil[month]
    gas_at_refrac = gas[month]
    water_at_refrac = water[month]

    if oil_at_refrac <= 0:
        return _clamp_nonneg(oil, gas, water)

    # GOR and WOR at refrac point for proportional scaling
    gor = gas_at_refrac / oil_at_refrac if oil_at_refrac > 0 else 0.0
    wor = water_at_refrac / oil_at_refrac if oil_at_refrac > 0 else 0.0

    # New initial rate
    new_qi_oil = oil_at_refrac * new_qi_multiplier

    # Generate new decline segment using petbox-dca MH model
    remaining_months = n - month
    qi_daily = new_qi_oil / 30.4375
    d_min = 0.05  # standard terminal decline

    model = dca.MH(qi=qi_daily, Di=new_Di, bi=new_b, Dterm=d_min)
    days_per_month = 30.4375
    t = np.arange(1, remaining_months + 1) * days_per_month
    new_oil_monthly = model.monthly_vol(t)

    # Replace production from refrac month onward
    oil[month:] = new_oil_monthly
    gas[month:] = new_oil_monthly * gor
    water[month:] = new_oil_monthly * wor * new_qi_multiplier  # water increases proportionally

    return _clamp_nonneg(oil, gas, water)


# ---------------------------------------------------------------------------
# 6. Partial Month  (spec 3.5)
# ---------------------------------------------------------------------------

def apply_partial_month(oil, gas, water, month, days_produced=15):
    """Partial production month: only days_produced out of 30 days reported.

    - Scale that month's volume by days_produced / 30.0
    - Affects all three streams equally
    """
    oil = oil.copy()
    gas = gas.copy()
    water = water.copy()
    n = len(oil)

    if month >= n:
        return _clamp_nonneg(oil, gas, water)

    scale = days_produced / 30.0
    oil[month] *= scale
    gas[month] *= scale
    water[month] *= scale

    return _clamp_nonneg(oil, gas, water)


# ---------------------------------------------------------------------------
# 7. Allocation Noise  (spec 3.5)
# ---------------------------------------------------------------------------

def apply_allocation_noise(oil, gas, water, months, magnitude=0.3):
    """Commingled/allocation noise: random multiplicative noise on specified months.

    months: list of month indices
    magnitude: max fractional change (e.g., 0.3 = +/-30%)
    - For each specified month, multiply production by uniform(1-magnitude, 1+magnitude)
    - Affects oil, gas, water independently
    """
    oil = oil.copy()
    gas = gas.copy()
    water = water.copy()
    n = len(oil)

    rng = np.random.default_rng()

    for m in months:
        if m >= n:
            continue
        oil[m] *= rng.uniform(1.0 - magnitude, 1.0 + magnitude)
        gas[m] *= rng.uniform(1.0 - magnitude, 1.0 + magnitude)
        water[m] *= rng.uniform(1.0 - magnitude, 1.0 + magnitude)

    return _clamp_nonneg(oil, gas, water)


# ---------------------------------------------------------------------------
# 8. ESP Installation  (spec 3.4)
# ---------------------------------------------------------------------------

def apply_esp_installation(oil, gas, water, month, rate_jump_fraction=0.4):
    """ESP installation: production jump with sustained elevated rate.

    rate_jump_fraction: e.g., 0.4 = 40% production increase
    - At installation month: oil jumps by rate_jump_fraction
    - Sustained: from month onward, production is elevated by rate_jump_fraction
      (multiplicative on the base decline)
    - Gas follows proportionally
    """
    oil = oil.copy()
    gas = gas.copy()
    water = water.copy()
    n = len(oil)

    if month >= n:
        return _clamp_nonneg(oil, gas, water)

    multiplier = 1.0 + rate_jump_fraction
    oil[month:] *= multiplier
    gas[month:] *= multiplier

    return _clamp_nonneg(oil, gas, water)


# ---------------------------------------------------------------------------
# 9. ESP Failure  (spec 3.4)
# ---------------------------------------------------------------------------

def apply_esp_failure(oil, gas, water, month, repair_months=1):
    """ESP failure: rapid drop to near-zero, then repair.

    - At failure month: production drops to 5% of rate
    - If repair_months > 0: production recovers to 90% of pre-failure rate
      over repair_months
    - Creates sharp V-shape in monthly data
    """
    oil = oil.copy()
    gas = gas.copy()
    water = water.copy()
    n = len(oil)

    if month >= n:
        return _clamp_nonneg(oil, gas, water)

    # Store pre-failure rates
    pre_oil = oil[month]
    pre_gas = gas[month]
    pre_water = water[month]

    # Failure: drop to 5%
    oil[month] *= 0.05
    gas[month] *= 0.05
    water[month] *= 0.05

    # Repair ramp from 5% to 90% of pre-failure rate
    if repair_months > 0:
        rec_start = month + 1
        rec_end = min(rec_start + repair_months, n)
        ramp_len = rec_end - rec_start

        if ramp_len > 0:
            ramp = np.linspace(0.05, 0.90, ramp_len)
            # Replace with ramped values based on pre-failure rate
            oil[rec_start:rec_end] = pre_oil * ramp
            gas[rec_start:rec_end] = pre_gas * ramp
            water[rec_start:rec_end] = pre_water * ramp

        # After repair: sustain at 90% of pre-failure projected rate
        after_repair = rec_end
        if after_repair < n:
            oil[after_repair:] *= 0.90
            gas[after_repair:] *= 0.90
            water[after_repair:] *= 0.90

    return _clamp_nonneg(oil, gas, water)


# ---------------------------------------------------------------------------
# 10. Liquid Loading Onset  (spec 3.6)
# ---------------------------------------------------------------------------

def apply_liquid_loading(oil, gas, water, start_month, steepening_factor=1.5):
    """Liquid loading: gradual decline steepening as liquids accumulate.

    steepening_factor: how much steeper the decline becomes (1.5 = 50% steeper)
    - From start_month onward, apply exponentially increasing decline multiplier
    - Oil declines faster; water may increase slightly
    - Effect is gradual over 3-6 months then stabilizes at steepened rate
    """
    oil = oil.copy()
    gas = gas.copy()
    water = water.copy()
    n = len(oil)

    if start_month >= n:
        return _clamp_nonneg(oil, gas, water)

    # Transition period: 6 months to reach full steepening
    transition = 6
    remaining = n - start_month

    t = np.arange(remaining, dtype=float)

    # Ramp the steepening factor from 1.0 to steepening_factor over transition months
    effective_factor = np.ones(remaining)
    for i in range(remaining):
        if i < transition:
            frac = i / transition
            effective_factor[i] = 1.0 + (steepening_factor - 1.0) * frac
        else:
            effective_factor[i] = steepening_factor

    # Apply as additional exponential decline on top of existing production
    # Each month gets multiplied by exp(-additional_decline) cumulatively
    additional_decline_rate = (steepening_factor - 1.0) * 0.05  # fractional monthly extra decline
    cumulative_decline = np.exp(-additional_decline_rate * t * effective_factor / steepening_factor)

    oil[start_month:] *= cumulative_decline
    gas[start_month:] *= cumulative_decline

    # Water tends to increase slightly with liquid loading
    water_increase = 1.0 + 0.1 * (1.0 - cumulative_decline)
    water[start_month:] *= water_increase

    return _clamp_nonneg(oil, gas, water)


# ---------------------------------------------------------------------------
# 11. Pressure Depletion Plateau  (spec 3.6)
# ---------------------------------------------------------------------------

def apply_pressure_plateau(oil, gas, water, start_month, duration_months, plateau_fraction=0.9):
    """Pressure plateau: production flattens for extended period before resuming decline.

    plateau_fraction: rate relative to start_month rate (0.9 = 90% of rate at start)
    - From start_month to start_month+duration_months: production holds near plateau_fraction * rate_at_start
    - After plateau: decline resumes from the plateau level
    - Very challenging for DCA because b exponent appears to shift
    """
    oil = oil.copy()
    gas = gas.copy()
    water = water.copy()
    n = len(oil)

    if start_month >= n:
        return _clamp_nonneg(oil, gas, water)

    # Rates at plateau start
    oil_at_start = oil[start_month]
    gas_at_start = gas[start_month]
    water_at_start = water[start_month]

    plateau_oil = oil_at_start * plateau_fraction
    plateau_gas = gas_at_start * plateau_fraction
    plateau_water = water_at_start * plateau_fraction

    end_plateau = min(start_month + duration_months, n)

    # Hold production flat during plateau
    oil[start_month:end_plateau] = plateau_oil
    gas[start_month:end_plateau] = plateau_gas
    water[start_month:end_plateau] = plateau_water

    # After plateau: resume decline from the plateau level
    # Scale remaining production so it continues from plateau_oil
    if end_plateau < n and oil[end_plateau] > 0:
        scale_oil = plateau_oil / oil[end_plateau]
        scale_gas = plateau_gas / gas[end_plateau] if gas[end_plateau] > 0 else 1.0
        scale_water = plateau_water / water[end_plateau] if water[end_plateau] > 0 else 1.0

        oil[end_plateau:] *= scale_oil
        gas[end_plateau:] *= scale_gas
        water[end_plateau:] *= scale_water

    return _clamp_nonneg(oil, gas, water)


# ---------------------------------------------------------------------------
# 12. GOR Increase / Gas Breakout  (spec 3.6)
# ---------------------------------------------------------------------------

def apply_gor_breakout(oil, gas, water, start_month, oil_decline_factor=1.3, gas_boost_factor=1.2):
    """GOR increase as reservoir pressure drops below bubble point.

    - From start_month onward: oil declines faster by oil_decline_factor
    - Gas rate temporarily increases by gas_boost_factor for 3-6 months then declines
    - Permanently decouples oil and gas decline rates
    """
    oil = oil.copy()
    gas = gas.copy()
    water = water.copy()
    n = len(oil)

    if start_month >= n:
        return _clamp_nonneg(oil, gas, water)

    remaining = n - start_month
    t = np.arange(remaining, dtype=float)

    # Oil: apply additional exponential decline
    extra_oil_decline = (oil_decline_factor - 1.0) * 0.08  # monthly extra decline rate
    oil_multiplier = np.exp(-extra_oil_decline * t)
    oil[start_month:] *= oil_multiplier

    # Gas: temporary boost over ~5 months then gradual decline
    gas_boost_duration = 5
    gas_multiplier = np.ones(remaining)
    for i in range(remaining):
        if i < gas_boost_duration:
            # Ramp up to gas_boost_factor then start declining
            ramp_up = min(i / 2.0, 1.0)  # reach peak in ~2 months
            gas_multiplier[i] = 1.0 + (gas_boost_factor - 1.0) * ramp_up
        else:
            # Gradual decay of the boost, but gas still declines slower than oil
            months_past_boost = i - gas_boost_duration
            decay = np.exp(-0.1 * months_past_boost)
            gas_multiplier[i] = 1.0 + (gas_boost_factor - 1.0) * decay * 0.5

    gas[start_month:] *= gas_multiplier

    return _clamp_nonneg(oil, gas, water)
