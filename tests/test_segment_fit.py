"""Tests for single-segment fitting."""

import numpy as np
import pytest

from segmented_decline.config import SegmentedDeclineConfig
from segmented_decline.fitting import (
    fit_global_power_segment,
    fit_shifted_power_segment,
    fit_exponential_segment,
)


def test_global_power_recovery():
    # Test 1: q(t) = 1000 * (t/1)^(-0.8)
    t = np.arange(1, 61, dtype=float)
    q = 1000.0 * (t / 1.0) ** (-0.8)
    cfg = SegmentedDeclineConfig(model_type="global_power", min_segment_length=12)
    fit = fit_global_power_segment(t, q, 0, len(t), cfg)
    assert fit.params["alpha"] == pytest.approx(0.8, abs=1e-6)
    assert fit.rss < 1e-12


def test_shifted_power_recovery():
    # Test 2: q(t) = 1000 * ((t+20)/(1+20))^(-1.2)
    t = np.arange(1, 61, dtype=float)
    alpha_true, c_true = 1.2, 20.0
    q = 1000.0 * ((t + c_true) / (1.0 + c_true)) ** (-alpha_true)
    cfg = SegmentedDeclineConfig(model_type="shifted_power", min_segment_length=12)
    fit = fit_shifted_power_segment(t, q, 0, len(t), cfg)
    assert fit.params["alpha"] == pytest.approx(alpha_true, abs=0.05)
    assert fit.params["c"] == pytest.approx(c_true, abs=2.0)
    assert fit.rmse < 1e-3


def test_shifted_power_reports_arps_equivalents():
    t = np.arange(1, 61, dtype=float)
    q = 1000.0 * ((t + 15.0) / (1.0 + 15.0)) ** (-1.0)
    cfg = SegmentedDeclineConfig(model_type="shifted_power")
    fit = fit_shifted_power_segment(t, q, 0, len(t), cfg)
    # b = 1/alpha, D = 1/(b*(T+c)) must be present and positive here.
    assert fit.params["b_equiv"] == pytest.approx(1.0 / fit.params["alpha"])
    assert fit.params["D_equiv"] > 0


def test_exponential_recovery():
    t = np.arange(1, 41, dtype=float)
    D_true = 0.1
    q = 500.0 * np.exp(-D_true * (t - 1.0))
    cfg = SegmentedDeclineConfig(model_type="exponential")
    fit = fit_exponential_segment(t, q, 0, len(t), cfg)
    assert fit.params["D"] == pytest.approx(D_true, abs=1e-6)
    assert fit.rss < 1e-12


def test_alpha_is_clamped():
    # Steeper-than-allowed decline should be clamped to max_alpha.
    t = np.arange(1, 31, dtype=float)
    q = 1000.0 * (t / 1.0) ** (-3.0)
    cfg = SegmentedDeclineConfig(model_type="global_power", max_alpha=1.0)
    fit = fit_global_power_segment(t, q, 0, len(t), cfg)
    assert fit.params["alpha"] == pytest.approx(1.0)
