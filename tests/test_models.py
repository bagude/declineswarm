"""Tests for the segment ratio models and parameter conversions."""

import numpy as np
import pytest

from segmented_decline.models import (
    global_power_ratio,
    shifted_power_ratio,
    restarted_hyperbolic_ratio,
    exponential_ratio,
    shifted_power_to_hyperbolic,
    hyperbolic_to_shifted_power,
)


def test_ratio_anchored_at_one():
    # Every ratio model equals 1 at the segment start T.
    T = 5.0
    assert global_power_ratio(np.array([T]), T, 0.8)[0] == pytest.approx(1.0)
    assert shifted_power_ratio(np.array([T]), T, 1.2, 20.0)[0] == pytest.approx(1.0)
    assert restarted_hyperbolic_ratio(np.array([T]), T, 0.1, 1.5)[0] == pytest.approx(1.0)
    assert exponential_ratio(np.array([T]), T, 0.1)[0] == pytest.approx(1.0)


def test_global_is_shifted_with_zero_c():
    t = np.arange(1, 40, dtype=float)
    T = 1.0
    g = global_power_ratio(t, T, 0.7)
    s = shifted_power_ratio(t, T, 0.7, 0.0)
    np.testing.assert_allclose(g, s, rtol=1e-12)


def test_hyperbolic_shifted_equivalence():
    # Test 3: restarted hyperbolic equals shifted power under the mapping.
    D, b, T = 0.12, 1.4, 3.0
    alpha, c = hyperbolic_to_shifted_power(D, b, T)
    t = np.arange(3, 80, dtype=float)
    hyp = restarted_hyperbolic_ratio(t, T, D, b)
    shi = shifted_power_ratio(t, T, alpha, c)
    np.testing.assert_allclose(hyp, shi, rtol=1e-10)


def test_conversion_roundtrip():
    D, b, T = 0.08, 0.9, 2.0
    alpha, c = hyperbolic_to_shifted_power(D, b, T)
    b2, D2 = shifted_power_to_hyperbolic(alpha, c, T)
    assert b2 == pytest.approx(b)
    assert D2 == pytest.approx(D)
