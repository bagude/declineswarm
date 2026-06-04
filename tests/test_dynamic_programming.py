"""Tests for the DP segmentation and model selection."""

import numpy as np
import pytest

from segmented_decline.config import SegmentedDeclineConfig
from segmented_decline.dp import (
    fit_segmented_decline,
    solve_best_segmentation,
    information_criteria,
    param_count,
)
from segmented_decline.fitting import build_cost_matrix
from segmented_decline.examples import make_two_segment_power, make_single_segment_power


def test_single_segment_dp_matches_cost():
    t = np.arange(1, 49, dtype=float)
    q = 1000.0 * (t / 1.0) ** (-0.7)
    cfg = SegmentedDeclineConfig(model_type="global_power", min_segment_length=12)
    cost, lookup = build_cost_matrix(t, q, cfg)
    rss, segs = solve_best_segmentation(cost, lookup, len(t), 1, cfg)
    assert len(segs) == 1
    assert rss == pytest.approx(cost[0, len(t)])


def test_two_segment_recovery():
    # Test 4: break at month 36, alpha 0.5 -> 1.4.
    well = make_two_segment_power(n=72, break_month=36, alpha1=0.5, alpha2=1.4,
                                  sigma=0.02, seed=3)
    cfg = SegmentedDeclineConfig(model_type="global_power", max_segments=3,
                                 min_segment_length=12)
    fit = fit_segmented_decline(well.t, well.q, cfg, well_id=well.well_id)
    assert fit.n_segments == 2
    assert len(fit.breakpoints_t) == 1
    assert fit.breakpoints_t[0] == pytest.approx(36.0, abs=2.0)


def test_bic_avoids_oversegmentation():
    # Test 5: clean single-segment data should select K=1 under BIC.
    well = make_single_segment_power(n=60, alpha=0.8, sigma=0.05, seed=11)
    cfg = SegmentedDeclineConfig(model_type="global_power", max_segments=5,
                                 min_segment_length=12, criterion="bic")
    fit = fit_segmented_decline(well.t, well.q, cfg, well_id=well.well_id)
    assert fit.n_segments == 1


def test_param_count_and_ic():
    cfg = SegmentedDeclineConfig(model_type="shifted_power")
    # K=2 shifted_power: 2*(level+alpha+c) + 1 breakpoint = 7
    assert param_count(2, cfg) == 7
    # global_power K=3: 3*(level+alpha) + 2 breakpoints = 8
    cfg_g = SegmentedDeclineConfig(model_type="global_power")
    assert param_count(3, cfg_g) == 8
    aic, bic = information_criteria(rss=10.0, n=50, p=5)
    assert bic > aic  # log(50) > 2


def test_infeasible_returns_none():
    t = np.arange(1, 20, dtype=float)
    q = 1000.0 * (t / 1.0) ** (-0.7)
    cfg = SegmentedDeclineConfig(min_segment_length=12)
    cost, lookup = build_cost_matrix(t, q, cfg)
    # 19 points cannot make 2 segments of length >= 12.
    assert solve_best_segmentation(cost, lookup, len(t), 2, cfg) is None
