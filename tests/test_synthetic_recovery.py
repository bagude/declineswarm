"""End-to-end synthetic recovery and preprocessing tests."""

import numpy as np
import pandas as pd
import pytest

from segmented_decline import (
    SegmentedDeclineConfig,
    fit_segmented_decline,
    prepare_production_data,
    summarize_fit,
    fit_many_wells,
)
from segmented_decline.examples import (
    make_single_segment_power,
    make_shifted_power,
    make_two_segment_power,
)


def test_single_segment_shifted_end_to_end():
    well = make_shifted_power(n=60, alpha=1.2, c=20.0, sigma=0.03, seed=5)
    cfg = SegmentedDeclineConfig(model_type="shifted_power", max_segments=4)
    fit = fit_segmented_decline(well.t, well.q, cfg, well_id=well.well_id)
    assert fit.n_segments == 1
    seg = fit.segments[0]
    assert seg.params["alpha"] == pytest.approx(1.2, abs=0.25)
    summary = summarize_fit(fit)
    assert summary["selected_n_segments"] == 1
    assert summary["equivalent_b"][0] == pytest.approx(1.0 / seg.params["alpha"])


def test_bad_data_handling():
    # Test 6: zeros, negatives, nulls, inf are dropped/flagged.
    df = pd.DataFrame({
        "well": ["A"] * 6,
        "month": [1, 2, 3, 4, 5, 6],
        "rate": [100.0, 0.0, -5.0, np.nan, np.inf, 50.0],
    })
    clean = prepare_production_data(
        df, well_col="well", month_index_col="month", rate_col="rate",
        drop_invalid=True,
    )
    assert list(clean["q"]) == [100.0, 50.0]
    assert list(clean["t"]) == [1.0, 6.0]

    flagged = prepare_production_data(
        df, well_col="well", month_index_col="month", rate_col="rate",
        drop_invalid=False,
    )
    assert flagged["valid"].tolist() == [True, False, False, False, False, True]


def test_rate_from_volume_and_days():
    df = pd.DataFrame({
        "well": ["A", "A"],
        "month": [1, 2],
        "vol": [3000.0, 1520.0],
        "days": [30.0, 30.4375],
    })
    clean = prepare_production_data(
        df, well_col="well", month_index_col="month",
        volume_col="vol", days_col="days", normalize_monthly=True,
    )
    # rate = vol/days * 30.4375
    assert clean["q"].iloc[0] == pytest.approx(3000.0 / 30.0 * 30.4375)
    assert clean["q"].iloc[1] == pytest.approx(1520.0)


def test_fit_many_wells():
    wells = [
        make_single_segment_power(seed=1, sigma=0.04),
        make_two_segment_power(seed=2, sigma=0.04),
    ]
    frames = []
    for w in wells:
        frames.append(pd.DataFrame({"well_id": w.well_id, "t": w.t, "q": w.q}))
    df = pd.concat(frames, ignore_index=True)
    cfg = SegmentedDeclineConfig(model_type="global_power", max_segments=3)
    summary, fits = fit_many_wells(df, "well_id", "t", "q", cfg)
    assert len(fits) == 2
    assert set(summary["well_id"]) == {"single_power", "two_segment"}
