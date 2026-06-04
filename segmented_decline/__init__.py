"""segmented_decline — multi-segment decline-curve fitting.

Infers whether a production profile is better explained by one continuous
decline curve or by several segments separated by breakpoints, treating the
number of segments, the breakpoint locations, and the segment parameters as a
single optimization problem (dynamic programming + BIC/AIC model selection).

Quick start::

    from segmented_decline import fit_segmented_decline, SegmentedDeclineConfig
    fit = fit_segmented_decline(t, q, SegmentedDeclineConfig(model_type="shifted_power"))
    print(fit.n_segments, fit.breakpoints_t)
"""

from __future__ import annotations

from .config import (
    SegmentedDeclineConfig,
    SegmentFit,
    SegmentedDeclineFit,
    MODEL_TYPES,
)
from .models import (
    global_power_ratio,
    shifted_power_ratio,
    restarted_hyperbolic_ratio,
    exponential_ratio,
    shifted_power_to_hyperbolic,
    hyperbolic_to_shifted_power,
)
from .fitting import (
    fit_segment,
    fit_global_power_segment,
    fit_shifted_power_segment,
    fit_exponential_segment,
    fit_restarted_hyperbolic_segment,
    build_cost_matrix,
)
from .dp import (
    fit_segmented_decline,
    solve_best_segmentation,
    information_criteria,
    param_count,
)
from .preprocessing import prepare_production_data
from .diagnostics import summarize_fit, segment_table, fit_many_wells

__all__ = [
    "SegmentedDeclineConfig",
    "SegmentFit",
    "SegmentedDeclineFit",
    "MODEL_TYPES",
    "global_power_ratio",
    "shifted_power_ratio",
    "restarted_hyperbolic_ratio",
    "exponential_ratio",
    "shifted_power_to_hyperbolic",
    "hyperbolic_to_shifted_power",
    "fit_segment",
    "fit_global_power_segment",
    "fit_shifted_power_segment",
    "fit_exponential_segment",
    "fit_restarted_hyperbolic_segment",
    "build_cost_matrix",
    "fit_segmented_decline",
    "solve_best_segmentation",
    "information_criteria",
    "param_count",
    "prepare_production_data",
    "summarize_fit",
    "segment_table",
    "fit_many_wells",
]
