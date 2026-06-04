"""Configuration and result dataclasses for segmented decline analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# Model types that fit_segment() understands.
MODEL_TYPES = ("global_power", "shifted_power", "restarted_hyperbolic", "exponential")

# Number of parameters each segment model contributes (excluding the breakpoint,
# which is counted separately). Each segment carries a free *level* (the anchor
# scale Q, fixed to the observed start point when jumps are off, or fitted when
# they are on); either way it consumes one degree of freedom, so it is counted
# here. Without it BIC systematically under-penalizes extra segments and the
# fitter over-segments low-noise data.
PARAMS_PER_SEGMENT = {
    "global_power": 2,          # level, alpha
    "shifted_power": 3,         # level, alpha, c
    "restarted_hyperbolic": 3,  # level, b, D
    "exponential": 2,           # level, D
}


@dataclass
class SegmentedDeclineConfig:
    """Knobs controlling the segmented fitter. All defaults are explicit."""

    model_type: str = "shifted_power"
    min_segment_length: int = 12
    max_segments: int = 5
    criterion: str = "bic"          # "bic" or "aic"
    allow_jumps: bool = False
    min_alpha: float = 0.01
    max_alpha: float = 5.0
    min_rate: float = 1e-6
    robust: bool = False            # use soft_l1 loss in the nonlinear refit

    def __post_init__(self) -> None:
        if self.model_type not in MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {MODEL_TYPES}, got {self.model_type!r}"
            )
        if self.criterion not in ("bic", "aic"):
            raise ValueError(f"criterion must be 'bic' or 'aic', got {self.criterion!r}")
        if self.min_segment_length < 2:
            raise ValueError("min_segment_length must be >= 2 (need a slope).")
        if self.max_segments < 1:
            raise ValueError("max_segments must be >= 1.")
        if not (0 < self.min_alpha < self.max_alpha):
            raise ValueError("require 0 < min_alpha < max_alpha.")

    @property
    def params_per_segment(self) -> int:
        return PARAMS_PER_SEGMENT[self.model_type]


@dataclass
class SegmentFit:
    """Fit of a single segment over the half-open index range [start_idx, end_idx)."""

    start_idx: int
    end_idx: int            # exclusive
    start_t: float
    end_t: float
    model_type: str
    params: dict
    rss: float              # sum of squared log residuals
    rmse: float             # sqrt(rss / n) in log space
    n: int
    y_obs: Optional[np.ndarray] = None   # observed log-rate
    y_hat: Optional[np.ndarray] = None   # fitted log-rate


@dataclass
class SegmentedDeclineFit:
    """Full segmented fit for one well."""

    well_id: Optional[str]
    model_type: str
    criterion: str
    n_segments: int
    breakpoints_idx: list[int]
    breakpoints_t: list[float]
    segments: list[SegmentFit]
    rss: float
    aic: float
    bic: float
    q_hat: np.ndarray
    residuals_log: np.ndarray
    config: dict
    k_results: list[dict] = field(default_factory=list)  # per-K model-selection trace
