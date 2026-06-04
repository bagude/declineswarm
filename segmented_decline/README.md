# segmented_decline

Multi-segment decline-curve analysis. Given a production-rate profile `q(t)`,
the package infers whether the data is better explained by one continuous
decline curve or by several decline segments separated by breakpoints —
treating the **number of segments**, the **breakpoint locations**, and the
**segment parameters** as a single optimization problem.

It is *not* a single Arps fit: segmentation is solved by dynamic programming and
the number of segments is chosen by an information criterion (BIC by default).

## Mathematical model

Each segment is a **ratio** curve anchored at its start `T_j` so that
`r_j(T_j) = 1`, and the segment rate is `q_hat(t) = Q_j * r_j(t)`. Working in
ratio form keeps the global time clock intact — we re-anchor at the segment
start without pretending the well restarted.

Supported segment models (`models.py`):

| model | ratio `r(t)` | params |
|-------|-------------|--------|
| `global_power` | `(t/T)^(-alpha)` | alpha |
| `shifted_power` (default) | `((t+c)/(T+c))^(-alpha)` | alpha, c |
| `restarted_hyperbolic` | `[1 + bD(t-T)]^(-1/b)` | b, D (mapped to shifted power) |
| `exponential` | `exp(-D(t-T))` | D |

The shifted-clock power law is equivalent to a restarted hyperbolic via
`alpha = 1/b`, `c = 1/(bD) - T`; conversions are provided both ways, so
`shifted_power` fits can report Arps-style `b` and `D`.

Fitting is done in **log-rate space** (`residual = log(q) - log(q_hat)`), so the
objective is a percentage error. For power-law models `alpha` is solved in
closed form (least squares through the origin); the shifted offset `c` is the
only nonlinear knob and is found with a 1-D bounded search. `robust=True` adds a
`soft_l1` refinement of the selected segments.

## Segmentation & model selection

`fitting.build_cost_matrix` fits every candidate segment `[a, b)` of length
`>= min_segment_length` and stores its squared-log-error RSS. `dp.py` then solves

    DP[k, end] = min over start { DP[k-1, start] + cost[start, end] }

for each `K = 1..max_segments`, scores each `K` with

    BIC = n*log(RSS/n) + p*log(n)      (AIC = n*log(RSS/n) + 2p)

and keeps the best. `p = K * params_per_segment + (K-1)` breakpoints; each
segment's level/anchor is counted as a parameter so BIC does not over-segment
low-noise data. BIC is the default because AIC tends to over-segment noisy
production.

## Quick start

```python
import numpy as np
from segmented_decline import fit_segmented_decline, SegmentedDeclineConfig, summarize_fit

t = np.arange(1, 73, dtype=float)
q = ...  # monthly rates

cfg = SegmentedDeclineConfig(model_type="shifted_power", criterion="bic", max_segments=4)
fit = fit_segmented_decline(t, q, cfg, well_id="my_well")

print(fit.n_segments, fit.breakpoints_t)
print(summarize_fit(fit))
```

Preprocess a raw DataFrame first if needed:

```python
from segmented_decline import prepare_production_data
clean = prepare_production_data(df, well_col="well_id", date_col="date",
                                volume_col="oil_volume", days_col="days_produced",
                                normalize_monthly=True)
```

Batch over many wells:

```python
from segmented_decline import fit_many_wells
summary_df, fits = fit_many_wells(clean, "well_id", "t", "q", cfg)
```

## Example script

```bash
python scripts/run_segmented_decline.py --output-dir outputs/segmented_decline
```

Generates three synthetic wells (single-segment, two-segment, shifted-clock),
fits them, prints summaries + per-K traces, and saves 4-panel diagnostic plots.

## Tests

```bash
python -m pytest tests/
```

Covers ratio models and conversions, single-segment recovery (global / shifted /
exponential), hyperbolic↔shifted equivalence, two-segment breakpoint recovery,
BIC anti-over-segmentation, and bad-data handling.

## Important caveat

A fitted breakpoint is **not automatically physical**. It is only where the
model improves the chosen objective; it may reflect a flow-regime change, an
operational event, a data artifact, or noise. Use the diagnostics and prefer the
smallest `K` that materially improves the fit.
