# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

**declineswarm** — an autonomous agent swarm that tunes oil/gas decline-curve parameters. Each agent reads experiment history, proposes one parameter change, scores it via 12-month hindcast MAPE, and commits improvements to git. The hypothesis: LLMs steer parameter search better than blind grid search by reading optimization traces.

## Commands

```bash
# Setup
pip install -r requirements.txt

# Score a single well (state + API number)
python run_eval.py NM 30-025-50364

# Launch agent swarm (4 workers, 50 iterations default)
python run_swarm.py NM
python run_swarm.py NM --max-workers 8 --iterations 20
python run_swarm.py NM --wells 30-025-50364 30-025-50366

# Portfolio summary
python read_results.py NM

# Generate hindcast plots
python plot_wells.py NM

# Add a new well from CSV
python setup_from_csv.py my_well.csv --state NM --api 30-025-99999

# Stop swarm
kill_claude.bat        # Windows
./kill_claude.sh       # Linux/macOS
```

No tests, no linter. Validation is via trace.jsonl audit trail, fit.json metrics, and git history.

## Architecture

### Pipeline (3 stages)

1. **`preprocessing.py`** — Cleans raw production: smooths curve, detects decline start via peak detection, removes outliers with rolling median
2. **`arps.py`** — Fits Modified Hyperbolic decline model (qi, Di, b, d_min) using `petbox-dca` + `scipy.optimize.curve_fit` with bounded optimization
3. **`evaluation.py`** — Orchestrates: reads production.csv, chains preprocessing → fitting, computes quality metrics (R², MAPE, convergence)

### Scoring (`run_eval.py`)

Fits on months 1..N-12, forecasts months N-11..N (12-month holdout), computes hindcast MAPE. Writes `fit.json`, appends to `hindcast.json` and `trace.jsonl`.

### Agent Loop (defined in `program.md`)

Each agent: read trace.jsonl → propose ONE param change in params.json (increment version, write hypothesis in description) → run `python run_eval.py` → commit if MAPE improved, revert if not → repeat until 10 consecutive non-improvements or max turns.

### Swarm (`run_swarm.py`)

Resets wells to baseline (`defaults.py`), spawns N Claude Code subprocesses with `program.md` template injected. Each subprocess runs autonomously with `--dangerously-skip-permissions`.

### Well Atom Structure

Each `wells/{STATE}/{API}/` directory contains:

| File | Role | Mutability |
|------|------|------------|
| `production.csv` | Raw monthly data (date, oil_bbl, gas_mcf, water_bbl) | Read-only |
| `params.json` | Current parameters (version, description, preprocessing, fitting) | Agent-edited |
| `trace.jsonl` | One JSON object per experiment | Append-only |
| `hindcast.json` | MAPE by version | Append-only |
| `fit.json` | Latest fit metrics | Overwritten each run |

### Baseline Parameters (`defaults.py`)

12 tunable parameters across preprocessing (significance_threshold, smoothing_window, outlier_window, outlier_threshold, peak_merge_distance) and fitting (d_min, di_initial, b_initial, qi_guess_strategy, qi_multiplier_upper, b_upper_bound, di_upper_bound).

## Hard Constraints

- **`di_upper_bound = 0.99` — NEVER change.** petbox-dca requires Di < 1.0. This is hardcoded across all baselines and excluded from agent search space.
- **params.json audit integrity:** Before every commit, validate that parameter values mentioned in the `description` field match the actual values in `fitting`/`preprocessing` blocks. A mismatch is a silent lie in the audit trail.
