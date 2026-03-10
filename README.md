# declineswarm

AI-driven decline curve analysis. Spawns a swarm of Claude Code agents that each optimize [Arps decline parameters](https://petbox-dca.readthedocs.io/) for a single oil well, using hindcast MAPE as the objective function.

Each agent runs an autonomous experiment loop — proposing one parameter change at a time, evaluating it against a 12-month holdout, and committing improvements to git. The result is a per-well parameter set tuned to that well's production signature.

## How it works

```
┌─────────────┐
│  run_swarm   │  Spawns N parallel Claude Code agents
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  program.md  │────▶│  run_eval.py  │────▶│  hindcast    │
│  (agent loop)│     │  (scoring)    │     │  MAPE / R²   │
└─────────────┘     └──────────────┘     └─────────────┘
       │                    │
       ▼                    ▼
  params.json          fit.json
  trace.jsonl          hindcast.json
```

1. **`setup.py`** — Pulls 50 stratified wells (highrate, shale, stripper, conventional) from a DuckDB warehouse and creates per-well atom directories
2. **`run_swarm.py`** — Launches parallel Claude Code agents, one per well
3. **`program.md`** — The agent prompt: read params, propose a change, run eval, commit if improved
4. **`run_eval.py`** — Fits Arps decline on training data (all months minus last 12), forecasts the holdout period, computes MAPE
5. **`read_results.py`** — Portfolio-level summary across all wells

## Project structure

```
declineswarm/
├── arps.py            # Arps decline models via petbox-dca (exp/hyp/harmonic)
├── preprocessing.py   # Peak detection + outlier filtering
├── evaluation.py      # Fit-and-score pipeline
├── run_eval.py        # Hindcast evaluator (train/holdout split)
├── run_swarm.py       # Parallel agent orchestrator
├── read_results.py    # Portfolio summary
├── setup.py           # Well selection + atom directory creation
├── program.md         # Agent prompt template
└── wells/
    └── {api}/
        ├── production.csv   # Monthly oil/gas/water history
        ├── params.json      # Current tuning parameters (versioned)
        ├── fit.json         # Best fit result
        ├── hindcast.json    # Scoring history across versions
        └── trace.jsonl      # Agent reasoning log
```

## Tunable parameters

Each well's `params.json` controls two stages:

**Preprocessing** — how production history is prepared before fitting:
- `significance_threshold` — peak detection sensitivity (0.30–0.75)
- `smoothing_window` — moving average width for peak finding
- `outlier_window` / `outlier_threshold` — rolling median outlier filter

**Fitting** — Arps curve fit controls:
- `d_min` — terminal decline rate (most impactful parameter)
- `di_initial` / `b_initial` — optimizer starting guesses
- `qi_guess_strategy` — how initial rate is estimated (`first`, `max3`, `peak_value`)

## Quick start

```bash
# Install dependencies
pip install petbox-dca scipy numpy duckdb

# Set up well atoms (requires a DuckDB warehouse)
python setup.py

# Run the swarm (requires Claude Code CLI)
python run_swarm.py --max-workers 4

# View results
python read_results.py
```

### Run a subset of wells

```bash
python run_swarm.py --wells 42-001-32769 42-003-37265 --iterations 20
```

## Requirements

- Python 3.10+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code)
- [petbox-dca](https://pypi.org/project/petbox-dca/)
- scipy, numpy
- duckdb (for `setup.py` only)

## License

MIT
