# declineswarm


## The idea

Any real-world optimization surface, whether it's neural network hyperparameters or oil well decline curves, is riddled with local minima and heuristic traps. Grid search can navigate these surfaces, though it only ever sees numbers while requiring human observability and steering.

LLM can read the history of what's been tried, form a hypothesis about why the last tweak helped or hurt, and pick the next experiment accordingly. The reasoning trace between steps is the interesting part. 

## What this repo does

declineswarm points Claude at oil well production data and asks it to fit decline curves. Each well is an isolated directory with a CSV of monthly production, a params.json the agent can edit, and a scoring script that computes hindcast error against a 12-month holdout.

The agent proposes one parameter change at a time, runs the scorer, reads the result, and decides whether to keep or revert. Every step gets logged to a trace file with the agent's reasoning. Run enough wells in parallel and you get a swarm.

### The agent loop

Each agent runs autonomously inside its well directory (`wells/{state}/{api}/`). The loop looks like this:

1. Read `trace.jsonl` to see what has been tried so far
2. Pick ONE parameter to change in `params.json`, increment the version, and write a hypothesis in the description field ("d_min=0.07 -- production tail is flatter than exponential after month 48")
3. Run the scorer: `python run_eval.py {state} {api}`
4. Check the result:
   - **MAPE improved**: `git add` + `git commit` the new params, fit, trace, and hindcast files
   - **MAPE got worse**: `git checkout -- params.json` to revert, but keep the trace entry so the agent remembers what didn't work
5. Go to step 1

The agent stops when MAPE stops improving for 10 consecutive iterations or it runs out of turns.

### The pipeline

When `run_eval.py` scores a well, three things happen in sequence:

**1. Decline detection** (`preprocessing.detect_decline_start`) -- Raw production histories are noisy. A well might have workover bumps, shut-in months, or secondary peaks years after first production. This function smooths the production with a rolling average, finds local maxima, merges peaks that are close together, and picks the most recent peak above a significance threshold. Everything before that peak is discarded. The idea is to anchor the decline curve at the start of the current decline trend, not the beginning of the well's life.

**2. Outlier filtering** (`preprocessing.filter_outliers`) -- Even after anchoring, individual months can be anomalous (equipment failures, reporting errors, shut-ins). This computes a rolling median over a window and flags any month where production drops below a fraction of that median. Flagged months are removed so they don't pull the curve fit off track. If too few clean months survive, the well errors out.

**3. Curve fitting** (`arps.fit_arps`) -- Fits a Modified Hyperbolic (MH) decline model from petbox-dca to the cleaned data using scipy's `curve_fit`. The MH model has four parameters: initial rate (qi), initial decline rate (Di), hyperbolic exponent (b), and terminal decline rate (Dterm/d_min). The fitter uses an initial guess for qi based on a configurable strategy (first month, max of first 3, or peak value), then optimizes Di and b within bounded ranges. The fit quality is measured by R-squared on the training data.

The hindcast score (MAPE) comes from holding out the last 12 months: fit on months 1..N-12, forecast months N-11..N, compare forecast to actual.

### Search space

The agent tunes these parameters one at a time. They split into two groups:

**Preprocessing** -- controls what data the fitter sees:

| Parameter | Range | What it does |
|---|---|---|
| `significance_threshold` | 0.30 -- 0.75 | Fraction of global peak. Only peaks above this matter for anchoring. Lower values let smaller secondary peaks become the anchor point. |
| `smoothing_window` | 2 -- 6 | Months averaged for peak detection. Wider windows smooth out noise but can blur real peaks together. |
| `outlier_window` | 3 -- 7 | Rolling median window for outlier detection. Wider windows are more forgiving of short dips. |
| `outlier_threshold` | 0.15 -- 0.50 | Fraction of rolling median below which a month is flagged. 0.15 only removes severe dropouts; 0.50 removes anything below half the local trend. |
| `peak_merge_distance` | 2 -- 8 | Months. Peaks closer than this are merged into one. Prevents a noisy plateau from creating many false peaks. |

**Fitting** -- controls how the curve is shaped:

| Parameter | Range | What it does |
|---|---|---|
| `d_min` | 0.03 -- 0.10 | Terminal decline rate (annual, nominal). The floor the decline rate converges to at late time. This is the single most impactful parameter. Shale wells with visible production flattening typically need 0.07--0.10; clean exponential declines work at 0.04--0.06. |
| `di_initial` | 0.30 -- 1.00 | Initial guess for Di passed to the optimizer. Doesn't change the bounds, just where the solver starts searching. |
| `b_initial` | 0.30 -- 1.20 | Initial guess for the hyperbolic exponent b. b=0 is exponential, b=1 is harmonic, b>1 is super-hyperbolic. Most oil wells land between 0.5 and 1.5. |
| `qi_guess_strategy` | `first`, `max3`, `peak_value` | How to seed the initial rate guess. `first` uses month 1 after the anchor, `max3` takes the max of the first 3 months, `peak_value` uses the global peak. |
| `qi_multiplier_upper` | 3.0 -- 10.0 | Upper bound for qi is `qi_guess * qi_multiplier_upper`. Prevents the optimizer from fitting unrealistically high initial rates. |
| `di_upper_bound` | **0.99 (fixed)** | Hard ceiling on Di. petbox-dca requires Di < 1.0. Not tunable. |
| `b_upper_bound` | 2.0 (default) | Upper bound on b. Rarely needs changing. |

### Well directory structure

```
wells/NM/30-025-50364/
  production.csv   # monthly oil/gas/water (read-only, never modified)
  params.json      # current parameters + version + description
  trace.jsonl      # append-only log of every experiment
  hindcast.json    # MAPE history by version
  fit.json         # latest best-fit DCA parameters
```

## Prior art

Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) does the same thing for GPT training. An agent edits train.py, runs a 5-minute training loop, checks validation loss, and keeps or reverts. The shared pattern: one knob, one score, one reasoning trace, repeat.


## What happened

![Example hindcast figure for NM well 30-025-50366](docs/example_nm_30-025-50366.png)

The agents do find improvements. Whether that's because the LLM is actually reasoning about the surface or just doing structured trial and error with better bookkeeping, I honestly don't know yet. That's the fun part.

<p align="center">
  <img src="docs/example_nm_30-025-50360.png" width="49%" />
  <img src="docs/example_nm_30-025-50362.png" width="49%" />
</p>

## Running it

> **Warning: fully autonomous agents.** The swarm runs with `--dangerously-skip-permissions`, meaning each agent can read files, write files, and run commands without asking. There is no human-in-the-loop confirmation. The only way to stop a running swarm is to kill the process (`kill_claude.sh` or `kill_claude.bat`). Start small, watch the output, and understand what the agents are doing before scaling up.

### Prerequisites

- Python 3.10+
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated

### Quick start

```bash
# 1. Clone and install
git clone <this-repo>
cd declineswarm
pip install -r requirements.txt

# 2. Run the scorer on a single well to verify everything works
python run_eval.py NM 30-025-50364

# 3. Run the swarm
python run_swarm.py NM

# 4. Plot the results
python plot_wells.py NM
```

The repo ships with 50 New Mexico wells already populated.

### Defaults and what they cost

`run_swarm.py` has two knobs that control how much work (and spend) the swarm does:

- **`--max-workers`** (default: **4**) -- number of Claude Code agents running in parallel. Each agent is a separate subprocess with its own context window. 4 workers means 4 wells are being optimized simultaneously. More workers finish faster but hit your API harder.
- **`--iterations`** (default: **50**) -- maximum LLM turns per agent. Each optimization experiment takes roughly 4-5 turns (read trace, edit params, run scorer, check result, commit or revert), so 50 turns gives each agent about 10 experiments. Agents also stop early if MAPE hasn't improved for 10 consecutive iterations.

With defaults (4 workers, 50 iterations, 50 wells), the swarm processes wells in batches of 4. That's 50 wells x 50 turns = up to 2,500 LLM calls total, though early stopping usually cuts this significantly.

```bash
# Start with a single well to see what happens
python run_swarm.py NM --wells 30-025-50364 --iterations 20

# Scale up once you're comfortable
python run_swarm.py NM --max-workers 8 --iterations 50
```

### Stopping the swarm

The agents run autonomously. To stop them:

```bash
# Linux/macOS
./kill_claude.sh

# Windows
kill_claude.bat
```

This kills all Claude Code subprocesses. There is no graceful shutdown. Work from the current turn is lost, but all previously committed experiments are safe in git.

### What each step does

1. **`run_eval.py`** fits a decline curve on all-but-the-last-12 months, forecasts those 12 months, and scores the error (MAPE). This is the scorer the agent calls in its loop.
2. **`run_swarm.py`** resets each well to baseline parameters, then spawns one Claude Code instance per well with `--dangerously-skip-permissions`. Each agent reads `program.md` for its instructions: propose a parameter change, run the scorer, keep or revert, repeat. No human approval is requested at any step.
3. **`plot_wells.py`** generates hindcast figures showing actual vs. forecast production with the optimization trace.

### Options

```bash
# Target specific wells
python run_swarm.py NM --wells 30-025-50364 30-025-50366

# Read portfolio summary after the swarm finishes
python read_results.py NM
```

### Adding your own wells

Bring your own production CSV (columns: `date, oil_bbl, gas_mcf, water_bbl`):

```bash
python setup_from_csv.py my_well.csv --state NM --api 30-025-99999
python run_swarm.py NM --wells 30-025-99999
```

MIT
