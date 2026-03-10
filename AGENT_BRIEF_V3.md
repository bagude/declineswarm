# res-agent: The Swarm Brief

## The Mental Model

Forget databases. Forget MCP servers for this. Forget global config.

The atom is a well.

Each well owns its own parameter set. The agent reads atoms, improves atoms, commits atoms. Git is the database. The directory is the schema. The swarm is just multiple agents working on different atoms simultaneously.

That's the whole system.

---

## The Stack

```
agent + git + atoms
```

Nothing else.

---

## Directory Structure

```
wells/
  {api_number}/
    production.csv     ← raw monthly production (immutable)
    params.json        ← DCA parameters for this well (mutable)
    fit.json           ← current best fit result (generated)
    hindcast.json      ← scoring history (generated)
    trace.jsonl        ← reasoning log for this well (generated)

run_eval.py            ← evaluate one atom (fixed)
run_swarm.py           ← spawn agents across atoms (orchestrator)
results.tsv            ← portfolio-level experiment log (generated)
```

### The Atom

Every well is a self-contained directory. An agent working on well `30-025-43838` touches nothing outside `wells/30-025-43838/`. This is the isolation guarantee that makes the swarm safe.

**`production.csv`** — immutable. Never written by the agent. Populated once from `warehouse.duckdb` at setup.

```csv
date,oil_bbl,gas_mcf,water_bbl
2018-01,4120,1823,210
2018-02,3987,1791,198
...
```

**`params.json`** — the atom's mutable state. This is what the agent optimizes.

```json
{
  "version": 1,
  "well_type": "shale",
  "description": "Baseline — global defaults",
  "preprocessing": {
    "significance_threshold": 0.50,
    "smoothing_window": 3,
    "outlier_window": 3,
    "outlier_threshold": 0.30,
    "peak_merge_distance": 3,
    "min_clean_months": 3
  },
  "fitting": {
    "d_min": 0.05,
    "di_initial": 0.50,
    "b_initial": 0.50,
    "qi_guess_strategy": "first",
    "qi_multiplier_upper": 5.0,
    "di_upper_bound": 5.0,
    "b_upper_bound": 2.0
  }
}
```

**`fit.json`** — written by the agent after each successful fit.

```json
{
  "timestamp": "2026-03-09T14:32:00Z",
  "param_version": 3,
  "model": "hyperbolic",
  "qi": 398.2,
  "di": 0.72,
  "b": 0.84,
  "d_min": 0.07,
  "r2_insample": 0.923,
  "hindcast_mape": 0.134,
  "convergence": true
}
```

**`hindcast.json`** — scoring history across all versions of params for this well.

```json
[
  {"version": 1, "mape": 0.158, "r2_outsample": 0.811},
  {"version": 2, "mape": 0.141, "r2_outsample": 0.847},
  {"version": 3, "mape": 0.134, "r2_outsample": 0.863}
]
```

**`trace.jsonl`** — one line per experiment. The reasoning record for this well.

```json
{"iteration": 2, "hypothesis": "d_min=0.07 — late-time flattening visible in production tail", "change": {"d_min": {"from": 0.05, "to": 0.07}}, "mape_before": 0.158, "mape_after": 0.141, "kept": true, "observation": "MAPE improved 10.8%. Late tail now matches observed production. d_min=0.05 was understating terminal decline for this shale well."}
{"iteration": 3, "hypothesis": "qi_guess_strategy=max3 — first month is partial (only 18 days production)", "change": {"qi_guess_strategy": {"from": "first", "to": "max3"}}, "mape_before": 0.141, "mape_after": 0.134, "kept": true, "observation": "Further improvement. First month was indeed low due to partial production period. Max of first 3 months is a better starting point for this well type."}
```

---

## The Experiment Loop (Per Atom)

```
┌─────────────────────────────────────────────────┐
│           ATOM EXPERIMENT LOOP                   │
│                                                  │
│  1. Read wells/{api}/params.json + trace.jsonl   │
│  2. Read wells/{api}/production.csv              │
│  3. Propose one parameter change with reasoning  │
│  4. Write updated params.json                    │
│  5. git add wells/{api}/params.json              │
│     git commit -m "{api} v{N}: {reasoning}"     │
│  6. python run_eval.py {api}                     │
│  7. Parse hindcast MAPE from output              │
│  8. If improved → keep commit                    │
│     If worse   → git reset HEAD~1                │
│                  git checkout wells/{api}/       │
│  9. Update trace.jsonl with observation          │
│  10. Repeat                                      │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

## The Evaluator

### `run_eval.py`

Takes one API number. Fits decline using the atom's current params. Runs hindcast. Returns scores. Fixed — never modified during experiments.

```bash
python run_eval.py 30-025-43838
```

Output (stdout, valid JSON):

```json
{
  "api": "30-025-43838",
  "param_version": 3,
  "hindcast_mape": 0.134,
  "r2_outsample": 0.863,
  "r2_insample": 0.923,
  "fit_months": 72,
  "holdout_months": 12,
  "convergence": true,
  "outlier_rejection_rate": 0.056
}
```

### Hindcast Protocol

For a well with N months of production:
1. Use months 1 through N-12 to fit the decline
2. Forecast months N-11 through N
3. Compute MAPE between forecast and actual

Wells with fewer than 18 months total history return `{"error": "insufficient_history"}` — treat as skip, not failure.

### Keep/Revert Rule

Keep if: `hindcast_mape` is lower than previous best for this atom.

That's it. One metric. One atom. No cross-well trade-offs at the atom level — those emerge at the portfolio level when you read all `fit.json` files.

---

## The Swarm

### `run_swarm.py`

Spawns one agent per atom (or subset). Agents are independent — no shared state, no coordination needed. Git isolation is per-well directory.

```python
import subprocess
import concurrent.futures
from pathlib import Path

def run_agent_on_atom(api: str, n_iterations: int = 50):
    """Spawn a Claude Code instance targeting one well."""
    subprocess.run([
        "claude",  # or whatever your agent CLI is
        "--directory", f"wells/{api}",
        "--prompt", open("program.md").read().format(api=api),
        "--max-turns", str(n_iterations)
    ])

def run_swarm(n_iterations: int = 50, max_workers: int = 4):
    apis = [p.name for p in Path("wells").iterdir() if p.is_dir()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(lambda api: run_agent_on_atom(api, n_iterations), apis)

if __name__ == "__main__":
    run_swarm()
```

`max_workers` controls how many wells are being optimized in parallel. Set to the number of agents you want running simultaneously. Start with 1, verify the loop works, then scale.

---

## `program.md` — The Agent's Instructions

This is what each agent instance receives. `{api}` is substituted at spawn time.

```markdown
You are optimizing DCA parameters for well {api}.

Your working directory is wells/{api}/. 
Do not read or write anything outside this directory.
Do not modify production.csv.

## Your loop

1. Read params.json — these are the current parameters
2. Read trace.jsonl — this is what has been tried before and what was learned
3. Read hindcast.json — this is the scoring history
4. Propose ONE parameter change. Write your reasoning in the description field.
5. Update params.json with the change (increment version)
6. Run: python run_eval.py {api}
7. If hindcast_mape improved: 
     git add wells/{api}/params.json wells/{api}/fit.json
     git commit -m "{api} v{version}: {description}"
   If hindcast_mape did not improve:
     git reset HEAD~1
     git checkout wells/{api}/params.json
8. Append your observation to trace.jsonl
9. Go to step 1

## Rules

- Change ONE parameter per experiment
- Your hypothesis must be specific — reference what you see in 
  the production data or trace history, not generic statements
- If a parameter has already been tried 3+ times with no improvement, 
  stop trying it and move to something else
- If hindcast_mape stops improving for 10 consecutive iterations, stop

## Parameter search space

Preprocessing:
  significance_threshold: 0.30 – 0.75
  smoothing_window: 2, 3, 4, 5, 6
  outlier_window: 3, 4, 5, 6, 7
  outlier_threshold: 0.15 – 0.50
  peak_merge_distance: 2, 3, 4, 5, 6, 7, 8

Fitting:
  d_min: 0.03 – 0.10          ← start here
  di_initial: 0.30 – 1.00
  b_initial: 0.30 – 1.20
  qi_guess_strategy: "first", "max3", "peak_value"
  qi_multiplier_upper: 3.0 – 10.0
  di_upper_bound: 2.0 – 5.0

## Priority

Try d_min first. It is the most impactful parameter and 
currently hardcoded wrong for most wells. 
A shale well with visible late-time production flattening 
likely needs d_min between 0.07 and 0.10.
A conventional well with clean exponential decline 
may be fine at 0.04 – 0.06.

## What good trace entries look like

Good: "d_min=0.07 — production tail from month 48 onward is 
flatter than exponential, consistent with terminal decline 
above 5%. MAPE improved from 0.158 to 0.141."

Bad: "tried different d_min value"
```

---

## Setup: Populating the Atoms

Before running the swarm, populate the well directories from `warehouse.duckdb`.

```python
# setup.py — run once
import duckdb
import json
import csv
from pathlib import Path

BASELINE_PARAMS = {
    "version": 1,
    "description": "Baseline — global defaults",
    "preprocessing": {
        "significance_threshold": 0.50,
        "smoothing_window": 3,
        "outlier_window": 3,
        "outlier_threshold": 0.30,
        "peak_merge_distance": 3,
        "min_clean_months": 3
    },
    "fitting": {
        "d_min": 0.05,
        "di_initial": 0.50,
        "b_initial": 0.50,
        "qi_guess_strategy": "first",
        "qi_multiplier_upper": 5.0,
        "di_upper_bound": 5.0,
        "b_upper_bound": 2.0
    }
}

def setup_atoms(well_list: list[str], well_types: dict[str, str]):
    con = duckdb.connect("warehouse.duckdb", read_only=True)
    
    for api in well_list:
        atom_dir = Path(f"wells/{api}")
        atom_dir.mkdir(parents=True, exist_ok=True)
        
        # Pull production history
        rows = con.execute("""
            SELECT date, oil_volume, gas_volume, water_volume
            FROM prod.production_monthly
            WHERE well_id = ?
            ORDER BY date
        """, [api]).fetchall()
        
        # Write production.csv
        with open(atom_dir / "production.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "oil_bbl", "gas_mcf", "water_bbl"])
            writer.writerows(rows)
        
        # Write params.json with well_type annotation
        params = {**BASELINE_PARAMS, "well_type": well_types.get(api, "unknown")}
        (atom_dir / "params.json").write_text(json.dumps(params, indent=2))
        
        # Initialize empty trace
        (atom_dir / "trace.jsonl").write_text("")
        (atom_dir / "hindcast.json").write_text("[]")
        
        print(f"  initialized {api} ({len(rows)} months)")
    
    con.close()
```

### Well Selection

Pick 50 wells stratified by type. Do not cherry-pick clean wells.

| Type | Count | Criteria |
|---|---|---|
| Shale | 15 | >24 months history |
| Conventional | 15 | >36 months history |
| Stripper | 10 | avg <10 BOE/day, >24 months |
| High-rate | 10 | avg >500 BOE/day, >18 months |

---

## Reading the Swarm Results

After the swarm runs, the portfolio picture emerges from the atoms:

```python
# read_results.py — run anytime to see current state
import json
from pathlib import Path
import statistics

results = []
for well_dir in Path("wells").iterdir():
    fit_file = well_dir / "fit.json"
    params_file = well_dir / "params.json"
    if not fit_file.exists():
        continue
    fit = json.loads(fit_file.read_text())
    params = json.loads(params_file.read_text())
    results.append({
        "api": well_dir.name,
        "well_type": params.get("well_type"),
        "param_version": params["version"],
        "hindcast_mape": fit["hindcast_mape"],
        "d_min": params["fitting"]["d_min"],
        "model": fit["model"]
    })

# What d_min did the swarm converge on by well type?
for well_type in ["shale", "conventional", "stripper", "highrate"]:
    subset = [r for r in results if r["well_type"] == well_type]
    if subset:
        d_mins = [r["d_min"] for r in subset]
        mapes = [r["hindcast_mape"] for r in subset]
        print(f"{well_type}: d_min mean={statistics.mean(d_mins):.3f}, "
              f"MAPE mean={statistics.mean(mapes):.3f}")
```

This is how you learn what the swarm discovered. If shale wells all converged to `d_min=0.08` and conventional wells to `d_min=0.04`, you now have empirical, well-type-specific defaults instead of one wrong global value.

---

## What Gets Committed to Git

Each agent commits to the same repo but only touches its own atom directory. The git log becomes a readable history of the swarm's discoveries:

```
git log --oneline

a3f2c1  30-025-43838 v4: outlier_threshold=0.22 — stripper variance, 0.30 was too aggressive
b7e9d2  30-025-99001 v3: qi_guess=max3 — partial first month confirmed in production.csv
c4a1f8  30-025-55201 v5: d_min=0.08 — late tail flattening, consistent with Eagle Ford behavior
d2b3e7  30-025-43838 v3: d_min=0.07 — improved MAPE 0.158→0.141
...
```

Each commit is an experiment. Each message is a hypothesis. The history is the research log.

---

## Implementation Order

1. **`setup.py`** — populate atoms from warehouse. Verify well directories look right.
2. **Update `arps.py`** — make `d_min` a parameter (not hardcoded). Default 0.05.
3. **Update `preprocessing.py`** — all thresholds accept parameters. Defaults unchanged.
4. **Update `evaluation.py`** — expose all params through `fit_decline_impl`.
5. **`run_eval.py`** — hindcast harness. Takes one API, returns JSON scores. Verify on 3-4 wells manually before trusting it.
6. **`program.md`** — agent instructions (template above).
7. **`run_swarm.py`** — orchestrator. Start with `max_workers=1`, run one well end-to-end, verify the loop works, then scale.
8. **`read_results.py`** — portfolio reader. Run after the swarm to see what was learned.

Do not build `run_swarm.py` until `run_eval.py` is verified correct on real wells. A broken evaluator running across 50 agents simultaneously is 50x harder to debug.

---

## Constraints

- Agents never write outside their atom directory
- `production.csv` is never modified
- One parameter change per commit
- `run_eval.py` is never modified during a run
- DuckDB (`warehouse.duckdb`) is read-only, only touched during setup

---

*v3 — March 2026. Agent + git + atoms. No database. No MCP. No global config. The swarm discovers per-well parameters. Git is the audit trail. The directory is the schema.*
