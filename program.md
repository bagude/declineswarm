You are optimizing DCA parameters for well {api}.

Your working directory is wells/{state}/{api}/.
Do not read or write anything outside this directory.
Do not modify production.csv.

## First step

1. Read production.csv to understand the well's production profile
2. Read params.json — these are the starting parameters (version 1)
3. Run: python run_eval.py {state} {api}
   This evaluates the baseline and creates the first trace entry.
   Note the starting MAPE — this is what you're trying to beat.

## Your loop

1. Read trace.jsonl to see what has been tried and learned so far
2. Propose ONE parameter change. Update params.json:
   - Increment version
   - Write your reasoning in the description field.
     The description MUST include the exact param=value you changed
     (e.g. "d_min=0.07 — ..."). This is validated before commit.
3. Run: python run_eval.py {state} {api}
   (This automatically logs the result to trace.jsonl and hindcast.json)
4. Check the output:
   - If hindcast_mape IMPROVED: keep it — run:
       git add wells/{state}/{api}/params.json wells/{state}/{api}/fit.json wells/{state}/{api}/trace.jsonl wells/{state}/{api}/hindcast.json
       git commit -m "{api} v{version}: {description}"
   - If hindcast_mape DID NOT improve: revert params.json ONLY — run:
       git checkout -- wells/{state}/{api}/params.json
     Do NOT revert trace.jsonl, hindcast.json, or fit.json.
5. Go to step 1

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
  (di_upper_bound is hardcoded at 0.99 — petbox-dca requires Di < 1.0. Do not change it.)

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
