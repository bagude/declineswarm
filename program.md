You are optimizing DCA parameters for well {api}.

Your working directory is wells/{state}/{api}/.
Do not read or write anything outside this directory.
Do not modify production.csv.

## Your loop

1. Read params.json — these are the current parameters
2. Read trace.jsonl — this is what has been tried before and what was learned
3. Read hindcast.json — this is the scoring history
4. Propose ONE parameter change. Write your reasoning in the description field.
5. Update params.json with the change (increment version)
6. Run: python run_eval.py {state} {api}
7. If hindcast_mape improved:
     git add wells/{state}/{api}/params.json wells/{state}/{api}/fit.json
     git commit -m "{api} v{version}: {description}"
   If hindcast_mape did not improve:
     git reset HEAD~1
     git checkout wells/{state}/{api}/params.json
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
