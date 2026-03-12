# declineswarm

A toy experiment in letting LLMs optimize things they probably shouldn't.

## The idea

What if you gave an LLM a knob, a score, and told it to improve?

Any real-world optimization surface, whether it's neural network hyperparameters or oil well decline curves, is riddled with local minima and heuristic traps. Grid search can navigate these surfaces, though it only ever sees numbers.

An LLM can read the history of what's been tried, form a hypothesis about why the last tweak helped or hurt, and pick the next experiment accordingly. The reasoning trace between steps is the interesting part. That's what separates "search" from "guessing."

## What this repo does

declineswarm points Claude at oil well production data and asks it to fit decline curves. Each well is an isolated directory with a CSV of monthly production, a params.json the agent can edit, and a scoring script that computes hindcast error against a 12-month holdout.

The agent proposes one parameter change at a time, runs the scorer, reads the result, and decides whether to keep or revert. Every step gets logged to a trace file with the agent's reasoning. Run enough wells in parallel and you get a swarm.

## Prior art

Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) does the same thing for GPT training. An agent edits train.py, runs a 5-minute training loop, checks validation loss, and keeps or reverts. The shared pattern: one knob, one score, one reasoning trace, repeat.

declineswarm swaps the domain from neural nets to petroleum engineering and trades depth for breadth, running many wells in parallel instead of iterating on a single model.

## What happened

![Example hindcast figure for NM well 30-025-50366](docs/example_nm_30-025-50366.png)

The agents do find improvements. Whether that's because the LLM is actually reasoning about the surface or just doing structured trial and error with better bookkeeping, I honestly don't know yet. That's the fun part.

<p align="center">
  <img src="docs/example_nm_30-025-50360.png" width="49%" />
  <img src="docs/example_nm_30-025-50362.png" width="49%" />
</p>

## Running it

```bash
pip install petbox-dca scipy numpy duckdb matplotlib
python setup_nm.py
python run_swarm.py NM --max-workers 5
```

MIT
