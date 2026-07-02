# hello-inspect-ai

A personal playground for learning the basics of [Inspect AI](https://inspect.aisi.org.uk/), the
open-source LLM evaluation framework from the UK AI Security Institute.

## What this is

I asked Claude to generate a set of practice problems for getting comfortable with the core
Inspect concepts — datasets, solvers, and scorers — and I'm working through them one at a time.
Each problem asks me to build a small `Task`: load some data into `Sample`s, wire up a solver
chain, and pick an appropriate scorer. The scripts here are **my own solutions**, so expect the
occasional rough edge or note about what didn't work.

## Layout

The practice set itself lives in three Markdown files:

- `inspect-practice-problems-set1.md` — the problems
- `inspect-practice-hints-set1.md` — hints
- `inspect-practice-answer-key-set1.md` — a worked answer key

My solutions, one Inspect `@task` per file (data files alongside where a problem reads from CSV/JSONL):

| Problem | Solution | Focus |
| --- | --- | --- |
| A1. Capital cities | `capital_cities.py` | in-code dataset, model-graded fact |
| A2. Secure-coding advice | `secure_coding_advice.py` | `system_message`, model-graded fact |
| A3. Theory-of-mind short-answers | `theory_of_mind_sa.py` | `self_critique`, model-graded QA |
| Brand-voice rewrite | `brand_voice_rewrite.py` (+ prompt/system `.txt`) | `system_message` + `prompt_template`, custom solver |
| B5. Geography QA | `geography_qa.py` (+ `geography.csv`) | loading a dataset from CSV |
| B6. Sycophancy probe | `sycophancy_probe.py` (+ `sycophancy.csv`) | CSV dataset, probing behavior |
| B7. Support-bot factuality | `support_bot_factuality.py` (+ `support.jsonl`) | loading a dataset from JSONL |
| B8. Behavior-matching | `behavior_matching.py` (+ `behavior.jsonl`) | JSONL dataset, matching behavior |

## Running an eval

The project uses [uv](https://docs.astral.sh/uv/) and requires Python 3.12.

```bash
# install dependencies
uv sync

# run a single task against a model (set the relevant provider API key first)
uv run inspect eval capital_cities.py --model anthropic/claude-opus-4-8

# run one task from a file that defines several
uv run inspect eval theory_of_mind_sa.py@theory_of_mind_sa --model anthropic/claude-opus-4-8

# browse results
uv run inspect view
```

## Status

Ongoing — a learning exercise, not a polished library.
