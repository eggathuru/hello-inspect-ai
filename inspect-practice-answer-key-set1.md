# Inspect AI — Practice Set 1: Answer Key & Coverage Map

Open this **only after attempting a set.** Each entry gives the intended **loader → solver(s) → scorer → metric**, the field args that matter, a short code fragment where the mapping is the hard part, and the one thing the problem is meant to teach. There's often more than one defensible design — if yours hits the same primitives, it counts.

Common imports assumed:
```python
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, csv_dataset, json_dataset, hf_dataset, FieldSpec
from inspect_ai.solver import (
    generate, system_message, prompt_template, chain_of_thought,
    self_critique, multiple_choice, use_tools,
)
from inspect_ai.scorer import (
    match, includes, exact, f1, choice, model_graded_qa, model_graded_fact,
    accuracy, stderr, mean, bootstrap_stderr, grouped, metric,
)
from inspect_ai.tool import python, web_search
```

---

## Set A

**A1 — Capitals**
- Loader: `Sample` list (in code). Solver: `generate()`. Scorer: `match()` (default `location="end"`). Metric: `accuracy()` + `stderr()` (default).
- Teaches: the minimal pipeline; `match` checks a *position*, so "ends with target" is the default and tolerates trailing words.

**A2 — Secure-coding advice**
- Loader: `Sample` list. Solver: `[system_message(SYSTEM), generate()]`. Scorer: `model_graded_fact()`. Metric: default.
- Teaches: `model_graded_fact` vs `model_graded_qa` — use *fact* when you're checking the output **contains** specific facts rather than is a fully correct open answer.

**A3 — Theory-of-mind**
- Loader: `Sample` list. Solver: `[generate(), self_critique()]` (a `system_message` is optional). Scorer: `model_graded_qa()`. Metric: default.
- Teaches: `self_critique` is a *second* stage after `generate` — it replays the first answer to a model and regenerates. Open-ended → QA grader.

**A4 — Brand-voice rewrite**
- Loader: `Sample` list. Solver: `[system_message(VOICE_GUIDE), prompt_template("Rewrite this note: {prompt}"), generate()]`. Scorer: `model_graded_qa()`. Metric: default.
- Teaches: two injection points — `system_message` (persistent persona) and `prompt_template` (per-prompt reshaping via `{prompt}`).

---

## Set B

**B5 — Geography CSV (direct)**
- Loader: `csv_dataset("geography.csv")` (columns already `input`/`target` → no mapping). Solver: `generate()`. Scorer: `match()`. Metric: default.
- Teaches: when columns match Inspect's names, just pick the loader for the format.

**B6 — Sycophancy CSV (fixed criterion)**
- Loader: `csv_dataset("sycophancy.csv", FieldSpec(input="prompt"))` — rename only, **no target**. Solver: `[system_message(NEUTRAL), prompt_template(...), generate()]`. Scorer: `model_graded_qa(template=SYCOPHANCY_TEMPLATE)` where the rubric is hardcoded in the template (uses `{question}`/`{answer}`/`{instructions}`, **omits** `{criterion}`). Metric: default.
- Teaches: `FieldSpec(input=...)` with no target; how to grade when the criterion is the same for every sample instead of per-row.

**B7 — Support-bot JSONL (grouped metric)**
```python
ds = json_dataset("support.jsonl", FieldSpec(
    input="ticket", target="gold", id="ticket_id", metadata=["product_area"]))
```
- Solver: `generate()`. Scorer: `model_graded_fact()`. Metric: `metrics=[grouped(accuracy(), "product_area"), stderr()]`.
- Teaches: full `FieldSpec` (input/target/id/metadata); `grouped(metric, key)` computes the metric per metadata group (also gives clustered SEs).

**B8 — Behavior-matching JSONL**
```python
ds = json_dataset("behavior.jsonl", FieldSpec(
    input="question", target="answer_matching_behavior",
    id="question_id", metadata=["label_confidence"]))
```
- Solver: `[system_message(...), generate()]`. Scorer: `model_graded_qa()`. Metric: default.
- Teaches: `target` can come from an oddly-named field; metadata can be a numeric confidence kept for later.

---

## Set C

**C9 — Trivia MCQ CSV (assemble choices)**
```python
def record_to_sample(r):
    return Sample(
        input=r["question"],
        choices=[r["opt_a"], r["opt_b"], r["opt_c"], r["opt_d"]],
        target=r["correct"],            # already a letter
    )
ds = csv_dataset("trivia_mc.csv", record_to_sample)
```
- Solver: `multiple_choice()`. Scorer: `choice()`. Metric: default.
- Teaches: when choices are spread across columns, `FieldSpec` can't gather them — use `record_to_sample`. `multiple_choice` + `choice` are a pair (don't call `generate()` yourself).

**C10 — HellaSwag**
```python
def record_to_sample(r):
    return Sample(input=r["ctx"], choices=r["endings"],
                  target=chr(ord("A") + int(r["label"])))
ds = hf_dataset(path="hellaswag", split="validation",
                sample_fields=record_to_sample, trust=True)
```
- Solver: `multiple_choice()`. Scorer: `choice()`. Metric: default.
- Teaches: `hf_dataset(path, split, sample_fields=...)`; string index → choice letter. (`trust=True` if the mirror needs remote code.)

**C11 — MMLU subject**
```python
def record_to_sample(r):
    return Sample(input=r["question"], choices=r["choices"],
                  target=chr(ord("A") + r["answer"]))   # answer is an int
ds = hf_dataset(path="cais/mmlu", name="high_school_mathematics",
                split="test", sample_fields=record_to_sample)
```
- Solver: `multiple_choice()`. Scorer: `choice()`. Metric: default.
- Teaches: `name=` selects the config/subset; int answer → letter.

**C12 — ARC-Challenge (messy answer key)**
```python
def record_to_sample(r):
    labels = r["choices"]["label"]          # e.g. ["A","B","C","D"] or ["1","2","3","4"]
    texts  = r["choices"]["text"]
    key    = r["answerKey"]
    idx    = labels.index(key)              # works whether key is letter or number
    return Sample(input=r["question"], choices=texts,
                  target=chr(ord("A") + idx))
ds = hf_dataset(path="allenai/ai2_arc", name="ARC-Challenge",
                split="test", sample_fields=record_to_sample)
```
- Solver: `multiple_choice()`. Scorer: `choice()`. Metric: default.
- Teaches: choices as a dict of parallel lists; normalize an inconsistent `answerKey` via the per-row `label` list.

**C13 — CommonsenseQA (shuffled)**
```python
def record_to_sample(r):
    labels = r["choices"]["label"]; texts = r["choices"]["text"]
    idx = labels.index(r["answerKey"])
    return Sample(input=r["question"], choices=texts,
                  target=chr(ord("A") + idx))
ds = hf_dataset(path="tau/commonsense_qa", split="validation",
                sample_fields=record_to_sample, shuffle_choices=True)
```
- Solver: `multiple_choice()`. Scorer: `choice()`. Metric: default.
- Teaches: `shuffle_choices=True` (or `ds.shuffle_choices(seed=42)`) re-orders options **and** re-points the target for you; `choice()` unshuffles before scoring.

---

## Set D

**D14 — GSM8K final answer**
```python
def record_to_sample(r):
    return Sample(input=r["question"],
                  target=r["answer"].split("####")[-1].strip(),
                  metadata={"solution": r["answer"]})
ds = hf_dataset(path="gsm8k", name="main", split="test",
                sample_fields=record_to_sample)
```
- Solver: `[chain_of_thought(), generate()]`. Scorer: `match(numeric=True)` (CoT puts the final answer last; `numeric=True` compares numbers and strips `$`/`,`). `math()` is a valid alternative. Metric: default.
- Teaches: extract a target from a formatted string; `chain_of_thought` rewrites the prompt then `generate` answers; numeric matching.

**D15 — GSM8K with self-correction (same data)**
- Loader: reuse D14's `hf_dataset`. Solver: `[chain_of_thought(), generate(), self_critique()]`. Scorer: `model_graded_qa()` (revised answers are free-form). Metric: default.
- Teaches: same dataset, different middle — add a self-revision stage and switch from string match to a model grader.

**D16 — TruthfulQA (grouped + bootstrap)**
```python
def record_to_sample(r):
    return Sample(input=r["question"], target=r["best_answer"],
                  metadata={"category": r["category"],
                            "correct": r["correct_answers"],
                            "incorrect": r["incorrect_answers"]})
ds = hf_dataset(path="truthful_qa", name="generation",
                split="validation", sample_fields=record_to_sample)
```
- Solver: `[system_message(...), chain_of_thought(), generate()]` (CoT is the natural reasoning aid here — that's its 3rd use). Scorer: `model_graded_qa()`. Metric: `metrics=[grouped(accuracy(), "category"), bootstrap_stderr()]`.
- Teaches: `best_answer` as target; group the headline metric by a metadata field and swap the analytic SE for `bootstrap_stderr`.

**D17 — Summarization faithfulness (partial credit)**
- Loader: `Sample` list or small `json_dataset`. Solver: `[system_message(GRADER_PERSONA), generate()]`. Scorer: `model_graded_qa(partial_credit=True)`. Metric: default (now `PARTIAL` → 0.5 in the mean).
- Teaches: `partial_credit=True` lets the grader return `P`, which `value_to_float` maps to 0.5 — a middle score in the aggregate.

**D18 — Essay rubric (panel of graders)**
- Loader: `Sample` list. Solver: `[system_message(STRICT), prompt_template(RUBRIC_TEMPLATE), generate()]`. Scorer:
```python
model_graded_qa(model=["openai/gpt-4o",
                       "anthropic/claude-sonnet-4-6",
                       "google/gemini-2.5-pro"])
```
Metric: default.
- Teaches: passing a **list** of grader models triggers majority vote (via `multi_scorer` with a mode reducer); rubric packaging is a templating job.

---

## Set E

**E19 — Arithmetic with a code tool**
- Loader: `Sample` list. Solver: `[use_tools([python()]), generate()]`. Scorer: `match(numeric=True)` (or `includes()`). Metric: default. (Needs a `sandbox`, e.g. `sandbox="docker"`, since `python()` runs untrusted code.)
- Teaches: `use_tools` exposes the tool; `generate` runs the tool-use loop automatically — no custom loop required.

**E20 — Unit-conversion with a tool**
- Loader: `Sample` list. Solver: `[system_message("Use the tool; do not estimate."), use_tools([python()]), generate()]`. Scorer: `model_graded_fact()`. Metric: default.
- Teaches: stacking persona + tool exposure + generation; fact-grading the numeric result.

**E21 — Web-grounded QA**
- Loader: `Sample` list. Solver: `[use_tools([web_search()]), generate()]`. Scorer: `model_graded_qa()`. Metric: default.
- Teaches: a different standard tool; `generate` may call it several times on its own — that built-in loop is the boundary *before* you'd reach for `react()`.

---

## Set F

**F22 — SQuAD extractive QA (f1)**
```python
def record_to_sample(r):
    return Sample(
        input=f"{r['context']}\n\nQuestion: {r['question']}",
        target=r["answers"]["text"][0],     # nested list
        id=r["id"])
ds = hf_dataset(path="rajpurkar/squad", split="validation",
                sample_fields=record_to_sample)
```
- Solver: `generate()`. Scorer: `f1()`. Metric: `mean()` + `stderr()` (what `f1` reports).
- Teaches: combine two fields into the input and pull a target out of a nested list (→ `record_to_sample`, not `FieldSpec`); `f1` for short free-text answers reports a **mean**, not accuracy.

**F23 — Closed-book trivia (exact)**
- Loader: `csv_dataset("trivia.csv")`. Solver: `generate()`. Scorer: `exact()`. Metric: `mean()` + `stderr()`.
- Teaches: `exact` requires the **whole normalized output** to equal the target (vs `includes`/`match`), and reports a mean.

**F24 — BoolQ as two-option MCQ**
```python
def record_to_sample(r):
    return Sample(
        input=f"{r['passage']}\n\nQuestion: {r['question']}",
        choices=["Yes", "No"],
        target="A" if r["answer"] else "B")
ds = hf_dataset(path="google/boolq", split="validation",
                sample_fields=record_to_sample)
```
- Solver: `multiple_choice()`. Scorer: `choice()`. Metric: default.
- Teaches: turn a boolean into a built-in two-item choice list with a letter target, then use the multiple-choice path.

---

## Set G

**G25 — Safety-advice self-critique**
- Loader: `Sample` list. Solver: `[system_message(CAREFUL_ADVISOR), generate(), self_critique()]`. Scorer: `model_graded_qa()`. Metric: default.
- Teaches: `self_critique` once more, after a persona'd first pass.

**G26 — Code-explanation with typed metadata**
```python
from pydantic import BaseModel

class CodeMeta(BaseModel, frozen=True):   # frozen → read-only metadata
    difficulty: str
    lang: str

ds = json_dataset("code_explain.jsonl", FieldSpec(
    input="snippet", target="reference", metadata=CodeMeta))
# read back type-safely in a solver/scorer: state.metadata_as(CodeMeta)
```
- Solver: `[system_message(SENIOR_REVIEWER), prompt_template("Explain this code:\n{prompt}"), generate()]`. Scorer: `model_graded_qa()`. Metric: `metrics=[grouped(accuracy(), "difficulty"), stderr()]`.
- Teaches: a `frozen` Pydantic model wired into `FieldSpec(metadata=...)` validates on load and is read via `metadata_as`; group the metric by a metadata key.

**G27 — Refusal-rate with a custom metric**
```python
@metric
def refusal_rate():
    def m(scores):
        disallowed = [s for s in scores if s.sample_metadata.get("disallowed")]
        refused = [s for s in disallowed if s.value == "C"]   # C == refused
        return len(refused) / len(disallowed) if disallowed else 0.0
    return m
```
- Loader: `Sample` list (set `metadata={"disallowed": True/False}` per item) or small CSV. Solver: `[system_message(...), generate()]`. Scorer: `model_graded_qa(template=REFUSAL_TEMPLATE)` (C = refused, I = complied). Metric: override on the Task → `metrics=[refusal_rate()]`.
- Teaches: a custom `@metric` summarizes per-sample scores your way; Task-level `metrics=` overrides the scorer's defaults. (Exact `Score`/sample-metadata accessor names: confirm against the `inspect_ai.scorer` reference — the *idea* is reading each score + its sample metadata.)

**G28 — ARC with epochs + reducer**
```python
from inspect_ai import Epochs

@task
def arc_epochs():
    return Task(
        dataset=ds,                       # the C12 ARC dataset
        solver=multiple_choice(),
        scorer=choice(),
        epochs=Epochs(5, reducer="at_least_1"),   # solved if right ≥1 of 5
    )
# CLI equivalent: inspect eval arc.py --epochs 5 --epochs-reducer at_least_1
```
- Teaches: `epochs` runs repeated passes; the **reducer** decides how per-epoch scores collapse — `at_least_1` ("right at least once") vs the default `mean`. Expect the headline number to rise vs a single epoch.

---

## Coverage matrix

Each target primitive lands **≥3 times** (excluding `generate`, which is everywhere by design).

### Dataset layer
| Primitive | Count | Problems |
|---|---|---|
| `Sample` (in-code source) | 8 | A1, A2, A3, A4, E19, E20, E21, G25 (+ D17/G27 optional) |
| `csv_dataset` | 4 | B5, B6, C9, F23 |
| `json_dataset` | 4 | B7, B8, D17*, G26 |
| `hf_dataset` | 10 | C10, C11, C12, C13, D14, D15, D16, F22, F24, G28 |
| `FieldSpec` | 4 | B6 (input-only), B7 (full+metadata), B8 (matching-behavior), G26 (typed) |
| `record_to_sample` | 10 | C9, C10, C11, C12, C13, D14, D16, F22, F24 (+ D15 reuse) |

`hf_dataset` arg coverage: `path`+`split`+`sample_fields` everywhere; `name`/config in C11, C12, D14, D16; `shuffle_choices` in C13; `trust` in C10.

### Solvers
| Primitive | Count | Problems |
|---|---|---|
| `generate` | ~all | every problem |
| `system_message` | 11 | A2, A4, B6, B8, D16, D17, D18, E20, G25, G26, G27 |
| `prompt_template` | 4 | A4, B6, D18, G26 |
| `chain_of_thought` | 3 | D14, D15, D16 |
| `self_critique` | 3 | A3, D15, G25 |
| `multiple_choice` | 6 | C9, C10, C11, C12, C13, F24 |
| `use_tools` | 3 | E19, E20, E21 |

### Scorers
| Primitive | Count | Problems |
|---|---|---|
| `model_graded_qa` | 12 | A3, A4, B6, B8, D15, D16, D17, D18, E21, G25, G26, G27 |
| `model_graded_fact` | 3 | A2, B7, E20 |
| `choice` | 7 | C9, C10, C11, C12, C13, F24, G28 |
| `match` | 3 | A1, B5, D14 (+ E19 alt) |
| `exact` | 1 | F23 |
| `f1` | 1 | F22 |

### Metrics (beyond default accuracy + stderr)
| Variant | Problems |
|---|---|
| `mean` + `stderr` (from `f1`/`exact`) | F22, F23 |
| `bootstrap_stderr` | D16 |
| `grouped(...)` / clustered SE | B7, D16, G26 |
| partial credit (`P` → 0.5) | D17 |
| multi-grader majority vote | D18 |
| custom `@metric` (+ Task-level override) | G27 |
| epochs + reducer (`at_least_1`) | G28 |

\* D17 can be `Sample` or `json_dataset` depending on how you set it up.

---

### A note on accuracy of details
Field names and HF paths reflect the common dataset schemas as of early 2026. A few exact API surface details worth confirming against the `inspect_ai` reference when you implement them: the precise `Score`/sample-metadata accessors inside a custom metric (G27), the exact `Epochs` reducer name spelling (G28), and whether your installed `hf_dataset` needs `trust=True` for a given mirror (C10). The primitives and the shape of each solution are correct; these are the spots where a signature might have drifted.
