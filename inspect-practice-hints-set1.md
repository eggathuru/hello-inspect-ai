# Inspect AI — Practice Set 1: Hints

Light nudges only — they point at the tricky part, not the tool to use. If a hint isn't enough, the full intended solution is in the **answer-key file**.

---

## A1 — Capital cities
The answer won't always be the model's last token — it might trail explanation. Think about *where* in the output you require the target to appear.

## A2 — Secure-coding advice
The "answer must *contain* these facts" framing matters — it points to a narrower kind of model grading than full open-ended QA.

## A3 — Theory-of-mind short-answers
"Take a pass, then improve it" is a two-stage solver chain — the second stage replays the first answer back to a model for a refined attempt.

## A4 — Brand-voice rewrite
There are two distinct ways to inject text here: one sets a persistent persona, the other reshapes each individual prompt with a `{prompt}` placeholder. You want both.

## B5 — Geography QA from CSV
When the columns are already `input`/`target`, there's nothing to remap — the question is which loader matches the file format.

## B6 — Sycophancy probe from CSV
The column isn't called `input`, so it needs mapping — but there's no `target` at all. How do you grade when the criterion is the same for every sample instead of living in the data?

## B7 — Support-bot factuality from JSONL
Two things to carry beyond input/target: a stable id and an extra field you'll *group the metric by* later. Both can be declared without writing a function.

## B8 — Behavior-matching from JSONL
This is the same declarative-mapping tool as B7 — the lesson is just that target can come from an oddly-named field, and metadata can be a numeric confidence you keep for later analysis.

## C9 — Trivia MCQ from CSV (split option columns)
Choices need to be a *list* on the sample, but they're spread across separate columns. A declarative rename can't gather columns into a list — you'll need the function form.

## C10 — HellaSwag commonsense completion
The label is an integer position into `endings` — you need to turn `2` into the corresponding letter (A/B/C/…) when building each sample.

## C11 — MMLU subject slice
Two new things vs. HellaSwag: the dataset is selected by a *config/subset name*, not just a path; and the answer is an int, not a string.

## C12 — ARC-Challenge science MCQ
Choices arrive as a dict of parallel lists, and the answer key's format isn't guaranteed. You'll want to normalize both inside your mapping function.

## C13 — CommonsenseQA with shuffled options
Shuffling choices is a built-in option on the loader (or a dataset method) that also re-points the target — you shouldn't have to re-map the answer yourself after shuffling.

## D14 — GSM8K final-answer accuracy
The target is buried at the end of a formatted string — extract the number after `####` in your mapping. The "reason step by step" part is a ready-made prompting solver.

## D15 — GSM8K with self-correction (same data, different chain)
Same dataset, different middle of the pipeline: add a self-revision stage after the reasoning stage, and switch to a grader that tolerates free-form answers.

## D16 — TruthfulQA truthfulness (grouped + bootstrapped)
Two metric tweaks: group the headline metric by a metadata field, and swap the default standard-error estimator for the resampling-based one.

## D17 — Summarization faithfulness with partial credit
"Partially right counts for something" is a switch on the model grader, and it changes how scores map to numbers in the metric.

## D18 — Essay rubric with a panel of graders
The model grader accepts more than one grader model; passing several triggers a vote. The rubric/essay packaging is a templating job.

## E19 — Arithmetic with a code tool
Exposing a tool to the model is one solver; the actual generation is another. Pair them. (No custom loop needed — the generation step runs the tool calls for you.)

## E20 — Unit-conversion assistant with a tool
Three pieces stack here: the persona, the tool exposure, and generation. Grading targets a specific fact in the output.

## E21 — Web-grounded factual QA with a search tool
Same shape as E19/E20 with a different tool. Note: the generation step may call the tool more than once on its own — that's the built-in loop, not an agent you wrote.

## F22 — SQuAD extractive QA (token-overlap metric)
You need to combine two fields into the input and pull the target out of a nested list — that's the function form of mapping. The scorer is the one designed for short free-text answers, and it reports a mean rather than accuracy.

## F23 — Closed-book trivia, exact match
"The whole answer must match after normalization" is a different scorer than "the target appears somewhere" — and it reports a mean, not accuracy.

## F24 — BoolQ yes/no as two-option MCQ
A boolean target becomes a two-item choice list. Build the choices yourself in the mapping and set the target letter from the bool, then use the multiple-choice path.

## G25 — Safety-advice self-critique (self-critique #3)
Persona + answer + self-revision + open-ended grading. You've built the revision stage before in Set D.

## G26 — Code-explanation grading with typed metadata
"Typed, read-only metadata validated on load" points at a Pydantic model (frozen) wired into the mapping spec, then read back type-safely. Grouping the metric by difficulty is the same grouping idea from D16/B7.

## G27 — Refusal-rate eval with a custom metric
The grader still emits per-sample scores; the novelty is supplying your *own* metric function to summarize them instead of the default accuracy.

## G28 — ARC with epochs and a pass-style reducer
Repeated passes are a Task option; how the repeated scores collapse into one is an *epoch reducer* — and "right at least once" is a specific reducer, not the default mean.
