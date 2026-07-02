"""
Run one subject of MMLU as multiple choice. Records have question, choices (a list of four), answer (an integer 0–3), and subject. MMLU is published as many configs (one per subject, plus all) — pick a single subject config. Convert the integer answer to a letter.

Source: path="cais/mmlu", name="high_school_mathematics", split="test".
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice


def record_to_sample(record):
    return Sample(
        input=record["question"],
        target=chr(record["answer"] + ord("a")),
        choices=record["choices"],
    )


@task
def mmlu_subject_slice(limit=10):
    return Task(
        dataset=hf_dataset(
            path="cais/mmlu",
            # On the HF website this is a "Subset". Does `hf_dataset.name` always correspond to a "Subset"?
            name="high_school_mathematics",
            split="test",
            sample_fields=record_to_sample,
            limit=limit,
        ),
        solver=multiple_choice(),
        scorer=choice(),
    )
