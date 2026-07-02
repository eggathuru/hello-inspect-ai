"""
Load HellaSwag from Hugging Face and run it as multiple choice. Each record has ctx (the context sentence), endings (a list of four candidate continuations), and label (the index of the correct ending, as a string like "2"). Convert the index into a choice letter.

Source: path="hellaswag", split="validation" (some mirrors require trust_remote_code/trust=True).
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice


def record_to_sample(record):
    return Sample(
        input=record["ctx"],
        target=chr(ord(record["label"]) + ord("a") - ord("0")),
        choices=record["endings"],
    )


@task
def hellaswag_commonsense_completion(limit=10):
    return Task(
        dataset=hf_dataset(
            path="Rowan/hellaswag",
            split="validation",
            sample_fields=record_to_sample,
            limit=limit,
        ),
        solver=multiple_choice(),
        scorer=choice(),
    )
