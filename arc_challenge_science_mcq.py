"""
Run the ARC-Challenge science questions. Records have question, choices shaped as {"text": [...], "label": [...]}, and answerKey. Watch out: answerKey is usually a letter ("A"–"D") but in some rows it's a number ("1"–"4"), and the per-row label list tells you the actual choice labels. Build clean choices + a correct letter regardless.

Source: path="allenai/ai2_arc", name="ARC-Challenge", split="test".
"""

from os import path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice


def record_to_sample(record):
    def answer_to_choice(answer: str) -> str:
        if answer.isalpha():
            return answer
        else:  # answer.isdigit() should be true
            return chr(ord(answer) + ord("A") - ord("1"))

    return Sample(
        id=record["id"],
        input=record["question"],
        choices=record["choices"]["text"],
        target=answer_to_choice(record["answerKey"]),
    )


@task
def arc_challenge_science_mcq(limit: int = 10):
    return Task(
        dataset=hf_dataset(
            path="allenai/ai2_arc",
            split="test",
            name="ARC-Challenge",
            sample_fields=record_to_sample,
            limit=limit,
        ),
        solver=multiple_choice(),
        scorer=choice(),
    )
