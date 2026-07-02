"""
Run CommonsenseQA as multiple choice, but shuffle the options on load so answer position can't leak. Records have question, choices as {"label": [...], "text": [...]}, and answerKey (a letter). Make sure shuffling keeps the correct target aligned.

Source: path="tau/commonsense_qa", split="validation".
"""

import random

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import answer, choice
from inspect_ai.solver import multiple_choice


def record_to_sample(record):
    def shuffle_choices(choices: list[str], target: str) -> tuple[list[str], str]:
        target_index = ord(target) - ord("A")
        index_to_choice = list(enumerate(choices))
        shuffled_index_to_choice = index_to_choice.copy()
        random.shuffle(shuffled_index_to_choice)
        new_target_index = target_index
        for i, choice in shuffled_index_to_choice:
            if choice == index_to_choice[target_index]:
                new_target_index = i
        return (
            [choice for _, choice in shuffled_index_to_choice],
            chr(new_target_index + ord("A")),
        )

    choices, target = shuffle_choices(record["choices"]["text"], record["answerKey"])

    return Sample(
        id=record["id"],
        input=record["question"],
        choices=choices,
        target=target,
    )


@task
def commonsenseqa_with_shuffled_options(limit: int = 10):
    return Task(
        dataset=hf_dataset(
            path="tau/commonsense_qa",
            split="validation",
            limit=limit,
            sample_fields=record_to_sample,
        ),
        solver=multiple_choice(),
        scorer=choice(),
    )
