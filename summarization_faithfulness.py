"""
You have article→summary pairs and want a grader to rate each summary as fully faithful, partially faithful, or unfaithful — not just pass/fail. Set a grading persona, grade open-ended, and allow partial credit so a "partially faithful" verdict lands between right and wrong in the aggregate.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import generate


@task
def summarization_faithfulness():
    return Task(
        dataset=json_dataset("summarization.jsonl"),
        solver=generate(),
        # The problem statement suggests using a custom template here but I omit it.
        scorer=model_graded_qa(partial_credit=True),
    )
