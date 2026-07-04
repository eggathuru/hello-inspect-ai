"""
A trivia.csv of short-answer questions where you want the whole normalized answer to match exactly (case/whitespace-insensitive) — partial overlap shouldn't count. Report the mean exact-match rate with a standard error.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import csv_dataset
from inspect_ai.scorer import match
from inspect_ai.solver import generate, prompt_template

PROMPT_TEMPLATE = """
Output only the answer. Not as a full sentence. All on one line.

QUESTION: {prompt}
"""


@task
def closed_book_trivia():
    return Task(
        dataset=csv_dataset("trivia.csv"),
        solver=[
            prompt_template(PROMPT_TEMPLATE),
            generate(),
        ],
        scorer=match("exact"),
    )
