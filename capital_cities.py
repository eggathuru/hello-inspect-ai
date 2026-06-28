"""
A1. Capital cities

Build a tiny eval from a handful of geography questions you hold in code (no files). Each item is a question and its one-word answer. Score a response correct if the expected answer shows up at the end of the model's reply. Default aggregate metrics are fine.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate

capital_cities_dataset = [
    Sample(input="What is the capital of France?", target="Paris"),
    Sample(input="What is the capital of Japan?", target="Tokyo"),
    Sample(input="What is the capital of Australia?", target="Canberra"),
]


@task
def capital_cities_eval():
    return Task(
        dataset=capital_cities_dataset,
        solver=generate(),
        scorer=model_graded_fact(),
    )
