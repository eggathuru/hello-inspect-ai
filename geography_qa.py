from inspect_ai import Task, task
from inspect_ai.dataset import csv_dataset
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate


@task
def geography_qa():
    return Task(
        dataset=csv_dataset("geography.csv"),
        solver=generate(),
        scorer=model_graded_fact(),
    )
