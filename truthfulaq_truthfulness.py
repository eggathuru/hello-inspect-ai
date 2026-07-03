"""
Run TruthfulQA's open-ended generation split. Records have question, best_answer, correct_answers (list), incorrect_answers (list), category, and source. Use best_answer as the grading target, keep the lists and category in metadata. Give the model a system framing, grade open-ended, and report accuracy broken down by category with a bootstrapped standard error rather than the analytic one.

Source: path="truthful_qa", name="generation", split="validation".
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import accuracy, bootstrap_stderr, grouped, model_graded_qa
from inspect_ai.solver import generate


def record_to_sample(record):
    return Sample(
        input=record["question"],
        target=record["best_answer"],
        metadata={
            key: record[key]
            for key in ["category", "correct_answers", "incorrect_answers"]
        },
    )


@task
def truthfulaq_truthfulness(limit: int = 10):
    return Task(
        dataset=hf_dataset(
            path="truthfulqa/truthful_qa",
            name="generation",
            split="validation",
            sample_fields=record_to_sample,
            limit=limit,
        ),
        solver=generate(),
        scorer=model_graded_qa(),
        metrics=[grouped(accuracy(), "category"), bootstrap_stderr()],
    )
