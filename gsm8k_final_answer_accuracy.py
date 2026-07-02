"""
D14. GSM8K final-answer accuracy

Run grade-school math word problems. Records have question and answer, where answer is a full worked solution ending in a line like #### 42. Your target should be just that final number. Prompt the model to reason step by step, then check the final answer. Keep the full worked solution in metadata for later inspection.

Source: path="gsm8k", name="main", split="test".
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import chain_of_thought, generate


def record_to_sample(record):
    reasoning, target = record["answer"].split("####")
    return Sample(
        input=record["question"],
        target=target,
        metadata={"reasoning": reasoning},
    )


@task
def gsm8k_final_answer_accuracy(limit: int = 10):
    return Task(
        dataset=hf_dataset(
            path="openai/gsm8k",
            name="main",
            split="test",
            sample_fields=record_to_sample,
            limit=limit,
        ),
        solver=[
            chain_of_thought(),
            generate(),
        ],
        scorer=model_graded_fact(),
    )
