"""
Run BoolQ. Records have question, passage, and answer (a boolean). Present each as a passage+question with exactly two options — Yes and No — and map the boolean to the correct option. Score the choice.

Source: path="google/boolq", split="validation".
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import exact
from inspect_ai.solver import generate, prompt_template

QUESTION_TEMPLATE = """
PASSAGE: {passage}

QUESTION: {question}
"""


def record_to_sample(record):
    return Sample(
        input=QUESTION_TEMPLATE.format(
            question=record["question"], passage=record["passage"]
        ),
        target="Yes" if record["answer"] else "No",
    )


PROMPT_TEMPLATE = """
Answer the question using the passage.

Simply answer with 'Yes' or 'No'. Output nothing else.

{prompt}
"""


@task
def boolq_yes_no_mcq(limit: int = 10):
    return Task(
        hf_dataset(
            path="google/boolq",
            split="validation",
            sample_fields=record_to_sample,
            limit=limit,
        ),
        solver=[prompt_template(PROMPT_TEMPLATE), generate()],
        scorer=exact(),
    )
