"""
Run reading-comprehension questions where the answer is a span of the passage. Records have id, title, context, question, and answers shaped as {"text": [...], "answer_start": [...]}. Build the input from context + question, set the target to the first gold answer string, and keep the id. Score by token-overlap between the model's answer and the gold span (not exact equality), and report the mean with a standard error.

Source: path="rajpurkar/squad", split="validation".
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import f1
from inspect_ai.solver import generate, prompt_template

QUESTION_TEMPLATE = """
PASSAGE: {context}

QUESTION: {question}
"""


def record_to_sample(record):
    return Sample(
        id=record["id"],
        input=QUESTION_TEMPLATE.format(
            context=record["context"], question=record["question"]
        ),
        target=record["answers"]["text"][0],
    )


PROMPT_TEMPLATE = """
Answer the following question with a direct quote from the passage.
Your response must be only the direct quote. Do not wrap your answer
in quote symbols. Try to extract the noun that answer the question
rather than whole phrases.

[BEGIN SAMPLE]
{prompt}
[END SAMPLE]
"""


@task
def squad_extractive_qa(limit: int = 10):
    return Task(
        dataset=hf_dataset(
            path="rajpurkar/squad",
            split="validation",
            sample_fields=record_to_sample,
            limit=limit,
        ),
        solver=[
            prompt_template(PROMPT_TEMPLATE),
            generate(),
        ],
        scorer=f1(),
    )
