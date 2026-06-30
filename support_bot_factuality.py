"""
support.jsonl records have fields ticket (the customer question), gold (the correct resolution), ticket_id, and product_area. Map these onto a sample, keep the id, and carry product_area along so you can later read accuracy broken out per product area. Grade whether the response contains the gold resolution facts.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, json_dataset
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate, system_message

SYSTEM_MESSAGE = """
You are a support-bot. Suggest one or several solutions to the app user's difficulties.
"""

SCORER_TEMPLATE = """
You are assessing support-bot.

[BEGIN DATA]
***
[Customer issue]: {question}
***
[Support-bot suggestions]: {answer}
***
[Ideal answer]: {criterion}
***
[END DATA]

Does the support-bot's answers contain the ideal answer?

{instructions}
"""


@task
def support_bot_factuality():
    return Task(
        dataset=json_dataset(
            "support.jsonl",
            sample_fields=FieldSpec(
                input="ticket", target="gold", id="ticket_id", metadata=["product_area"]
            ),
        ),
        solver=[system_message(SYSTEM_MESSAGE), generate()],
        scorer=model_graded_fact(SCORER_TEMPLATE, partial_credit=True),
    )
