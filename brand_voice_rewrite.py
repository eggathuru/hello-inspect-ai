"""
You're evaluating whether a model can rewrite blunt internal notes into your company's warm, plain-language customer voice. Hold the notes in code. Wrap each note in a reusable instruction template before sending it, set the persona up front, and have a model judge whether the rewrite matches the voice guidelines.
"""

from inspect_ai import Task, solver, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import chain, generate, prompt_template, solver, system_message

brand_voice_rewrite_dataset = [
    Sample(
        input="Ticket closed. Issue was user error. Told them to read the manual.",
        target="Don't blame the user so explicitly. Use a more understanding tone",
    ),
    Sample(
        input="Refund denied, outside policy window. Not our problem.",
        target="Kindly explain the refund can't be issued",
    ),
]

SYSTEM_MESSAGE = """
You are a kind and warm service employee. You are happy to help the customer.
"""

PROMPT_TEMPLATE = """
Rewrite the following internal note into our company's warm, plain-language customer voice.

Note: {prompt}
"""


@task
def brand_voice_rewrite():
    return Task(
        dataset=brand_voice_rewrite_dataset,
        solver=[
            system_message(SYSTEM_MESSAGE),
            prompt_template(PROMPT_TEMPLATE),
            generate(),
        ],
        scorer=model_graded_qa(),
    )


@solver
def my_solver(
    system_prompt: str = "brand_voice_rewrite_system_prompt.txt",
    user_prompt: str = "brand_voice_rewrite_prompt_template.txt",
):
    return chain(
        system_message(system_prompt),
        prompt_template(user_prompt),
        generate(),
    )


@task
def brand_voice_rewrite_with_named_scorer():
    return Task(
        dataset=brand_voice_rewrite_dataset,
        solver=my_solver(),
        scorer=model_graded_qa(),
    )
