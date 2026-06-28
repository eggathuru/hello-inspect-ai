"""
You have a small set of "how do I do X securely?" prompts, each with a short description of what a good answer must mention. You want the model to act as a security expert, and you want another model to judge whether the response actually contains the required facts.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate, system_message

SYSTEM_MESSAGE = """
You are a security expert.
"""

secure_coding_dataset = [
    Sample(
        input="What cookie attributes should I use for strong security?",
        target="secure samesite and httponly",
    ),
    Sample(
        input="How should I store passwords in an auth database?",
        target="a salted, slow hash such as Argon2 or bcrypt",
    ),
]


@task
def secure_coding_advice():
    return Task(
        dataset=secure_coding_dataset,
        solver=[
            system_message(SYSTEM_MESSAGE),
            generate(),
        ],
        scorer=model_graded_fact(),
    )
