"""
A3. Theory-of-mind short-answers

A set of social-reasoning vignettes, each with an ideal answer. These are open-ended, so a model should grade them. You also want the answering model to take a first pass, then critique and revise its own answer before it's graded.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import get_model
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import generate, self_critique

theory_of_mind_sa_dataset = [
    Sample(
        input="Anya hides her toy in the blue box, then leaves. Ravi moves it to the red box. Anya returns. Where will Anya look for her toy first?",
        target="The blue box, because Anya didn't see Ravi move it.",
    ),
]


@task
def theory_of_mind_sa():
    return Task(
        dataset=theory_of_mind_sa_dataset,
        solver=[
            generate(),
            self_critique(),
        ],
        scorer=model_graded_qa(),
    )


@task
def theory_of_mind_sa_with_hardcoded_critique_model():
    return Task(
        dataset=theory_of_mind_sa_dataset,
        solver=[
            generate(),
            self_critique(model="anthropic/claude-haiku-4-5"),
        ],
        scorer=model_graded_qa(),
    )


@task
def theory_of_mind_sa_with_critique_model_param(critique_model: str | None = None):
    return Task(
        dataset=theory_of_mind_sa_dataset,
        solver=[
            generate(),
            self_critique(model=critique_model),
        ],
        scorer=model_graded_qa(),
    )


@task
def theory_of_mind_sa_with_critique_model_role():
    return Task(
        dataset=theory_of_mind_sa_dataset,
        solver=[
            generate(),
            self_critique(model=get_model(role="critique")),
        ],
        scorer=model_graded_qa(),
    )
