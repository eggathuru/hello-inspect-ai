"""
A3. Theory-of-mind short-answers

A set of social-reasoning vignettes, each with an ideal answer. These are open-ended, so a model should grade them. You also want the answering model to take a first pass, then critique and revise its own answer before it's graded.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
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
