"""
Evaluate short argumentative essays against a rubric. Wrap the essay and rubric in a template, set a strict-grader persona, and — because one judge is noisy — have three different models grade each essay and take the majority verdict.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import accuracy, model_graded_qa, multi_scorer, scorer, stderr
from inspect_ai.solver import generate

essay_rubric_dataset = [
    Sample(input="Write a 150-word argument for four-day work weeks."),
]

GRADER_TEMPLATE = """
You are assessing an argumentative essay. You are a pursuadable skeptic.

[BEGIN DATA]
***
[Task]: {question}
***
[Submission]: {answer}
***
[END DATA]

Does the submission meet the constraints described in the task description? Is the argument valid? Would it be convincing to a reader who was not convinced of the thesis beforehand?

{instructions}
"""


@scorer(metrics=[accuracy(), stderr()])
def grader_panel():
    return multi_scorer(
        scorers=[
            # I use identical scorers for this purpose but I should probably
            # experiment by configuring them separately.
            model_graded_qa(GRADER_TEMPLATE),
            model_graded_qa(GRADER_TEMPLATE),
            model_graded_qa(GRADER_TEMPLATE),
        ],
        reducer="mode",
    )


@task
def essay_rubric_grader_panel():
    return Task(
        dataset=essay_rubric_dataset,
        solver=generate(),
        scorer=grader_panel(),
    )
