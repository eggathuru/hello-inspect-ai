"""
trivia_mc.csv stores each question across columns: question, opt_a, opt_b, opt_c, opt_d, and correct (a letter). A flat field rename won't help here — you need to assemble the four options into a list and set the target to the right letter. Present it as a multiple-choice question and score the choice.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, csv_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice


def record_to_sample(record):
    return Sample(
        input=record["question"],
        target=record["correct"],
        choices=[
            record["opt_a"],
            record["opt_b"],
            record["opt_c"],
            record["opt_d"],
        ],
    )


@task
def trivia_mc():
    return Task(
        dataset=csv_dataset("trivia_mc.csv", sample_fields=record_to_sample),
        solver=multiple_choice(),
        scorer=choice(),
    )
