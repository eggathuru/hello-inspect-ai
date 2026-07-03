"""
Give the model a sandboxed code-execution tool and a set of arithmetic word problems it should solve by running code rather than guessing. Score correct if the expected number appears in the final answer.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate, prompt_template, use_tools
from inspect_ai.tool import python

arithmetic_dataset = [
    Sample(input="What is 47 * 89 - 312?", target="3871"),
    Sample(input="What is the 12th Fibonacci number (F1=1, F2=1)?", target="144"),
]

TEMPLATE = """
Output the answer to the following match question your the final line.
Do not use commas. You may use the decimal if the answer isn't a whole number.

{prompt}
"""


@task
def arithmetic_with_code_tool():
    return Task(
        dataset=arithmetic_dataset,
        solver=[
            prompt_template(TEMPLATE),
            use_tools(python()),
            generate(),
        ],
        scorer=includes(),
        sandbox="docker",
    )
