"""
Same idea, different domain: the model gets a code tool and must answer conversion questions precisely. Set a "use the tool, don't estimate" system instruction, and have a model grade whether the stated result contains the correct converted value.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import python

unit_conversion_dataset = [
    Sample(
        input="How many liters are in 5 US gallons? Give the number to 2 decimals.",
        target="18.93",
    ),
    Sample(
        input="Convert 100 km/h to m/s, rounded to 2 decimals.",
        target="27.78",
    ),
]


@task
def unit_conversion_with_tool():
    return Task(
        dataset=unit_conversion_dataset,
        solver=[
            # Problem statement suggests `system_message` but I omit it here.
            use_tools(python()),
            generate(),
        ],
        scorer=includes(),
        sandbox="docker",
    )
