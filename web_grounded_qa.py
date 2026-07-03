"""
Give the model a web-search tool and ask current-events-style questions whose answers it must look up. Grade open-ended against a known answer.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import chain, generate, solver, use_tools
from inspect_ai.tool import web_search

qa_dataset = [
    Sample(
        input="Who is the 2026 Democratic nominee for Colorado's 1st congressional district?",
        target="Melat Kiros",
    )
]


@solver
def noweb_solver():
    return generate()


@solver
def web_solver():
    return chain(
        use_tools(web_search()),
        generate(),
    )


@task
def web_grounded_qa():
    return Task(
        dataset=qa_dataset,
        solver=web_solver(),
        scorer=model_graded_fact(),
    )
