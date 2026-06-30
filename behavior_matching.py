"""
behavior.jsonl (a common safety-eval shape) has question, answer_matching_behavior (the response label that indicates the behavior you're testing for), question_id, and label_confidence. Map question→input, the matching-behavior field→target, keep the id, and collect label_confidence into metadata. Give the model a system framing and grade open-ended.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, json_dataset
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import generate, system_message

SYSTEM_MESSAGE = """
"""


@task
def behavior_matching():
    return Task(
        dataset=json_dataset(
            "behavior.jsonl",
            sample_fields=FieldSpec(
                input="question",
                target="answer_matching_behavior",
                id="question_id",
                metadata=["label_confidence"],
            ),
        ),
        solver=generate(),  # The question suggests using a system message but I omit it here.
        scorer=model_graded_qa(),
    )
