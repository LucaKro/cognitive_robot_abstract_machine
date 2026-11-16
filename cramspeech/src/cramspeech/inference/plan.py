"""
The plans the llm model writes for a spoken instruction, in the JSON format the model
was trained on.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import Any


# %% JSON keys
class PlanKey(StrEnum):
    """
    The keys of the plan JSON.
    """

    PLANS = "plans"
    """
    The list of plans, one per task of the instruction.
    """

    ACTION = "action"
    """
    What a step does.
    """

    TEXT = "text"
    """
    What the robot says.
    """

    LOCATION = "location"
    """
    The place a step goes to or acts on.
    """

    OBJECT = "object"
    """
    The object a step acts on.
    """

    PERSON = "person"
    """
    The person a step goes to, looks for or hands something to.
    """


# %% Steps
class PlanAction(StrEnum):
    """
    What a step of a plan does.
    """

    SAY = "say"
    """
    Say :attr:`PlanStep.text` out loud.
    """

    NAVIGATE = "navigate"
    """
    Go to :attr:`PlanStep.location` or to :attr:`PlanStep.person`.
    """

    PICK_UP = "pick_up"
    """
    Pick up :attr:`PlanStep.object_name`.
    """

    PLACE = "place"
    """
    Put the held :attr:`PlanStep.object_name` down at :attr:`PlanStep.location`.
    """

    OPEN = "open"
    """
    Open :attr:`PlanStep.location`, such as a door or a drawer.
    """

    CLOSE = "close"
    """
    Close :attr:`PlanStep.location`.
    """

    COUNT = "count"
    """
    Count the :attr:`PlanStep.object_name` in sight. A later step says the count.
    """

    FIND = "find"
    """
    Look for :attr:`PlanStep.object_name` or :attr:`PlanStep.person`.
    """

    GIVE = "give"
    """
    Hand the held :attr:`PlanStep.object_name` to :attr:`PlanStep.person`.
    """


@dataclass
class PlanStep:
    """
    One action of a plan, with the words of the instruction it acts on.
    """

    action: PlanAction
    """
    What the step does.
    """

    text: str | None = None
    """
    What the robot says: plain words, or a placeholder in angle brackets such as
    ``<result>`` that the robot fills in when it speaks.
    """

    location: str | None = None
    """
    The place the step goes to or acts on, as spoken.
    """

    object_name: str | None = None
    """
    The object the step acts on, as spoken.
    """

    person: str | None = None
    """
    The person the step goes to, looks for or hands something to, as spoken. ``me`` is
    the person who gave the instruction.
    """

    @classmethod
    def from_json(cls, payload: dict[str, str]) -> PlanStep:
        """
        :param payload: One step of the plan JSON.
        :return: The step the payload describes.
        """
        return cls(
            action=PlanAction(payload[PlanKey.ACTION]),
            text=payload.get(PlanKey.TEXT),
            location=payload.get(PlanKey.LOCATION),
            object_name=payload.get(PlanKey.OBJECT),
            person=payload.get(PlanKey.PERSON),
        )


# %% Plans
@dataclass
class Plan:
    """
    The steps that carry out one task of an instruction.
    """

    summary: str
    """
    What the robot says it will do before it starts.
    """

    steps: list[PlanStep]
    """
    The steps, in the order the robot carries them out.
    """

    @classmethod
    def from_json(cls, payload: list[dict[str, str]]) -> Plan:
        """
        :param payload: One plan of the plan JSON, opening with a ``say`` step that holds
            the summary.
        :return: The plan the payload describes.
        """
        summary, *steps = payload
        return cls(
            summary=summary[PlanKey.TEXT],
            steps=[PlanStep.from_json(step) for step in steps],
        )


@dataclass
class InstructionPlans:
    """
    The plans for one spoken instruction, one per task it asks for.
    """

    plans: list[Plan]
    """
    The plans, in the order the tasks were spoken.
    """

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> InstructionPlans:
        """
        :param payload: The plan JSON the model wrote for the instruction.
        :return: The plans the payload describes.
        """
        return cls(plans=[Plan.from_json(plan) for plan in payload[PlanKey.PLANS]])
