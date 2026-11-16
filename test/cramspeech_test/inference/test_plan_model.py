"""
What the plan model makes of the text its language model writes for an instruction.
"""

from __future__ import annotations

from pathlib import Path

from cramspeech.inference.plan import InstructionPlans, Plan, PlanAction, PlanStep
from cramspeech.inference.plan_model import PlanModel

from ..dataset.dataset_files import ModelOutputFile
from ..dataset.language_model_stand_ins import LanguageModelWritingText


# %% Plans
def test_instruction_becomes_its_plans() -> None:
    """
    The model writes under the plan grammar one plan per task of the instruction, each
    opening with a spoken summary, which the plan keeps apart from its steps.
    """
    plan_model = PlanModel(model_path=Path("trained_plan_model.gguf"))
    language_model = LanguageModelWritingText(
        text=ModelOutputFile.TWO_PLANS.read_text()
    )
    plan_model.language_model = language_model

    instruction_plans = plan_model.write_plans(
        "From the bed, get me a red bell pepper and then how many fasteners do you see "
        "on the kitchen cabinet."
    )

    [settings] = language_model.completion_settings
    assert settings["grammar"] is plan_model.grammar
    assert instruction_plans == InstructionPlans(
        plans=[
            Plan(
                summary="I will bring you a red bell pepper from the bed",
                steps=[
                    PlanStep(action=PlanAction.NAVIGATE, location="bed"),
                    PlanStep(action=PlanAction.PICK_UP, object_name="red bell pepper"),
                    PlanStep(action=PlanAction.NAVIGATE, person="me"),
                    PlanStep(
                        action=PlanAction.GIVE,
                        object_name="red bell pepper",
                        person="me",
                    ),
                ],
            ),
            Plan(
                summary="I will count the fasteners on the kitchen cabinet and tell you",
                steps=[
                    PlanStep(action=PlanAction.NAVIGATE, location="kitchen cabinet"),
                    PlanStep(action=PlanAction.COUNT, object_name="fasteners"),
                    PlanStep(action=PlanAction.NAVIGATE, person="me"),
                    PlanStep(action=PlanAction.SAY, text="<result>"),
                ],
            ),
        ]
    )
