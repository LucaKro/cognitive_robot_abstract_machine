"""
Turning a spoken instruction into plans with a trained language model, held to the plan
format by a grammar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from functools import cached_property
from pathlib import Path

from krrood.ormatic.utils import classproperty
from krrood.patterns.path_enumeration import PathEnumeration
from llama_cpp import Llama, LlamaGrammar
from typing_extensions import Any

from cramspeech.inference.plan import InstructionPlans


# %% Resource files
class InferenceResourcePath(PathEnumeration):
    """
    The files the inference package ships and reads.
    """

    PLAN_GRAMMAR = Path(__file__).parent / "resources" / "plan_grammar.gbnf"
    """
    The llama.cpp grammar that holds the model to the plan format.
    """


# %% Completions
class CompletionKey(StrEnum):
    """
    The keys of a completion as llama.cpp returns it.
    """

    CHOICES = "choices"
    """
    The generated alternatives, of which the plan model asks for one.
    """

    TEXT = "text"
    """
    What the model wrote, without the prompt.
    """


@dataclass
class ModelAnswer:
    """
    What the language model wrote for one instruction.
    """

    text: str
    """
    The plan JSON the model wrote.
    """

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ModelAnswer:
        """
        :param payload: The model's completion with one choice, as llama.cpp returns it.
        :return: The answer the completion holds.
        """
        [choice] = payload[CompletionKey.CHOICES]
        return cls(text=choice[CompletionKey.TEXT])


# %% Plan model
class ComputeDevice(IntEnum):
    """
    Where the language model runs, as the number of its layers llama.cpp moves to the
    graphics card.
    """

    GPU = -1
    """
    Every layer on the graphics card, or on the CPU where llama.cpp was built without
    graphics card support.
    """

    CPU = 0
    """
    Every layer on the CPU.
    """


@dataclass
class PlanModel:
    """
    Turns a spoken instruction into plans with a trained language model in GGUF format.
    """

    model_path: Path
    """
    The GGUF file of the trained model.
    """

    device: ComputeDevice = ComputeDevice.GPU
    """
    Where the model runs.
    """

    context_token_count: int = 2048
    """
    The most tokens the prompt and the output may hold together.
    """

    maximum_output_token_count: int = 1024
    """
    The most tokens the model writes for one instruction.
    """

    @classproperty
    def prompt_template(cls) -> str:
        """
        :return: The text the model continues: the instruction as the user's turn in the
            chat format of Qwen, without a system message, as in training.
        """
        return "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

    @classproperty
    def greedy_temperature(cls) -> float:
        """
        :return: The sampling temperature that always takes the likeliest token, so an
            instruction always gives the same plans.
        """
        return 0.0

    @classproperty
    def no_repetition_penalty(cls) -> float:
        """
        :return: The repetition penalty that leaves the token scores unchanged, since a
            plan repeats the words of the instruction on purpose.
        """
        return 1.0

    @cached_property
    def language_model(self) -> Llama:
        """
        :return: The loaded model.
        """
        return Llama(
            model_path=str(self.model_path),
            n_gpu_layers=self.device,
            n_ctx=self.context_token_count,
            verbose=False,
        )

    @cached_property
    def grammar(self) -> LlamaGrammar:
        """
        :return: The grammar that holds the model to the plan format.
        """
        return LlamaGrammar.from_file(InferenceResourcePath.PLAN_GRAMMAR, verbose=False)

    def write_plans(self, instruction: str) -> InstructionPlans:
        """
        :param instruction: The spoken instruction.
        :return: One plan per task of the instruction.
        :raises json.JSONDecodeError: If the output reached
            :attr:`maximum_output_token_count`, so the plan JSON is cut off.
        """
        completion = self.language_model.create_completion(
            self.prompt_template.format(instruction=instruction),
            max_tokens=self.maximum_output_token_count,
            temperature=self.greedy_temperature,
            repeat_penalty=self.no_repetition_penalty,
            grammar=self.grammar,
        )
        answer = ModelAnswer.from_json(completion)
        return InstructionPlans.from_json(json.loads(answer.text))
