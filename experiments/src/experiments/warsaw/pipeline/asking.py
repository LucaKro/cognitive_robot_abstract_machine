"""
Putting a question to a model, and putting an unusable answer back to it.

Four of the pipeline's steps ask a model something: which class a label means, whose a
contested face is, which whole a part belongs to, what each body is. They differ in what
is said and how the answer is read, and agree on everything around that -- keeping the
reply as it came back, reading a kept one instead of asking again, checking the answer is
usable, and asking once more with what was wrong when it is not.

An answer naming a class that is not in the taxonomy is worth nothing, and saying so is
cheaper than either dropping the question or letting a person correct it by hand.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from semantic_digital_twin.adapters.vision_language_model.client import (
    ModelResponse,
    VisionLanguageModel,
)
from semantic_digital_twin.adapters.vision_language_model.exceptions import (
    ModelRefusedError,
)
from semantic_digital_twin.adapters.vision_language_model.message import (
    MessagePart,
    TextPart,
)
from typing_extensions import Any, Dict, Generic, List, Sequence, Type, TypeVar

from experiments.warsaw.pipeline.reporting import Reporting
from experiments.warsaw.pipeline.templates import PipelineTemplates

AnswerType = TypeVar("AnswerType")
"""
What reading one kind of question's answer produces.
"""

# %% the two halves of a prompt


class PromptHalf(StrEnum):
    """
    One of the two halves of what a question puts to a model.

    They go into the request as two messages, in two roles, and they have different
    lifetimes: the standing instruction is sent unchanged however often an unusable answer
    is put back, while what is being asked about is built afresh each time. Both are kept
    in one directory per question, so the pair is found together.

    The value is the file each is kept in. Only one of them interpolates anything, so only
    one of them is named as a template; both are read the same way, since a template with
    nothing to fill in renders as itself.
    """

    SYSTEM = "system.md"
    """
    What the model is told it is doing.
    """

    MESSAGE = "message.md.jinja"
    """
    What it is being asked about this time.
    """


# %% one thing to ask


@dataclass(kw_only=True)
class Question(ABC, Generic[AnswerType]):
    """
    One thing to ask a model, and the reading of what comes back.

    A question names the prompt it is put with, and both halves of that prompt are found
    from the name: naming one thing rather than two files is what keeps a question from
    being asked with one half of one prompt and one half of another.
    """

    templates: PipelineTemplates = field(default_factory=PipelineTemplates)
    """
    Where the words put to the model are written from.
    """

    prompt: str = ""
    """
    The prompt this question is put with, by the directory its two halves are kept in.
    """

    prompts_directory: str = "prompts"
    """
    Where those directories are, under the templates.
    """

    def prompt_template(self, half: PromptHalf) -> str:
        """
        :param half: Which half of the prompt is wanted.
        :return: The template it is written from, by file name.
        """
        return f"{self.prompts_directory}/{self.prompt}/{half.value}"

    @property
    def message_template(self) -> str:
        """
        :return: What the model is asked about, by template name.
        """
        return self.prompt_template(PromptHalf.MESSAGE)

    @property
    @abstractmethod
    def key(self) -> str:
        """
        :return: What names this question, which is what its kept reply is filed under.
        """

    @property
    def system_prompt(self) -> str:
        """
        :return: What the model is told it is doing.
        """
        return self.templates.render_document(self.prompt_template(PromptHalf.SYSTEM))

    @abstractmethod
    def message(self) -> List[MessagePart]:
        """
        :return: The question, in the order it is to be read.
        """

    @abstractmethod
    def read(self, response: ModelResponse) -> AnswerType:
        """
        :param response: What the model answered with.
        :return: The answer, in the shape the pipeline keeps it.
        :raises ModelRefusedError: If the reply holds no answer at all.
        """

    @abstractmethod
    def problems_with(self, answer: AnswerType) -> List[str]:
        """
        :param answer: What was read out of the reply.
        :return: One sentence per thing that makes it unusable, empty when nothing does.
        """

    @abstractmethod
    def refusal(self, refused: ModelRefusedError) -> AnswerType:
        """
        :param refused: The reply that held no answer.
        :return: The blank answer to keep in its place, so a refusal costs one question
            rather than the run.
        """


# %% a question asked about a class of the ontology


@dataclass(kw_only=True)
class QuestionAboutTheOntology(Question[AnswerType], Generic[AnswerType]):
    """
    A question about what a class of the ontology is or what it may hold.

    Both such questions are put the same way: the model is shown the taxonomy, is checked
    against the classes standing in it, and is given renders of an object of each class
    being discussed.
    """

    taxonomy: Dict[str, Any] = field(default_factory=dict)
    """
    The ontology as a model reads it.
    """

    known: Dict[str, Type] = field(default_factory=dict)
    """
    The ontology's classes by name, for checking an answer against.
    """

    renders_directory: Path = field(default_factory=Path)
    """
    Where the renders shown with the question are kept.
    """


# %% what came back


@dataclass
class Answered(Generic[AnswerType]):
    """
    What one question was answered with, after as many attempts as it was worth.
    """

    answer: AnswerType
    """
    What was read out of the last reply.
    """

    problems: List[str] = field(default_factory=list)
    """
    What still makes it unusable, empty when nothing does.
    """

    attempts: int = 1
    """
    How many times it was asked.
    """

    @property
    def is_usable(self) -> bool:
        """
        :return: Whether the answer can be acted on.
        """
        return not self.problems


# %% putting a question, and putting it again


@dataclass
class Questioner(Reporting):
    """
    A model being asked the pipeline's questions, with every reply kept.
    """

    model: VisionLanguageModel
    """
    The model to ask.
    """

    answers_directory: Path
    """
    Where the replies are kept, as they came back.
    """

    corrections: int = 1
    """
    How often an unusable answer is put back to the model with what was wrong with it.
    """

    reuse_answers: bool = False
    """
    Whether to read a kept reply rather than ask again, which re-reads a run without
    spending anything on it.
    """

    correction_preamble: str = "Your previous answer could not be used:"
    """
    How the model is told what was wrong with what it said.
    """

    correction_closing: str = "Answer the same question again, correcting that."
    """
    How that correction ends.
    """

    def answer(self, question: Question[AnswerType]) -> Answered[AnswerType]:
        """
        Put one question, correcting an unusable answer while it is worth doing.

        :param question: What to ask and how to read the answer.
        :return: The answer, and what is still wrong with it.
        """
        problems: Sequence[str] = ()
        answered = None
        for attempt in range(1 + self.corrections):
            response = self.respond_to(question, problems)
            try:
                answer = question.read(response)
            except ModelRefusedError as refused:
                answer, problems = question.refusal(refused), [str(refused)]
            else:
                problems = question.problems_with(answer)
            answered = Answered(
                answer=answer, problems=list(problems), attempts=attempt + 1
            )
            if answered.is_usable:
                return answered
            if attempt < self.corrections:
                self.logger.info("%s: %s, asking again ...", question.key, problems[0])
        return answered

    def respond_to(
        self, question: Question[AnswerType], problems: Sequence[str]
    ) -> ModelResponse:
        """
        Ask one question, or read back what the model already said about it.

        :param question: What to ask.
        :param problems: What was wrong with the answer to the same question, when this is
            another attempt at it.
        :return: The reply.
        """
        kept = self.kept_path(question)
        if self.reuse_answers and kept.exists() and not problems:
            return ModelResponse.from_json(json.loads(kept.read_text()))

        message = list(question.message())
        if problems:
            message.append(TextPart(self.correction_of(problems)))
        response = self.model.ask(message, question.system_prompt)
        self.answers_directory.mkdir(parents=True, exist_ok=True)
        kept.write_text(json.dumps(response.to_json(), indent=2))
        return response

    def kept_path(self, question: Question[AnswerType]) -> Path:
        """
        :param question: The question whose reply is kept.
        :return: Where that reply is kept.
        """
        return self.answers_directory / f"{question.key}.json"

    def correction_of(self, problems: Sequence[str]) -> str:
        """
        :param problems: What was wrong with the previous answer.
        :return: What to tell the model about it.
        """
        return (
            f"{self.correction_preamble}\n- "
            + "\n- ".join(problems)
            + f"\n{self.correction_closing}"
        )
