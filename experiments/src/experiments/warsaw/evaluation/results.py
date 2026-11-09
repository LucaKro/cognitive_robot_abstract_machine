"""
What finished runs come to, as tables for a document.

Any run reports what it built, what it asked of its model and how long that took,
whatever scene it was made of. A run that was scored against a ground truth reports that
as well. Runs made under the same settings by the same code are one method, so a
comparison needs no more than the runs themselves.

Run ``python -m experiments.warsaw.evaluation.results --help`` for its options.
"""

from __future__ import annotations

import argparse
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from krrood.ormatic.utils import classproperty
from semantic_digital_twin.adapters.vision_language_model.client import ModelResponse
from typing_extensions import Any, Callable, Sequence

from experiments.experiment_definitions import (
    ExperimentResult,
    ExperimentsTable,
    LatexRenderer,
    MeanAndStandardDeviation,
)
from experiments.warsaw.articulate3d.convert import ConvertedScene
from experiments.warsaw.articulate3d.rule_ceiling import CandidateOutcome, RuleCeiling
from experiments.warsaw.articulate3d.run_answers import (
    ConfirmationOutcome,
    RunAnswersJudged,
)
from experiments.warsaw.articulate3d.score import SceneScore, ScoreFile, Share
from experiments.warsaw.evaluation.exceptions import RunNotFinishedError
from experiments.warsaw.evaluation.graph import EvaluationGraph
from experiments.warsaw.pipeline.model_calls import ModelCallTrace
from experiments.warsaw.pipeline.records import StepDurations
from experiments.warsaw.pipeline.report import RunReport
from experiments.warsaw.pipeline.run import Run, RunFile
from experiments.warsaw.pipeline.settings import Model

# %% the names in what a run wrote


class ProvenanceKey(StrEnum):
    """
    The parts of a run's provenance that say what the run was.
    """

    SETTINGS = "settings"
    """
    What the run was told.
    """

    SOURCE = "source"
    """
    The code it ran.
    """

    COMMIT = "commit"
    """
    The commit of that code.
    """

    PATCH_DIGEST = "patch_sha256"
    """
    The digest of the local changes on top of that commit.
    """


class SettingOutsideConditions(StrEnum):
    """
    The settings that say where or how a run was kept rather than what it did to a
    scene.
    """

    SCENE_DIRECTORY = "scene_directory"
    """
    Which scene it was made of, which a condition is compared across.
    """

    RUNS_DIRECTORY = "runs_directory"
    """
    Where it was kept.
    """

    REPOSITORY = "repository"
    """
    Which checkout it ran from.
    """

    HEADLESS = "headless"
    """
    Whether its renders were drawn in a hidden window.
    """

    IGNORE_AMENDMENTS = "ignore_amendments"
    """
    Whether it started although the ontology's files carried local changes, which the
    code version already tells apart.
    """


class SettingName(StrEnum):
    """
    The settings a method is called something shorter than their names for.
    """

    MODEL = "model"
    """
    Which model every question goes to.
    """

    SHOW_THE_PICTURES = "show_the_pictures"
    """
    Whether the objects are shown to the model.
    """

    SHOW_THE_CONTESTED_FACES = "show_the_contested_faces"
    """
    Whether the questions about overlaps are put with pictures.
    """

    DEFER_NEW_CLASSES = "defer_new_classes"
    """
    Whether a new class is proposed only after every label was read.
    """

    SETTLE_CLASSES_BY_LABEL = "settle_classes_by_label"
    """
    Whether a body takes the class its label was read as without being asked.
    """

    CONFIRM_WHOLES_WITHIN_THE_GAP = "confirm_wholes_within_the_gap"
    """
    Whether a whole only found nearby is asked about.
    """

    CONSIDER_ALTERNATIVE_CLASSES = "consider_alternative_classes"
    """
    Whether a pair is judged by the classes its labels may also be.
    """

    PREFER_NESTED_WHOLES = "prefer_nested_wholes"
    """
    Whether a candidate whole another candidate is fixed to is dropped.
    """

    USE_PART_WHOLE_KNOWLEDGE = "use_part_whole_knowledge"
    """
    Whether the ontology's knowledge of which class can hold which as a part is used.
    """

    LARGEST_GAP_TO_A_WHOLE = "largest_gap_to_a_whole"
    """
    How far a whole a part touches no whole may be from it.
    """

    LARGEST_GAP_COUNTED_AS_TOUCHING = "largest_gap_counted_as_touching"
    """
    How far apart a part and a whole may be and still count as touching.
    """

    def short_description(self, value: Any) -> str:
        """
        :param value: The setting's value in a run, nothing where the run recorded none.
        :return: What the name of a method made with that value says about it.
        """
        match self:
            case SettingName.MODEL:
                known = {one.value: one.short_name for one in Model}
                return known.get(value, str(value))
            case SettingName.LARGEST_GAP_TO_A_WHOLE:
                return self.distance_described("wholes within", value)
            case SettingName.LARGEST_GAP_COUNTED_AS_TOUCHING:
                return self.distance_described("touching within", value)
            case SettingName.USE_PART_WHOLE_KNOWLEDGE:
                # A run made before the setting existed used the knowledge.
                return "ontology" if value is not False else "no ontology"
        on, off = {
            SettingName.SHOW_THE_PICTURES: ("images", "text"),
            SettingName.SHOW_THE_CONTESTED_FACES: ("face images", "face text"),
            SettingName.DEFER_NEW_CLASSES: ("deferred", "immediate"),
            SettingName.SETTLE_CLASSES_BY_LABEL: ("settled", "asked"),
            SettingName.CONFIRM_WHOLES_WITHIN_THE_GAP: ("confirmed", "unconfirmed"),
            SettingName.CONSIDER_ALTERNATIVE_CLASSES: (
                "alternatives",
                "first readings",
            ),
            SettingName.PREFER_NESTED_WHOLES: ("nested wholes", "every candidate"),
        }[self]
        return on if value else off

    def distance_described(self, words: str, metres: float | None) -> str:
        """
        :param words: What the distance is said after.
        :param metres: The distance, nothing where a run recorded none.
        :return: The distance in centimetres after those words.
        """
        if metres is None:
            return f"{self.value.replace('_', ' ')} not recorded"
        return f"{words} {100 * metres:g} cm"


class UsageKey(StrEnum):
    """
    What a model's response says it cost.
    """

    PROMPT_TOKENS = "prompt_tokens"
    """
    The tokens of the question.
    """

    COMPLETION_TOKENS = "completion_tokens"
    """
    The tokens of the answer, its reasoning included.
    """

    COST = "cost"
    """
    What it cost in US dollars, on a service that charges and says so.
    """


# %% what one run asked of its model


@dataclass(frozen=True)
class ModelUse:
    """
    What a run asked of its model, added up over every attempt it kept.
    """

    questions: int
    """
    How many questions it asked.
    """

    calls: int
    """
    How many attempts those took, a correction being one more.
    """

    prompt_tokens: int
    """
    The tokens sent.
    """

    completion_tokens: int
    """
    The tokens answered.
    """

    cost_cents: float | None
    """
    What it cost in US cents, or nothing where the service named no price.
    """

    seconds: float
    """
    How long the model took to answer.
    """

    @classmethod
    def of(cls, run: Run) -> ModelUse:
        """
        :param run: A run.
        :return: What it asked of its model.
        """
        paths = sorted(run.path(RunFile.MODEL_CALLS).rglob("*.json"))
        traces = [ModelCallTrace.from_json(json.loads(p.read_text())) for p in paths]
        usages = [ModelResponse.from_json(one.response).usage or {} for one in traces]
        prices = [one[UsageKey.COST] for one in usages if UsageKey.COST in one]
        return cls(
            questions=len({path.parent for path in paths}),
            calls=len(traces),
            prompt_tokens=sum(one.get(UsageKey.PROMPT_TOKENS, 0) for one in usages),
            completion_tokens=sum(
                one.get(UsageKey.COMPLETION_TOKENS, 0) for one in usages
            ),
            cost_cents=100 * sum(prices) if prices else None,
            seconds=sum(one.elapsed_seconds for one in traces),
        )


# %% what one run comes to


@dataclass(frozen=True)
class RunMeasurements:
    """
    Everything one finished run is reported by.
    """

    run: str
    """
    The name of the run.
    """

    scene: str
    """
    The scene it was made of.
    """

    settings: dict[str, Any]
    """
    What it was told, as its provenance records it.
    """

    code: str
    """
    The version of the code it ran.
    """

    labels: int
    """
    How many labels the scene has.
    """

    labels_mapped: int
    """
    How many of them were read as a class.
    """

    new_classes: int
    """
    How many of those classes the run proposed itself.
    """

    bodies: int
    """
    How many bodies it cut the scene into.
    """

    bodies_annotated: int
    """
    How many of them hold the class they were given in the final world.
    """

    distinct_classes: int
    """
    How many different classes the bodies were given.
    """

    parts_mounted: int
    """
    How many parts it mounted into a whole.
    """

    mounts_refused: int
    """
    How many mounts the ontology refused.
    """

    answers_with_problems: int
    """
    How many of the answers about overlaps stayed unusable after being corrected.
    """

    model_use: ModelUse
    """
    What it asked of its model.
    """

    total_seconds: float
    """
    How long its steps took together.
    """

    score: SceneScore | None
    """
    How it agrees with the scene's ground truth, or nothing where it was not scored.
    """

    answers: RunAnswersJudged | None
    """
    It judged with and without its model's answers about wholes, or nothing where that
    was not judged.
    """

    @classmethod
    def of(cls, run: Run) -> RunMeasurements:
        """
        :param run: A finished run.
        :return: What it comes to.
        :raises RunNotFinishedError: If it stopped before its end.
        """
        if not run.holds(RunFile.EVALUATION_GRAPH):
            raise RunNotFinishedError(run_directory=run.directory)
        provenance = run.read_json(RunFile.PROVENANCE)
        counted = RunReport(run=run).context()
        graph = run.read_record(RunFile.EVALUATION_GRAPH, EvaluationGraph)
        durations = run.read_record(RunFile.STEP_DURATIONS, StepDurations)
        return cls(
            run=run.directory.name,
            scene=ConvertedScene.scene_id_of(run.scene_directory),
            settings=provenance[ProvenanceKey.SETTINGS],
            code=cls.code_version(provenance[ProvenanceKey.SOURCE]),
            labels=counted["labels"],
            labels_mapped=counted["labels_mapped"],
            new_classes=counted["labels_new"],
            bodies=counted["bodies"],
            bodies_annotated=sum(one.annotation_applied for one in graph.nodes),
            distinct_classes=counted["distinct_classes"],
            parts_mounted=counted["mounted"],
            mounts_refused=len(counted["refused"]),
            answers_with_problems=counted["answers_with_problems"],
            model_use=ModelUse.of(run),
            total_seconds=durations.total_seconds,
            score=ScoreFile.SCENE.read(run, SceneScore),
            answers=ScoreFile.ANSWERS.read(run, RunAnswersJudged),
        )

    @staticmethod
    def code_version(source: dict[str, Any]) -> str:
        """
        :param source: What a run's provenance says about its code.
        :return: The commit, and the local changes on top of it where there were any.
        """
        commit = str(source.get(ProvenanceKey.COMMIT))[:10]
        patch = source.get(ProvenanceKey.PATCH_DIGEST)
        return f"{commit}+{patch[:6]}" if patch else commit


# %% runs reported together


@dataclass(frozen=True)
class ScoredGroup(ABC):
    """
    Scores reported as one row.
    """

    name: str
    """
    What the row is called.
    """

    @property
    @abstractmethod
    def scores(self) -> list[SceneScore | None]:
        """
        :return: The score of each run the row covers, nothing for a run not scored.
        """

    @property
    @abstractmethod
    def scene_of_each_run(self) -> list[str]:
        """
        :return: The scene each run the row covers was made of.
        """

    @property
    def scenes(self) -> str | int:
        """
        :return: The name of the one scene the runs were made of, or how many there were.
        """
        scenes = set(self.scene_of_each_run)
        return next(iter(scenes)) if len(scenes) == 1 else len(scenes)

    @property
    def run_count(self) -> int:
        """
        :return: How many runs the row covers.
        """
        return len(self.scene_of_each_run)

    @staticmethod
    def summarised(
        values: list[float | None],
    ) -> float | MeanAndStandardDeviation | None:
        """
        :param values: One value per run, nothing where a run has none.
        :return: The only value, the mean and the standard deviation of several, and
            nothing where no run has one.
        """
        present = [one for one in values if one is not None]
        if not present:
            return None
        if len(present) == 1:
            return present[0]
        return MeanAndStandardDeviation.from_measurements(present)

    def over_scores(
        self, measured: Callable[[SceneScore], float | None]
    ) -> float | MeanAndStandardDeviation | None:
        """
        :param measured: What to read of a score, which a score may leave undefined.
        :return: :meth:`summarised` that, for the runs that were scored.
        """
        return self.summarised([measured(one) if one else None for one in self.scores])


@dataclass(frozen=True)
class RunGroup(ScoredGroup):
    """
    Runs reported as one row.
    """

    runs: list[RunMeasurements]
    """
    The runs it reports.
    """

    @property
    def scores(self) -> list[SceneScore | None]:
        return [one.score for one in self.runs]

    @property
    def scene_of_each_run(self) -> list[str]:
        return [one.scene for one in self.runs]

    def over(
        self, measured: Callable[[RunMeasurements], float | None]
    ) -> float | MeanAndStandardDeviation | None:
        """
        :param measured: What to read of a run, which a run may not have.
        :return: :meth:`summarised` that, over the runs.
        """
        return self.summarised([measured(one) for one in self.runs])

    def total(self, measured: Callable[[RunMeasurements], int | None]) -> int | None:
        """
        :param measured: What to count of a run, which a run may not have.
        :return: The counts of the runs holding it added up, nothing where none does.
        """
        present = [one for one in map(measured, self.runs) if one is not None]
        return sum(present) if present else None


@dataclass(frozen=True)
class BaselineGroup(ScoredGroup):
    """
    A baseline reported as one row, judged once on each scene.
    """

    scene_scores: list[SceneScore]
    """
    The baseline's score on each scene.
    """

    @property
    def scores(self) -> list[SceneScore | None]:
        return list(self.scene_scores)

    @property
    def scene_of_each_run(self) -> list[str]:
        return [one.scene_id for one in self.scene_scores]


@dataclass
class RunComparison:
    """
    Finished runs compared by what they were told to do and the code they ran.
    """

    runs: list[RunMeasurements]
    """
    The runs compared.
    """

    renamed_conditions: dict[str, str] = field(default_factory=dict)
    """
    Per name a condition is given on its own, what to call it instead.
    """

    everything_alike: str = "all runs"
    """
    What the one condition is called where nothing tells the runs apart.
    """

    @property
    def _comparable_settings(self) -> list[str]:
        """
        :return: The settings that say what a run did to a scene, in the order a run records
            them.
        """
        outside = {one.value for one in SettingOutsideConditions}
        every_setting = dict.fromkeys(key for one in self.runs for key in one.settings)
        return [key for key in every_setting if key not in outside]

    @property
    def varying_settings(self) -> list[str]:
        """
        :return: The settings a method is told apart by: those not all runs agree on.
        """
        return [
            key
            for key in self._comparable_settings
            if len({json.dumps(one.settings.get(key)) for one in self.runs}) > 1
        ]

    @property
    def shared_settings(self) -> dict[str, Any]:
        """
        :return: The settings every method has in common, by name.
        """
        varying = self.varying_settings
        return {
            key: self.runs[0].settings.get(key)
            for key in self._comparable_settings
            if key not in varying
        }

    @property
    def code_versions(self) -> list[str]:
        """
        :return: The versions of the code the runs ran, the one run earliest first.
        """
        by_time = sorted(self.runs, key=lambda one: one.run)
        return list(dict.fromkeys(one.code for one in by_time))

    def condition_of(self, run: RunMeasurements) -> str:
        """
        :param run: One of the runs.
        :return: The short name of the method it was made with, which says only what tells
            it apart from the other methods.
        """
        said = [
            self.named(key, run.settings.get(key), run.settings)
            for key in self.varying_settings
        ]
        versions = self.code_versions
        if len(versions) > 1:
            said.append(f"v{versions.index(run.code) + 1}")
        name = ", ".join(one for one in said if one) or self.everything_alike
        return self.renamed_conditions.get(name, name)

    @staticmethod
    def named(setting: str, value: Any, settings: dict[str, Any]) -> str:
        """
        :param setting: The name of a setting.
        :param value: Its value in a run.
        :param settings: Every setting of that run.
        :return: What the name of a method says about it, nothing where another of its
            settings already says it.
        """
        pictures = settings.get(SettingName.SHOW_THE_PICTURES)
        if setting == SettingName.SHOW_THE_CONTESTED_FACES and value == pictures:
            return ""
        if setting in set(SettingName):
            return SettingName(setting).short_description(value)
        words = setting.replace("_", " ")
        if value is True:
            return words
        if value is False:
            return f"no {words}"
        return f"{words} {value}"

    def definition_of(self, run: RunMeasurements) -> str:
        """
        :param run: One of the runs.
        :return: Everything that tells the method it was made with apart from the others,
            spelled out.
        """
        said = [
            f"{key.replace('_', ' ')} "
            + (str(run.settings[key]) if key in run.settings else "not recorded")
            for key in self.varying_settings
        ]
        if len(self.code_versions) > 1:
            said.append(f"code {run.code}")
        return "; ".join(said) or "the only method"

    @property
    def by_condition(self) -> list[RunGroup]:
        """
        :return: One group per condition, in the order of their names.
        """
        conditions: dict[str, list[RunMeasurements]] = defaultdict(list)
        for run in self.runs:
            conditions[self.condition_of(run)].append(run)
        return [RunGroup(name, conditions[name]) for name in sorted(conditions)]

    @property
    def by_run(self) -> list[RunGroup]:
        """
        :return: One group per run, in the order of their conditions.
        """
        return [
            RunGroup(f"{group.name} / {run.run}", [run])
            for group in self.by_condition
            for run in group.runs
        ]

    @property
    def any_scored(self) -> bool:
        """
        :return: Whether any of the runs has a score.
        """
        return any(one.score for one in self.runs)

    @property
    def any_answers_judged(self) -> bool:
        """
        :return: Whether any of the runs was judged with and without its answers.
        """
        return any(one.answers for one in self.runs)


# %% what each method is


@dataclass
class MethodDefinition(ExperimentResult):
    """
    What the short name of one method stands for.
    """

    method: str
    """
    What the tables call it.
    """

    differs_by: str
    """
    Everything that tells it apart from the others.
    """


# %% the rows of the tables


@dataclass
class TableRow(ExperimentResult, ABC):
    """
    One row of a table of results.
    """

    method: str
    """
    The method the row reports, and the run where it reports only one.
    """

    @classproperty
    def table_name(cls) -> str:
        """
        :return: What the table of such rows is named after in its files.
        """
        raise NotImplementedError

    @classproperty
    def what_it_shows(cls) -> str:
        """
        :return: What the table's caption says it shows, in LaTeX.
        """
        raise NotImplementedError

    @classproperty
    def context_columns(cls) -> list[str]:
        """
        :return: The columns that say what a row covers rather than what it measured.
        """
        return ["scenes"]

    @property
    def covers_several_runs(self) -> bool:
        """
        :return: Whether the row gives a mean and a standard deviation over runs.
        """
        return False


@dataclass
class RunGroupResult(TableRow, ABC):
    """
    One row reporting one :class:`RunGroup`.
    """

    runs: int
    """
    How many runs it covers.
    """

    scenes: str | int
    """
    The scene they were made of, or how many there were.
    """

    @property
    def covers_several_runs(self) -> bool:
        return self.runs > 1

    @classproperty
    def reads_only_the_score(cls) -> bool:
        """
        :return: Whether every column of such a row comes from a score alone, so a
            baseline can be reported by it as well.
        """
        return False

    @classproperty
    def context_columns(cls) -> list[str]:
        return ["runs", "scenes"]

    @classmethod
    def can_report(cls, comparison: RunComparison) -> bool:
        """
        :param comparison: The runs compared.
        :return: Whether any of them holds what such a row reports.
        """
        return True

    @classmethod
    @abstractmethod
    def of(cls, group: RunGroup) -> RunGroupResult:
        """
        :param group: The runs to report.
        :return: The row reporting them.
        """


@dataclass
class ScoreResult(RunGroupResult, ABC):
    """
    A row read from scores alone, of runs or of a baseline.
    """

    @classproperty
    def reads_only_the_score(cls) -> bool:
        return True

    @classmethod
    def can_report(cls, comparison: RunComparison) -> bool:
        return comparison.any_scored

    @classmethod
    @abstractmethod
    def of(cls, group: ScoredGroup) -> ScoreResult:
        """
        :param group: The scores to report.
        :return: The row reporting them.
        """


@dataclass
class ClassResult(ScoreResult):
    """
    How the objects and classes of runs agree with a ground truth.
    """

    parts_found: float | MeanAndStandardDeviation | None
    """
    The share of the annotated parts a body was paired with.
    """

    class_accuracy: float | MeanAndStandardDeviation | None
    """
    The share of the found parts given exactly their class.
    """

    class_hierarchy_f1: float | MeanAndStandardDeviation | None
    """
    The partial credit for answers on the right branch of the class hierarchy.
    """

    @classproperty
    def table_name(cls) -> str:
        return "classes"

    @classproperty
    def what_it_shows(cls) -> str:
        return "Whether the objects and their classes are right, judged against the ground truth."

    @classmethod
    def of(cls, group: ScoredGroup) -> ClassResult:
        return cls(
            method=group.name,
            runs=group.run_count,
            scenes=group.scenes,
            parts_found=group.over_scores(
                lambda one: one.correspondence.found_share.value
            ),
            class_accuracy=group.over_scores(lambda one: one.classes.accuracy.value),
            class_hierarchy_f1=group.over_scores(
                lambda one: one.classes.class_hierarchy_f1
            ),
        )


@dataclass
class PartWholeResult(ScoreResult):
    """
    How the part hierarchies of runs agree with a ground truth.
    """

    part_whole_precision: float | MeanAndStandardDeviation | None
    """
    The parts hung from another part that hang from their annotated whole.
    """

    part_whole_recall: float | MeanAndStandardDeviation | None
    """
    The relations of the part hierarchy that were made.
    """

    part_whole_f1: float | MeanAndStandardDeviation | None
    """
    The harmonic mean of the two.
    """

    representable_recall: float | MeanAndStandardDeviation | None
    """
    The relations the ontology can hold that were made.
    """

    whole_chain_f1: float | MeanAndStandardDeviation | None
    """
    The partial credit for a part hung on the right branch.
    """

    part_whole_with_classes_f1: float | MeanAndStandardDeviation | None
    """
    The relations made with both ends given exactly their class.
    """

    @classproperty
    def table_name(cls) -> str:
        return "part_whole"

    @classproperty
    def what_it_shows(cls) -> str:
        return "Whether the part hierarchy is right, judged against the ground truth."

    @classmethod
    def of(cls, group: ScoredGroup) -> PartWholeResult:
        return cls(
            method=group.name,
            runs=group.run_count,
            scenes=group.scenes,
            part_whole_precision=group.over_scores(
                lambda one: one.links.precision.value
            ),
            part_whole_recall=group.over_scores(lambda one: one.links.recall.value),
            part_whole_f1=group.over_scores(lambda one: one.links.f1),
            representable_recall=group.over_scores(
                lambda one: one.links.representable_recall.value
            ),
            whole_chain_f1=group.over_scores(lambda one: one.links.whole_chain_f1),
            part_whole_with_classes_f1=group.over_scores(
                lambda one: one.links.f1_with_classes
            ),
        )


@dataclass
class AnswersResult(RunGroupResult):
    """
    What the model's answers to the questions about wholes added, and how the questions
    that may be answered with none came out.
    """

    part_whole_f1_answered: float | MeanAndStandardDeviation | None
    """
    The part-whole F1 of the runs as they are.
    """

    part_whole_f1_first_candidates: float | MeanAndStandardDeviation | None
    """
    The part-whole F1 with every question about wholes answered by its first candidate.
    """

    true_whole_kept: int | None
    """
    How many questions that may be answered with none kept the true whole.
    """

    true_whole_declined: int | None
    """
    How many declined every whole although the true one was offered.
    """

    other_whole_taken: int | None
    """
    How many took another whole although the true one was offered.
    """

    wrong_whole_avoided: int | None
    """
    How many declined every whole where the true one was not offered.
    """

    wrong_whole_taken: int | None
    """
    How many took a whole where the true one was not offered.
    """

    true_whole_offered: float | None
    """
    The share of those questions that offered the true whole: how often always taking a
    candidate would have been right.
    """

    @classproperty
    def table_name(cls) -> str:
        return "answers"

    @classproperty
    def what_it_shows(cls) -> str:
        return (
            "What the model's answers about wholes add, against answering every "
            "question by its first candidate, and how the questions that may be "
            "answered with none came out, counted over every run."
        )

    @classmethod
    def can_report(cls, comparison: RunComparison) -> bool:
        return comparison.any_answers_judged

    @classmethod
    def of(cls, group: RunGroup) -> AnswersResult:
        def counted(outcome: ConfirmationOutcome) -> int | None:
            return group.total(
                lambda one: one.answers.confirmations(outcome) if one.answers else None
            )

        offered = [one.answers.true_whole_offered for one in group.runs if one.answers]
        return cls(
            method=group.name,
            runs=group.run_count,
            scenes=group.scenes,
            part_whole_f1_answered=group.over(
                lambda one: one.answers.with_answers.links.f1 if one.answers else None
            ),
            part_whole_f1_first_candidates=group.over(
                lambda one: (
                    one.answers.without_answers.links.f1 if one.answers else None
                )
            ),
            true_whole_kept=counted(ConfirmationOutcome.TRUE_WHOLE_KEPT),
            true_whole_declined=counted(ConfirmationOutcome.TRUE_WHOLE_DECLINED),
            other_whole_taken=counted(ConfirmationOutcome.OTHER_WHOLE_TAKEN),
            wrong_whole_avoided=counted(ConfirmationOutcome.WRONG_WHOLE_AVOIDED),
            wrong_whole_taken=counted(ConfirmationOutcome.WRONG_WHOLE_TAKEN),
            true_whole_offered=Share(
                sum(one.count for one in offered), sum(one.total for one in offered)
            ).value,
        )


@dataclass
class StructureResult(RunGroupResult):
    """
    How much of their scenes runs made something of, with no ground truth involved.
    """

    labels_mapped: float | MeanAndStandardDeviation | None
    """
    The share of the labels read as a class.
    """

    new_classes: float | MeanAndStandardDeviation | None
    """
    How many of those classes the run proposed itself.
    """

    bodies_annotated: float | MeanAndStandardDeviation | None
    """
    The share of the bodies holding their class in the final world.
    """

    bodies_mounted: float | MeanAndStandardDeviation | None
    """
    The share of the bodies mounted into a whole.
    """

    mounts_accepted: float | MeanAndStandardDeviation | None
    """
    The share of the mounts named that the ontology accepted.
    """

    @classproperty
    def table_name(cls) -> str:
        return "structure"

    @classproperty
    def what_it_shows(cls) -> str:
        return (
            "How much was built from the scene, counted without any ground truth: this "
            "says how much a method asserted, not whether it is right."
        )

    @classmethod
    def of(cls, group: RunGroup) -> StructureResult:
        return cls(
            method=group.name,
            runs=group.run_count,
            scenes=group.scenes,
            labels_mapped=group.over(
                lambda one: Share(one.labels_mapped, one.labels).value
            ),
            new_classes=group.over(lambda one: one.new_classes),
            bodies_annotated=group.over(
                lambda one: Share(one.bodies_annotated, one.bodies).value
            ),
            bodies_mounted=group.over(
                lambda one: Share(one.parts_mounted, one.bodies).value
            ),
            mounts_accepted=group.over(
                lambda one: (
                    Share(
                        one.parts_mounted, one.parts_mounted + one.mounts_refused
                    ).value
                )
            ),
        )


@dataclass
class EffortResult(RunGroupResult):
    """
    What runs asked of their model and how long they took.
    """

    questions: float | MeanAndStandardDeviation | None
    """
    How many questions were asked.
    """

    corrected_answers: float | MeanAndStandardDeviation | None
    """
    How many answers had to be asked for again.
    """

    unusable_answers: float | MeanAndStandardDeviation | None
    """
    How many of the answers about overlaps stayed unusable.
    """

    thousand_tokens: float | MeanAndStandardDeviation | None
    """
    The tokens sent and answered, in thousands.
    """

    cost_cents: float | MeanAndStandardDeviation | None
    """
    What it cost in US cents, where the service named a price.
    """

    model_minutes: float | MeanAndStandardDeviation | None
    """
    How long the model took to answer.
    """

    total_minutes: float | MeanAndStandardDeviation | None
    """
    How long the whole run took.
    """

    @classproperty
    def table_name(cls) -> str:
        return "effort"

    @classproperty
    def what_it_shows(cls) -> str:
        return "What was asked of the model and how long it took."

    @classmethod
    def of(cls, group: RunGroup) -> EffortResult:
        return cls(
            method=group.name,
            runs=group.run_count,
            scenes=group.scenes,
            questions=group.over(lambda one: one.model_use.questions),
            corrected_answers=group.over(
                lambda one: one.model_use.calls - one.model_use.questions
            ),
            unusable_answers=group.over(lambda one: one.answers_with_problems),
            thousand_tokens=group.over(
                lambda one: (
                    (one.model_use.prompt_tokens + one.model_use.completion_tokens)
                    / 1000
                )
            ),
            cost_cents=group.over(lambda one: one.model_use.cost_cents),
            model_minutes=group.over(lambda one: one.model_use.seconds / 60),
            total_minutes=group.over(lambda one: one.total_seconds / 60),
        )


@dataclass
class RuleCeilingResult(TableRow):
    """
    What the pipeline's rule does with the relations the ontology can hold, given the
    true classes and a method's settings, and so the most of them any model can make.
    """

    scenes: str | int
    """
    The scene the method's runs were made of, or how many there were.
    """

    taken_right: int
    """
    How many relations the rule settles without a question.
    """

    left_to_the_model: int
    """
    How many it asks about with the true whole offered.
    """

    wrong_whole_taken: int
    """
    How many it settles wrongly without a question.
    """

    true_whole_not_offered: int
    """
    How many it asks about without offering the true whole.
    """

    no_candidate: int
    """
    How many it offers no whole for.
    """

    ceiling: float | None
    """
    The share of the relations settled right or asked about with the true whole offered.
    """

    @classproperty
    def table_name(cls) -> str:
        return "rule_ceiling"

    @classproperty
    def what_it_shows(cls) -> str:
        return (
            "What the rule does with the relations the ontology can hold, given the true "
            "classes and each method's settings: the most a model answering its "
            "questions can make."
        )

    @classmethod
    def of(cls, group: RunGroup, ceiling: RuleCeiling) -> RuleCeilingResult:
        """
        :param group: The runs of a method.
        :param ceiling: What the rule did on their scenes, with their settings.
        :return: The row reporting it.
        """
        return cls(
            method=group.name,
            scenes=group.scenes,
            taken_right=ceiling.count(CandidateOutcome.TAKEN_RIGHT, True),
            left_to_the_model=ceiling.count(CandidateOutcome.LEFT_TO_THE_MODEL, True),
            wrong_whole_taken=ceiling.count(CandidateOutcome.WRONG_WHOLE_TAKEN, True),
            true_whole_not_offered=ceiling.count(
                CandidateOutcome.TRUE_WHOLE_NOT_OFFERED, True
            ),
            no_candidate=ceiling.count(CandidateOutcome.NO_CANDIDATE, True),
            ceiling=ceiling.ceiling.value,
        )


# %% the tables


class TableFile(StrEnum):
    """
    The files written beside the tables of each row.
    """

    EVERY_TABLE = "all_tables.tex"
    """
    Every table, one after another, to be read into a document at once.
    """

    METHODS = "methods"
    """
    What the tables' names of the methods stand for.
    """

    SHARED_SETTINGS = "shared_settings.json"
    """
    The settings every method has in common.
    """


@dataclass
class ResultTables:
    """
    The tables of a comparison, per method and per run.

    What holds for every row of a table is said once in its caption rather than in a
    column, so the columns hold only what tells the rows apart.
    """

    comparison: RunComparison
    """
    The runs compared.
    """

    baselines: list[BaselineGroup] = field(default_factory=list)
    """
    The baselines reported beside the methods in every table read from scores alone.
    """

    ceilings: dict[str, RuleCeiling] = field(default_factory=dict)
    """
    Per method, by name, what the rule does on its scenes with its settings.
    """

    @classproperty
    def row_kinds(cls) -> list[type[RunGroupResult]]:
        """
        :return: Each kind of row a table is made of, so a table fits on a page.
        """
        return [
            ClassResult,
            PartWholeResult,
            AnswersResult,
            StructureResult,
            EffortResult,
        ]

    def write(self, directory: Path) -> list[Path]:
        """
        Write each table as LaTeX, with all of its rows and columns as JSON beside it,
        and every table again in one file.

        :param directory: Where to write them, made where it is not there.
        :return: The tables written, without the one holding them all.
        """
        directory.mkdir(parents=True, exist_ok=True)
        (directory / TableFile.SHARED_SETTINGS).write_text(
            json.dumps(self.comparison.shared_settings, indent=2)
        )
        written = [self.write_methods(directory)]
        levels = [
            ("conditions", "Per method.", self.comparison.by_condition),
            ("runs", "Per run.", self.comparison.by_run),
        ]
        for level, level_caption, groups in levels:
            for kind in self.row_kinds:
                if not kind.can_report(self.comparison):
                    continue
                rows = [kind.of(group) for group in groups]
                if kind.reads_only_the_score and level == "conditions":
                    rows += [kind.of(baseline) for baseline in self.baselines]
                written.append(
                    self.write_table(
                        directory, f"{level}_{kind.table_name}", rows, level_caption
                    )
                )
        ceilings = [
            RuleCeilingResult.of(group, self.ceilings[group.name])
            for group in self.comparison.by_condition
            if group.name in self.ceilings
        ]
        if ceilings:
            written.append(
                self.write_table(
                    directory,
                    f"conditions_{RuleCeilingResult.table_name}",
                    ceilings,
                    "Per method.",
                )
            )
        (directory / TableFile.EVERY_TABLE).write_text(
            "\n".join(path.read_text() for path in written)
        )
        return written

    def write_table(
        self,
        directory: Path,
        name: str,
        rows: Sequence[TableRow],
        level_caption: str,
    ) -> Path:
        """
        :param directory: Where to write the table.
        :param name: What it is named after.
        :param rows: Its rows, all of one kind.
        :param level_caption: What the caption says the rows are, after what they show.
        :return: The table, written as LaTeX with its rows as JSON beside it.
        """
        table = ExperimentsTable(list(rows))
        kind = type(rows[0])
        path = directory / f"{name}.tex"
        path.write_text(
            self.figure_of(table, kind, f"{kind.what_it_shows} {level_caption}", name)
            + "\n"
        )
        table.write_manifest(directory, f"{name}.json")
        return path

    def write_methods(self, directory: Path) -> Path:
        """
        :param directory: Where to write the table.
        :return: The table saying what each method's short name stands for.
        """
        definitions = {
            self.comparison.condition_of(run): self.comparison.definition_of(run)
            for run in self.comparison.runs
        }
        table = ExperimentsTable(
            [
                MethodDefinition(method, definitions[method])
                for method in sorted(definitions)
            ]
        )
        path = directory / f"{TableFile.METHODS}.tex"
        path.write_text(
            LatexRenderer(table, fit_to_line_width=True).render_figure(
                "The methods compared. Every other setting is the same for all of them.",
                label=f"tab:{TableFile.METHODS}",
            )
            + "\n"
        )
        table.write_manifest(directory, f"{TableFile.METHODS}.json")
        return path

    @staticmethod
    def figure_of(
        table: ExperimentsTable,
        kind: type[TableRow],
        what_it_shows: str,
        name: str,
    ) -> str:
        """
        :param table: The rows.
        :param kind: The kind of rows they are.
        :param what_it_shows: The start of the caption.
        :param name: What the table is named after.
        :return: The table, with what holds for every row in its caption.
        """
        renderer = LatexRenderer(table, fit_to_line_width=True)
        alike = renderer.columns_showing_one_value()
        alike.pop("method", None)
        absent = {k for k, v in alike.items() if v == renderer.absent_measurement}
        measured = set(kind.get_column_names()) - set(kind.context_columns) - {"method"}
        if measured - absent <= set(alike):
            alike = {k: v for k, v in alike.items() if k in kind.context_columns}
        left_out = set(alike) | absent
        said = [
            f"{key.replace('_', ' ').title()} {value}"
            for key, value in alike.items()
            if key not in absent
        ]
        caption = [what_it_shows]
        if said:
            caption.append(f"For every row: {', '.join(said)}.")
        if any(row.covers_several_runs for row in table.experiments):
            caption.append(r"Several runs are given as mean $\pm$ standard deviation.")
        return LatexRenderer(
            table, left_out_columns=frozenset(left_out), fit_to_line_width=True
        ).render_figure(" ".join(caption), label=f"tab:{name}")


# %% command-line entry point


def argument_parser() -> argparse.ArgumentParser:
    """
    :return: The command-line interface for writing the tables of finished runs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs", type=Path, nargs="+", help="The directories of finished runs"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Where to write the tables"
    )
    parser.add_argument(
        "--rename",
        action="append",
        default=[],
        metavar="CONDITION=NAME",
        help="What to call a condition in the tables, instead of what tells it apart",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """
    Write the tables of finished runs.

    :param arguments: Command-line arguments without the program name.
    :return: Zero once the tables are written.
    """
    parsed = argument_parser().parse_args(arguments)
    comparison = RunComparison(
        runs=[RunMeasurements.of(Run(directory=one)) for one in parsed.runs],
        renamed_conditions=dict(one.split("=", 1) for one in parsed.rename),
    )
    for group in comparison.by_condition:
        print(f"{group.name}: {len(group.runs)} runs")
    for path in ResultTables(comparison).write(parsed.output):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
