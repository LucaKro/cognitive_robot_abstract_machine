"""Count-based semantic-label metrics compatible with the earlier HM3D study."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

# %% results


@dataclass(frozen=True)
class ClassLabelMetrics:
    """Counts and intersection over union for one semantic class."""

    class_name: str
    """The semantic class being counted."""

    predicted: int
    """How many predictions carry the class."""

    ground_truth: int
    """How many ground-truth entities carry the class."""

    true_positives: int
    """The shared multiplicity of the class."""

    false_positives: int
    """Predictions beyond the ground-truth multiplicity."""

    false_negatives: int
    """Ground-truth entities beyond the predicted multiplicity."""

    intersection_over_union: float
    """The class-count intersection divided by its union."""


@dataclass(frozen=True)
class LabelCountMetrics:
    """Bag-of-label metrics over predicted and ground-truth class multiplicities."""

    true_positives: int
    """The sum of shared class multiplicities."""

    false_positives: int
    """Predictions not covered by a shared class multiplicity."""

    false_negatives: int
    """Ground-truth entities not covered by a shared class multiplicity."""

    precision: float
    """The true-positive share of all predictions."""

    recall: float
    """The true-positive share of all ground-truth entities."""

    f1_score: float
    """The harmonic mean of precision and recall."""

    mean_intersection_over_union: float
    """The macro average over every class present in either input."""

    per_class: dict[str, ClassLabelMetrics] = field(default_factory=dict)
    """The counts and intersection over union of every class."""


# %% count comparison


def compare_label_counts(
    predicted: Sequence[str], ground_truth: Sequence[str]
) -> LabelCountMetrics:
    """Compare label multiplicities as the earlier HM3D evaluation did.

    This intentionally ignores object identity. It exists to reproduce the earlier
    result alongside identity-aware metrics, rather than to stand in for them.

    :param predicted: The predicted semantic class of every counted entity.
    :param ground_truth: The ground-truth semantic class of every counted entity.
    :return: Aggregate and per-class count metrics.
    """
    predicted_counts = Counter(predicted)
    ground_truth_counts = Counter(ground_truth)
    class_names = sorted(set(predicted_counts) | set(ground_truth_counts))
    per_class = {
        class_name: _compare_class_counts(
            class_name,
            predicted_counts[class_name],
            ground_truth_counts[class_name],
        )
        for class_name in class_names
    }

    true_positives = sum(one.true_positives for one in per_class.values())
    false_positives = len(predicted) - true_positives
    false_negatives = len(ground_truth) - true_positives
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    f1_score = _ratio(2.0 * precision * recall, precision + recall)
    mean_intersection_over_union = _ratio(
        sum(one.intersection_over_union for one in per_class.values()),
        len(per_class),
    )
    return LabelCountMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        mean_intersection_over_union=mean_intersection_over_union,
        per_class=per_class,
    )


def _compare_class_counts(
    class_name: str, predicted: int, ground_truth: int
) -> ClassLabelMetrics:
    """Compare the multiplicity of one class.

    :param class_name: The semantic class being counted.
    :param predicted: How many predictions carry the class.
    :param ground_truth: How many ground-truth entities carry the class.
    :return: Counts and intersection over union for the class.
    """
    true_positives = min(predicted, ground_truth)
    false_positives = predicted - true_positives
    false_negatives = ground_truth - true_positives
    union = predicted + ground_truth - true_positives
    return ClassLabelMetrics(
        class_name=class_name,
        predicted=predicted,
        ground_truth=ground_truth,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        intersection_over_union=_ratio(true_positives, union),
    )


def _ratio(numerator: float, denominator: float) -> float:
    """Divide while defining an empty ratio as zero.

    :param numerator: The value above the division line.
    :param denominator: The value below the division line.
    :return: Their ratio, or zero when the denominator is zero.
    """
    return numerator / denominator if denominator else 0.0
