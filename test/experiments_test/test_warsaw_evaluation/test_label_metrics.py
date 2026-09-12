"""Semantic-label metrics retained from the earlier HM3D evaluation."""

from __future__ import annotations

from experiments.warsaw.evaluation.label_metrics import compare_label_counts

# %% matching label multisets


def test_label_counts_reproduce_the_previous_hm3d_metric() -> None:
    """Repeated labels are matched by count, independently of object identity."""
    measured = compare_label_counts(
        predicted=["Drawer", "Drawer", "Door", "Handle"],
        ground_truth=["Drawer", "Door", "Door", "Cabinet"],
    )

    assert measured.true_positives == 2
    assert measured.false_positives == 2
    assert measured.false_negatives == 2
    assert measured.precision == 0.5
    assert measured.recall == 0.5
    assert measured.f1_score == 0.5
    assert measured.per_class["Drawer"].intersection_over_union == 0.5
    assert measured.per_class["Door"].intersection_over_union == 0.5
    assert measured.per_class["Handle"].intersection_over_union == 0.0
    assert measured.per_class["Cabinet"].intersection_over_union == 0.0
    assert measured.mean_intersection_over_union == 0.25


def test_empty_label_counts_have_defined_zero_metrics() -> None:
    """An empty comparison is represented explicitly rather than divided by zero."""
    measured = compare_label_counts(predicted=[], ground_truth=[])

    assert measured.true_positives == 0
    assert measured.false_positives == 0
    assert measured.false_negatives == 0
    assert measured.precision == 0.0
    assert measured.recall == 0.0
    assert measured.f1_score == 0.0
    assert measured.mean_intersection_over_union == 0.0
    assert measured.per_class == {}


def test_per_class_counts_expose_false_positives_and_false_negatives() -> None:
    """The atomic class counts remain available for later aggregation."""
    measured = compare_label_counts(
        predicted=["Drawer", "Drawer", "Handle"],
        ground_truth=["Drawer", "Door"],
    )

    drawer = measured.per_class["Drawer"]
    assert drawer.predicted == 2
    assert drawer.ground_truth == 1
    assert drawer.true_positives == 1
    assert drawer.false_positives == 1
    assert drawer.false_negatives == 0
