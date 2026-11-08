"""
Measuring an Articulate3D scene once, as the pipeline measures it, so that baselines and
the rule's ceiling can be worked out on every scene without converting it again.
"""

from __future__ import annotations

from experiments.warsaw.articulate3d.measured_scene import (
    MeasuredScene,
    MeasuredSceneFile,
    main,
)
from experiments.warsaw.articulate3d.protocol import EvaluationProtocol
from experiments.warsaw.articulate3d.rules import AnnotationRules

from .scenes import (
    SceneName,
    kitchen_corner_parts,
    measured_kitchen_corner,
    row_of_squares,
)

# %% the record


def test_a_measured_scene_reads_back_as_it_was_written(tmp_path):
    measured = measured_kitchen_corner()
    path = MeasuredSceneFile.SUFFIX.path(tmp_path, measured.converted.scene_id)

    measured.write(path)

    assert MeasuredScene.read(path) == measured


def test_each_segment_is_known_by_the_label_it_carries():
    labels = measured_kitchen_corner().label_of_segment

    assert labels["switch_base_1"] == "switch_base"
    assert labels["handle_2"] == "handle"


def test_each_segment_is_known_as_the_part_it_was_written_from():
    assert measured_kitchen_corner().converted.segment_of_part[5] == "handle_2"


# %% measuring a scene from the datasets


def measure_kitchen_corner(dataset, tmp_path):
    """
    :return: The directory the scene's measurement is written into.
    """
    dataset.write_scene(
        SceneName.KITCHEN_CORNER, kitchen_corner_parts(), row_of_squares(8)
    )
    protocol_path = tmp_path / "protocol.json"
    EvaluationProtocol(annotation_rules=AnnotationRules()).write(protocol_path)
    output = tmp_path / "measured"
    main(
        [
            "--dataset",
            str(dataset.articulate3d_directory),
            "--scannetpp",
            str(dataset.scannetpp_directory),
            "--protocol",
            str(protocol_path),
            "--output",
            str(output),
            SceneName.KITCHEN_CORNER,
        ]
    )
    return output


def test_measuring_a_scene_records_one_segment_per_part_with_faces(dataset, tmp_path):
    output = measure_kitchen_corner(dataset, tmp_path)

    measured = MeasuredScene.read(
        MeasuredSceneFile.SUFFIX.path(output, SceneName.KITCHEN_CORNER)
    )

    assert {one.name for one in measured.segments} == {
        one.segment for one in measured.converted.parts
    }


def test_measuring_a_scene_finds_the_neighbours_that_touch(dataset, tmp_path):
    """
    The cabinet's squares and the door's square lie side by side in the row.
    """
    output = measure_kitchen_corner(dataset, tmp_path)
    measured = MeasuredScene.read(
        MeasuredSceneFile.SUFFIX.path(output, SceneName.KITCHEN_CORNER)
    )
    segment = measured.converted.segment_of_part
    cabinet, door = segment[1], segment[2]

    [pair] = [one for one in measured.pairs if {one.one, one.other} == {cabinet, door}]

    assert pair.touching_edges > 0


def test_measuring_leaves_nothing_but_the_measurements_behind(dataset, tmp_path):
    output = measure_kitchen_corner(dataset, tmp_path)

    assert [one.name for one in output.iterdir()] == [
        MeasuredSceneFile.SUFFIX.path(output, SceneName.KITCHEN_CORNER).name
    ]
