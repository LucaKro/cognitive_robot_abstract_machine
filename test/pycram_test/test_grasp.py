import pytest
import numpy as np
from pycram.datastructures.grasp import GraspDescription, PreferredGraspAlignment
from pycram.datastructures.enums import (
    ApproachDirection,
    VerticalAlignment,
    AxisIdentifier,
)
from pycram.datastructures.pose import PoseStamped, PyCramVector3
from pycram.testing import ApartmentWorldTestCase


class TestGrasp(ApartmentWorldTestCase):
    """
    Test suite for the GraspDescription class and its related functionalities.
    """

    def test_grasp_description_hash(self):
        """
        Tests that the GraspDescription can be hashed and that equality works as expected.
        """
        grasp_description_1 = GraspDescription(
            ApproachDirection.FRONT, VerticalAlignment.TOP, False
        )
        grasp_description_2 = GraspDescription(
            ApproachDirection.FRONT, VerticalAlignment.TOP, False
        )
        grasp_description_3 = GraspDescription(
            ApproachDirection.BACK, VerticalAlignment.TOP, False
        )

        assert hash(grasp_description_1) == hash(grasp_description_2)
        assert hash(grasp_description_1) != hash(grasp_description_3)
        assert grasp_description_1 == grasp_description_2
        assert grasp_description_1 != grasp_description_3

    def test_as_list(self):
        """
        Tests the as_list method of GraspDescription.
        """
        grasp_description = GraspDescription(
            ApproachDirection.FRONT, VerticalAlignment.TOP, True
        )
        assert grasp_description.as_list() == [
            ApproachDirection.FRONT,
            VerticalAlignment.TOP,
            True,
        ]

    def test_calculate_grasp_orientation(self):
        """
        Tests that the calculated grasp orientation is a valid normalized quaternion.
        """
        front_orientation = np.array([0, 0, 0, 1])
        grasp_description = GraspDescription(
            ApproachDirection.FRONT, VerticalAlignment.NoAlignment, False
        )
        orientation = grasp_description.calculate_grasp_orientation(front_orientation)

        assert len(orientation) == 4
        assert np.isclose(np.linalg.norm(orientation), 1.0)

    def test_calculate_closest_faces_primary_secondary(self):
        """
        Tests the calculation of closest faces based on a vector.
        """
        # Robot is at +X relative to object (object's BACK face is facing the robot)
        # Use np.nan to isolate the X axis for testing secondary as opposite face
        vector = PyCramVector3(1, np.nan, np.nan)
        primary, secondary = GraspDescription.calculate_closest_faces(vector)
        assert primary == ApproachDirection.BACK
        assert secondary == ApproachDirection.FRONT

        # Robot is at -X relative to object
        vector = PyCramVector3(-1, np.nan, np.nan)
        primary, secondary = GraspDescription.calculate_closest_faces(vector)
        assert primary == ApproachDirection.FRONT
        assert secondary == ApproachDirection.BACK

        # Mixed axes, primary should be the one with larger magnitude
        # 1.0 > 0.5, so X is primary. Y is next best (secondary).
        vector = PyCramVector3(1, 0.5, np.nan)
        primary, secondary = GraspDescription.calculate_closest_faces(vector)
        assert primary == ApproachDirection.BACK
        assert secondary == ApproachDirection.LEFT  # (AxisIdentifier.Y, 1) is LEFT

    def test_calculate_closest_faces_vertical(self):
        """
        Tests the calculation of closest faces for vertical alignment.
        """
        # Robot is above the object (+Z)
        vector = PyCramVector3(np.nan, np.nan, 1)
        primary, secondary = GraspDescription.calculate_closest_faces(vector)
        assert primary == VerticalAlignment.BOTTOM
        assert secondary == VerticalAlignment.TOP

    def test_calculate_closest_faces_specified_axis(self):
        """
        Tests that specifying an axis overrides the default magnitude-based selection.
        """
        vector = PyCramVector3(1, 10, np.nan)  # Y has larger magnitude
        primary, secondary = GraspDescription.calculate_closest_faces(
            vector, specified_grasp_axis=AxisIdentifier.X
        )
        assert primary == ApproachDirection.BACK
        assert secondary == ApproachDirection.FRONT

    def test_get_grasp_pose_for_pose(self):
        """
        Tests the generation of a grasp pose from a target pose.
        """
        manipulator = self.robot_view.left_arm.manipulator
        target_pose = PoseStamped.from_list([1, 1, 1], [0, 0, 0, 1], self.world.root)

        # Use BACK approach direction to ensure a non-identity rotation
        grasp_description = GraspDescription(
            ApproachDirection.BACK, VerticalAlignment.NoAlignment, False
        )
        grasp_pose = grasp_description.get_grasp_pose_for_pose(manipulator, target_pose)

        assert isinstance(grasp_pose, PoseStamped)
        assert np.allclose(grasp_pose.position.to_list(), [1, 1, 1])
        # Orientation should be different from target orientation
        assert not np.allclose(
            grasp_pose.orientation.to_list(), target_pose.orientation.to_list()
        )

    def test_get_grasp_pose_for_body_with_offset(self):
        """
        Tests the generation of a grasp pose from a body, including rim offset translation.
        """
        milk = self.world.get_body_by_name("milk.stl")
        manipulator = self.robot_view.left_arm.manipulator

        grasp_description = GraspDescription(
            ApproachDirection.FRONT, VerticalAlignment.NoAlignment, False
        )

        # Without rim offset
        grasp_pose_no_offset = grasp_description.get_grasp_pose_for_body(
            manipulator, milk, translate_rim_offset=False
        )
        milk_pose = PoseStamped().from_spatial_type(milk.global_pose)
        assert np.allclose(
            grasp_pose_no_offset.position.to_list(), milk_pose.position.to_list()
        )

        # With rim offset
        grasp_pose_offset = grasp_description.get_grasp_pose_for_body(
            manipulator, milk, translate_rim_offset=True
        )
        assert not np.allclose(
            grasp_pose_offset.position.to_list(), milk_pose.position.to_list()
        )

    def test_calculate_grasp_descriptions_generation(self):
        """
        Tests the generation of multiple grasp descriptions based on robot and target pose.
        """
        target_pose = PoseStamped.from_list([2, 2, 0.5], [0, 0, 0, 1], self.world.root)

        grasp_descriptions = GraspDescription.calculate_grasp_descriptions(
            self.robot_view, target_pose
        )
        assert len(grasp_descriptions) > 0
        assert all(isinstance(gd, GraspDescription) for gd in grasp_descriptions)

        # Test with alignment preference
        alignment = PreferredGraspAlignment(
            AxisIdentifier.X, with_vertical_alignment=True, with_rotated_gripper=True
        )
        grasp_descriptions_aligned = GraspDescription.calculate_grasp_descriptions(
            self.robot_view, target_pose, alignment
        )

        assert len(grasp_descriptions_aligned) > 0
        for gd in grasp_descriptions_aligned:
            assert gd.vertical_alignment != VerticalAlignment.NoAlignment
            assert gd.rotate_gripper is True
            assert gd.approach_direction.axis == AxisIdentifier.X
