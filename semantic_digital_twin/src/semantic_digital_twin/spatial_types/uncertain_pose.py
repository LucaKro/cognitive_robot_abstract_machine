from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Self

from semantic_digital_twin.spatial_types.pose_covariance import PoseCovariance
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
)

# %% a pose that says how well it is known


@dataclass
class UncertainPose:
    """
    A pose together with how uncertain it is.

    The two are held together because neither survives an operation without the other:
    a covariance is expressed in a frame, so moving the pose has to move the covariance
    with it. Holding them apart lets one be re-expressed while the other is not, which
    is worse than carrying no uncertainty at all - an uncertainty read in the wrong
    frame still reads as an answer.

    This is also where the uncertainty gets a frame. :class:`PoseCovariance` is
    deliberately frame-naive; pairing it with the pose says which frame it belongs to
    without that type having to carry one.
    """

    pose: Pose
    """
    Where the pose was reported to be.
    """

    covariance: PoseCovariance
    """
    How far the true pose may be from that, in the frame the pose is expressed in.
    """

    def transformed_by(
        self, new_reference_T_reference: HomogeneousTransformationMatrix
    ) -> Self:
        """
        Re-express both halves in another frame.

        :param new_reference_T_reference: The transform from the frame this pose is
            expressed in to the frame to express it in, whose own numbers are taken to
            be certain.
        :return: The same pose and the same uncertainty, read in the new frame.
        :raises HasFreeVariablesError: If the transform is still symbolic, so that the
            uncertainty cannot be carried through it.
        """
        return type(self)(
            pose=new_reference_T_reference @ self.pose,
            covariance=self.covariance.transformed_by(new_reference_T_reference),
        )

    def inverse(self) -> Self:
        """
        Turn the pose around, so that it locates its own reference frame instead.

        The uncertainty is unchanged in size and is read from the other end, which is
        the same propagation applied with the pose's own inverse.

        ..note:: The reference frame follows
            :meth:`HomogeneousTransformationMatrix.inverse`. A pose names the frame it
            is expressed in and not the one it locates, so the result has no frame to
            name and does not get one back on a second inversion; the uncertainty does.

        :return: The pose read from the other end, and its uncertainty there.
        :raises HasFreeVariablesError: If the pose is still symbolic, so that it has no
            numeric inverse to carry the uncertainty through.
        """
        reference_T_pose_inverse = self.pose.to_homogeneous_matrix().inverse()
        return type(self)(
            pose=reference_T_pose_inverse.to_pose(),
            covariance=self.covariance.transformed_by(reference_T_pose_inverse),
        )
