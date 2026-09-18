from __future__ import annotations

from dataclasses import dataclass

from krrood.symbolic_math.exceptions import UnsupportedOperationError
from typing_extensions import Self

from semantic_digital_twin.exceptions import UncertaintyCorrelationUnknownError
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

    Uncertainty travels along a kinematic chain: a bottle held in a drawer whose position
    is uncertain is itself uncertain. Which operation carries it depends on which side the
    certain transform is on, and the two are not the same:

    - :meth:`transformed_by` re-expresses the pose in another frame, and has to carry the
      displacement through that frame change with a :class:`PoseDisplacementMap`.
    - :meth:`dot` extends the pose further along the chain, and carries the displacement
      unchanged, because a rigid offset from an uncertain pose does not change how the
      assembly as a whole is displaced in the frame it is reported in.
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

    def dot(self, pose_T_further: HomogeneousTransformationMatrix) -> Self:
        """
        Extend this pose further along the chain by a transform that is certain.

        The uncertainty is carried unchanged. It describes how far the true pose is from
        the reported one, as a displacement in the frame this pose is reported in, and a
        rigid offset from it is displaced by that same amount.

        ..note:: Unchanged does not mean the far end is no less well located. The
            displacement acts about the reference frame's origin, so a pose further from
            it ends up further from where it was reported. Reading that as uncertainty
            about the far end's own position is :meth:`transformed_by`.

        :param pose_T_further: The transform from this pose to the one to extend it to,
            whose own numbers are taken to be certain.
        :return: The pose at the far end, carrying this pose's uncertainty.
        :raises UncertaintyCorrelationUnknownError: If the transform is itself uncertain.
        :raises UnsupportedOperationError: If it is not a transform at all.
        """
        if isinstance(pose_T_further, UncertainPose):
            raise UncertaintyCorrelationUnknownError()
        if not isinstance(pose_T_further, HomogeneousTransformationMatrix):
            raise UnsupportedOperationError("dot", self.pose, pose_T_further)
        return type(self)(
            pose=(self.pose.to_homogeneous_matrix() @ pose_T_further).to_pose(),
            covariance=self.covariance,
        )

    def __matmul__(self, other: HomogeneousTransformationMatrix) -> Self:
        return self.dot(other)
