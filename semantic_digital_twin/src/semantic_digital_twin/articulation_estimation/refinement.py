"""
The refinement of a motion along a screw axis by robust least squares: damped Gauss-
Newton steps that solve for the shared pose and screw axis first and then for each
position on its own.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.articulation_estimation.observations import (
    ObservationNoise,
    PoseObservations,
)

if TYPE_CHECKING:
    from semantic_digital_twin.articulation_estimation.fits import (
        ScrewMotionParameters,
    )


# %% The robust loss
@dataclass(kw_only=True)
class SoftL1Loss:
    r"""
    The cost :math:`c(r) = s^2 (\sqrt{1 + (r / s)^2} - 1)` of an error :math:`r`:
    about :math:`r^2 / 2` for small errors and growing only linearly beyond the scale
    :math:`s`, so that a few outliers do not dominate the sum.
    """

    scale: float
    """
    The error :math:`s` beyond which the cost grows only linearly.
    """

    def costs(self, *, errors: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        :param errors: The errors.
        :return: The cost of each.
        """
        return self.scale**2 * (np.sqrt(1 + (errors / self.scale) ** 2) - 1)

    def slopes(self, *, errors: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""
        :param errors: The errors.
        :return: The derivative :math:`c'(r) = r / \sqrt{1 + (r / s)^2}` at each.
        """
        return errors / np.sqrt(1 + (errors / self.scale) ** 2)

    def curvatures(self, *, errors: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""
        :param errors: The errors.
        :return: The second derivative :math:`c''(r) = (1 + (r / s)^2)^{-3/2}` at each,
            always positive.
        """
        return (1 + (errors / self.scale) ** 2) ** -1.5


# %% Linearized errors
@dataclass(kw_only=True)
class LinearizedPoseErrors:
    """
    The standardized errors of each observation and their derivatives, see
    :meth:`ScrewMotionParameters.linearized_errors`.
    """

    errors: npt.NDArray[np.float64]
    """
    The position and then the orientation errors of each observation, shape (n, 6).
    """

    reference_pose_jacobians: npt.NDArray[np.float64]
    """
    The derivatives of each observation's errors by a twist that moves the pose at
    position zero in the parent frame, shape (n, 6, 6).
    """

    position_jacobians: npt.NDArray[np.float64]
    """
    The derivatives of each observation's errors by its own position, shape (n, 6).
    """

    screw_axis_jacobians: npt.NDArray[np.float64]
    """
    The derivatives of each observation's errors by the screw axis entries, shape (n, 6,
    6).
    """

    @property
    def shared_jacobians(self) -> npt.NDArray[np.float64]:
        """
        The derivatives by the twist of the pose at position zero and then by the screw
        axis entries, the twelve numbers every observation depends on, shape (n, 6, 12).
        """
        return np.concatenate(
            [self.reference_pose_jacobians, self.screw_axis_jacobians], axis=2
        )


# %% Steps
@dataclass(kw_only=True)
class ScrewMotionStep:
    """
    A change of :class:`ScrewMotionParameters`, see
    :meth:`ScrewMotionParameters.copy_with_step`.
    """

    reference_pose_twist: npt.NDArray[np.float64]
    """
    The exponential coordinates of the move of the pose at position zero, applied in the
    parent frame.
    """

    position_changes: npt.NDArray[np.float64]
    """
    The change of the position at each observation.
    """

    screw_axis_change: npt.NDArray[np.float64]
    """
    The change of the six screw axis entries.
    """


@dataclass(kw_only=True)
class NormalEquations:
    r"""
    The Gauss-Newton equations of a step, in blocks: the twelve shared numbers :math:`a`
    and the positions :math:`p`, of which each error depends on one only.

    .. math::
        \begin{bmatrix} H_{aa} & H_{ap} \\ H_{ap}^T & H_{pp} \end{bmatrix}
        \begin{bmatrix} \delta a \\ \delta p \end{bmatrix} =
        -\begin{bmatrix} g_a \\ g_p \end{bmatrix}

    with :math:`H_{pp}` diagonal, so that a step costs time linear in the observations.
    """

    shared_hessian: npt.NDArray[np.float64]
    """
    :math:`H_{aa}`, shape (12, 12).
    """

    shared_gradient: npt.NDArray[np.float64]
    """
    :math:`g_a`, shape (12,).
    """

    position_hessians: npt.NDArray[np.float64]
    """
    The diagonal of :math:`H_{pp}`, shape (n,).
    """

    coupling: npt.NDArray[np.float64]
    """
    :math:`H_{ap}`, shape (12, n).
    """

    position_gradients: npt.NDArray[np.float64]
    """
    :math:`g_p`, shape (n,).
    """

    @classmethod
    def from_linearized_errors(
        cls, *, linearized_errors: LinearizedPoseErrors, loss: SoftL1Loss
    ) -> NormalEquations:
        """
        :param linearized_errors: The errors and their derivatives.
        :param loss: The loss of the errors.
        :return: The equations of the Gauss-Newton step of the summed loss, with the
            loss's second derivative weighting each error.
        """
        shared_jacobians = linearized_errors.shared_jacobians
        position_jacobians = linearized_errors.position_jacobians
        curvatures = loss.curvatures(errors=linearized_errors.errors)
        slopes = loss.slopes(errors=linearized_errors.errors)
        stacked_jacobians = shared_jacobians.reshape(-1, 12)
        return cls(
            shared_hessian=stacked_jacobians.T
            @ (curvatures.reshape(-1, 1) * stacked_jacobians),
            shared_gradient=stacked_jacobians.T @ slopes.ravel(),
            position_hessians=np.sum(curvatures * position_jacobians**2, axis=1),
            coupling=np.sum(
                shared_jacobians * (curvatures * position_jacobians)[:, :, None], axis=1
            ).T,
            position_gradients=np.sum(slopes * position_jacobians, axis=1),
        )

    def damped_step(self, *, damping: float) -> ScrewMotionStep:
        r"""
        :param damping: Marquardt's factor :math:`\lambda`, which adds :math:`\lambda`
            times the diagonal of the equations to it.
        :return: The solution, through the Schur complement: the twelve shared numbers
            first, then each position from them.
        """
        position_hessians = self.position_hessians * (1 + damping)
        reduced_hessian = (
            self.shared_hessian
            + damping * np.diag(np.diag(self.shared_hessian))
            - (self.coupling / position_hessians) @ self.coupling.T
        )
        shared_change = np.linalg.solve(
            reduced_hessian,
            self.coupling @ (self.position_gradients / position_hessians)
            - self.shared_gradient,
        )
        return ScrewMotionStep(
            reference_pose_twist=shared_change[:6],
            position_changes=-(
                self.position_gradients + self.coupling.T @ shared_change
            )
            / position_hessians,
            screw_axis_change=shared_change[6:],
        )


# %% The refinement
@dataclass(kw_only=True)
class RefinementIterate:
    """
    Where the refinement stands after a step.
    """

    parameters: ScrewMotionParameters
    """
    The current motion.
    """

    cost: float
    """
    Its summed loss.
    """

    damping: float
    """
    The damping the next step starts from.
    """


@dataclass(kw_only=True)
class ScrewMotionRefinement:
    """
    Levenberg-Marquardt on the summed :class:`SoftL1Loss` of the standardized errors.

    All positions are free; after each step the motion is reparametrized to the first
    position zero and a screw axis of unit length, which changes no prediction.
    """

    observations: PoseObservations
    """
    The poses the motion explains.
    """

    noise: ObservationNoise
    """
    The noise the observations carry.
    """

    loss: SoftL1Loss
    """
    The loss of each standardized error.
    """

    relative_cost_tolerance: float = 1e-8
    """
    The refinement stops once a step lowers the cost by less than this fraction.
    """

    screening_iteration_count: int = 10
    """
    The steps :meth:`refine_lowest` takes from each start before it goes on with only
    the lowest.
    """

    maximum_iteration_count: int = 100
    """
    The most steps :meth:`refine_lowest` takes from the lowest start, whether or not it
    converged.
    """

    initial_damping: float = 1e-3
    """
    The damping of the first step.
    """

    damping_decrease: float = 3.0
    """
    The factor the damping shrinks by after a step that lowered the cost.
    """

    damping_increase: float = 4.0
    """
    The factor the damping grows by after a step that did not.
    """

    smallest_damping: float = 1e-10
    """
    The damping never shrinks below this, which keeps the equations solvable although
    the free positions leave the motion's parametrization ambiguous.
    """

    largest_damping: float = 1e12
    """
    Once no step with up to this damping lowers the cost, the refinement has converged.
    """

    def refine_lowest(
        self, *, starts: list[ScrewMotionParameters]
    ) -> ScrewMotionParameters:
        """
        :param starts: The motions to start from.
        :return: The refinement of the start that is lowest after
            :attr:`screening_iteration_count` steps each: different starts can end in
            different minima.
        """
        screened = [
            self.refine(
                parameters=start, iteration_count=self.screening_iteration_count
            )
            for start in starts
        ]
        return self.refine(
            parameters=min(
                screened, key=lambda parameters: self.summed_loss(parameters=parameters)
            ),
            iteration_count=self.maximum_iteration_count,
        )

    def refine(
        self, *, parameters: ScrewMotionParameters, iteration_count: int
    ) -> ScrewMotionParameters:
        """
        :param parameters: The motion to start from.
        :param iteration_count: The most steps to take.
        :return: The motion at which the refinement stopped.
        """
        parameters = parameters.copy_with_fixed_gauge(noise=self.noise)
        iterate = RefinementIterate(
            parameters=parameters,
            cost=self.summed_loss(parameters=parameters),
            damping=self.initial_damping,
        )
        for _ in range(iteration_count):
            next_iterate = self.next_iterate(iterate=iterate)
            if next_iterate is None:
                break
            converged = (
                iterate.cost - next_iterate.cost
                < self.relative_cost_tolerance * next_iterate.cost
            )
            iterate = next_iterate
            if converged:
                break
        return iterate.parameters

    def next_iterate(self, *, iterate: RefinementIterate) -> RefinementIterate | None:
        """
        :param iterate: Where the refinement stands.
        :return: Where the first step that lowers the cost leads, raising the damping
            until one does, or None if none does up to :attr:`largest_damping`.
        """
        equations = NormalEquations.from_linearized_errors(
            linearized_errors=iterate.parameters.linearized_errors(
                observations=self.observations, noise=self.noise
            ),
            loss=self.loss,
        )
        damping = iterate.damping
        while damping <= self.largest_damping:
            candidate = iterate.parameters.copy_with_step(
                step=equations.damped_step(damping=damping)
            )
            cost = self.summed_loss(parameters=candidate)
            if cost < iterate.cost:
                return RefinementIterate(
                    parameters=candidate.copy_with_fixed_gauge(noise=self.noise),
                    cost=cost,
                    damping=max(damping / self.damping_decrease, self.smallest_damping),
                )
            damping *= self.damping_increase
        return None

    def summed_loss(self, *, parameters: ScrewMotionParameters) -> float:
        """
        :param parameters: A motion.
        :return: The summed loss of its standardized errors.
        """
        return float(
            np.sum(
                self.loss.costs(
                    errors=self.noise.standardized_pose_errors(
                        observed_stack=self.observations.parent_T_child_stack,
                        predicted_stack=parameters.predicted_parent_T_child_stack,
                    )
                )
            )
        )
