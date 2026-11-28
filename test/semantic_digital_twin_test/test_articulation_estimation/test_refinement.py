"""
Tests of the robust least squares refinement of a motion along a screw axis.
"""

from __future__ import annotations

# %% Imports
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import RigidTransform, Rotation

from semantic_digital_twin.articulation_estimation.fits import ScrewMotionParameters
from semantic_digital_twin.articulation_estimation.observations import (
    ObservationNoise,
    PoseObservations,
)
from semantic_digital_twin.articulation_estimation.refinement import (
    RefinementIterate,
    ScrewMotionRefinement,
    ScrewMotionStep,
    SoftL1Loss,
)
from .test_articulation_estimation import screw_motions


# %% Fixtures
@pytest.fixture()
def tracker_noise() -> ObservationNoise:
    """
    The noise of a good motion tracker: a millimeter and a quarter of a degree.
    """
    return ObservationNoise(
        position_standard_deviation=0.001, orientation_standard_deviation=0.005
    )


def perturbed_guess(
    *, observations: PoseObservations, noise: ObservationNoise
) -> ScrewMotionParameters:
    """
    :return: The initial guess for the observations, moved off the fit so that every
        derivative is taken away from a minimum.
    """
    guess = ScrewMotionParameters.initial_guesses(
        observations=observations, noise=noise, count=1
    )[0]
    return guess.copy_with_step(
        step=ScrewMotionStep(
            reference_pose_twist=np.array([0.02, -0.01, 0.03, 0.01, 0.02, -0.01]),
            position_changes=np.linspace(-0.02, 0.03, len(guess.positions)),
            screw_axis_change=np.array([0.01, 0.02, -0.01, 0.02, -0.01, 0.01]),
        )
    )


# %% The loss
def test_soft_l1_loss_derivatives_match_central_differences():
    """
    The slopes and curvatures are the derivatives of the costs.
    """
    loss = SoftL1Loss(scale=3.0)
    errors = np.array([-40.0, -3.0, -0.1, 0.0, 0.5, 3.0, 12.0])
    step = 1e-5
    assert_allclose(
        loss.slopes(errors=errors),
        (loss.costs(errors=errors + step) - loss.costs(errors=errors - step))
        / (2 * step),
        atol=1e-8,
    )
    assert_allclose(
        loss.curvatures(errors=errors),
        (loss.slopes(errors=errors + step) - loss.slopes(errors=errors - step))
        / (2 * step),
        atol=1e-8,
    )


# %% Linearized errors
def central_difference(
    *,
    parameters: ScrewMotionParameters,
    observations: PoseObservations,
    noise: ObservationNoise,
    change: ScrewMotionStep,
) -> np.ndarray:
    """
    :return: The change of the errors per unit of ``change``, by central differences.
    """
    errors = [
        parameters.copy_with_step(
            step=ScrewMotionStep(
                reference_pose_twist=sign * change.reference_pose_twist,
                position_changes=sign * change.position_changes,
                screw_axis_change=sign * change.screw_axis_change,
            )
        )
        .linearized_errors(observations=observations, noise=noise)
        .errors
        for sign in (1.0, -1.0)
    ]
    return (errors[0] - errors[1]) / 2


@pytest.mark.parametrize("motion_name", list(screw_motions()))
def test_linearized_errors_match_central_differences(
    motion_name: str, tracker_noise: ObservationNoise
):
    """
    The derivatives by the twist of the pose at position zero, by each position and by
    the screw axis agree with central differences of the errors.
    """
    observations = screw_motions()[motion_name].observe(noise=tracker_noise, seed=1)
    parameters = perturbed_guess(observations=observations, noise=tracker_noise)
    linearized = parameters.linearized_errors(
        observations=observations, noise=tracker_noise
    )
    step = 1e-6
    no_positions = np.zeros(len(parameters.positions))

    def derivative(*, change: ScrewMotionStep) -> np.ndarray:
        """
        :return: The central difference of the errors along ``change``, per unit.
        """
        return (
            central_difference(
                parameters=parameters,
                observations=observations,
                noise=tracker_noise,
                change=change,
            )
            / step
        )

    reference_pose = np.stack(
        [
            derivative(
                change=ScrewMotionStep(
                    reference_pose_twist=step * unit,
                    position_changes=no_positions,
                    screw_axis_change=np.zeros(6),
                )
            )
            for unit in np.eye(6)
        ],
        axis=-1,
    )
    screw_axis = np.stack(
        [
            derivative(
                change=ScrewMotionStep(
                    reference_pose_twist=np.zeros(6),
                    position_changes=no_positions,
                    screw_axis_change=step * unit,
                )
            )
            for unit in np.eye(6)
        ],
        axis=-1,
    )
    positions = derivative(
        change=ScrewMotionStep(
            reference_pose_twist=np.zeros(6),
            position_changes=np.full(len(parameters.positions), step),
            screw_axis_change=np.zeros(6),
        )
    )

    tolerance = 1e-6 * np.max(np.abs(linearized.shared_jacobians))
    assert_allclose(linearized.reference_pose_jacobians, reference_pose, atol=tolerance)
    assert_allclose(linearized.screw_axis_jacobians, screw_axis, atol=tolerance)
    assert_allclose(linearized.position_jacobians, positions, atol=tolerance)


# %% The gauge
@pytest.mark.parametrize("motion_name", list(screw_motions()))
def test_fixing_the_gauge_keeps_the_motion(
    motion_name: str, tracker_noise: ObservationNoise
):
    """
    The reparametrized motion predicts the same poses, with the first position zero, the
    last positive and the standardized screw axis of length one.
    """
    observations = screw_motions()[motion_name].observe(noise=tracker_noise, seed=1)
    parameters = perturbed_guess(observations=observations, noise=tracker_noise)
    shifted = replace(
        parameters,
        positions=-(parameters.positions + 0.3),
        screw_axis_entries=-parameters.screw_axis_entries,
        reference_parent_T_child=RigidTransform.from_exp_coords(
            -0.3 * parameters.screw_axis_entries
        ).as_matrix()
        @ parameters.reference_parent_T_child,
    )

    fixed = shifted.copy_with_fixed_gauge(noise=tracker_noise)

    assert_allclose(
        fixed.predicted_parent_T_child_stack,
        parameters.predicted_parent_T_child_stack,
        atol=1e-12,
    )
    assert fixed.positions[0] == 0
    assert fixed.positions[-1] > 0
    assert np.linalg.norm(
        fixed.screw_axis_entries
        / tracker_noise.exponential_coordinate_standard_deviations
    ) == pytest.approx(1)


# %% The refinement
@pytest.mark.parametrize("motion_name", list(screw_motions()))
def test_refinement_ends_where_no_step_lowers_the_loss(
    motion_name: str, tracker_noise: ObservationNoise
):
    """
    With the first and the last observation jumped, which alone fix where position zero
    lies and the unit of the positions, the refinement still ends at a minimum.
    """
    observations = screw_motions()[motion_name].observe(noise=tracker_noise, seed=5)
    jumped_stack = observations.parent_T_child_stack.copy()
    jumped_stack[[0, -1], :3, 3] += 0.05 * np.array([[0, 0.6, 0.8], [1, 0, 0]])
    jumped_stack[[0, -1], :3, :3] = (
        Rotation.from_rotvec([[0.2, 0, 0], [0, 0, -0.2]])
        * Rotation.from_matrix(jumped_stack[[0, -1], :3, :3])
    ).as_matrix()
    jumped = PoseObservations(parent_T_child_stack=jumped_stack)
    refinement = ScrewMotionRefinement(
        observations=jumped,
        noise=tracker_noise,
        loss=SoftL1Loss(scale=ScrewMotionParameters.outlier_threshold),
    )

    refined = refinement.refine_lowest(
        starts=ScrewMotionParameters.initial_guesses(
            observations=jumped,
            noise=tracker_noise,
            count=ScrewMotionParameters.refined_guess_count,
        )
    )

    loss = refinement.summed_loss(parameters=refined)
    next_iterate = refinement.next_iterate(
        iterate=RefinementIterate(
            parameters=refined, cost=loss, damping=refinement.smallest_damping
        )
    )
    assert next_iterate is None or loss - next_iterate.cost < 1e-6 * loss


def test_refinement_goes_on_with_the_lowest_start(tracker_noise: ObservationNoise):
    """
    Of a start far off the motion and the best initial guess, the refinement goes on
    with the one that is lower after screening, and ends no higher than either refined
    alone.
    """
    observations = screw_motions()["revolute"].observe(noise=tracker_noise, seed=4)
    best_guess = ScrewMotionParameters.initial_guesses(
        observations=observations, noise=tracker_noise, count=1
    )[0]
    far_off = replace(
        best_guess,
        screw_axis_entries=np.array([0.0, 0.3, -1.0, 0.5, 0.2, 0.1]),
    )
    refinement = ScrewMotionRefinement(
        observations=observations,
        noise=tracker_noise,
        loss=SoftL1Loss(scale=ScrewMotionParameters.outlier_threshold),
    )

    refined = refinement.refine_lowest(starts=[far_off, best_guess])

    alone = [
        refinement.summed_loss(
            parameters=refinement.refine(
                parameters=start,
                iteration_count=refinement.screening_iteration_count
                + refinement.maximum_iteration_count,
            )
        )
        for start in (far_off, best_guess)
    ]
    assert refinement.summed_loss(parameters=refined) <= min(alone) * (1 + 1e-6)
