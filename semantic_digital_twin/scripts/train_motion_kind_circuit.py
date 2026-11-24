"""
Trains the circuit that
:class:`~semantic_digital_twin.articulation_estimation.motion_kind_classification.MotionKindCircuit`
loads: simulated motions of known kind, summarized with
:class:`~semantic_digital_twin.articulation_estimation.motion_kind_classification.MotionSummary`,
a region graph circuit trained on them by gradient descent on the cross entropy of the
kind, the best circuit on validation motions, and a calibration fitted on the same
motions.

Needs JAX and optax, which semantic_digital_twin does not depend on; install them with
probabilistic_model's ``jax`` extra: ``pip install "probabilistic_model[jax]"``.

Usage: ``python train_motion_kind_circuit.py OUTPUT``, where ``OUTPUT`` is the JSON file
to write. Write it outside the checkout: trained circuits are not part of the
repository.
"""

# %% Imports
import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax
from random_events.variable import Continuous
from scipy.optimize import minimize
from scipy.spatial.transform import RigidTransform, Rotation
from scipy.special import logsumexp
from sortedcontainers import SortedSet

from probabilistic_model.learning.gradient_descent import CrossEntropy
from probabilistic_model.learning.jax_gradient_descent import JaxGradientDescent
from probabilistic_model.learning.region_graph.region_graph import RegionGraph
from probabilistic_model.learning.training_schedule import MiniBatches
from semantic_digital_twin.articulation_estimation.motion_kind_classification import (
    MotionKindCircuit,
    MotionSummary,
    ProbabilityCalibration,
)
from semantic_digital_twin.articulation_estimation.motion_kinds import (
    MotionKind,
    MotionKindProbabilities,
)
from semantic_digital_twin.articulation_estimation.observations import (
    ObservationNoise,
    PoseObservations,
)


# %% Simulated motions
@dataclass(kw_only=True)
class SensorConditions:
    """
    How a simulated motion is observed and how large it is.
    """

    small_motion: bool = False
    """
    Whether the motion turns or slides only a little, see
    :attr:`MotionSimulator.small_turn_degrees` and :attr:`MotionSimulator.small_slide`.
    """

    noise_factor: float = 1.0
    """
    The factor the simulated noise is multiplied by.
    """

    stated_noise_divisor: float = 1.0
    """
    The number the noise stated to the classifier is divided by; above one, the noise
    is stated too small.
    """

    correlation: float = 0.0
    """
    The correlation of the noise of one observation with the one before.
    """


@dataclass(kw_only=True)
class MotionSimulator:
    """
    Simulates observed revolute, helical and sliding motions of a child whose pose and
    axis are random, with noise that may be correlated over time and stated wrongly.
    """

    rng: np.random.Generator
    """
    The random numbers the simulation draws.
    """

    observation_counts: tuple[int, int] = (8, 30)
    """
    The smallest and the largest number of observed poses.
    """

    position_noise: tuple[float, float] = (0.0005, 0.002)
    """
    The range, in meters, of the standard deviation of the position noise.
    """

    orientation_noise: tuple[float, float] = (0.0025, 0.01)
    """
    The range, in radians, of the standard deviation of the orientation noise.
    """

    turn_degrees: tuple[float, float] = (0.25, 270.0)
    """
    The range of the angle a turning motion turns by, in degrees.
    """

    small_turn_degrees: tuple[float, float] = (0.25, 2.0)
    """
    The range of the angle a small turning motion turns by, in degrees.
    """

    slide: tuple[float, float] = (0.002, 0.6)
    """
    The range of the distance a sliding motion slides, in meters.
    """

    small_slide: tuple[float, float] = (0.002, 0.02)
    """
    The range of the distance a small sliding motion slides, in meters.
    """

    axis_distance: tuple[float, float] = (0.05, 1.0)
    """
    The range of the distance of a turning axis from the child's origin, in meters.
    """

    pitch_per_turn: tuple[float, float] = (0.001, 0.02)
    """
    The range of how far a helical motion advances per revolution, in meters.
    """

    noise_factors: tuple[float, float] = (0.5, 4.0)
    """
    The range of :attr:`SensorConditions.noise_factor` of randomized conditions.
    """

    correlated_share: float = 0.5
    """
    The share of randomized conditions whose noise is correlated over time.
    """

    largest_correlation: float = 0.9
    """
    The largest :attr:`SensorConditions.correlation` of randomized conditions.
    """

    understated_share: float = 0.3
    """
    The share of randomized conditions whose noise is stated too small.
    """

    stated_noise_divisors: tuple[float, float] = (1.0, 3.0)
    """
    The range of :attr:`SensorConditions.stated_noise_divisor` of randomized conditions
    whose noise is stated too small.
    """

    def log_uniform(self, *, bounds: tuple[float, float]) -> float:
        """
        :return: A number drawn uniformly in its logarithm between ``bounds``.
        """
        low, high = bounds
        return float(np.exp(self.rng.uniform(np.log(low), np.log(high))))

    def unit_vector(self) -> npt.NDArray[np.float64]:
        """
        :return: A direction drawn uniformly.
        """
        vector = self.rng.normal(size=3)
        return vector / np.linalg.norm(vector)

    def randomized_conditions(self) -> SensorConditions:
        """
        :return: Sensor conditions for one training motion: wider noise, correlated in
            :attr:`correlated_share` of the motions and stated too small in
            :attr:`understated_share` of them.
        """
        noise_factor = self.log_uniform(bounds=self.noise_factors)
        correlation = (
            float(self.rng.uniform(0.0, self.largest_correlation))
            if self.rng.random() < self.correlated_share
            else 0.0
        )
        stated_noise_divisor = (
            self.log_uniform(bounds=self.stated_noise_divisors)
            if self.rng.random() < self.understated_share
            else 1.0
        )
        return SensorConditions(
            noise_factor=noise_factor,
            stated_noise_divisor=stated_noise_divisor,
            correlation=correlation,
        )

    def simulate(
        self, *, kind: MotionKind, conditions: SensorConditions
    ) -> tuple[PoseObservations, ObservationNoise]:
        """
        :param kind: The kind of the simulated motion.
        :param conditions: How the motion is observed and how large it is.
        :return: The observed poses and the noise stated for them.
        """
        rng = self.rng
        low_count, high_count = self.observation_counts
        count = int(rng.integers(low_count, high_count + 1))
        position_noise = (
            self.log_uniform(bounds=self.position_noise) * conditions.noise_factor
        )
        orientation_noise = (
            self.log_uniform(bounds=self.orientation_noise) * conditions.noise_factor
        )
        pose = RigidTransform.from_components(
            rng.uniform(-1, 1, 3),
            Rotation.random(random_state=int(rng.integers(2**31))),
        ).as_matrix()
        direction = self.unit_vector()
        if kind == MotionKind.SLIDING:
            end = self.log_uniform(
                bounds=self.small_slide if conditions.small_motion else self.slide
            )
            entries = np.concatenate([np.zeros(3), direction])
        else:
            end = np.radians(
                self.log_uniform(
                    bounds=(
                        self.small_turn_degrees
                        if conditions.small_motion
                        else self.turn_degrees
                    )
                )
            )
            radial = np.cross(direction, self.unit_vector())
            radial /= np.linalg.norm(radial)
            point = pose[:3, 3] + radial * self.log_uniform(bounds=self.axis_distance)
            pitch = (
                self.log_uniform(bounds=self.pitch_per_turn) / (2 * np.pi)
                if kind == MotionKind.HELICAL
                else 0.0
            )
            entries = np.concatenate(
                [direction, -np.cross(direction, point) + pitch * direction]
            )
        poses = (
            RigidTransform.from_exp_coords(
                np.outer(np.linspace(0.0, end, count), entries)
            ).as_matrix()
            @ pose
        )
        observed = poses.copy()
        observed[:, :3, 3] += position_noise * self.correlate(
            white=rng.normal(size=(count, 3)),
            correlation=conditions.correlation,
        )
        observed[:, :3, :3] = (
            Rotation.from_matrix(poses[:, :3, :3])
            * Rotation.from_rotvec(
                orientation_noise
                * self.correlate(
                    white=rng.normal(size=(count, 3)),
                    correlation=conditions.correlation,
                )
            )
        ).as_matrix()
        return PoseObservations(parent_T_child_stack=observed), ObservationNoise(
            position_standard_deviation=position_noise
            / conditions.stated_noise_divisor,
            orientation_standard_deviation=orientation_noise
            / conditions.stated_noise_divisor,
        )

    @staticmethod
    def correlate(
        *, white: npt.NDArray[np.float64], correlation: float
    ) -> npt.NDArray[np.float64]:
        """
        :param white: Independent standard normal errors, one row per observation.
        :param correlation: The correlation of each row with the one before.
        :return: The errors correlated over the rows, still of unit variance.
        """
        correlated = white.copy()
        for index in range(1, len(white)):
            correlated[index] = (
                correlation * correlated[index - 1]
                + np.sqrt(1 - correlation**2) * white[index]
            )
        return correlated


def summary_values(
    *, observations: PoseObservations, noise: ObservationNoise
) -> npt.NDArray[np.float64]:
    """
    :return: The numbers of the summary of ``observations``.
    """
    return MotionSummary.from_observations(
        observations=observations, noise=noise
    ).values


# %% Training
@dataclass(kw_only=True)
class MotionKindCircuitTraining:
    """
    The recipe of the circuit: how much data, the circuit's size and the training
    schedule.
    """

    data_seed: int = 2026
    """
    The seed of the simulated motions.
    """

    seed: int = 8
    """
    The seed of the circuit's structure and of the mini-batches.
    """

    training_motions_per_kind: int = 5000
    """
    The number of training motions of each kind, each under randomized conditions.
    """

    validation_motions_per_kind_and_set: int = 400
    """
    The number of validation motions of each kind in each of
    :attr:`validation_conditions`.
    """

    test_motions_per_kind: int = 80
    """
    The number of fresh motions of each kind the written circuit is tested on.
    """

    validation_conditions: list[SensorConditions] = field(
        default_factory=lambda: [
            SensorConditions(),
            SensorConditions(small_motion=True),
            SensorConditions(noise_factor=3.0),
            SensorConditions(correlation=0.8),
            SensorConditions(noise_factor=2.0, stated_noise_divisor=2.0),
        ]
    )
    """
    The conditions of the validation motions, which choose the best circuit and fit
    the calibration.
    """

    count_jitter: float = 0.02
    """
    The half width of the uniform noise added to the logarithm of the observation
    count of training motions, which keeps the Gaussian leaves of an integer from
    collapsing.
    """

    depth: int = 2
    """
    The depth of the region graph.
    """

    repetitions: int = 3
    """
    The number of repetitions of the region graph.
    """

    units: int = 4
    """
    The number of input and sum units per region.
    """

    steps: int = 4000
    """
    The number of gradient steps.
    """

    batch_size: int = 512
    """
    The number of motions per gradient step.
    """

    learning_rate: float = 0.01
    """
    The learning rate of Adam.
    """

    validate_every: int = 250
    """
    The number of steps between two validations.
    """

    def train(self, *, output: Path) -> None:
        """
        Trains the circuit, writes it to ``output`` and prints how it does on fresh
        motions.
        """
        start = time.perf_counter()
        simulator = MotionSimulator(rng=np.random.default_rng(self.data_seed))
        training_summaries, training_kinds = [], []
        for kind_index, kind in enumerate(MotionKind):
            for _ in range(self.training_motions_per_kind):
                conditions = simulator.randomized_conditions()
                observations, noise = simulator.simulate(
                    kind=kind, conditions=conditions
                )
                training_summaries.append(
                    summary_values(observations=observations, noise=noise)
                )
                training_kinds.append(kind_index)
        training_summaries = np.array(training_summaries)
        training_kinds = np.array(training_kinds)
        count_column = MotionSummary.field_names.index("log_observation_count")
        training_summaries[:, count_column] += simulator.rng.uniform(
            -self.count_jitter, self.count_jitter, len(training_kinds)
        )
        mean = training_summaries.mean(axis=0)
        deviation = training_summaries.std(axis=0)
        validation_summaries, validation_kinds = [], []
        for conditions in self.validation_conditions:
            for kind_index, kind in enumerate(MotionKind):
                for _ in range(self.validation_motions_per_kind_and_set):
                    observations, noise = simulator.simulate(
                        kind=kind, conditions=conditions
                    )
                    validation_summaries.append(
                        summary_values(observations=observations, noise=noise)
                    )
                    validation_kinds.append(kind_index)
        validation_summaries = np.array(validation_summaries)
        validation_kinds = np.array(validation_kinds)
        print(
            f"data: {len(training_kinds)} training and {len(validation_kinds)} "
            f"validation motions, {time.perf_counter() - start:.0f} s",
            flush=True,
        )

        kind_variable = MotionKindProbabilities.kind_variable
        variables = SortedSet([Continuous(name) for name in MotionSummary.field_names])
        circuit = (
            RegionGraph(
                variables=variables,
                partitions=2,
                depth=self.depth,
                repetitions=self.repetitions,
                seed=self.seed,
            )
            .create_random_region_graph()
            .to_probabilistic_circuit(
                input_units=self.units,
                sum_units=self.units,
                predicted_variable=kind_variable,
            )
            .to_namespace(jnp, floating_point_type=jnp.float64)
        )
        names = [variable.name for variable in circuit.variables]
        kind_codes = np.array(
            [
                circuit.encoded_point({kind_variable: kind})[kind_variable]
                for kind in MotionKind
            ],
            dtype=float,
        )

        def rows(
            summaries: npt.NDArray[np.float64], kinds: npt.NDArray[np.int64]
        ) -> npt.NDArray[np.float64]:
            """
            :return: The circuit's rows of the standardized ``summaries`` and ``kinds``.
            """
            standardized = dict(
                zip(MotionSummary.field_names, ((summaries - mean) / deviation).T)
            )
            return np.column_stack(
                [
                    (
                        kind_codes[kinds]
                        if name == kind_variable.name
                        else standardized[name]
                    )
                    for name in names
                ]
            )

        def log_posterior(
            root, summaries: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
            """
            :return: The log probability of each kind given each of ``summaries``.
            """
            joint = np.stack(
                [
                    np.asarray(
                        root.log_likelihood_of_nodes(
                            events=jnp.asarray(
                                rows(summaries, np.full(len(summaries), kind_index))
                            )
                        )
                    )[:, 0]
                    for kind_index in range(len(MotionKind))
                ],
                axis=1,
            )
            return joint - logsumexp(joint, axis=1, keepdims=True)

        training_rows = jnp.asarray(rows(training_summaries, training_kinds))
        loss = CrossEntropy.from_predicted_variable(
            predicted_variable=kind_variable, variables=circuit.variables
        )
        training = JaxGradientDescent(
            circuit.root, loss, optax.adam(self.learning_rate)
        ).start_training(arguments=(training_rows,), mini_batches=MiniBatches())
        batches = np.random.default_rng(self.seed)
        best, best_root = np.inf, None
        for step in range(1, self.steps + 1):
            training.step_on_batch(
                batch=(
                    training_rows[
                        batches.integers(0, len(training_kinds), self.batch_size)
                    ],
                )
            )
            if step % self.validate_every == 0:
                root = training.trained_root()
                value = -log_posterior(root, validation_summaries)[
                    np.arange(len(validation_kinds)), validation_kinds
                ].mean()
                if value < best:
                    best, best_root = value, root
                print(f"step {step}: validation cross entropy {value:.4f}", flush=True)
        validation_log_posterior = log_posterior(best_root, validation_summaries)

        def calibrated_loss(parameters: npt.NDArray[np.float64]) -> float:
            """
            :param parameters: The logarithm of the temperature, then the revolute and
                the helical offset.
            :return: The cross entropy of the calibrated validation probabilities.
            """
            scaled = validation_log_posterior / np.exp(parameters[0]) + np.array(
                [parameters[1], parameters[2], 0.0]
            )
            return -(scaled - logsumexp(scaled, axis=1, keepdims=True))[
                np.arange(len(validation_kinds)), validation_kinds
            ].mean()

        parameters = minimize(
            calibrated_loss,
            np.zeros(3),
            method="Nelder-Mead",
            options=dict(maxiter=600, xatol=1e-3, fatol=1e-5),
        ).x
        circuit.root = best_root
        share = 1 / len(MotionKind)
        classifier = MotionKindCircuit(
            circuit=circuit.to_namespace(np, floating_point_type=np.float64),
            summary_mean=mean,
            summary_standard_deviation=deviation,
            calibration=ProbabilityCalibration(
                temperature=float(np.exp(parameters[0])),
                revolute_offset=float(parameters[1]),
                helical_offset=float(parameters[2]),
            ),
            training_kind_share=MotionKindProbabilities(
                revolute=share, helical=share, sliding=share
            ),
        )
        classifier.to_file(path=output)
        print(
            f"validation cross entropy {best:.4f}, calibrated "
            f"{calibrated_loss(parameters):.4f}; wrote {output} "
            f"({output.stat().st_size / 1000:.0f} kB), "
            f"{time.perf_counter() - start:.0f} s",
            flush=True,
        )
        self.test(
            classifier=MotionKindCircuit.from_file(path=output), simulator=simulator
        )

    def test(
        self, *, classifier: MotionKindCircuit, simulator: MotionSimulator
    ) -> None:
        """
        Prints how often ``classifier`` tells turning from sliding on fresh motions, and
        how often it is more than 90% sure and wrong.
        """
        for name, conditions in [
            ("as trained", SensorConditions()),
            ("small motions", SensorConditions(small_motion=True)),
        ]:
            turning_probabilities, turns = [], []
            for kind in MotionKind:
                for _ in range(self.test_motions_per_kind):
                    observations, noise = simulator.simulate(
                        kind=kind, conditions=conditions
                    )
                    probabilities = classifier.kind_probabilities(
                        observations=observations, noise=noise
                    )
                    turning_probabilities.append(
                        probabilities.revolute + probabilities.helical
                    )
                    turns.append(kind.turns_child)
            turning_probabilities = np.array(turning_probabilities)
            turns = np.array(turns)
            correct = (turning_probabilities > 0.5) == turns
            confidently_wrong = ((turning_probabilities > 0.9) & ~turns) | (
                (turning_probabilities < 0.1) & turns
            )
            print(
                f"fresh test, {name}: turning vs sliding correct {correct.mean():.3f}, "
                f"more than 90% sure and wrong {confidently_wrong.mean():.3f}",
                flush=True,
            )


# %% Command line
def main() -> None:
    """
    Trains the circuit with the default recipe and writes it where the command line
    says.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "output", type=Path, help="The JSON file to write, outside the checkout."
    )
    jax.config.update("jax_enable_x64", True)
    MotionKindCircuitTraining().train(output=parser.parse_args().output)


if __name__ == "__main__":
    main()
