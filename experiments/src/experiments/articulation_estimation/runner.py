"""
Runs the screw motion fit on the joint motions of the RBO dataset and writes the results
as JSON.

A rerun reuses the files already downloaded.
"""

from __future__ import annotations

# %% imports
import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from experiments.articulation_estimation.dataset import (
    Interaction,
    JointKind,
    RBODataset,
)
from experiments.articulation_estimation.evaluation import (
    FullFit,
    JointEvaluation,
    JointMotionResult,
)
from krrood.adapters.json_serializer import to_json
from krrood.patterns.called_positionally import called_positionally
from semantic_digital_twin.articulation_estimation.observations import ObservationNoise


# %% the experiment


@dataclass(kw_only=True)
class RBOExperiment:
    """
    Fits every joint motion of the chosen objects, then evaluates each against the
    others of the same object, recording date and joint.
    """

    dataset: RBODataset
    """
    The dataset.
    """

    object_names: list[str] | None = None
    """
    The objects to evaluate, or all of them.
    """

    process_count: int = 4
    """
    How many processes fit at once.
    """

    smallest_excursion: dict[JointKind, float] = field(
        default_factory=lambda: {
            JointKind.REVOLUTE: float(np.radians(15.0)),
            JointKind.PRISMATIC: 0.05,
        }
    )
    """
    The excursion, in radians or meters, below which a joint motion is left out: a part
    that barely moved shows little of its axis.
    """

    smallest_group_size: int = 3
    """
    The fewest motions of one joint for which they are evaluated, so that each reference
    combines at least two other fits.
    """

    initial_noise: ObservationNoise = field(
        default_factory=lambda: ObservationNoise(
            position_standard_deviation=1e-3, orientation_standard_deviation=5e-3
        )
    )
    """
    The noise assumed for the first fit of each motion, before it is estimated.
    """

    def run(self) -> list[JointMotionResult]:
        """
        :return: The evaluation of every joint motion that moved far enough, of joints
            with at least :attr:`smallest_group_size` such motions.
        """
        interactions = [
            interaction
            for interaction in self.dataset.interactions
            if self.object_names is None or interaction.object_name in self.object_names
        ]
        with Pool(self.process_count) as pool:
            full_fits = [
                fit
                for fits in pool.imap_unordered(self.full_fits, interactions)
                for fit in fits
            ]
            groups = defaultdict(list)
            for fit in full_fits:
                if fit.excursion[-1] >= self.smallest_excursion[fit.motion.joint.kind]:
                    interaction = fit.motion.interaction
                    groups[
                        (
                            interaction.object_name,
                            interaction.recording_date,
                            fit.motion.joint_index,
                        )
                    ].append(fit)
            evaluations = [
                JointEvaluation(
                    full_fit=fit,
                    other_fits=[other for other in group if other is not fit],
                )
                for group in groups.values()
                if len(group) >= self.smallest_group_size
                for fit in group
            ]
            return list(pool.imap_unordered(JointEvaluation.evaluate, evaluations))

    @called_positionally
    def full_fits(self, interaction: Interaction) -> list[FullFit]:
        """
        :return: The full fit of each joint motion of the interaction.
        """
        return [
            FullFit.from_motion(motion=motion, initial_noise=self.initial_noise)
            for motion in self.dataset.joint_motions(interaction=interaction)
        ]


# %% command line


def main() -> None:
    """
    Runs the experiment as the command line arguments say and writes the results.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objects", nargs="*")
    parser.add_argument("--processes", type=int, default=4)
    arguments = parser.parse_args()
    results = RBOExperiment(
        dataset=RBODataset(directory=arguments.directory),
        object_names=arguments.objects,
        process_count=arguments.processes,
    ).run()
    arguments.output.write_text(json.dumps(to_json(results)))


if __name__ == "__main__":
    main()
