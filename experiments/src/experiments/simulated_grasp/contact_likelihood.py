"""
What the physics says about whether the gripper is holding the block.

The belief layer filters a share of samples that found the body between the fingers. In
a simulation that share is measured rather than sampled: the physics reports exactly
which bodies are touching which, so the share is how many of the gripper's fingers are
against the block.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import List, Optional, Set

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.grasp_likelihood_source import GraspLikelihoodSource
from giskardpy.motion_statechart.graph_node import MotionStatechartNode, NodeArtifacts
from krrood.symbolic_math.symbolic_math import FloatVariable
from semantic_digital_twin.adapters.multi_sim import MujocoSim

# %% where contact is read from


@dataclass
class ContactSensor(ABC):
    """
    Something that reports which bodies are currently touching a given one.
    """

    @abstractmethod
    def bodies_touching(self, body_name: str) -> Set[str]:
        """
        :param body_name: The body to report contact against.
        :return: The names of the bodies touching it.
        """


@dataclass
class SimulatedContactSensor(ContactSensor):
    """
    Reads contact out of a running physics simulation.
    """

    simulation: MujocoSim
    """
    The physics whose contacts are reported.
    """

    def bodies_touching(self, body_name: str) -> Set[str]:
        return set(
            self.simulation.simulator.get_contact_bodies(body_name=body_name).result
        )


# %% a likelihood measured rather than sampled

FINGERS_OF_A_PARALLEL_GRIPPER = 2
"""
How many fingers a parallel gripper has, and so how many measurements one cycle of its
contact amounts to at face value.
"""


@dataclass(eq=False, repr=False)
class ContactLikelihood(MotionStatechartNode, GraspLikelihoodSource):
    """
    Publishes what share of the gripper's fingers the physics reports against the body
    it is meant to be holding.

    A grasp that has failed reports nothing touching, for as long as it goes on failing,
    which is the evidence a belief about the grasp is then left with.
    """

    contacts: ContactSensor = field(kw_only=True)
    """
    Where the contact between the fingers and the body is read from.
    """

    held_body: str = field(kw_only=True)
    """
    The body the gripper is meant to be holding.
    """

    fingers: List[str] = field(kw_only=True)
    """
    The gripper's fingers, whose contact with that body is what the share is taken over.
    """

    sample_size: int = field(default=FINGERS_OF_A_PARALLEL_GRIPPER, kw_only=True)
    """
    How many independent samples one cycle's reading is trusted as, defaulting to the
    fingers it is taken over.

    Contact is measured rather than sampled, so the reading itself carries no sample
    count, and what it is worth is really a statement about how far a moment of contact
    settles whether the grasp will go on holding. Stating more than the face value above
    is how a caller says it settles more than that.
    """

    _likelihood: Optional[FloatVariable] = field(default=None, init=False, repr=False)
    """
    The variable the share is published to, created while building.
    """

    @property
    def likelihood(self) -> FloatVariable:
        return self._likelihood

    @property
    def share_of_fingers_holding(self) -> float:
        """
        :return: What share of the fingers are against the body right now.
        """
        touching = self.contacts.bodies_touching(self.held_body)
        return len([finger for finger in self.fingers if finger in touching]) / len(
            self.fingers
        )

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        self._likelihood = FloatVariable(f"{self.unique_name}_contact_likelihood")
        context.float_variable_data.register_expression(self._likelihood)
        return NodeArtifacts()

    def on_start(self, context: MotionStatechartContext) -> None:
        self._publish(context)

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        self._publish(context)
        return None

    def _publish(self, context: MotionStatechartContext) -> None:
        """
        Writes the measured share into the variable carrying it.

        :param context: The context holding the float variable data to write to.
        """
        context.float_variable_data.set_value(
            self._likelihood, self.share_of_fingers_holding
        )
