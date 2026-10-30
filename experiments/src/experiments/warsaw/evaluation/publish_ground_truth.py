"""
Publish a modelled world for RViz, so it can be looked at beside a run's.

A run leaves its own ``publish_world.py`` behind, which reads the world it built out of
the database. This publishes the other side of the comparison: a world modelled by hand,
built afresh from what defines it rather than read from anywhere.

Nothing here touches the database or the ORM, so it starts in seconds and can be stopped
and started freely. Give it a different topic from a run's publisher and RViz can show
both at once.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import rclpy
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color
from typing_extensions import Dict, List, Optional

from experiments.warsaw.bases import HasLogger
from experiments.warsaw.evaluation.ground_truth import (
    world_from_provider,
    world_from_urdf,
)
from experiments.warsaw.painting import Coloring, paint

# %% publishing one modelled world


@dataclass
class GroundTruthPublication(HasLogger):
    """
    One modelled world, painted and published until interrupted.
    """

    world: World
    """
    The world to publish.
    """

    coloring: Coloring = Coloring.BY_CLASS
    """
    What its bodies are painted by.
    """

    topic_name: str = "/semworld/ground_truth/viz_marker"
    """
    Where the markers are published.

    Not where a run's publisher puts its own, so RViz can be shown both at once and the
    modelled world compared with the built one rather than replaced by it.
    """

    node_name: str = "ground_truth_world"
    """
    What this calls itself on the ROS graph.
    """

    republish_every_seconds: Optional[float] = 5.0
    """
    How often to publish the markers again, or None to publish only when the world
    changes.

    A world built once never changes again, so it is published once and nothing follows
    it. Publishing on a timer means a viewer started afterwards is shown it within a few
    seconds rather than staying empty.
    """

    alpha: float = 1.0
    """
    How solid the markers are.

    Below one every body shows through the ones in front of it.
    """

    def say_the_legend(self, painted: Dict[str, Color]) -> None:
        """
        Say what each colour means, where the colour is the only thing that says it.

        Said for classes and not for bodies: RViz lists every marker under the name of
        the body it belongs to, so painting by body needs no key, and a world's worth of
        them would bury everything else this prints.

        :param painted: What each name was painted.
        """
        if not painted or self.coloring is not Coloring.BY_CLASS:
            return
        self.logger.info("painted %s:", self.coloring.value.replace("_", " "))
        for name, color in sorted(painted.items()):
            self.logger.info(
                "  %-16s #%02x%02x%02x",
                name,
                round(color.R * 255),
                round(color.G * 255),
                round(color.B * 255),
            )

    def say_what_rviz_needs(self) -> None:
        """
        Say what to set up in RViz, which shows nothing at all until it is set.
        """
        for line in (
            "in RViz:",
            "  add a MarkerArray display",
            f"  set its topic to {self.topic_name}",
            "  set its durability policy to Transient Local",
            "  set the fixed frame to the root of the tf tree",
        ):
            self.logger.info(line)

    def carry_out(self) -> None:
        """
        Publish the world until interrupted.
        """
        self.logger.info(
            "publishing %s bodies and %s annotations",
            len(self.world.bodies),
            len(self.world.semantic_annotations),
        )
        self.say_the_legend(paint(self.world, self.coloring))
        self.say_what_rviz_needs()

        rclpy.init()
        node = rclpy.create_node(self.node_name)
        publisher = VizMarkerPublisher(
            _world=self.world,
            node=node,
            topic_name=self.topic_name,
            alpha=self.alpha,
        )
        if self.republish_every_seconds is not None:
            node.create_timer(self.republish_every_seconds, publisher.on_model_change)
            self.logger.info(
                "publishing every %s seconds; stop with ctrl-c",
                self.republish_every_seconds,
            )
        else:
            self.logger.info("publishing once; stop with ctrl-c")
        try:
            rclpy.spin(node)
        finally:
            # Reached however this ends, including the interrupt that is how it is meant
            # to end: a node left registered keeps the topic and the next start cannot
            # take it.
            publisher.stop()
            node.destroy_node()
            rclpy.shutdown()


# %% command-line entry point


def argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for publishing a modelled world.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--world-provider",
        help="What builds the world, as module.path:ClassName",
    )
    source.add_argument("--urdf", type=Path, help="A URDF file describing the world")
    parser.add_argument(
        "--coloring",
        type=Coloring,
        choices=list(Coloring),
        default=Coloring.BY_CLASS,
        help="What the bodies are painted by",
    )
    parser.add_argument(
        "--topic",
        default=GroundTruthPublication.topic_name,
        help="Where to publish the markers",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="How solid the markers are"
    )
    parser.add_argument(
        "--no-infer-semantics",
        dest="infer_semantics",
        action="store_false",
        help="Read a URDF's own classes without inferring any",
    )
    return parser


def world_asked_for(parsed: argparse.Namespace) -> World:
    """
    Build the world the command names.

    :param parsed: What the command was given.
    :return: The world, with its classes named.
    """
    if parsed.urdf is not None:
        return world_from_urdf(parsed.urdf, infer_semantics=parsed.infer_semantics)
    return world_from_provider(parsed.world_provider)


def main(arguments: Optional[List[str]] = None) -> int:
    """
    Publish a modelled world for RViz.

    :param arguments: Command-line arguments without the program name.
    :return: Zero once publishing has finished.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parsed = argument_parser().parse_args(arguments)
    GroundTruthPublication(
        world=world_asked_for(parsed),
        coloring=parsed.coloring,
        topic_name=parsed.topic,
        alpha=parsed.alpha,
    ).carry_out()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
