"""
Load a Warsaw scene and report what it holds.

A scene directory holds one mesh of the whole room, segmented into objects by a set of
classes. This loads it into a world, one body per object, and prints what came out.

Run it against a scene directory::

    python -m experiments.warsaw.demo dataset/kitchenlab_meshes_out_20260816

``--view`` opens an interactive viewer on the scene, and ``--render <directory>`` writes
images of it from the loader's four camera poses without opening a window::

    python -m experiments.warsaw.demo dataset/kitchenlab_meshes_out_20260816 --view

..note:: Building the world writes one mesh file per object, so a scene of a few hundred
    objects takes some tens of seconds.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from experiments.warsaw.world_loader import WarsawWorldLoader
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


def height_of(body: Body, world: World) -> float:
    """
    :param body: The body to measure.
    :param world: The world the body belongs to.
    :return: The body's mean height above the world's origin.
    """
    return float(body.collision[0].mesh_in_frame(world.root).vertices[:, 2].mean())


def report(loader: WarsawWorldLoader) -> None:
    """
    Print what a loaded scene holds, tallest body last.

    :param loader: The loader holding the scene's world.
    """
    bodies = loader.world.bodies_with_collision
    print(f"bodies: {len(bodies)}")

    per_class = Counter(str(body.name).rsplit("_", 1)[0] for body in bodies)
    print(f"classes: {len(per_class)}")
    for class_name, count in sorted(per_class.items()):
        print(f"  {class_name:<20} {count}")

    # Measuring a body reads its mesh, so each is measured once and then sorted.
    measured = [(height_of(body, loader.world), str(body.name)) for body in bodies]
    print("\nbodies by height:")
    for height, name in sorted(measured):
        print(f"  {name:<24} {height:6.2f} m")


def view(world: World) -> None:
    """
    Open the world in the trimesh viewer, returning when its window is closed.

    :param world: The world to look at.
    """
    scene = RayTracer(world=world).scene
    # Smoothing recomputes vertex normals for every mesh before the first frame is drawn,
    # which on a scanned room costs minutes of an apparently black window and buys
    # nothing: the meshes already carry the colors they are drawn with.
    #
    # The scene's own camera is sized for rendering, which fills the screen in a window.
    scene.show(smooth=False, resolution=(1280, 960))


def main() -> None:
    """
    Load the scene named on the command line and report it.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "scene_directory",
        type=Path,
        help="Directory holding the scene mesh and its per-class segmentations.",
    )
    parser.add_argument(
        "--render",
        type=Path,
        default=None,
        help="Directory to write renders of the scene into.",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Open the scene in the trimesh viewer.",
    )
    arguments = parser.parse_args()

    print(f"Loading {arguments.scene_directory} ...")
    loader = WarsawWorldLoader(arguments.scene_directory)
    report(loader)

    if arguments.render is not None:
        arguments.render.mkdir(parents=True, exist_ok=True)
        loader.render_scene_from_predefined_poses(
            arguments.render, "warsaw_scene", headless=True
        )
        print(f"\nrenders written to {arguments.render}")

    if arguments.view:
        print("\nopening the viewer, close its window to finish ...")
        view(loader.world)


if __name__ == "__main__":
    main()
