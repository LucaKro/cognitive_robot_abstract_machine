"""
Load a Warsaw scene and report what it holds.

A scene directory holds one mesh of the whole room, whose faces carry, per class, the
instance they belong to. This loads it into a world and prints what came out.

Run it against a scene directory::

    python -m experiments.warsaw.demo dataset/kitchenlab_new_mesh_agreement_dataset

``--view`` opens an interactive viewer on the scene, and ``--render <directory>`` writes
images of it from the loader's four camera poses without opening a window::

    python -m experiments.warsaw.demo dataset/kitchenlab_new_mesh_agreement_dataset --view
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.warsaw.world_loader import WarsawWorldLoader
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.world import World


def report(loader: WarsawWorldLoader) -> None:
    """
    Print what a loaded scene holds, class by class.

    :param loader: The loader holding the scene's world.
    """
    scene = loader.scene
    segments = loader.label_segments
    face_count = len(scene.mesh.faces)

    print(f"mesh: {scene.mesh_path.name}")
    print(f"vertices: {len(scene.mesh.vertices)}  faces: {face_count}")
    print(f"classes: {len(scene.class_names)}  segments: {len(segments)}")

    segments_per_class = Counter(segment.class_name for segment in segments)
    faces_per_class = Counter()
    for segment in segments:
        faces_per_class[segment.class_name] += len(segment)

    print(f"\n{'class':<20} {'segments':>8} {'faces':>10}")
    for class_name in scene.class_names:
        print(
            f"  {class_name:<18} {segments_per_class[class_name]:>8} "
            f"{faces_per_class[class_name]:>10}"
        )

    # A face can carry several classes at once, since a scene labels a drawer front both
    # as the drawer and as the cabinet holding it.
    labels_per_face = np.zeros(face_count, dtype=np.int32)
    for instances in scene.face_labels.values():
        labels_per_face += (instances != scene.UNSEGMENTED).astype(np.int32)
    labelled = int((labels_per_face > 0).sum())
    print(
        f"\nlabelled faces: {labelled} / {face_count} "
        f"({100 * labelled / face_count:.1f}%), "
        f"of which {int((labels_per_face > 1).sum())} carry more than one class"
    )


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
        help="Directory holding the scene's labelled mesh.",
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
