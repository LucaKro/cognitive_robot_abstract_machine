import os
from pathlib import Path
from typing import Dict, List, Tuple
from uuid import UUID

import numpy as np
import trimesh
import distinctipy
from matplotlib.colors import CSS4_COLORS


from semantic_digital_twin.adapters.mesh import OBJParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.pipeline.pipeline import (
    Pipeline,
    TransformGeometry,
    CenterLocalGeometryAndPreserveWorldPose,
)
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.spatial_types import TransformationMatrix
from semantic_digital_twin.utils import InheritanceStructureExporter
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)


def load_world(obj_dir: Path) -> World:
    """Load OBJ files from directory and return a World with applied pipeline."""
    files = [f for f in os.listdir(obj_dir) if f.endswith(".obj")]
    world = World()
    root = Body(name=PrefixedName("root_body"))
    with world.modify_world():
        world.add_body(root)
    for file in files:
        obj_world = OBJParser(os.path.join(obj_dir, file)).parse()
        with world.modify_world():
            world.merge_world(obj_world)

    pipeline = Pipeline(
        steps=[
            TransformGeometry(
                TransformationMatrix.from_xyz_rpy(roll=np.pi / 2, pitch=0, yaw=0)
            ),
            CenterLocalGeometryAndPreserveWorldPose(),
        ]
    )
    return pipeline.apply(world)


def export_world_metadata(world: World, export_dir: Path) -> None:
    """Export kinematic structure and semantic annotations JSON."""
    export_dir.mkdir(parents=True, exist_ok=True)
    world.export_kinematic_structure_tree_to_json(
        export_dir / "kinematic_structure.json",
        include_connections=False,
    )
    InheritanceStructureExporter(
        SemanticAnnotation, export_dir / "semantic_annotations.json"
    ).export()


def distinct_html_colors(n, seed=None) -> List[Color]:
    """
    Generate n maximally distinct CSS4 color names
    by projecting distinctipy colors to nearest HTML color names.
    Ensures the final names are distinct.
    """
    # generate maximally distinct base colors
    if seed is not None:
        colors = distinctipy.get_colors(n, rng=seed)
    else:
        colors = distinctipy.get_colors(n)

    return [Color(*c) for c in colors]


def get_camera_poses():
    """Return list of camera poses for rendering."""
    rotate = trimesh.transformations.rotation_matrix(
        angle=np.radians(-90.0), direction=[0, 1, 0]
    )
    rotate_x = trimesh.transformations.rotation_matrix(
        angle=np.radians(180.0), direction=[1, 0, 0]
    )

    camera_poses = []
    camera_pose1 = TransformationMatrix.from_xyz_rpy(
        x=-3, y=0, z=2.5, roll=-np.pi / 2, pitch=np.pi / 4, yaw=0
    ).to_np()
    camera_poses.append(camera_pose1 @ rotate_x @ rotate)

    camera_pose2 = TransformationMatrix.from_xyz_rpy(
        x=3, y=0, z=2.5, roll=-np.pi / 2, pitch=np.pi / 4, yaw=np.pi
    ).to_np()
    camera_poses.append(camera_pose2 @ rotate_x @ rotate)

    camera_pose3 = TransformationMatrix.from_xyz_rpy(
        x=0, y=-3.5, z=3, roll=-np.pi / 2, pitch=np.pi / 4, yaw=np.pi / 2
    ).to_np()
    camera_poses.append(camera_pose3 @ rotate_x @ rotate)

    camera_pose4 = TransformationMatrix.from_xyz_rpy(
        x=0, y=3.5, z=3, roll=-np.pi / 2, pitch=np.pi / 4, yaw=-np.pi / 2
    ).to_np()
    camera_poses.append(camera_pose4 @ rotate_x @ rotate)

    return camera_poses


def render_scene(world, camera_pose, output_filepath=None) -> bytes:
    """Render world from a single camera pose, return PNG bytes."""
    rt = RayTracer(world=world)
    scene = rt.scene
    scene.camera.fov = [60, 45]
    scene.graph[scene.camera.name] = camera_pose
    png = scene.save_image(resolution=(1024, 768), visible=True)
    if output_filepath:
        with open(output_filepath, "wb") as f:
            f.write(png)
    return png


def render_scene_multi_pose(
    world, camera_poses, output_path=None, filename_prefix="render"
):
    """
    Renders the given world for each camera pose, optionally saves images,
    and returns a list of the images as bytes.
    """
    rt = RayTracer(world=world)
    scene = rt.scene
    scene.camera.fov = [60, 45]
    images = []
    for j, pose in enumerate(camera_poses):
        scene.graph[scene.camera.name] = pose
        png = scene.save_image(resolution=(1024, 768), visible=True)
        images.append(png)
        if output_path:
            with open(
                os.path.join(output_path, f"{filename_prefix}_{j}.png"), "wb"
            ) as f:
                f.write(png)
    return images


class SceneVisualState:
    """Helper to save/restore mesh visuals for a world's bodies."""

    def __init__(self, world: World):
        self.world = world
        self.bodies = list(world.bodies_with_enabled_collision)
        self.original_state = {}
        for body in self.bodies:
            self.original_state[body.id] = body.collision[0].mesh.visual.copy()

    def reset(self):
        """Restore all bodies to their original visuals."""
        for body in self.bodies:
            body.collision[0].mesh.visual = self.original_state[body.id]

    def apply_highlight_to_group(self, group) -> Dict[UUID, Color]:
        """Apply highlight colors to a group of bodies."""
        colors = distinct_html_colors(len(group))

        bodies_colors = zip(group, colors)
        for body, color in bodies_colors:
            body.collision[0].override_mesh_with_color(color)
        return {body.id: color for body, color in zip(group, colors)}


def render_warsaw_scene(world, number_of_bodies=4, write_images=True):
    """Render scene with original textures and highlighted groups."""
    visual_state = SceneVisualState(world)
    camera_poses = get_camera_poses()

    output_path = Path("../resources/warsaw_data/scene_images/")
    if write_images:
        output_path.mkdir(parents=True, exist_ok=True)

    # Initial render with original textures
    original_images = render_scene_multi_pose(
        world,
        camera_poses,
        output_path if write_images else None,
        "original_render",
    )

    # Iterate over groups of bodies and repaint group
    grouped_images = []
    for i, start in enumerate(range(0, len(visual_state.bodies), number_of_bodies)):
        group = visual_state.bodies[start : start + number_of_bodies]
        visual_state.reset()
        visual_state.apply_highlight_to_group(group)
        images = render_scene_multi_pose(
            world,
            camera_poses,
            output_path if write_images else None,
            f"group_{i}_render",
        )
        grouped_images.append(images)

    return original_images, grouped_images


if __name__ == "__main__":
    obj_dir = "/home/ben/devel/iai/src/cognitive_robot_abstract_machine/Objects"
    export_dir = Path("../resources/warsaw_data/json_exports/")
    world = load_world(obj_dir)
    export_world_metadata(world, export_dir)
    render_warsaw_scene(world, number_of_bodies=4)
