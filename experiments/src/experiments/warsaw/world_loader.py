from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterator, List, Tuple
from uuid import UUID

import numpy as np
import trimesh

from experiments.warsaw.exceptions import (
    WarsawSceneNotFoundError,
    WarsawSegmentationMissingError,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.pipeline.pipeline import (
    Pipeline,
    TransformGeometry,
    CenterLocalGeometryAndPreserveWorldPose,
)
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.utils import InheritanceStructureExporter
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Color, Mesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)


class WarsawSceneFile(StrEnum):
    """
    Names the files a Warsaw scene directory is made of.
    """

    SCENE_MESH = "0_mesh.npz"
    """
    The mesh of the whole scene, together with the classes it is segmented by.
    """

    @staticmethod
    def segmentation_of(class_name: str) -> str:
        """
        :param class_name: A class the scene mesh declares.
        :return: The name of the file segmenting the scene by that class, whose spaces
            are written as underscores.
        """
        return f"{class_name.replace(' ', '_')}.npz"


class SceneMeshField(StrEnum):
    """
    Names the arrays a Warsaw scene's files hold.
    """

    VERTICES = "vertices"
    """
    The scene's vertex positions.
    """

    FACES = "faces"
    """
    The vertex indices of each of the scene's triangles.
    """

    VERTEX_COLORS = "verts_colors"
    """
    The color of each vertex, as fractions between 0 and 1.
    """

    CLASSES = "classes"
    """
    The names of the classes the scene is segmented by.
    """

    FACE_INSTANCES = "seg_indices"
    """
    Per face, the instance of one class it belongs to.
    """


@dataclass
class WarsawSceneObject:
    """
    One object a Warsaw scene segments out of its mesh.
    """

    class_name: str
    """
    The class this object was segmented as.
    """

    instance: int
    """
    Which object of that class this is.
    """

    mesh: trimesh.Trimesh
    """
    The object's own geometry, carrying the scene's colors.
    """

    @property
    def name(self) -> PrefixedName:
        """
        :return: The name identifying this object among the scene's objects.
        """
        return PrefixedName(f"{self.class_name}_{self.instance}")


@dataclass
class WarsawScene:
    """
    A Warsaw scene: one mesh, segmented into objects by a set of classes.

    The scene is stored as a single mesh plus, per class, the instance each face belongs
    to. Instance :attr:`UNSEGMENTED` marks the faces a class does not cover.
    """

    UNSEGMENTED: ClassVar[int] = 0
    """
    The instance marking every face a class does not cover.
    """

    vertices: np.ndarray
    """
    The scene's vertex positions.
    """

    faces: np.ndarray
    """
    The vertex indices of each of the scene's triangles.
    """

    vertex_colors: np.ndarray
    """
    The color of each vertex, as fractions between 0 and 1.
    """

    face_instances: Dict[str, np.ndarray]
    """
    Per class, the instance each face belongs to.
    """

    @classmethod
    def from_directory(cls, directory: Path) -> WarsawScene:
        """
        Read the scene a directory holds.

        :param directory: The directory holding the scene mesh and its segmentations.
        :raises WarsawSceneNotFoundError: If the directory holds no scene mesh.
        :raises WarsawSegmentationMissingError: If a declared class has no segmentation.
        """
        scene_mesh_path = Path(directory) / WarsawSceneFile.SCENE_MESH.value
        if not scene_mesh_path.exists():
            raise WarsawSceneNotFoundError(
                directory=Path(directory),
                scene_mesh_file=WarsawSceneFile.SCENE_MESH.value,
            )

        scene_mesh = np.load(scene_mesh_path, allow_pickle=True)
        face_instances = {}
        for class_name in (
            str(name) for name in scene_mesh[SceneMeshField.CLASSES.value]
        ):
            segmentation_file = WarsawSceneFile.segmentation_of(class_name)
            segmentation_path = Path(directory) / segmentation_file
            if not segmentation_path.exists():
                raise WarsawSegmentationMissingError(
                    class_name=class_name, segmentation_file=segmentation_file
                )
            face_instances[class_name] = np.load(segmentation_path)[
                SceneMeshField.FACE_INSTANCES.value
            ]

        return cls(
            vertices=scene_mesh[SceneMeshField.VERTICES.value],
            faces=scene_mesh[SceneMeshField.FACES.value],
            vertex_colors=scene_mesh[SceneMeshField.VERTEX_COLORS.value],
            face_instances=face_instances,
        )

    def objects(self) -> Iterator[WarsawSceneObject]:
        """
        :return: Every object the scene segments out of its mesh, in class order.
        """
        colors = (self.vertex_colors * 255).astype(np.uint8)
        for class_name, instances in self.face_instances.items():
            for instance in np.unique(instances):
                if instance == self.UNSEGMENTED:
                    continue
                # Carry over only the vertices this instance's faces reach, so an
                # object costs its own size rather than the whole scene's.
                faces = self.faces[instances == instance]
                used_vertices, rebased_faces = np.unique(faces, return_inverse=True)
                yield WarsawSceneObject(
                    class_name=class_name,
                    instance=int(instance),
                    mesh=trimesh.Trimesh(
                        vertices=self.vertices[used_vertices],
                        faces=rebased_faces.reshape(-1, 3),
                        vertex_colors=colors[used_vertices],
                        process=False,
                    ),
                )


@dataclass
class WarsawWorldLoader:
    """
    Load a Warsaw scene directory into a single World, or wrap an existing World for
    rendering and export operations.
    """

    input_directory: Path = field(default=None)
    """
    Directory holding the scene to load. Can be None if world is provided directly.
    """

    world: World = field(default=None)
    """
    Loaded World object. Can be provided directly or loaded from input_directory.
    """

    _camera_field_of_view: Tuple[float, float] = field(default=(60, 45))
    """
    Camera field of view for rendering.
    """

    framed_fraction: float = field(default=0.99)
    """
    The share of the scene that has to fall inside each view.

    Below 1 the cameras stand closer and the scene's outermost fringe falls outside the
    view, which on a scanned room crops the stray fragments its walls trail off into
    rather than anything standing in it.
    """

    original_state: Dict[UUID, Any] = field(init=False, default_factory=dict)
    """
    Original visual states of bodies before highlighting.
    """

    SOURCE_TO_WORLD: ClassVar[HomogeneousTransformationMatrix] = (
        HomogeneousTransformationMatrix.from_xyz_rpy(roll=-np.pi / 2)
    )
    """
    Turns the scene from the frame it is written in into the world's.

    The scene measures height down its own y axis, so a floor is written at a greater y
    than the ceiling above it. This rolls that axis onto the world's upward z.
    """

    def __post_init__(self):
        if self.world is None and self.input_directory is None:
            raise ValueError("Either input_directory or world must be provided")

        if self.world is None:
            # Load from directory
            self.world = self._load_world_from_directory(self.input_directory)

        # Cache original visual states of bodies
        for body in self.world.bodies_with_collision:
            if (
                body.collision
                and len(body.collision) > 0
                and hasattr(body.collision[0], "mesh")
            ):
                self.original_state[body.id] = body.collision[0].mesh.visual.copy()

    @classmethod
    def from_world(
        cls, world: World, camera_field_of_view: Tuple[float, float] = (60, 45)
    ) -> WarsawWorldLoader:
        """
        Create a WarsawWorldLoader from an existing World object.

        :param world: An existing World object to wrap.
        :param camera_field_of_view: Camera field of view for rendering.
        :return: A WarsawWorldLoader instance wrapping the given world.
        """
        return cls(
            input_directory=None,
            world=world,
            _camera_field_of_view=camera_field_of_view,
        )

    @staticmethod
    def _load_world_from_directory(input_directory: Path) -> World:
        """
        Load the scene a directory holds, one body per segmented object.

        :param input_directory: The scene directory to read.
        :return: A world holding the scene's objects under a single root.
        """
        scene = WarsawScene.from_directory(input_directory)

        main_world = World()
        root = Body(name=PrefixedName("root_body"))
        with main_world.modify_world():
            main_world.add_body(root)
            for scene_object in scene.objects():
                shape = Mesh.from_trimesh(
                    mesh=scene_object.mesh,
                    origin=HomogeneousTransformationMatrix(),
                )
                shapes = ShapeCollection([shape])
                body = Body(name=scene_object.name, collision=shapes, visual=shapes)
                main_world.add_body(body)
                main_world.add_connection(
                    FixedConnection(
                        parent=root,
                        child=body,
                        name=PrefixedName(f"root_to_{scene_object.name.name}"),
                    )
                )

        pipeline = Pipeline(
            steps=[
                TransformGeometry(WarsawWorldLoader.SOURCE_TO_WORLD),
                CenterLocalGeometryAndPreserveWorldPose(),
            ]
        )
        return pipeline.apply(main_world)

    # %% Public API
    def export_semantic_annotation_inheritance_structure(
        self, output_directory: Path
    ) -> None:
        """
        Export kinematic structure and semantic annotations to JSON files.

        :param output_directory: Directory to write JSON files to.
        """
        output_directory.mkdir(parents=True, exist_ok=True)
        self.world.export_kinematic_structure_tree_to_json(
            output_directory / "kinematic_structure.json",
            include_connections=False,
        )
        InheritanceStructureExporter(
            SemanticAnnotation, output_directory / "semantic_annotations.json"
        ).export()

    def export_scene_to_pngs(
        self,
        number_of_bodies: int,
        output_directory: Path,
        headless: bool = False,
    ):
        """
        Export rendered images of the scene with highlighted groups of bodies.

        :param number_of_bodies: Number of bodies to highlight in each group.
        :param output_directory: Directory to write images to.
        :param headless: Whether to render without opening a window.
        """
        output_directory.mkdir(parents=True, exist_ok=True)

        self.render_scene_from_predefined_poses(
            output_directory, "original_render", headless=headless
        )

        for i, start in enumerate(
            range(
                0,
                len(bodies := self.world.bodies_with_collision),
                number_of_bodies,
            )
        ):
            group = bodies[start : start + number_of_bodies]
            self._reset_body_colors()
            self._apply_highlight_to_group(group)
            self.render_scene_from_predefined_poses(
                output_directory, f"group_{i}_render", headless=headless
            )

    def render_scene_from_predefined_poses(
        self, output_path: Path, filename_prefix: str, headless: bool = False
    ):
        """
        Render the world from each of the predefined camera poses, writing one image per
        pose.

        :param output_path: Directory to save images.
        :param filename_prefix: Prefix for image filenames.
        :param headless: Whether to render without opening a window.
        """
        for index, pose in enumerate(self._predefined_camera_transforms):
            self.render_scene_from_camera_pose(
                pose,
                os.path.join(output_path, f"{filename_prefix}_{index}.png"),
                headless=headless,
            )

    def render_scene_from_camera_pose(
        self,
        camera_transform: HomogeneousTransformationMatrix,
        output_filepath=None,
        headless: bool = False,
    ) -> bytes:
        """
        Render the world from a single camera pose.

        :param camera_transform: Where the camera stands and what it faces.
        :param output_filepath: Where to write the image, or None to only return it.
        :param headless: Whether to render without opening a window.
        :return: The rendered image as PNG bytes.
        """
        rt = RayTracer(world=self.world)
        scene = rt.scene
        scene.camera.fov = self._camera_field_of_view
        scene.graph[scene.camera.name] = camera_transform
        png = scene.save_image(resolution=(1024, 768), visible=not headless)
        if output_filepath:
            with open(output_filepath, "wb") as f:
                f.write(png)
        return png

    # %% Export Helpers

    VIEWPOINT_ELEVATION: ClassVar[float] = np.radians(30)
    """
    How far above the scene's middle the predefined cameras are lifted.
    """

    VIEWPOINT_AZIMUTHS: ClassVar[Dict[str, float]] = {
        "front_left": np.radians(45),
        "back_left": np.radians(135),
        "back_right": np.radians(225),
        "front_right": np.radians(315),
    }
    """
    The direction each named viewpoint looks at the scene from.
    """

    @cached_property
    def _scene_points(self) -> np.ndarray:
        """
        :return: Every vertex the scene draws, in the world's frame.
        """
        return np.concatenate(
            [
                shape.mesh_in_frame(self.world.root).vertices
                for body in self.world.bodies_with_collision
                for shape in body.collision
            ]
        )

    @cached_property
    def _scene_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: The lowest and highest corner the scene's geometry reaches.
        """
        return self._scene_points.min(axis=0), self._scene_points.max(axis=0)

    @staticmethod
    def _looking_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Place a camera at *eye* facing *target*.

        :param eye: Where the camera stands.
        :param target: What it faces.
        :return: The camera's transform in the world's frame.
        """
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right = right / np.linalg.norm(right)

        transform = np.eye(4)
        transform[:3, 0] = right
        transform[:3, 1] = np.cross(right, forward)
        # A camera looks down its own negative z.
        transform[:3, 2] = -forward
        transform[:3, 3] = eye
        return transform

    @cached_property
    def _predefined_camera_transforms(self) -> List[np.ndarray]:
        """
        :return: The camera poses of :meth:`compute_camera_poses`, in viewpoint order.
        """
        return list(self.compute_camera_poses().values())

    def compute_camera_poses(self) -> Dict[str, np.ndarray]:
        """
        :return: One camera pose per named viewpoint, each standing just far enough from
            the scene for :attr:`framed_fraction` of it to fall inside the view.

        The distance is measured from the scene rather than fixed, so a scene of any size
        is framed rather than viewed from within itself.
        """
        low, high = self._scene_bounds
        middle = (low + high) / 2

        poses = {}
        for name, azimuth in self.VIEWPOINT_AZIMUTHS.items():
            direction = np.array(
                [
                    np.cos(self.VIEWPOINT_ELEVATION) * np.cos(azimuth),
                    np.cos(self.VIEWPOINT_ELEVATION) * np.sin(azimuth),
                    np.sin(self.VIEWPOINT_ELEVATION),
                ]
            )
            distance = self._framing_distance(self._scene_points - middle, direction)
            poses[name] = self._looking_at(middle + distance * direction, middle)
        return poses

    def _framing_distance(self, points: np.ndarray, direction: np.ndarray) -> float:
        """
        Measure how far along *direction* a camera has to stand for
        :attr:`framed_fraction` of the points to fall inside its view.

        A point's room in the view grows with its distance from the camera, so one on the
        near side needs more room than one equally far off the view's axis on the far
        side. Each point is therefore solved for separately and the furthest answer wins.

        ..note:: This measures the scene's own geometry rather than the corners of the box
            around it, which stand in empty air and would push the camera needlessly far
            back.

        :param points: The scene's points, relative to what the camera faces.
        :param direction: The direction from the scene to the camera.
        :return: The distance from the scene to stand at.
        """
        orientation = self._looking_at(direction, np.zeros(3))
        right, up = orientation[:3, 0], orientation[:3, 1]
        horizontal, vertical = (
            np.tan(np.radians(angle) / 2) for angle in self._camera_field_of_view
        )

        # A point at depth ``distance - along`` fits when its offset across the view is
        # within that depth's half-width, so it needs ``along + offset / half_angle``.
        along = points @ direction
        needed = np.concatenate(
            [
                along + np.abs(points @ right) / horizontal,
                along + np.abs(points @ up) / vertical,
            ]
        )
        return float(np.percentile(needed, self.framed_fraction * 100))

    # %% Body Highlighting
    def _reset_body_colors(self):
        """
        Reset all bodies to their original visual states.
        """
        for body in self.world.bodies_with_collision:
            body.collision[0].mesh.visual = self.original_state[body.id]

    @staticmethod
    def _apply_highlight_to_group(bodies: List[Body]) -> Dict[UUID, Color]:
        """
        Apply distinct highlight colors to a group of bodies.
        """
        colors = Color.distinct_colors(len(bodies))
        for body, color in zip(bodies, colors):
            body_mesh: Mesh = body.collision[0]
            body_mesh.dye(color)
        return {body.id: color for body, color in zip(bodies, colors)}
