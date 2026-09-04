from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)
from uuid import UUID

import numpy as np
import trimesh
from PIL import Image

from experiments.warsaw.exceptions import (
    AmbiguousWarsawSceneError,
    WarsawLabelsMisalignedError,
    WarsawLabelsMissingError,
    WarsawSceneNotFoundError,
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

SCENE_MESH_PATTERN = "*.ply"
"""
How a scene directory's one mesh is named.
"""

GEOMETRY_PROPERTY = "vertex_indices"
"""
The face property holding a face's geometry rather than one of its labels.
"""

SCENE_BODY_NAME = "scene"
"""
The name of the body carrying the whole scene.
"""

VIEWPOINT_IN_ROOM = "in-room"
"""
Choose a viewpoint by what is visible of the segments in the scene around them.
"""

VIEWPOINT_ALONE = "alone"
"""
Choose a viewpoint by what is visible of the segments on their own.
"""

MAXIMUM_LABEL_LENGTH = 120
"""
How many characters of segment names a render's filename carries.
"""


def segment_label(
    segments: Iterable[LabelSegment], maximum_length: int = MAXIMUM_LABEL_LENGTH
) -> str:
    """
    Name the segments a render highlights, so its filename says what is colored in it.

    :param segments: The segments highlighted in the render.
    :param maximum_length: How many characters of names the filename can hold.
    :return: Their names joined by dashes, cut short of that length.
    """
    names = [str(segment.name).replace(" ", "_") for segment in segments]
    if len("-".join(names)) <= maximum_length:
        return "-".join(names)

    kept: List[str] = []
    length = 0
    for name in names:
        if length + len(name) + 1 > maximum_length:
            break
        kept.append(name)
        length += len(name) + 1
    return "-".join(kept + [f"and_{len(names) - len(kept)}_more"])


def changed_pixels(one: bytes, other: bytes) -> int:
    """
    Count how many pixels two renders of the same pose differ in.

    Comparing a render against the same view painted in the scene's own colors counts
    exactly the pixels the highlight is responsible for, which is what "how much of this
    is visible from here" means. Matching the highlight's color instead would need a
    tolerance, since a renderer shades one color across a range of them.

    :param one: One render, as PNG bytes.
    :param other: The other render of the same pose.
    :return: How many pixels differ.
    """
    first = np.asarray(Image.open(io.BytesIO(one)).convert("RGB"))
    second = np.asarray(Image.open(io.BytesIO(other)).convert("RGB"))
    if first.shape != second.shape:
        return 0
    return int((first != second).any(axis=-1).sum())


@dataclass
class LabelSegment:
    """
    One object a Warsaw scene labels: the faces one of its classes marks as one instance.

    A face can belong to segments of several classes at once, since a scene labels, for
    example, a drawer's front both as ``drawer`` and as the ``cabinet`` holding it.
    """

    class_name: str
    """
    The class that labels this object.
    """

    instance: int
    """
    Which object of that class this is.
    """

    faces: np.ndarray
    """
    The indices of the scene mesh's faces this object is made of.
    """

    @property
    def name(self) -> PrefixedName:
        """
        :return: The name identifying this object among the scene's objects.
        """
        return PrefixedName(f"{self.class_name}_{self.instance}")

    def __len__(self) -> int:
        """
        :return: How many of the scene's faces this object is made of.
        """
        return len(self.faces)


@dataclass
class WarsawScene:
    """
    A Warsaw scene: one mesh whose faces carry, per class, the instance they belong to.

    The scene is stored as a single mesh with one integer face property per class.
    Instance :attr:`UNSEGMENTED` marks the faces a class does not cover.
    """

    UNSEGMENTED: ClassVar[int] = 0
    """
    The instance marking every face a class does not cover.
    """

    mesh_path: Path
    """
    The file the scene was read from.
    """

    mesh: trimesh.Trimesh
    """
    The scene's geometry, carrying the colors it was scanned in.
    """

    face_labels: Dict[str, np.ndarray]
    """
    Per class, the instance each face belongs to.
    """

    @classmethod
    def from_directory(cls, directory: Path) -> WarsawScene:
        """
        Read the scene a directory holds.

        :param directory: The directory holding the scene's mesh.
        :raises WarsawSceneNotFoundError: If the directory holds no mesh.
        :raises AmbiguousWarsawSceneError: If it holds more than one.
        """
        directory = Path(directory)
        scene_meshes = sorted(directory.glob(SCENE_MESH_PATTERN))
        if not scene_meshes:
            raise WarsawSceneNotFoundError(
                directory=directory, scene_mesh_pattern=SCENE_MESH_PATTERN
            )
        if len(scene_meshes) > 1:
            raise AmbiguousWarsawSceneError(
                directory=directory, scene_meshes=scene_meshes
            )
        return cls.from_file(scene_meshes[0])

    @classmethod
    def from_file(cls, scene_mesh_path: Path) -> WarsawScene:
        """
        Read the scene one mesh file holds.

        :param scene_mesh_path: The mesh to read.
        :raises WarsawLabelsMissingError: If the mesh carries no per-face class labels.
        """
        scene_mesh_path = Path(scene_mesh_path)
        # Processing welds vertices and drops degenerate faces, which renumbers the
        # faces and would leave every label pointing at another face than the one it
        # was written for.
        mesh = trimesh.load(scene_mesh_path, process=False)
        return cls(
            mesh_path=scene_mesh_path,
            mesh=mesh,
            face_labels=cls._read_face_labels(mesh, scene_mesh_path),
        )

    @staticmethod
    def _read_face_labels(
        mesh: trimesh.Trimesh, scene_mesh_path: Path
    ) -> Dict[str, np.ndarray]:
        """
        Read the instance each face belongs to, per class.

        :param mesh: The mesh the scene was read from.
        :param scene_mesh_path: The file it was read from, for the error message.
        :return: Per class, the instance each face belongs to.
        :raises WarsawLabelsMissingError: If the mesh carries no labels.
        """
        try:
            faces = mesh.metadata["_ply_raw"]["face"]["data"]
            class_names = [
                name for name in faces.dtype.names if name != GEOMETRY_PROPERTY
            ]
        except (AttributeError, KeyError, TypeError):
            raise WarsawLabelsMissingError(scene_mesh=scene_mesh_path)

        if not class_names:
            raise WarsawLabelsMissingError(scene_mesh=scene_mesh_path)
        return {name: np.asarray(faces[name]) for name in class_names}

    @property
    def class_names(self) -> List[str]:
        """
        :return: The classes the scene is labelled by, in the order it declares them.
        """
        return list(self.face_labels)

    def segments(self) -> Iterator[LabelSegment]:
        """
        :return: Every object the scene labels, in class order.
        """
        for class_name, instances in self.face_labels.items():
            for instance in np.unique(instances):
                if instance == self.UNSEGMENTED:
                    continue
                yield LabelSegment(
                    class_name=class_name,
                    instance=int(instance),
                    faces=np.flatnonzero(instances == instance),
                )


@dataclass
class RenderedSegmentGroup:
    """
    One group of label segments, colored and rendered from every viewpoint.
    """

    index: int
    """
    Which group of the scene's segments this is.
    """

    segments: List[LabelSegment]
    """
    The segments colored in the renders.
    """

    colors: Dict[PrefixedName, Color]
    """
    The color each of them was given.
    """

    images: Dict[str, bytes]
    """
    Per viewpoint, the render taken from it, as PNG bytes.
    """

    @property
    def label(self) -> str:
        """
        :return: The segments' names, short enough for a filename to carry.
        """
        return segment_label(self.segments)


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

    render_resolution: Tuple[int, int] = field(default=(1024, 768))

    deciding_resolution: Optional[Tuple[int, int]] = field(default=None)
    """
    How large to draw a render made only to choose between viewpoints, if not the size
    a kept picture is drawn at.

    Which viewpoint shows more of something is a question about proportions, and
    proportions survive being asked small -- but only asking it small can answer it
    differently, and the renders turned out not to be the cost they looked like, so it
    is not done unless asked for.
    """
    """
    The size of the images the renders are written as.
    """

    unhighlighted_dimming: float = field(default=0.35)
    """
    How much of its own color the scene keeps where nothing is highlighted.

    The scene is scanned in its own colors, against which a highlight of a similar hue
    is hard to make out. Dimming everything else leaves the room recognizable while the
    highlighted segments are the only part of it in full color.
    """

    scene: Optional[WarsawScene] = field(default=None)
    """
    The scene the world was loaded from, or None for a world provided directly.
    """

    original_state: Dict[UUID, Any] = field(init=False, default_factory=dict)
    """
    Original visual states of bodies before highlighting.
    """

    _original_face_colors: Optional[np.ndarray] = field(init=False, default=None)
    """
    The color of each of the scene body's faces before anything was highlighted.
    """

    _plain_views: Dict[Tuple[str, ...], Dict[str, bytes]] = field(
        init=False, default_factory=dict
    )
    """
    The scene rendered dimmed and unhighlighted, kept so it is rendered once rather than
    once per region measured against it.
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
            if self.scene is None:
                self.scene = WarsawScene.from_directory(self.input_directory)
            self.world = self._world_from_scene(self.scene)
            self._verify_faces_carry_the_labels()

        # Cache original visual states of bodies
        for body in self.world.bodies_with_collision:
            if (
                body.collision
                and len(body.collision) > 0
                and hasattr(body.collision[0], "mesh")
            ):
                self.original_state[body.id] = body.collision[0].mesh.visual.copy()

        # Face colors are read once here, while the mesh still carries the colors it was
        # scanned in, because highlighting replaces them.
        if self.world.bodies_with_collision:
            self._original_face_colors = np.asarray(
                self.scene_mesh.visual.face_colors
            ).copy()

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
    def _world_from_scene(scene: WarsawScene) -> World:
        """
        Build the world a scene describes: its whole mesh, under a single root.

        The scene stays one body rather than one body per object, because its objects
        are labels over shared faces: a drawer front is part of the drawer and of the
        cabinet holding it, and cutting the mesh into bodies would have to duplicate
        such geometry or take it from one of them.

        The body reads the scene's file directly, so its faces arrive in the order the
        labels were written for.

        :param scene: The scene to build a world from.
        :return: A world holding the scene under a single root.
        """
        main_world = World()
        root = Body(name=PrefixedName("root_body"))
        shapes = ShapeCollection([Mesh.from_file(str(scene.mesh_path))])
        scene_body = Body(
            name=PrefixedName(SCENE_BODY_NAME), collision=shapes, visual=shapes
        )
        with main_world.modify_world():
            main_world.add_body(root)
            main_world.add_body(scene_body)
            main_world.add_connection(
                FixedConnection(
                    parent=root,
                    child=scene_body,
                    name=PrefixedName(f"root_to_{SCENE_BODY_NAME}"),
                )
            )

        pipeline = Pipeline(
            steps=[
                TransformGeometry(WarsawWorldLoader.SOURCE_TO_WORLD),
                CenterLocalGeometryAndPreserveWorldPose(),
            ]
        )
        return pipeline.apply(main_world)

    def _verify_faces_carry_the_labels(self) -> None:
        """
        Check that the world was built from the faces the labels were written for.

        :raises WarsawLabelsMisalignedError: If it was built from other faces.
        """
        loaded_faces = self.scene_mesh.faces
        if not np.array_equal(loaded_faces, self.scene.mesh.faces):
            raise WarsawLabelsMisalignedError(
                scene_mesh=self.scene.mesh_path,
                labelled_faces=len(self.scene.mesh.faces),
                loaded_faces=len(loaded_faces),
            )

    # %% The Scene's Body and Segments

    @cached_property
    def scene_body(self) -> Body:
        """
        :return: The body carrying the scene's geometry.
        """
        for body in self.world.bodies_with_collision:
            if body.name.name == SCENE_BODY_NAME:
                return body
        return self.world.bodies_with_collision[0]

    @property
    def scene_mesh(self) -> trimesh.Trimesh:
        """
        :return: The scene body's mesh, as the world holds it.
        """
        return self.scene_body.collision[0].mesh

    @cached_property
    def label_segments(self) -> List[LabelSegment]:
        """
        :return: Every object the scene labels, in class order.
        """
        return list(self.scene.segments())

    def segments_of_classes(self, class_names: Iterable[str]) -> List[LabelSegment]:
        """
        :param class_names: The classes to take the segments of.
        :return: The scene's segments of those classes, in class order.
        """
        wanted = set(class_names)
        return [
            segment for segment in self.label_segments if segment.class_name in wanted
        ]

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

    def render_label_segment_groups(
        self,
        group_size: int = 8,
        segments: Optional[Iterable[LabelSegment]] = None,
        headless: bool = False,
    ) -> Iterator[RenderedSegmentGroup]:
        """
        Color the scene's objects a group at a time and render each group from every
        viewpoint.

        The scene is left in its own colors once the last group has been rendered.

        :param group_size: How many segments to color at once.
        :param segments: The segments to walk through, defaulting to all of the scene's.
        :param headless: Whether to render without opening a window.
        :return: One rendered group at a time.
        """
        segments = list(self.label_segments if segments is None else segments)
        camera_poses = self.compute_camera_poses()

        try:
            for index, start in enumerate(range(0, len(segments), group_size)):
                group = segments[start : start + group_size]
                colors = self._apply_highlight_to_segments(group)
                yield RenderedSegmentGroup(
                    index=index,
                    segments=group,
                    colors=colors,
                    images=self.render_scene_from_camera_poses(
                        camera_poses, headless=headless
                    ),
                )
        finally:
            self._reset_segment_colors()

    def export_scene_to_pngs(
        self,
        group_size: int,
        output_directory: Path,
        headless: bool = False,
    ) -> None:
        """
        Export rendered images of the scene with highlighted groups of label segments.

        :param group_size: Number of label segments to highlight in each group.
        :param output_directory: Directory to write images to.
        :param headless: Whether to render without opening a window.
        """
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        for pose_name, image in self.render_scene_from_camera_poses(
            self.compute_camera_poses(), headless=headless
        ).items():
            (output_directory / f"original_{pose_name}.png").write_bytes(image)

        for group in self.render_label_segment_groups(
            group_size=group_size, headless=headless
        ):
            for pose_name, image in group.images.items():
                (
                    output_directory
                    / f"group_{group.index}_{pose_name}__{group.label}.png"
                ).write_bytes(image)

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

    def render_scene_from_camera_poses(
        self,
        camera_poses: Dict[str, HomogeneousTransformationMatrix],
        headless: bool = False,
    ) -> Dict[str, bytes]:
        """
        Render the world from several camera poses at once.

        The scene is built once for all of them, which on a scanned room is most of the
        work a render costs.

        :param camera_poses: The poses to render from, by the name of each viewpoint.
        :param headless: Whether to render without opening a window.
        :return: Per viewpoint, its render as PNG bytes.
        """
        return self._render_from_poses(self._render_scene(), camera_poses, headless)

    def _render_from_poses(
        self,
        scene: trimesh.Scene,
        camera_poses: Dict[str, HomogeneousTransformationMatrix],
        headless: bool = False,
        resolution: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, bytes]:
        """
        Render one scene from several camera poses.

        :param scene: The scene to render.
        :param camera_poses: The poses to render from, by the name of each viewpoint.
        :param headless: Whether to render without opening a window.
        :param resolution: How large to render, defaulting to the size a picture is
            kept at. A render made only to be measured and thrown away costs what its
            pixels cost and needs none of them.
        :return: Per viewpoint, its render as PNG bytes.
        """
        scene.camera.fov = self._camera_field_of_view
        images = {}
        for pose_name, camera_pose in camera_poses.items():
            scene.graph[scene.camera.name] = camera_pose
            images[pose_name] = scene.save_image(
                resolution=resolution or self.render_resolution,
                visible=not headless,
                # A scan is a single layer of surface whose faces point whichever way
                # they were reconstructed, so culling the ones facing away hides parts of
                # the room for no reason a viewer could tell -- a drawer front vanishes
                # while the handle on it stays.
                flags={"cull": False},
            )
        return images

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
        scene = self._render_scene()
        scene.graph[scene.camera.name] = camera_transform
        png = scene.save_image(resolution=self.render_resolution, visible=not headless)
        if output_filepath:
            with open(output_filepath, "wb") as f:
                f.write(png)
        return png

    @cached_property
    def _ray_tracer(self) -> RayTracer:
        """
        :return: The ray tracer whose scene the renders are taken from.

        It holds the world's own meshes rather than copies of them, so it keeps up with
        recoloring on its own and is built once instead of once per image.
        """
        return RayTracer(world=self.world)

    def _render_scene(self) -> trimesh.Scene:
        """
        :return: The scene to render, brought up to date with the world.
        """
        self._ray_tracer.update_scene()
        return self._ray_tracer.scene

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

    def compute_camera_poses(
        self, points: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        :param points: The points to frame, defaulting to the whole scene's. Framing a
            few segments instead is what makes a handle of two hundred faces visible at
            all, rather than three pixels of a picture of a room.
        :return: One camera pose per named viewpoint, each standing just far enough from
            those points for :attr:`framed_fraction` of them to fall inside the view.

        The distance is measured from what is framed rather than fixed, so a scene of any
        size is framed rather than viewed from within itself.
        """
        framed = self._scene_points if points is None else points
        middle = (framed.min(axis=0) + framed.max(axis=0)) / 2

        poses = {}
        for name, azimuth in self.VIEWPOINT_AZIMUTHS.items():
            direction = np.array(
                [
                    np.cos(self.VIEWPOINT_ELEVATION) * np.cos(azimuth),
                    np.cos(self.VIEWPOINT_ELEVATION) * np.sin(azimuth),
                    np.sin(self.VIEWPOINT_ELEVATION),
                ]
            )
            distance = self._framing_distance(framed - middle, direction)
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

    # %% Label Segment Highlighting

    def _apply_highlight_to_faces(
        self, highlights: Sequence[Tuple[Color, np.ndarray]]
    ) -> None:
        """
        Paint sets of the scene's faces, dimming everything else.

        :param highlights: The color to paint each set of faces in. Later sets paint over
            earlier ones where they overlap.
        """
        face_colors = self._dimmed_face_colors.copy()
        for color, faces in highlights:
            face_colors[faces] = trimesh.visual.color.to_rgba(color.to_rgba())
        self.scene_mesh.visual.face_colors = face_colors

    def _apply_highlight_to_segments(
        self, segments: Iterable[LabelSegment]
    ) -> Dict[PrefixedName, Color]:
        """
        Color a group of the scene's objects, dimming everything else.

        :param segments: The segments to color.
        :return: The color each of them was given.
        """
        segments = list(segments)
        colors = Color.distinct_colors(len(segments))
        self._apply_highlight_to_faces(
            [(color, segment.faces) for segment, color in zip(segments, colors)]
        )
        return {segment.name: color for segment, color in zip(segments, colors)}

    @cached_property
    def _dimmed_face_colors(self) -> np.ndarray:
        """
        :return: The scene's own face colors, dimmed by :attr:`unhighlighted_dimming`.

        Every group of highlights is painted over these, so they are dimmed once rather
        than once per group, which on a scanned room is seconds of work each time.
        """
        face_colors = self._original_face_colors.copy()
        face_colors[:, :3] = (
            face_colors[:, :3].astype(np.float64) * self.unhighlighted_dimming
        ).astype(np.uint8)
        return face_colors

    def _reset_segment_colors(self) -> None:
        """
        Paint the scene back in the colors it was scanned in.
        """
        if self._original_face_colors is not None:
            self.scene_mesh.visual.face_colors = self._original_face_colors.copy()

    def pair_highlights(
        self, one: LabelSegment, other: LabelSegment
    ) -> Tuple[List[Tuple[Color, np.ndarray]], Dict[str, Color]]:
        """
        Work out how to color two segments so that what they disagree about can be seen.

        A face two segments both claim can only take one color, so a picture painted in
        two colors shows an overlap as though it belonged to whichever was painted last.
        The faces both claim therefore take a third color, which is the only way the
        question being asked is visible in the picture at all.

        :param one: One of the segments.
        :param other: The other segment.
        :return: What to paint, and what each color stands for: each segment's name for
            the faces it alone claims, and ``both`` for the faces they share, which is
            absent when they share none.
        """
        shared = np.intersect1d(one.faces, other.faces, assume_unique=True)
        colors = Color.distinct_colors(3)
        highlights = [
            (colors[0], np.setdiff1d(one.faces, other.faces, assume_unique=True)),
            (colors[1], np.setdiff1d(other.faces, one.faces, assume_unique=True)),
            (colors[2], shared),
        ]

        legend = {str(one.name): colors[0], str(other.name): colors[1]}
        if len(shared):
            legend["both"] = colors[2]
        return highlights, legend

    def highlight_pair(
        self, one: LabelSegment, other: LabelSegment
    ) -> Dict[str, Color]:
        """
        Color two segments so that what they disagree about can be seen.

        :param one: One of the segments.
        :param other: The other segment.
        :return: What each color stands for, as :meth:`pair_highlights` describes it.
        """
        highlights, legend = self.pair_highlights(one, other)
        self._apply_highlight_to_faces(highlights)
        return legend

    def points_of(self, segments: Iterable[LabelSegment]) -> np.ndarray:
        """
        :param segments: The segments to take the geometry of.
        :return: The vertices their faces are drawn from, in the world's frame.
        """
        faces = np.concatenate([segment.faces for segment in segments])
        return self.scene_mesh.vertices[self.scene_mesh.faces[faces].ravel()]

    def render_segments_alone(
        self,
        segments: Iterable[LabelSegment],
        face_colors: np.ndarray,
        viewpoints: Optional[Sequence[str]] = None,
        headless: bool = False,
    ) -> Dict[str, bytes]:
        """
        Render some segments' geometry by itself, with the rest of the scene left out.

        Framing a camera closely on a few hundred faces inside a scanned room puts it
        behind the room's own walls and cabinet fronts, so a close-up of the whole scene
        shows whatever stands in the way rather than what was asked about. Rendering the
        segments alone is what makes them visible; where they sit in the room is what a
        context view is for.

        :param segments: The segments to render.
        :param face_colors: The color of every face of the whole scene, of which the
            segments' own are taken.
        :param viewpoints: Which named viewpoints to render, defaulting to all of them.
        :param headless: Whether to render without opening a window.
        :return: Per viewpoint, its render as PNG bytes.
        """
        faces = np.unique(np.concatenate([segment.faces for segment in segments]))
        alone = self.scene_mesh.submesh([faces], append=True)
        alone.visual.face_colors = face_colors[faces]

        poses = self._chosen_viewpoints(
            self.compute_camera_poses(
                self.scene_mesh.vertices[self.scene_mesh.faces[faces].ravel()]
            ),
            viewpoints,
        )
        return self._render_from_poses(trimesh.Scene(alone), poses, headless)

    def _chosen_viewpoints(
        self,
        poses: Dict[str, HomogeneousTransformationMatrix],
        viewpoints: Optional[Sequence[str]],
    ) -> Dict[str, HomogeneousTransformationMatrix]:
        """
        :param poses: The poses of every named viewpoint.
        :param viewpoints: The names to keep, or None to keep all of them.
        :return: The poses that were asked for.
        """
        if viewpoints is None:
            return poses
        wanted = set(viewpoints)
        return {name: pose for name, pose in poses.items() if name in wanted}

    def render_region(
        self,
        segments: Iterable[LabelSegment],
        highlights: Sequence[Tuple[Color, np.ndarray]],
        viewpoints: Optional[Sequence[str]] = None,
        headless: bool = False,
        choose_viewpoint: Optional[str] = None,
        context_segments: Optional[Iterable[LabelSegment]] = None,
    ) -> Dict[str, bytes]:
        """
        Render a part of the scene the three ways a question about it needs answering.

        The close-up shows what is being asked about, on its own so that nothing stands
        in front of it; the context view shows where in the room it is; and the plain
        close-up shows the same geometry in the colors it was scanned in, without which a
        viewer is judging paint rather than an object.

        :param segments: The segments the close-ups show.
        :param highlights: The colors to paint, as :meth:`_apply_highlight_to_faces`
            takes them.
        :param viewpoints: Which named viewpoints to render, defaulting to all of them.
        :param headless: Whether to render without opening a window.
        :param context_segments: What the context view is framed on, defaulting to the
            whole scene. Framing it on a segment together with its measured neighbours
            is what makes a small object visible in it at all: a mug fills 0.00% of a
            picture of the room and 9.6% of a picture of the counter it stands on.
        :param choose_viewpoint: How to keep only the viewpoint that shows the most,
            measured rather than chosen: a cabinet standing against a wall is invisible
            from three of the four, and a fixed viewpoint then hands a viewer a picture
            of a room with nothing marked in it. ``in-room`` measures it in the scene, so
            that what stands in front of the segments counts against a viewpoint;
            ``alone`` measures it on the segments by themselves, which is what a hundred
            of these can afford. None keeps every viewpoint.
        :return: The renders, keyed ``<kind>_<viewpoint>``.
        """
        segments = list(segments)
        frame = (
            None if context_segments is None else self.points_of(context_segments)
        )
        if choose_viewpoint == VIEWPOINT_IN_ROOM:
            viewpoints = (
                self.viewpoint_showing_all(segments, viewpoints, headless, frame),
            )
        elif choose_viewpoint == VIEWPOINT_ALONE:
            viewpoints = (
                self.viewpoint_showing_all_alone(segments, viewpoints, headless),
            )
        elif choose_viewpoint is not None:
            raise ValueError(
                f"{choose_viewpoint!r} is no way of choosing a viewpoint; it is "
                f"{VIEWPOINT_IN_ROOM!r}, {VIEWPOINT_ALONE!r} or None"
            )

        self._apply_highlight_to_faces(highlights)
        painted = np.asarray(self.scene_mesh.visual.face_colors).copy()
        closeups = self.render_segments_alone(segments, painted, viewpoints, headless)
        contexts = self._render_from_poses(
            self._render_scene(),
            self._chosen_viewpoints(self.compute_camera_poses(frame), viewpoints),
            headless,
        )

        self._reset_segment_colors()
        plains = self.render_segments_alone(
            segments, self._original_face_colors, viewpoints, headless
        )

        return {
            **{f"closeup_{name}": image for name, image in closeups.items()},
            **{f"context_{name}": image for name, image in contexts.items()},
            **{f"plain_{name}": image for name, image in plains.items()},
        }

    def viewpoint_showing_all(
        self,
        segments: Iterable[LabelSegment],
        viewpoints: Optional[Sequence[str]] = None,
        headless: bool = False,
        frame: Optional[np.ndarray] = None,
    ) -> str:
        """
        Find the viewpoint from which the least-visible of some segments is seen best.

        Each segment is measured on its own, and the viewpoints are ranked by their
        *worst* segment rather than by their total. Ranking by the total lets a large
        object decide for a small one: a door of thirty thousand faces and the handle on
        it are seen best from opposite sides, and the sum picks the side showing the
        whole door and none of the handle -- which is the object whose membership was in
        question. Whichever segment is hardest to see is the one that decides.

        :param segments: The segments that all have to be visible.
        :param viewpoints: Which named viewpoints to consider, defaulting to all.
        :param headless: Whether to render without opening a window.
        :param frame: The points the views are framed on, defaulting to the whole scene.
            The choice has to be made at the framing it is made for, since which
            viewpoint shows an object best depends on how closely it is framed.
        :return: The name of the viewpoint that shows the least-visible segment best.
        """
        poses = self._chosen_viewpoints(self.compute_camera_poses(frame), viewpoints)
        rooms = self._dimmed_views(poses, headless, cacheable=frame is None)
        color = Color.distinct_colors(1)[0]

        worst = {name: None for name in poses}
        for segment in segments:
            self._apply_highlight_to_faces([(color, segment.faces)])
            shown = self._render_from_poses(self._render_scene(), poses, headless)
            for name, image in shown.items():
                visible = changed_pixels(image, rooms[name])
                if worst[name] is None or visible < worst[name]:
                    worst[name] = visible
        self._reset_segment_colors()
        return max(worst, key=lambda name: worst[name])

    def presented_area(
        self, segment: LabelSegment, direction: np.ndarray
    ) -> float:
        """
        Measure how much of a segment turns towards a direction, without drawing it.

        How large a face looks from a direction is its area foreshortened by how squarely
        it faces -- ``area * |n . d|`` summed over the segment. It is arithmetic over
        arrays the mesh already carries, where a render is a tenth of a second.

        Which way a face points does not enter into it, because the renders are made
        without culling: a scan is a single layer of surface whose faces point whichever
        way they were reconstructed, so one pointing away is drawn exactly like one
        pointing towards. Counting only the faces whose normals face the camera measures
        a picture nobody is taking, and picks a different viewpoint than the render does.

        ..note:: It knows nothing of what stands in front of the segment, nor of the
            segment standing in front of itself. It ranks; it does not decide.

        :param segment: The segment to measure.
        :param direction: The direction it is looked at from, from the camera outwards.
        :return: The area of it presented to that direction, in square metres.
        """
        mesh = self.scene_mesh
        towards = np.abs(mesh.face_normals[segment.faces] @ direction)
        return float(towards @ mesh.area_faces[segment.faces])

    def viewpoint_showing_all_alone(
        self,
        segments: Iterable[LabelSegment],
        viewpoints: Optional[Sequence[str]] = None,
        headless: bool = False,
        considered: int = 4,
    ) -> str:
        """
        Choose the viewpoint showing the least-visible of some segments best, measured
        on the segments by themselves rather than in the room.

        The same choice as :meth:`viewpoint_showing_all` and far cheaper. Every viewpoint
        is first ranked by :meth:`presented_area`, which draws nothing at all; only the
        best few are then drawn, and drawn small, since what is wanted of those renders
        is which of them shows more and not a picture anyone keeps.

        What it gives up against :meth:`viewpoint_showing_all` is occlusion by everything
        else -- a cabinet standing in front of the pair no longer counts against a
        viewpoint -- which is the right trade when the picture it is chosen for shows the
        segments alone.

        :param segments: The segments that all have to be visible.
        :param viewpoints: Which named viewpoints to consider, defaulting to all.
        :param headless: Whether to render without opening a window.
        :param considered: How many of the ranked viewpoints to draw and compare, by
            default all of them. Drawing fewer saved four percent of the time on this
            scene and picked a different viewpoint than drawing all four in two cases of
            six, the renders having never been the cost they looked like; the ranking is
            left for a caller who has more viewpoints than four to choose between. One
            skips drawing altogether and trusts the arithmetic, which cannot see a
            segment hiding behind itself.
        :return: The name of the viewpoint that shows the least-visible segment best.
        """
        segments = list(segments)
        faces = np.unique(np.concatenate([segment.faces for segment in segments]))
        poses = self._chosen_viewpoints(
            self.compute_camera_poses(
                self.scene_mesh.vertices[self.scene_mesh.faces[faces].ravel()]
            ),
            viewpoints,
        )

        # Ranked by the segment each viewpoint shows worst, so that a handle is not
        # outvoted by the door it is screwed to.
        middle = self.scene_mesh.vertices[
            self.scene_mesh.faces[faces].ravel()
        ].mean(axis=0)
        ranked = sorted(
            poses,
            key=lambda name: min(
                self.presented_area(
                    segment,
                    (middle - poses[name][:3, 3])
                    / np.linalg.norm(middle - poses[name][:3, 3]),
                )
                for segment in segments
            ),
            reverse=True,
        )
        looked_at = {name: poses[name] for name in ranked[: max(considered, 1)]}
        if len(looked_at) == 1:
            return next(iter(looked_at))

        alone = self.scene_mesh.submesh([faces], append=True)
        dimmed = self._dimmed_face_colors[faces]
        alone.visual.face_colors = dimmed
        rooms = self._render_from_poses(
            trimesh.Scene(alone), looked_at, headless, self.deciding_resolution
        )

        highlight = trimesh.visual.color.to_rgba(Color.distinct_colors(1)[0].to_rgba())
        worst: Dict[str, Optional[int]] = {name: None for name in looked_at}
        for segment in segments:
            painted = dimmed.copy()
            painted[np.searchsorted(faces, segment.faces)] = highlight
            alone.visual.face_colors = painted
            shown = self._render_from_poses(
                trimesh.Scene(alone), looked_at, headless, self.deciding_resolution
            )
            for name, image in shown.items():
                visible = changed_pixels(image, rooms[name])
                if worst[name] is None or visible < worst[name]:
                    worst[name] = visible
        return max(worst, key=lambda name: worst[name])

    def _dimmed_views(
        self,
        poses: Dict[str, HomogeneousTransformationMatrix],
        headless: bool,
        cacheable: bool,
    ) -> Dict[str, bytes]:
        """
        Render the scene dimmed and unhighlighted, from given poses.

        A context view is measured against this rather than against the scene in its own
        colors, because a highlight render dims everything else: measured against the
        undimmed scene every pixel differs, and the measurement reports the dimming
        instead of the highlight.

        :param poses: The camera poses to render from.
        :param headless: Whether to render without opening a window.
        :param cacheable: Whether these poses are the whole scene's, which every region
            shares and which is therefore worth rendering once.
        :return: Per viewpoint, the scene dimmed and unhighlighted.
        """
        key = tuple(sorted(poses))
        if cacheable and key in self._plain_views:
            return self._plain_views[key]

        self._apply_highlight_to_faces([])
        views = self._render_from_poses(self._render_scene(), poses, headless)
        self._reset_segment_colors()
        if cacheable:
            self._plain_views[key] = views
        return views

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
