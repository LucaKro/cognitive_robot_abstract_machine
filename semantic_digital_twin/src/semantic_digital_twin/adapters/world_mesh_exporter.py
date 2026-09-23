"""Extract labeled triangle meshes from a semantic digital twin world."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from uuid import UUID

import numpy as np
import trimesh
from krrood.utils import get_full_class_name
from numpy.typing import DTypeLike

from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootKinematicStructureEntity,
    IsPerceivable,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

# %% geometry selection


class GeometrySource(StrEnum):
    """Geometry collection used to construct exported body meshes."""

    VISUAL = "visual"
    """Use visual geometry only."""

    COLLISION = "collision"
    """Use collision geometry only."""

    VISUAL_WITH_COLLISION_FALLBACK = "visual_with_collision_fallback"
    """Use visual geometry when present and collision geometry otherwise."""


class SemanticAnnotationRelation(StrEnum):
    """How a body participates in a semantic annotation."""

    DIRECT = "direct"
    """The body is the annotation's root entity."""

    CONTEXT = "context"
    """The body belongs to the annotation without being its root entity."""


@dataclass(frozen=True)
class SelectedBodyGeometry:
    """Geometry selected from one body for a mesh snapshot."""

    shape_collection: ShapeCollection
    """The selected body-local shapes."""

    source: GeometrySource | None
    """The collection that supplied shapes, or ``None`` when none were available."""


# %% extracted mesh records


@dataclass(frozen=True)
class BodyMeshSnapshot:
    """A body's geometry and identity at one world state."""

    identifier: int
    """The body identifier within this export."""

    name: str
    """The body's fully qualified name."""

    source_identifier: UUID
    """The body's identifier in the source world."""

    parent_identifier: int | None
    """The parent body's export identifier, when the parent is a body."""

    direct_semantic_annotation_identifiers: tuple[int, ...]
    """Annotations for which this body is the root entity."""

    context_semantic_annotation_identifiers: tuple[int, ...]
    """Annotations that contain this body without using it as their root."""

    used_geometry_source: GeometrySource | None
    """The collection from which geometry was extracted."""

    shape_types: tuple[str, ...]
    """The concrete shape types combined into :attr:`local_mesh`."""

    local_mesh: trimesh.Trimesh | None
    """The body's selected shapes expressed in the body frame."""

    world_transform: np.ndarray
    """The body's transform from its local frame to the world root frame."""

    @property
    def world_mesh(self) -> trimesh.Trimesh | None:
        """Return a copy of the body mesh expressed in the world root frame."""
        if self.local_mesh is None:
            return None
        mesh = self.local_mesh.copy()
        mesh.apply_transform(self.world_transform)
        return mesh


@dataclass(frozen=True)
class WorldMeshSnapshot:
    """Per-body meshes extracted from one world state."""

    body_meshes: tuple[BodyMeshSnapshot, ...]
    """The extracted bodies in topological order."""

    semantic_annotations: tuple[SemanticAnnotationSnapshot, ...]
    """The world's semantic annotations in registration order."""


@dataclass(frozen=True)
class SemanticAnnotationSnapshot:
    """Serializable identity and body membership of one semantic annotation."""

    identifier: int
    """The annotation identifier within this export."""

    name: str
    """The annotation instance's fully qualified name."""

    source_identifier: UUID
    """The annotation's identifier in the source world."""

    type_name: str
    """The concrete semantic annotation class name."""

    qualified_type_name: str
    """The import-qualified concrete annotation class name."""

    root_body_identifier: int | None
    """The root body's export identifier, when the root entity is a body."""

    body_identifiers: tuple[int, ...]
    """All exported bodies participating in this annotation."""

    class_label: str | None
    """The external perception label when the annotation supplies one."""


# %% world extraction


@dataclass(frozen=True)
class WorldMeshExtractor:
    """Extract body-local meshes and current transforms from a world."""

    geometry_source: GeometrySource = GeometrySource.VISUAL_WITH_COLLISION_FALLBACK
    """The policy used to select geometry from every body."""

    def extract(self, world: World) -> WorldMeshSnapshot:
        """Capture every body and its selected geometry at the current world state."""
        bodies = world.bodies_topologically_sorted
        identifier_by_body = {
            body: identifier for identifier, body in enumerate(bodies)
        }
        semantic_annotations = tuple(
            self._extract_semantic_annotation(
                annotation,
                identifier,
                identifier_by_body,
            )
            for identifier, annotation in enumerate(
                world.semantic_annotations
            )
        )
        direct_annotations_by_body = {body: [] for body in bodies}
        context_annotations_by_body = {body: [] for body in bodies}
        body_by_identifier = {
            identifier: body for body, identifier in identifier_by_body.items()
        }
        for annotation in semantic_annotations:
            if annotation.root_body_identifier is not None:
                root_body = body_by_identifier[annotation.root_body_identifier]
                direct_annotations_by_body[root_body].append(annotation.identifier)
            for body_identifier in annotation.body_identifiers:
                if body_identifier == annotation.root_body_identifier:
                    continue
                body = body_by_identifier[body_identifier]
                context_annotations_by_body[body].append(annotation.identifier)

        return WorldMeshSnapshot(
            body_meshes=tuple(
                self._extract_body(
                    world,
                    body,
                    identifier_by_body,
                    tuple(direct_annotations_by_body[body]),
                    tuple(context_annotations_by_body[body]),
                )
                for body in bodies
            ),
            semantic_annotations=semantic_annotations,
        )

    @staticmethod
    def _extract_semantic_annotation(
        annotation: SemanticAnnotation,
        identifier: int,
        identifier_by_body: dict[Body, int],
    ) -> SemanticAnnotationSnapshot:
        """Capture one annotation's type and relationships to exported bodies."""
        root_body = (
            annotation.root
            if isinstance(annotation, HasRootKinematicStructureEntity)
            and isinstance(annotation.root, Body)
            else None
        )
        body_identifiers = tuple(
            sorted(
                {
                    identifier_by_body[body]
                    for body in annotation.bodies
                    if body in identifier_by_body
                }
            )
        )
        return SemanticAnnotationSnapshot(
            identifier=identifier,
            name=str(annotation.name),
            source_identifier=annotation.id,
            type_name=type(annotation).__name__,
            qualified_type_name=get_full_class_name(type(annotation)),
            root_body_identifier=(
                identifier_by_body[root_body] if root_body is not None else None
            ),
            body_identifiers=body_identifiers,
            class_label=(
                annotation.class_label
                if isinstance(annotation, IsPerceivable)
                else None
            ),
        )

    def _extract_body(
        self,
        world: World,
        body: Body,
        identifier_by_body: dict[Body, int],
        direct_semantic_annotation_identifiers: tuple[int, ...],
        context_semantic_annotation_identifiers: tuple[int, ...],
    ) -> BodyMeshSnapshot:
        """Capture one body's own shapes without including child bodies."""
        selected_geometry = self._select_geometry(body)
        local_mesh = (
            selected_geometry.shape_collection.combined_mesh.copy()
            if selected_geometry.source is not None
            else None
        )
        parent = None if body is world.root else body.parent_kinematic_structure_entity
        parent_identifier = (
            identifier_by_body[parent] if isinstance(parent, Body) else None
        )
        world_transform = (
            np.eye(4)
            if body is world.root
            else world.compute_forward_kinematics_np(world.root, body).copy()
        )
        return BodyMeshSnapshot(
            identifier=identifier_by_body[body],
            name=str(body.name),
            source_identifier=body.id,
            parent_identifier=parent_identifier,
            direct_semantic_annotation_identifiers=(
                direct_semantic_annotation_identifiers
            ),
            context_semantic_annotation_identifiers=(
                context_semantic_annotation_identifiers
            ),
            used_geometry_source=selected_geometry.source,
            shape_types=tuple(
                type(shape).__name__
                for shape in selected_geometry.shape_collection.shapes
            ),
            local_mesh=local_mesh,
            world_transform=world_transform,
        )

    def _select_geometry(self, body: Body) -> SelectedBodyGeometry:
        """Select one of a body's visual or collision shape collections."""
        if self.geometry_source is GeometrySource.VISUAL:
            return self._selection(body.visual, GeometrySource.VISUAL)
        if self.geometry_source is GeometrySource.COLLISION:
            return self._selection(body.collision, GeometrySource.COLLISION)
        if body.visual:
            return SelectedBodyGeometry(body.visual, GeometrySource.VISUAL)
        return self._selection(body.collision, GeometrySource.COLLISION)

    @staticmethod
    def _selection(
        shape_collection: ShapeCollection, geometry_source: GeometrySource
    ) -> SelectedBodyGeometry:
        """Describe a shape collection while representing an empty one explicitly."""
        return SelectedBodyGeometry(
            shape_collection=shape_collection,
            source=geometry_source if shape_collection else None,
        )


# %% GLB and manifest export


@dataclass(frozen=True)
class WorldMeshExportPaths:
    """Paths produced by :class:`WorldMeshExporter`."""

    scene: Path
    """The GLB scene path."""

    manifest: Path
    """The JSON manifest path."""

    labeled_mesh: Path
    """The combined labeled NumPy archive path."""

    bodies_directory: Path
    """The directory containing standalone body meshes."""


@dataclass(frozen=True)
class WorldMeshExporter:
    """Serialize a world mesh snapshot without merging its body meshes."""

    scene_file_name: str = "scene.glb"
    """File name used for the GLB scene."""

    manifest_file_name: str = "manifest.json"
    """File name used for the JSON manifest."""

    labeled_mesh_file_name: str = "mesh.npz"
    """File name used for the combined labeled mesh arrays."""

    bodies_directory_name: str = "bodies"
    """Directory name used for standalone body meshes."""

    base_frame: str = "__semantic_digital_twin_world__"
    """Synthetic GLB node above all source-world bodies."""

    def export(
        self, snapshot: WorldMeshSnapshot, output_directory: Path
    ) -> WorldMeshExportPaths:
        """Write a reloadable GLB scene and its body-label manifest."""
        output_directory.mkdir(parents=True, exist_ok=True)
        paths = WorldMeshExportPaths(
            scene=output_directory / self.scene_file_name,
            manifest=output_directory / self.manifest_file_name,
            labeled_mesh=output_directory / self.labeled_mesh_file_name,
            bodies_directory=output_directory / self.bodies_directory_name,
        )
        paths.bodies_directory.mkdir(parents=True, exist_ok=True)
        scene = self.to_scene(snapshot)
        scene.export(paths.scene, file_type="glb")
        self._export_body_meshes(snapshot, output_directory)
        np.savez_compressed(
            paths.labeled_mesh,
            **self._labeled_mesh_arrays(snapshot),  # type: ignore[arg-type]
        )
        paths.manifest.write_text(
            json.dumps(self._manifest(snapshot), indent=2) + "\n", encoding="utf-8"
        )
        return paths

    def _export_body_meshes(
        self, snapshot: WorldMeshSnapshot, output_directory: Path
    ) -> None:
        """Write every nonempty body as a world-aligned PLY mesh."""
        for body in snapshot.body_meshes:
            mesh = body.world_mesh
            if mesh is None:
                continue
            mesh.export(output_directory / self._body_mesh_relative_path(body))

    @staticmethod
    def _labeled_mesh_arrays(
        snapshot: WorldMeshSnapshot,
    ) -> dict[str, np.ndarray]:
        """Combine world meshes while retaining face and vertex body labels."""
        vertices: list[np.ndarray] = []
        faces: list[np.ndarray] = []
        vertex_body_ids: list[np.ndarray] = []
        face_body_ids: list[np.ndarray] = []
        vertex_offset = 0
        for body in snapshot.body_meshes:
            mesh = body.world_mesh
            if mesh is None:
                continue
            body_vertices = np.asarray(mesh.vertices)
            body_faces = np.asarray(mesh.faces, dtype=np.int64)
            vertices.append(body_vertices)
            faces.append(body_faces + vertex_offset)
            vertex_body_ids.append(
                np.full(len(body_vertices), body.identifier, dtype=np.int64)
            )
            face_body_ids.append(
                np.full(len(body_faces), body.identifier, dtype=np.int64)
            )
            vertex_offset += len(body_vertices)

        body_annotation_body_ids: list[int] = []
        body_annotation_ids: list[int] = []
        body_annotation_relations: list[SemanticAnnotationRelation] = []
        for body in snapshot.body_meshes:
            for annotation_identifier in body.direct_semantic_annotation_identifiers:
                body_annotation_body_ids.append(body.identifier)
                body_annotation_ids.append(annotation_identifier)
                body_annotation_relations.append(SemanticAnnotationRelation.DIRECT)
            for annotation_identifier in body.context_semantic_annotation_identifiers:
                body_annotation_body_ids.append(body.identifier)
                body_annotation_ids.append(annotation_identifier)
                body_annotation_relations.append(SemanticAnnotationRelation.CONTEXT)

        return {
            "vertices": WorldMeshExporter._concatenate_or_empty(
                vertices, (0, 3), np.float64
            ),
            "faces": WorldMeshExporter._concatenate_or_empty(faces, (0, 3), np.int64),
            "vertex_body_ids": WorldMeshExporter._concatenate_or_empty(
                vertex_body_ids, (0,), np.int64
            ),
            "face_body_ids": WorldMeshExporter._concatenate_or_empty(
                face_body_ids, (0,), np.int64
            ),
            "body_ids": np.asarray(
                [body.identifier for body in snapshot.body_meshes], dtype=np.int64
            ),
            "body_names": np.asarray(
                [body.name for body in snapshot.body_meshes], dtype=np.str_
            ),
            "body_source_ids": np.asarray(
                [str(body.source_identifier) for body in snapshot.body_meshes],
                dtype=np.str_,
            ),
            "body_parent_ids": np.asarray(
                [
                    -1 if body.parent_identifier is None else body.parent_identifier
                    for body in snapshot.body_meshes
                ],
                dtype=np.int64,
            ),
            "body_world_transforms": np.asarray(
                [body.world_transform for body in snapshot.body_meshes],
                dtype=np.float64,
            ),
            "annotation_ids": np.asarray(
                [annotation.identifier for annotation in snapshot.semantic_annotations],
                dtype=np.int64,
            ),
            "annotation_names": np.asarray(
                [annotation.name for annotation in snapshot.semantic_annotations],
                dtype=np.str_,
            ),
            "annotation_source_ids": np.asarray(
                [
                    str(annotation.source_identifier)
                    for annotation in snapshot.semantic_annotations
                ],
                dtype=np.str_,
            ),
            "annotation_type_names": np.asarray(
                [annotation.type_name for annotation in snapshot.semantic_annotations],
                dtype=np.str_,
            ),
            "annotation_qualified_type_names": np.asarray(
                [
                    annotation.qualified_type_name
                    for annotation in snapshot.semantic_annotations
                ],
                dtype=np.str_,
            ),
            "annotation_root_body_ids": np.asarray(
                [
                    -1
                    if annotation.root_body_identifier is None
                    else annotation.root_body_identifier
                    for annotation in snapshot.semantic_annotations
                ],
                dtype=np.int64,
            ),
            "annotation_class_labels": np.asarray(
                [
                    annotation.class_label or ""
                    for annotation in snapshot.semantic_annotations
                ],
                dtype=np.str_,
            ),
            "body_annotation_body_ids": np.asarray(
                body_annotation_body_ids, dtype=np.int64
            ),
            "body_annotation_ids": np.asarray(body_annotation_ids, dtype=np.int64),
            "body_annotation_relations": np.asarray(
                body_annotation_relations, dtype=np.str_
            ),
        }

    @staticmethod
    def _concatenate_or_empty(
        arrays: list[np.ndarray], empty_shape: tuple[int, ...], dtype: DTypeLike
    ) -> np.ndarray:
        """Concatenate array chunks or create a correctly shaped empty array."""
        if arrays:
            return np.concatenate(arrays).astype(dtype, copy=False)
        return np.empty(empty_shape, dtype=dtype)

    def to_scene(self, snapshot: WorldMeshSnapshot) -> trimesh.Scene:
        """Convert a snapshot to a scene with one named node per body."""
        body_by_identifier = {body.identifier: body for body in snapshot.body_meshes}
        self._validate_node_names(snapshot)
        scene = trimesh.Scene(base_frame=self.base_frame)
        for body in snapshot.body_meshes:
            parent = (
                body_by_identifier.get(body.parent_identifier)
                if body.parent_identifier is not None
                else None
            )
            parent_node_name = self.base_frame if parent is None else parent.name
            transform = self._relative_transform(body, parent)
            if body.local_mesh is None:
                scene.graph.update(
                    frame_from=parent_node_name,
                    frame_to=body.name,
                    matrix=transform,
                )
                continue
            scene.add_geometry(
                body.local_mesh.copy(),
                node_name=body.name,
                geom_name=self._geometry_name(body),
                parent_node_name=parent_node_name,
                transform=transform,
            )
        return scene

    def _validate_node_names(self, snapshot: WorldMeshSnapshot) -> None:
        """Reject body names that cannot uniquely identify GLB nodes."""
        names = [body.name for body in snapshot.body_meshes]
        if self.base_frame in names:
            raise ValueError(
                f"Body name conflicts with GLB base frame: {self.base_frame}"
            )
        duplicate_names = sorted({name for name in names if names.count(name) > 1})
        if duplicate_names:
            raise ValueError(f"Duplicate body names: {duplicate_names}")

    @staticmethod
    def _relative_transform(
        body: BodyMeshSnapshot, parent: BodyMeshSnapshot | None
    ) -> np.ndarray:
        """Return the body's transform relative to its exported parent node."""
        if parent is None:
            return body.world_transform
        return np.linalg.solve(parent.world_transform, body.world_transform)

    @staticmethod
    def _geometry_name(body: BodyMeshSnapshot) -> str:
        """Return the stable GLB geometry name for a body."""
        return f"body_{body.identifier}"

    def _body_mesh_relative_path(self, body: BodyMeshSnapshot) -> Path:
        """Return a unique, filesystem-safe path for a body's PLY mesh."""
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", body.name).strip("._")
        safe_name = safe_name or "body"
        return Path(self.bodies_directory_name) / (
            f"{body.identifier:06d}_{safe_name}.ply"
        )

    def _manifest(self, snapshot: WorldMeshSnapshot) -> dict[str, object]:
        """Build the JSON-compatible sidecar manifest."""
        return {
            "schema_version": 2,
            "scene": self.scene_file_name,
            "labeled_mesh": self.labeled_mesh_file_name,
            "body_mesh_directory": self.bodies_directory_name,
            "mesh_coordinate_frame": "world_root",
            "base_frame": self.base_frame,
            "bodies": [self._body_manifest(body) for body in snapshot.body_meshes],
            "semantic_annotations": [
                self._semantic_annotation_manifest(annotation)
                for annotation in snapshot.semantic_annotations
            ],
        }

    def _body_manifest(self, body: BodyMeshSnapshot) -> dict[str, object]:
        """Build one body's JSON-compatible manifest record."""
        mesh = body.local_mesh
        return {
            "id": body.identifier,
            "name": body.name,
            "source_id": str(body.source_identifier),
            "parent_id": body.parent_identifier,
            "direct_semantic_annotation_ids": list(
                body.direct_semantic_annotation_identifiers
            ),
            "context_semantic_annotation_ids": list(
                body.context_semantic_annotation_identifiers
            ),
            "node_name": body.name,
            "geometry_name": self._geometry_name(body) if mesh is not None else None,
            "mesh_file": (
                self._body_mesh_relative_path(body).as_posix()
                if mesh is not None
                else None
            ),
            "geometry_source": (
                body.used_geometry_source.value
                if body.used_geometry_source is not None
                else None
            ),
            "shape_types": list(body.shape_types),
            "vertex_count": len(mesh.vertices) if mesh is not None else 0,
            "face_count": len(mesh.faces) if mesh is not None else 0,
            "world_transform": body.world_transform.tolist(),
        }

    @staticmethod
    def _semantic_annotation_manifest(
        annotation: SemanticAnnotationSnapshot,
    ) -> dict[str, object]:
        """Build one semantic annotation's JSON-compatible manifest record."""
        return {
            "id": annotation.identifier,
            "name": annotation.name,
            "source_id": str(annotation.source_identifier),
            "type_name": annotation.type_name,
            "qualified_type_name": annotation.qualified_type_name,
            "root_body_id": annotation.root_body_identifier,
            "body_ids": list(annotation.body_identifiers),
            "class_label": annotation.class_label,
        }
