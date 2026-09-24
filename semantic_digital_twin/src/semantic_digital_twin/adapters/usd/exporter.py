from __future__ import annotations

import logging
import math
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Dict, Generic, List, TypeVar

from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric

from semantic_digital_twin.adapters.usd.exceptions import (
    UnsupportedShapeForUsdExportError,
    UsdPrimNameCollisionError,
)
from semantic_digital_twin.adapters.usd.parser import UsdAxis, UsdPhysicsJointType
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.usd_semantics import UsdSemanticLabels
from semantic_digital_twin.spatial_types.spatial_types import RotationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Cylinder,
    Mesh,
    Shape,
    Sphere,
)
from semantic_digital_twin.world_description.world_entity import Body, Connection

logger = logging.getLogger(__name__)

try:
    from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdPhysics, UsdSemantics
except ImportError:
    logger.warning(
        "usd-core 24.11 or newer is required for USD export. Please install it using "
        "'pip install usd-core'"
    )

ShapeType = TypeVar("ShapeType", bound=Shape)


class UsdSemanticTaxonomy(StrEnum):
    """
    The ``UsdSemantics.LabelsAPI`` taxonomies the exporter writes labels under.
    """

    CLASS = "class"
    """
    The semantic class of a prim, the taxonomy Isaac Sim's semantic segmentation reads.
    """


class UsdPrimName(StrEnum):
    """
    Names of the prims the exporter adds besides one per body.
    """

    JOINTS = "joints"
    """
    The scope every physics joint is written under.
    """


# %% transforms


def _to_gf_matrix(matrix: NDArray[np.float64]) -> Gf.Matrix4d:
    """
    :param matrix: A 4x4 homogeneous transform acting on column vectors.
    :return: The same transform as a USD matrix, which acts on row vectors.
    """
    return Gf.Matrix4d(matrix.T.tolist())


def _set_local_transform(prim: Usd.Prim, parent_T_prim: NDArray[np.float64]) -> None:
    """
    :param prim: The prim to place.
    :param parent_T_prim: The prim's pose relative to its parent prim.
    """
    UsdGeom.Xformable(prim).AddTransformOp().Set(_to_gf_matrix(parent_T_prim))


# %% shapes


@dataclass
class UsdShapeWriter(Generic[ShapeType], SubClassSafeGeneric):
    """
    Writes one type of Shape as a USD geometry prim, positioned relative to the link
    prim it is written under.
    """

    shape: ShapeType
    """
    The shape to write.
    """

    @classmethod
    def for_shape(cls, shape: Shape, body: Body) -> UsdShapeWriter:
        """
        :param shape: The shape to write.
        :param body: The body the shape belongs to.
        :return: A writer for ``shape``.
        :raises UnsupportedShapeForUsdExportError: If no writer writes ``shape``'s type.
        """
        writer_types = cls.__subclasses__()
        for writer_type in writer_types:
            if isinstance(shape, writer_type.written_shape_type()):
                return writer_type(shape=shape)
        raise UnsupportedShapeForUsdExportError(
            body_name=str(body.name),
            shape_type=type(shape),
            supported_types=[
                writer_type.written_shape_type() for writer_type in writer_types
            ],
        )

    @classmethod
    def written_shape_type(cls) -> type[Shape]:
        """
        :return: The shape type this writer writes.
        """
        [shape_type] = cls.get_generic_type_parameters()
        return shape_type

    def write(self, stage: Usd.Stage, path: Sdf.Path) -> Usd.Prim:
        """
        Writes :attr:`shape` as a geometry prim.

        :param stage: The stage to write to.
        :param path: The path of the prim to define.
        :return: The written prim.
        """
        prim = self._define(stage, path)
        _set_local_transform(prim, self.shape.origin.to_np())
        return prim

    @abstractmethod
    def _define(self, stage: Usd.Stage, path: Sdf.Path) -> Usd.Prim:
        """
        Defines the geometry prim for :attr:`shape`, in the shape's own frame.

        :param stage: The stage to write to.
        :param path: The path of the prim to define.
        :return: The defined prim.
        """


@dataclass
class UsdBoxWriter(UsdShapeWriter[Box]):
    """
    Writes a Box as a unit ``UsdGeom.Cube`` scaled to the box's size.
    """

    def write(self, stage: Usd.Stage, path: Sdf.Path) -> Usd.Prim:
        prim = super().write(stage, path)
        scale = self.shape.scale
        UsdGeom.Xformable(prim).AddScaleOp().Set(Gf.Vec3f(scale.x, scale.y, scale.z))
        return prim

    def _define(self, stage: Usd.Stage, path: Sdf.Path) -> Usd.Prim:
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        return cube.GetPrim()


@dataclass
class UsdSphereWriter(UsdShapeWriter[Sphere]):
    """
    Writes a Sphere as a ``UsdGeom.Sphere``.
    """

    def _define(self, stage: Usd.Stage, path: Sdf.Path) -> Usd.Prim:
        sphere = UsdGeom.Sphere.Define(stage, path)
        sphere.CreateRadiusAttr(self.shape.radius)
        return sphere.GetPrim()


@dataclass
class UsdCylinderWriter(UsdShapeWriter[Cylinder]):
    """
    Writes a Cylinder as a ``UsdGeom.Cylinder`` along its local z axis.
    """

    def _define(self, stage: Usd.Stage, path: Sdf.Path) -> Usd.Prim:
        cylinder = UsdGeom.Cylinder.Define(stage, path)
        cylinder.CreateAxisAttr(UsdAxis.Z)
        cylinder.CreateRadiusAttr(self.shape.radius)
        cylinder.CreateHeightAttr(self.shape.height)
        return cylinder.GetPrim()


@dataclass
class UsdMeshWriter(UsdShapeWriter[Mesh]):
    """
    Writes a Mesh as a triangulated ``UsdGeom.Mesh``, with its scale applied to the
    points.
    """

    def _define(self, stage: Usd.Stage, path: Sdf.Path) -> Usd.Prim:
        triangles = self.shape.mesh
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr([Gf.Vec3f(*vertex) for vertex in triangles.vertices])
        mesh.CreateFaceVertexCountsAttr([3] * len(triangles.faces))
        mesh.CreateFaceVertexIndicesAttr(triangles.faces.reshape(-1).tolist())
        return mesh.GetPrim()


# %% exporter


@dataclass
class USDExporter:
    """
    Writes a world to a Universal Scene Description (USD) stage, as one physically
    articulated asset laid out the way :class:`~semantic_digital_twin.adapters.usd.parser.USDParser`
    reads it and Isaac Sim simulates it.

    The stage's default prim is the articulation root. Every body becomes a rigid-body
    link prim directly under it, placed at the body's current pose relative to the
    world's root. Every connection becomes a physics joint, and the world's root is
    fixed to the stage by a joint without a ``body0``, so the asset has a fixed base.
    Regions have no physical presence and are not written.

    Geometry that is both visual and collision geometry is written once, as a collider.
    Visual-only geometry has no :class:`~pxr.UsdPhysics.CollisionAPI`, and
    collision-only geometry has the ``guide`` purpose, which keeps it from being
    rendered.

    Every annotation rooted at a body labels the body's prim through
    ``UsdSemantics.LabelsAPI``: a
    :class:`~semantic_digital_twin.semantic_annotations.usd_semantics.UsdSemanticLabels`
    under its own taxonomy, any other annotation by its class name under
    :attr:`UsdSemanticTaxonomy.CLASS`.

    .. note::
        Requires the ``usd-core`` package (``pxr``), version 24.11 or newer.

    .. note::
        Parsing the stage back yields one body more than the world has: the default
        prim becomes the parsed world's root, with the world's root fixed to it.
    """

    world: World
    """
    The world to export.
    """

    default_prim_name: str = field(kw_only=True)
    """
    The name of the stage's default prim, which names the asset.
    """

    # %% entry points

    def export(self, file_path: str) -> None:
        """
        Writes the stage :meth:`build_stage` builds to a file.

        :param file_path: The path of the file to write, whose extension (``.usda``,
            ``.usdc``, ``.usd``) selects the file format.
        """
        stage = self.build_stage()
        stage.GetRootLayer().Export(file_path)

    def build_stage(self) -> Usd.Stage:
        """
        Builds an in-memory stage describing :attr:`world`.

        :return: The built stage.
        :raises UsdPrimNameCollisionError: If two bodies, or a body and the default
            prim, share a prim name.
        :raises UnsupportedConnectionForUsdExportError: If a connection has no USD
            physics joint counterpart.
        :raises UnsupportedShapeForUsdExportError: If a shape has no USD geometry
            counterpart.
        """
        prim_names = self._prim_names_by_body()
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.LinearUnits.meters)

        root_path = Sdf.Path.absoluteRootPath.AppendChild(self.default_prim_name)
        default_prim = UsdGeom.Xform.Define(stage, root_path).GetPrim()
        stage.SetDefaultPrim(default_prim)
        UsdPhysics.ArticulationRootAPI.Apply(default_prim)

        link_paths = {
            body: root_path.AppendChild(prim_name)
            for body, prim_name in prim_names.items()
        }
        for body, link_path in link_paths.items():
            self._write_link(stage, link_path, body)

        joints_path = root_path.AppendChild(UsdPrimName.JOINTS)
        UsdGeom.Scope.Define(stage, joints_path)
        self._write_base_joint(stage, joints_path, link_paths[self.world.root])
        for connection in self.world.connections:
            if connection.child not in link_paths:
                continue
            self._write_joint(stage, joints_path, connection, link_paths)
        return stage

    # %% names

    def _prim_names_by_body(self) -> Dict[Body, str]:
        """
        :return: The prim name of every body.
        :raises UsdPrimNameCollisionError: If two bodies, or a body and the default
            prim, share a prim name.
        """
        prim_names = {
            body: Tf.MakeValidIdentifier(body.name.name) for body in self.world.bodies
        }
        entity_names_by_prim_name: Dict[str, List[str]] = {
            self.default_prim_name: ["the default prim"]
        }
        for body, prim_name in prim_names.items():
            entity_names_by_prim_name.setdefault(prim_name, []).append(str(body.name))
        for prim_name, entity_names in entity_names_by_prim_name.items():
            if len(entity_names) > 1:
                raise UsdPrimNameCollisionError(
                    prim_name=prim_name, entity_names=entity_names
                )
        return prim_names

    # %% links

    def _write_link(self, stage: Usd.Stage, link_path: Sdf.Path, body: Body) -> None:
        """
        Writes a body as a rigid-body link prim with its geometry, inertial properties
        and semantic labels.

        :param stage: The stage to write to.
        :param link_path: The path of the link prim.
        :param body: The body to write.
        """
        link_prim = UsdGeom.Xform.Define(stage, link_path).GetPrim()
        _set_local_transform(
            link_prim, self.world.compute_forward_kinematics_np(self.world.root, body)
        )
        UsdPhysics.RigidBodyAPI.Apply(link_prim)
        self._write_geometry(stage, link_path, body)
        self._write_inertial(link_prim, body)
        self._write_semantic_labels(link_prim, body)

    @staticmethod
    def _write_geometry(stage: Usd.Stage, link_path: Sdf.Path, body: Body) -> None:
        """
        Writes a body's visual and collision shapes under its link prim.

        :param stage: The stage to write to.
        :param link_path: The path of the body's link prim.
        :param body: The body whose shapes to write.
        """
        visual_shape_ids = {id(shape) for shape in body.visual}
        collision_shape_ids = {id(shape) for shape in body.collision}
        collision_only_shapes = [
            shape for shape in body.collision if id(shape) not in visual_shape_ids
        ]
        for index, shape in enumerate(body.visual.shapes + collision_only_shapes):
            prim = UsdShapeWriter.for_shape(shape, body).write(
                stage, link_path.AppendChild(f"shape_{index}")
            )
            if id(shape) not in collision_shape_ids:
                continue
            UsdPhysics.CollisionAPI.Apply(prim)
            if id(shape) not in visual_shape_ids:
                UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)

    @staticmethod
    def _write_inertial(link_prim: Usd.Prim, body: Body) -> None:
        """
        Writes a body's inertial properties as the link prim's
        :class:`~pxr.UsdPhysics.MassAPI`.

        :param link_prim: The body's link prim.
        :param body: The body whose inertial properties to write.
        """
        inertial = body.inertial
        if inertial is None:
            return
        moments, axes = inertial.inertia.to_principal_moments_and_axes()
        axes_rotation = Gf.Matrix4d(1.0)
        axes_rotation.SetRotateOnly(Gf.Matrix3d(axes.data.T.tolist()))
        mass_api = UsdPhysics.MassAPI.Apply(link_prim)
        mass_api.CreateMassAttr(inertial.mass)
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*inertial.center_of_mass.to_np()[:3]))
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*moments.data))
        mass_api.CreatePrincipalAxesAttr(Gf.Quatf(axes_rotation.ExtractRotationQuat()))

    def _write_semantic_labels(self, link_prim: Usd.Prim, body: Body) -> None:
        """
        Labels a body's link prim with every annotation rooted at the body.

        :param link_prim: The body's link prim.
        :param body: The body whose annotations to write.
        """
        labels_by_taxonomy: Dict[str, List[str]] = {}
        for annotation in self.world.semantic_annotations:
            if not isinstance(annotation, HasRootBody) or annotation.root is not body:
                continue
            if isinstance(annotation, UsdSemanticLabels):
                taxonomy, labels = annotation.taxonomy, annotation.labels
            else:
                taxonomy, labels = UsdSemanticTaxonomy.CLASS, [
                    type(annotation).__name__
                ]
            taxonomy_labels = labels_by_taxonomy.setdefault(taxonomy, [])
            taxonomy_labels.extend(
                label for label in labels if label not in taxonomy_labels
            )
        for taxonomy, labels in labels_by_taxonomy.items():
            UsdSemantics.LabelsAPI.Apply(link_prim, taxonomy).CreateLabelsAttr(labels)

    # %% joints

    @staticmethod
    def _write_base_joint(
        stage: Usd.Stage, joints_path: Sdf.Path, root_link_path: Sdf.Path
    ) -> None:
        """
        Fixes the world's root link to the stage.

        :param stage: The stage to write to.
        :param joints_path: The path of the scope joints are written under.
        :param root_link_path: The path of the world root's link prim.
        """
        joint = UsdPhysics.FixedJoint.Define(
            stage, joints_path.AppendChild(f"{root_link_path.name}_joint")
        )
        joint.CreateBody1Rel().SetTargets([root_link_path])

    def _write_joint(
        self,
        stage: Usd.Stage,
        joints_path: Sdf.Path,
        connection: Connection,
        link_paths: Dict[Body, Sdf.Path],
    ) -> None:
        """
        Writes a connection as the physics joint between its parent's and its child's
        link prims.

        A USD joint moves along the x axis of its joint frame, so the frame of a
        one-degree-of-freedom connection is rotated to align its axis with x, and the
        joint limits are the ones the connection's multiplier and offset apply to its
        degree of freedom.

        :param stage: The stage to write to.
        :param joints_path: The path of the scope joints are written under.
        :param connection: The connection to write.
        :param link_paths: The path of every body's link prim.
        :raises UnsupportedConnectionForUsdExportError: If the connection has no USD
            physics joint counterpart.
        """
        joint_type = UsdPhysicsJointType.for_connection(connection)
        child_link_path = link_paths[connection.child]
        joint_path = joints_path.AppendChild(f"{child_link_path.name}_joint")
        joint = UsdPhysics.Joint(stage.DefinePrim(joint_path, joint_type.value))
        joint.CreateBody0Rel().SetTargets([link_paths[connection.parent]])
        joint.CreateBody1Rel().SetTargets([child_link_path])

        parent_T_joint = connection.parent_T_connection_expression.to_np()
        child_T_joint = connection.connection_T_child_expression.inverse().to_np()
        if isinstance(connection, ActiveConnection1DOF):
            connection_T_joint = RotationMatrix.from_x_axis(connection.axis).to_np()
            parent_T_joint = parent_T_joint @ connection_T_joint
            child_T_joint = child_T_joint @ connection_T_joint
            self._write_axis_and_limits(joint.GetPrim(), connection)
        self._write_joint_frame(joint, parent_T_joint, child_T_joint)

    @staticmethod
    def _write_joint_frame(
        joint: UsdPhysics.Joint,
        parent_T_joint: NDArray[np.float64],
        child_T_joint: NDArray[np.float64],
    ) -> None:
        """
        Writes a joint frame's pose relative to both bodies it connects.

        :param joint: The joint to write to.
        :param parent_T_joint: The joint frame's pose relative to ``body0``.
        :param child_T_joint: The joint frame's pose relative to ``body1``.
        """
        parent_matrix = _to_gf_matrix(parent_T_joint)
        child_matrix = _to_gf_matrix(child_T_joint)
        joint.CreateLocalPos0Attr(Gf.Vec3f(parent_matrix.ExtractTranslation()))
        joint.CreateLocalRot0Attr(Gf.Quatf(parent_matrix.ExtractRotationQuat()))
        joint.CreateLocalPos1Attr(Gf.Vec3f(child_matrix.ExtractTranslation()))
        joint.CreateLocalRot1Attr(Gf.Quatf(child_matrix.ExtractRotationQuat()))

    @staticmethod
    def _write_axis_and_limits(
        joint_prim: Usd.Prim, connection: ActiveConnection1DOF
    ) -> None:
        """
        Writes a one-degree-of-freedom joint's axis and its limits, in degrees for a
        revolute joint and in meters for a prismatic one.

        :param joint_prim: The joint prim to write to.
        :param connection: The connection the joint describes.
        """
        axis_joint = (
            UsdPhysics.RevoluteJoint(joint_prim)
            if isinstance(connection, RevoluteConnection)
            else UsdPhysics.PrismaticJoint(joint_prim)
        )
        axis_joint.CreateAxisAttr(UsdAxis.X)
        limits = connection.dof.limits
        for limit, create_attribute in (
            (limits.lower.position, axis_joint.CreateLowerLimitAttr),
            (limits.upper.position, axis_joint.CreateUpperLimitAttr),
        ):
            if limit is None:
                continue
            if isinstance(connection, RevoluteConnection):
                limit = math.degrees(limit)
            create_attribute(limit)
