import numpy as np
import pytest

from semantic_digital_twin.adapters.usd.exceptions import (
    UnsupportedConnectionForUsdExportError,
    UnsupportedShapeForUsdExportError,
    UsdPrimNameCollisionError,
)
from semantic_digital_twin.adapters.usd.exporter import (
    USDExporter,
    UsdSemanticTaxonomy,
)
from semantic_digital_twin.adapters.usd.parser import USDParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle
from semantic_digital_twin.semantic_annotations.usd_semantics import UsdSemanticLabels
from semantic_digital_twin.spatial_types.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    Connection6DoF,
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Mesh, Sphere
from semantic_digital_twin.world_description.world_entity import Body

from .usd_export_worlds import (
    UnwritableShape,
    build_body_with_unwritable_shape,
    joint_limits,
    build_body_with_inertial,
    build_body_with_separate_collision,
    build_cabinet_world,
    build_mesh_body,
    build_primitive_shapes_body,
    build_single_body_world,
    mesh_vertices_in_body_frame,
)
from .usd_stages import PXR_AVAILABLE, USD_SEMANTICS_AVAILABLE

if PXR_AVAILABLE:
    from pxr import Usd, UsdGeom, UsdPhysics

pytestmark = pytest.mark.skipif(
    not PXR_AVAILABLE, reason="usd-core (pxr) not installed"
)

DEFAULT_PRIM_NAME = "asset"


def build_stage(world: World) -> Usd.Stage:
    return USDExporter(world=world, default_prim_name=DEFAULT_PRIM_NAME).build_stage()


def round_trip(world: World) -> World:
    return USDParser(stage=build_stage(world), prefix=DEFAULT_PRIM_NAME).parse()


def connection_to(world: World, child_name: str):
    child = world.get_body_by_name(child_name)
    [connection] = [
        connection for connection in world.connections if connection.child is child
    ]
    return connection


def body_names(world: World) -> set[str]:
    return {body.name.name for body in world.bodies}


# %% stage conventions


def test_export_declares_the_worlds_z_up_meter_convention():
    stage = build_stage(build_cabinet_world())

    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == UsdGeom.LinearUnits.meters


def test_export_makes_the_default_prim_the_articulation_root():
    stage = build_stage(build_cabinet_world())

    default_prim = stage.GetDefaultPrim()
    assert default_prim.GetName() == DEFAULT_PRIM_NAME
    assert default_prim.HasAPI(UsdPhysics.ArticulationRootAPI)


def test_export_makes_every_body_a_rigid_body():
    world = build_cabinet_world()
    stage = build_stage(world)

    rigid_body_names = {
        prim.GetName()
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    }
    assert rigid_body_names == body_names(world)


# %% kinematics


def test_round_trip_keeps_every_body_under_the_default_prim():
    world = build_cabinet_world()

    parsed = round_trip(world)

    assert body_names(parsed) == body_names(world) | {DEFAULT_PRIM_NAME}
    assert parsed.root.name.name == DEFAULT_PRIM_NAME


def test_round_trip_anchors_the_root_body_to_the_default_prim():
    world = build_cabinet_world()

    parsed = round_trip(world)

    anchor = connection_to(parsed, world.root.name.name)
    assert isinstance(anchor, FixedConnection)
    assert anchor.parent is parsed.root
    np.testing.assert_allclose(
        parsed.compute_forward_kinematics_np(anchor.parent, anchor.child),
        np.eye(4),
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "child_name",
    ["door", "drawer", "knob"],
)
def test_round_trip_keeps_the_connection_type(child_name):
    world = build_cabinet_world()

    parsed = round_trip(world)

    assert type(connection_to(parsed, child_name)) is type(
        connection_to(world, child_name)
    )


@pytest.mark.parametrize("child_name", ["door", "drawer"])
def test_round_trip_keeps_the_joint_limits(child_name):
    world = build_cabinet_world()
    original_limits = connection_to(world, child_name).dof.limits

    parsed_limits = connection_to(round_trip(world), child_name).dof.limits

    assert parsed_limits.lower.position == pytest.approx(original_limits.lower.position)
    assert parsed_limits.upper.position == pytest.approx(original_limits.upper.position)


@pytest.mark.parametrize("child_name", ["door", "drawer"])
def test_round_trip_keeps_the_kinematics_across_the_joint_range(child_name):
    world = build_cabinet_world()
    parsed = round_trip(world)
    original_connection = connection_to(world, child_name)
    parsed_connection = connection_to(parsed, child_name)
    limits = original_connection.dof.limits

    for position in np.linspace(limits.lower.position, limits.upper.position, 5):
        original_connection.position = position
        parsed_connection.position = position
        for tip_name in ("door", "drawer", "knob"):
            np.testing.assert_allclose(
                parsed.compute_forward_kinematics_np(
                    parsed.get_body_by_name("carcass"),
                    parsed.get_body_by_name(tip_name),
                ),
                world.compute_forward_kinematics_np(
                    world.root, world.get_body_by_name(tip_name)
                ),
                atol=1e-6,
            )


def test_export_places_every_link_at_its_current_pose():
    world = build_cabinet_world()
    connection_to(world, "door").position = -1.0
    knob = world.get_body_by_name("knob")

    stage = build_stage(world)

    knob_prim = stage.GetPrimAtPath(f"/{DEFAULT_PRIM_NAME}/knob")
    stage_T_knob = np.array(
        UsdGeom.Xformable(knob_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
    ).T
    np.testing.assert_allclose(
        stage_T_knob,
        world.compute_forward_kinematics_np(world.root, knob),
        atol=1e-6,
    )


def test_export_folds_a_connections_multiplier_and_offset_into_its_joint():
    world = World()
    base = Body(name=PrefixedName("base"))
    lid = Body(name=PrefixedName("lid"))
    with world.modify_world():
        world.add_body(base)
        world.add_body(lid)
        world.add_connection(
            RevoluteConnection.create_with_dofs(
                world=world,
                parent=base,
                child=lid,
                axis=Vector3.Y(),
                multiplier=-2.0,
                offset=0.1,
                dof_limits=joint_limits(-0.5, 0.5),
            )
        )
    original_connection = connection_to(world, "lid")

    parsed = round_trip(world)

    parsed_connection = connection_to(parsed, "lid")
    original_limits = original_connection.dof.limits
    parsed_limits = parsed_connection.dof.limits
    assert parsed_limits.lower.position == pytest.approx(original_limits.lower.position)
    assert parsed_limits.upper.position == pytest.approx(original_limits.upper.position)
    original_connection.position = 0.7
    parsed_connection.position = 0.7
    np.testing.assert_allclose(
        parsed.compute_forward_kinematics_np(
            parsed.get_body_by_name("base"), parsed_connection.child
        ),
        world.compute_forward_kinematics_np(base, lid),
        atol=1e-6,
    )


# %% geometry


def test_round_trip_keeps_primitive_shapes():
    body = build_primitive_shapes_body()
    world = build_single_body_world(body)

    parsed_body = round_trip(world).get_body_by_name(body.name.name)

    assert [type(shape) for shape in parsed_body.collision.shapes] == [
        type(shape) for shape in body.collision.shapes
    ]
    for parsed_shape, shape in zip(parsed_body.collision.shapes, body.collision.shapes):
        np.testing.assert_allclose(
            parsed_shape.origin.to_np(), shape.origin.to_np(), atol=1e-6
        )
        np.testing.assert_allclose(
            parsed_shape.local_frame_bounding_box.dimensions,
            shape.local_frame_bounding_box.dimensions,
            atol=1e-6,
        )


def test_round_trip_keeps_mesh_vertices():
    body = build_mesh_body()
    world = build_single_body_world(body)

    parsed_body = round_trip(world).get_body_by_name(body.name.name)

    [parsed_mesh] = parsed_body.collision.shapes
    [mesh] = body.collision.shapes
    assert isinstance(parsed_mesh, Mesh)
    np.testing.assert_allclose(
        mesh_vertices_in_body_frame(parsed_mesh),
        mesh_vertices_in_body_frame(mesh),
        atol=1e-6,
    )


def test_round_trip_keeps_separate_visual_and_collision_geometry():
    body = build_body_with_separate_collision()
    world = build_single_body_world(body)

    parsed_body = round_trip(world).get_body_by_name(body.name.name)

    assert [type(shape) for shape in parsed_body.visual.shapes] == [Sphere]
    assert [type(shape) for shape in parsed_body.collision.shapes] == [Box]


def test_export_applies_the_collision_api_to_collision_geometry_only():
    body = build_body_with_separate_collision()
    stage = build_stage(build_single_body_world(body))

    collision_prims = [
        prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    assert [prim.GetTypeName() for prim in collision_prims] == ["Cube"]


# %% inertial


def test_round_trip_keeps_the_inertial():
    body = build_body_with_inertial()
    world = build_single_body_world(body)

    parsed_inertial = round_trip(world).get_body_by_name(body.name.name).inertial

    assert parsed_inertial.mass == pytest.approx(body.inertial.mass)
    np.testing.assert_allclose(
        parsed_inertial.center_of_mass.to_np(),
        body.inertial.center_of_mass.to_np(),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        parsed_inertial.inertia.data, body.inertial.inertia.data, atol=1e-6
    )


# %% semantics


@pytest.mark.skipif(
    not USD_SEMANTICS_AVAILABLE, reason="UsdSemantics not available in this usd-core"
)
def test_round_trip_keeps_usd_semantic_labels():
    world = build_cabinet_world()
    labels = UsdSemanticLabels(
        root=world.get_body_by_name("door"),
        taxonomy="category",
        labels=["door", "furniture_part"],
    )
    with world.modify_world():
        world.add_semantic_annotation(labels)

    parsed = round_trip(world)

    [parsed_labels] = parsed.get_semantic_annotations_by_type(UsdSemanticLabels)
    assert parsed_labels.root is parsed.get_body_by_name("door")
    assert parsed_labels.taxonomy == labels.taxonomy
    assert parsed_labels.labels == labels.labels


@pytest.mark.skipif(
    not USD_SEMANTICS_AVAILABLE, reason="UsdSemantics not available in this usd-core"
)
def test_export_labels_a_body_with_the_class_of_each_annotation_rooted_at_it():
    world = build_cabinet_world()
    with world.modify_world():
        world.add_semantic_annotation(Handle(root=world.get_body_by_name("knob")))

    parsed = round_trip(world)

    [parsed_labels] = parsed.get_semantic_annotations_by_type(UsdSemanticLabels)
    assert parsed_labels.root is parsed.get_body_by_name("knob")
    assert parsed_labels.taxonomy == UsdSemanticTaxonomy.CLASS
    assert parsed_labels.labels == [Handle.__name__]


# %% errors


def test_export_raises_on_a_connection_usd_has_no_joint_for():
    world = World()
    base = Body(name=PrefixedName("base"))
    loose = Body(name=PrefixedName("loose"))
    with world.modify_world():
        world.add_body(base)
        world.add_body(loose)
        world.add_connection(
            Connection6DoF.create_with_dofs(world=world, parent=base, child=loose)
        )

    with pytest.raises(UnsupportedConnectionForUsdExportError) as error:
        build_stage(world)

    assert error.value.connection_type is Connection6DoF


def test_export_raises_on_a_shape_no_usd_geometry_is_written_for():
    world = build_single_body_world(build_body_with_unwritable_shape())

    with pytest.raises(UnsupportedShapeForUsdExportError) as error:
        build_stage(world)

    assert error.value.shape_type is UnwritableShape


def test_export_raises_when_the_default_prim_name_is_a_body_name():
    world = build_cabinet_world()

    with pytest.raises(UsdPrimNameCollisionError) as error:
        USDExporter(world=world, default_prim_name="door").build_stage()

    assert error.value.prim_name == "door"


def test_export_raises_when_two_bodies_share_a_prim_name():
    world = World()
    left = Body(name=PrefixedName("door", "left"))
    right = Body(name=PrefixedName("door", "right"))
    with world.modify_world():
        world.add_body(left)
        world.add_body(right)
        world.add_connection(FixedConnection(parent=left, child=right))

    with pytest.raises(UsdPrimNameCollisionError) as error:
        build_stage(world)

    assert error.value.prim_name == "door"


# %% files


def test_export_writes_a_file_the_parser_reads(tmp_path):
    world = build_cabinet_world()
    file_path = tmp_path / "cabinet.usda"

    USDExporter(world=world, default_prim_name=DEFAULT_PRIM_NAME).export(str(file_path))

    parsed = USDParser.from_file(str(file_path), prefix=DEFAULT_PRIM_NAME).parse()
    assert body_names(parsed) == body_names(world) | {DEFAULT_PRIM_NAME}


# %% kitchen


def test_round_trip_keeps_the_kitchen_kinematics(kitchen_world):
    parsed = round_trip(kitchen_world)

    assert body_names(parsed) == body_names(kitchen_world) | {DEFAULT_PRIM_NAME}
    for connection in kitchen_world.get_connections_by_type(ActiveConnection1DOF):
        connection_to(parsed, connection.child.name.name).position = connection.position
    parsed_root = parsed.get_body_by_name(kitchen_world.root.name.name)
    for body in kitchen_world.bodies:
        np.testing.assert_allclose(
            parsed.compute_forward_kinematics_np(
                parsed_root, parsed.get_body_by_name(body.name.name)
            ),
            kitchen_world.compute_forward_kinematics_np(kitchen_world.root, body),
            atol=1e-6,
        )
