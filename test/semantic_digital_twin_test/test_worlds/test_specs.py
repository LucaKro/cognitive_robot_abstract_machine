import numpy as np
import pytest

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Aperture,
    Apple,
    Fridge,
    Handle,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.specs import BodySpec, RegionSpec
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.inertial_properties import Inertial
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture
def world_with_root():
    world = World()
    root = Body(name=PrefixedName("root", prefix="test"))
    with world.modify_world():
        world.add_body(root)
    return world, root


def test_to_body_is_reusable():
    spec = BodySpec.box("crate", Scale(0.2, 0.2, 0.2))
    body1 = spec.to_body()
    body2 = spec.to_body(name="crate2")

    assert body1.name == PrefixedName("crate")
    assert body2.name == PrefixedName("crate2")
    assert body1.collision[0] is not body2.collision[0]
    assert body1.collision[0] is not spec.shapes[0]
    # the prototype shapes must never become bound to a created body
    assert spec.shapes[0].origin.reference_frame is None
    assert body1.collision[0].origin.reference_frame is body1


def test_to_body_shares_collision_and_visual():
    spec = BodySpec.sphere("ball", radius=0.1)
    body = spec.to_body()
    assert body.collision is body.visual
    assert body.collision[0].radius == 0.1


def test_multi_shape_spec():
    spec = BodySpec(
        name="compound",
        shapes=[
            Box(
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.05),
                scale=Scale(0.1, 0.2, 0.2),
            ),
            Box(
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.05),
                scale=Scale(0.1, 0.2, 0.2),
            ),
        ],
    )
    body = spec.to_body()
    assert len(body.collision) == 2
    assert np.allclose(body.combined_mesh.bounds, [[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]])


def test_inertial_passthrough():
    spec = BodySpec.box("crate", Scale(0.2, 0.2, 0.2))
    assert spec.to_body().inertial.mass == Inertial().mass

    inertial = Inertial(mass=2.0)
    spec_with_inertial = BodySpec(
        name="heavy_crate", shapes=[Box(scale=Scale(0.2, 0.2, 0.2))], inertial=inertial
    )
    body = spec_with_inertial.to_body()
    assert body.inertial.mass == 2.0
    assert body.inertial is not inertial


def test_spawn_defaults(world_with_root):
    world, root = world_with_root
    body = BodySpec.box("crate", Scale(0.2, 0.2, 0.2)).spawn(world)

    assert body in world.kinematic_structure_entities
    assert body.parent_kinematic_structure_entity is root
    assert isinstance(body.parent_connection, FixedConnection)
    assert np.allclose(body.global_transform.to_np(), np.eye(4))


def test_spawn_with_parent_and_pose(world_with_root):
    world, root = world_with_root
    table = BodySpec.box("table", Scale(1.0, 1.0, 0.8)).spawn(world)
    cup = BodySpec.cylinder("cup", width=0.08, height=0.1).spawn(
        world,
        parent=table,
        pose=HomogeneousTransformationMatrix.from_xyz_rpy(z=0.45),
    )

    assert cup.parent_kinematic_structure_entity is table
    assert np.allclose(cup.global_transform.to_np()[:3, 3], [0.0, 0.0, 0.45])


def test_spawn_with_6dof_connection(world_with_root):
    world, root = world_with_root
    body = BodySpec.sphere("ball", radius=0.1).spawn(
        world, connection_type=Connection6DoF
    )
    assert isinstance(body.parent_connection, Connection6DoF)
    assert len(body.parent_connection.dofs) > 0


def test_spawn_inside_open_modification_block(world_with_root):
    world, root = world_with_root
    spec = BodySpec.box("crate", Scale(0.2, 0.2, 0.2))
    with world.modify_world():
        body1 = spec.spawn(world, name="crate1")
        body2 = spec.spawn(
            world,
            name="crate2",
            parent=body1,
            pose=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.5),
        )
    assert body1 in world.kinematic_structure_entities
    assert body2.parent_kinematic_structure_entity is body1
    assert np.allclose(body2.global_transform.to_np()[:3, 3], [0.5, 0.0, 0.0])


def test_spawn_many_from_one_spec(world_with_root):
    world, root = world_with_root
    spec = BodySpec.box("crate", Scale(0.2, 0.2, 0.2))
    bodies = [spec.spawn(world, name=f"crate_{i}") for i in range(3)]

    names = {body.name.name for body in bodies}
    assert names == {"crate_0", "crate_1", "crate_2"}
    shapes = [body.collision[0] for body in bodies]
    assert len(set(map(id, shapes))) == 3


def test_from_event():
    event = Scale(1.0, 2.0, 3.0).to_simple_event().as_composite_set()
    spec = BodySpec.from_event("box_event", event)
    body = spec.to_body()

    assert np.allclose(body.combined_mesh.bounds, [[-0.5, -1.0, -1.5], [0.5, 1.0, 1.5]])
    assert all(shape.origin.reference_frame is None for shape in spec.shapes)


def test_annotation_body_spec_matches_factory_geometry(world_with_root):
    world, root = world_with_root
    spec_body = Handle.create_body_spec(PrefixedName("handle")).to_body()
    with world.modify_world():
        handle = Handle.create_with_new_body_in_world(
            world=world, body_spec=Handle.create_body_spec(PrefixedName("handle2"))
        )

    assert len(spec_body.collision) == len(handle.root.collision)
    assert np.allclose(
        spec_body.combined_mesh.bounds, handle.root.combined_mesh.bounds
    )


def test_case_body_spec_is_hollow():
    spec = Fridge.create_body_spec(PrefixedName("fridge"), Scale(1.0, 1.0, 2.0))
    body = spec.to_body()

    assert len(body.collision) > 1
    assert np.allclose(body.combined_mesh.bounds, [[-0.5, -0.5, -1.0], [0.5, 0.5, 1.0]])


def test_create_with_new_body_in_world_parity(world_with_root):
    world, root = world_with_root
    with world.modify_world():
        apple = Apple.create_with_new_body_in_world(
            world=world,
            body_spec=Apple.create_body_spec(
                PrefixedName("apple", prefix="test"), scale=Scale(1.0, 2.0, 3.0)
            ),
        )

    # geometry must match the previous BoundingBoxCollection.from_event construction
    assert np.allclose(
        apple.root.combined_mesh.bounds, [[-0.5, -1.0, -1.5], [0.5, 1.0, 1.5]]
    )
    assert apple.root.visual is apple.root.collision
    assert apple.root.parent_kinematic_structure_entity is root
    assert apple in world.semantic_annotations


def test_factory_name_comes_from_spec(world_with_root):
    world, root = world_with_root
    spec = Apple.create_body_spec(PrefixedName("apple", prefix="test"))
    with world.modify_world():
        apple = Apple.create_with_new_body_in_world(world=world, body_spec=spec)

    assert apple.name == spec.name
    assert apple.root.name == spec.name


def test_spec_pose_is_default_placement(world_with_root):
    world, root = world_with_root
    spec = BodySpec.box("crate", Scale(0.2, 0.2, 0.2))
    spec.pose = HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0)

    body1 = spec.spawn(world, name="crate1")
    body2 = spec.spawn(
        world, name="crate2", pose=HomogeneousTransformationMatrix.from_xyz_rpy(x=2.0)
    )

    assert np.allclose(body1.global_transform.to_np()[:3, 3], [1.0, 0.0, 0.0])
    assert np.allclose(body2.global_transform.to_np()[:3, 3], [2.0, 0.0, 0.0])
    # the spec's pose stays unbound and reusable
    assert spec.pose.reference_frame is None
    assert spec.pose.child_frame is None


def test_factory_pose_precedence(world_with_root):
    world, root = world_with_root
    spec = Apple.create_body_spec(PrefixedName("apple1"), scale=Scale(0.1, 0.1, 0.1))
    spec.pose = HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0)
    with world.modify_world():
        from_spec_pose = Apple.create_with_new_body_in_world(
            world=world, body_spec=spec
        )
        spec2 = Apple.create_body_spec(PrefixedName("apple2"), scale=Scale(0.1, 0.1, 0.1))
        spec2.pose = HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0)
        overridden = Apple.create_with_new_body_in_world(
            world=world,
            body_spec=spec2,
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=3.0),
        )

    assert np.allclose(from_spec_pose.root.global_transform.to_np()[:3, 3], [1.0, 0.0, 0.0])
    assert np.allclose(overridden.root.global_transform.to_np()[:3, 3], [3.0, 0.0, 0.0])


def test_spawn_does_not_mutate_caller_pose(world_with_root):
    world, root = world_with_root
    pose = HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0)
    spec = BodySpec.box("crate", Scale(0.2, 0.2, 0.2))

    spec.spawn(world, name="crate1", pose=pose)
    spec.spawn(world, name="crate2", pose=pose)

    assert pose.reference_frame is None
    assert pose.child_frame is None


def test_region_spec_is_reusable():
    spec = RegionSpec(name="area", shapes=[Box(scale=Scale(1.0, 1.0, 0.01))])
    region1 = spec.to_region()
    region2 = spec.to_region(name="area2")

    assert region1.name == PrefixedName("area")
    assert region2.name == PrefixedName("area2")
    assert region1.area[0] is not region2.area[0]
    assert spec.shapes[0].origin.reference_frame is None
    assert region1.area[0].origin.reference_frame is region1


def test_region_spec_spawn_with_parent(world_with_root):
    world, root = world_with_root
    table = BodySpec.box("table", Scale(1.0, 1.0, 0.8)).spawn(world)
    region = RegionSpec(name="tabletop", shapes=[Box(scale=Scale(1.0, 1.0, 0.01))]).spawn(
        world,
        parent=table,
        pose=HomogeneousTransformationMatrix.from_xyz_rpy(z=0.405),
    )

    assert region.parent_kinematic_structure_entity is table
    assert np.allclose(region.global_transform.to_np()[:3, 3], [0.0, 0.0, 0.405])


def test_aperture_region_spec_parity(world_with_root):
    world, root = world_with_root
    spec = Aperture.create_region_spec(PrefixedName("aperture"), scale=Scale(0.1, 1.0, 1.0))
    with world.modify_world():
        aperture = Aperture.create_with_new_region_in_world(
            world=world, region_spec=spec
        )

    assert np.allclose(
        aperture.root.combined_mesh.bounds, [[-0.05, -0.5, -0.5], [0.05, 0.5, 0.5]]
    )
    assert aperture.name == spec.name


def test_visual_shapes_none_shares_collection():
    body = BodySpec.box("crate", Scale(0.2, 0.2, 0.2)).to_body()
    assert body.visual is body.collision


def test_visual_shapes_separate_collections():
    spec = BodySpec(
        name="crate",
        shapes=[Box(scale=Scale(0.2, 0.2, 0.2))],
        visual_shapes=[Box(scale=Scale(0.3, 0.3, 0.3))],
    )
    body1 = spec.to_body()
    body2 = spec.to_body(name="crate2")

    assert body1.visual is not body1.collision
    assert body1.collision[0].scale.x == 0.2
    assert body1.visual[0].scale.x == 0.3
    assert body1.visual[0].origin.reference_frame is body1
    assert body1.visual[0] is not body2.visual[0]
    # prototypes stay unbound
    assert spec.visual_shapes[0].origin.reference_frame is None


def test_visual_shapes_empty_means_collision_only():
    spec = BodySpec(
        name="invisible",
        shapes=[Box(scale=Scale(0.2, 0.2, 0.2))],
        visual_shapes=[],
    )
    body = spec.to_body()
    assert len(body.collision) == 1
    assert len(body.visual) == 0
