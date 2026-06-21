from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connection_properties import JointDynamics
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    Connection6DoF,
    DifferentialDrive,
    FixedConnection,
    OmniDrive,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.degree_of_freedom_ownership import (
    DegreeOfFreedomOwnership,
    DegreeOfFreedomRole,
    OwnedDegreeOfFreedom,
)
from semantic_digital_twin.world_description.world_entity import Body, Connection


def _dof(name: str) -> DegreeOfFreedom:
    """Builds a standalone degree of freedom with the given name."""
    return DegreeOfFreedom(name=PrefixedName(name))


def _make_fixed(world: World, parent: Body, child: Body) -> FixedConnection:
    return FixedConnection.create_with_dofs(world, parent, child)


def _make_revolute(world: World, parent: Body, child: Body) -> RevoluteConnection:
    return RevoluteConnection.create_with_dofs(
        world, parent, child, axis=Vector3.Z(reference_frame=parent)
    )


def _make_prismatic(world: World, parent: Body, child: Body) -> PrismaticConnection:
    return PrismaticConnection.create_with_dofs(
        world,
        parent,
        child,
        axis=Vector3.X(reference_frame=parent),
        multiplier=2.0,
        offset=0.5,
    )


def _make_6dof(world: World, parent: Body, child: Body) -> Connection6DoF:
    return Connection6DoF.create_with_dofs(world, parent, child)


def _make_omni(world: World, parent: Body, child: Body) -> OmniDrive:
    return OmniDrive.create_with_dofs(world, parent, child)


def _make_differential(world: World, parent: Body, child: Body) -> DifferentialDrive:
    return DifferentialDrive.create_with_dofs(world, parent, child)


ALL_CONNECTION_FACTORIES = [
    _make_fixed,
    _make_revolute,
    _make_prismatic,
    _make_6dof,
    _make_omni,
    _make_differential,
]


def _build_world(factory) -> tuple[World, Connection]:
    """Builds a world with two bodies connected by the connection the factory produces."""
    world = World()
    parent = Body(name=PrefixedName("parent", prefix="ser"))
    child = Body(name=PrefixedName("child", prefix="ser"))
    with world.modify_world():
        world.add_kinematic_structure_entity(parent)
        world.add_kinematic_structure_entity(child)
        connection = factory(world, parent, child)
        world.add_connection(connection)
    return world, connection


def _round_trip(connection: Connection, world: World) -> tuple[Connection, World]:
    """
    Serializes ``connection`` and deserializes it against a deep copy of ``world``.

    The copy stands in for a freshly loaded world: it already holds the bodies and degrees of
    freedom (by id) that the connection references, so deserialization re-resolves them there.
    """
    json_data = connection.to_json()
    target_world = deepcopy(world)
    tracker = WorldEntityWithIDKwargsTracker.from_world(target_world)
    rebuilt = type(connection).from_json(json_data, **tracker.create_kwargs())
    return rebuilt, target_world


class TestOwnedDegreeOfFreedom:
    """A single owned degree of freedom pairs a role with a degree of freedom."""

    def test_stores_role_and_dof(self):
        dof = _dof("yaw")
        owned = OwnedDegreeOfFreedom(role=DegreeOfFreedomRole.YAW, degree_of_fredom=dof)
        assert owned.role is DegreeOfFreedomRole.YAW
        assert owned.degree_of_fredom is dof

    def test_equal_when_role_and_dof_match(self):
        dof = _dof("yaw")
        assert OwnedDegreeOfFreedom(
            DegreeOfFreedomRole.YAW, dof
        ) == OwnedDegreeOfFreedom(DegreeOfFreedomRole.YAW, dof)

    def test_differs_when_role_differs(self):
        dof = _dof("yaw")
        assert OwnedDegreeOfFreedom(
            DegreeOfFreedomRole.YAW, dof
        ) != OwnedDegreeOfFreedom(DegreeOfFreedomRole.X, dof)


class TestDegreeOfFreedomOwnership:
    """The ownership splits a connection's degrees of freedom into active and passive ones."""

    def test_empty_by_default(self):
        ownership = DegreeOfFreedomOwnership()
        assert ownership.active == []
        assert ownership.passive == []
        assert ownership.all_dofs() == []

    def test_create_without_arguments_is_empty(self):
        ownership = DegreeOfFreedomOwnership.create()
        assert ownership.active == []
        assert ownership.passive == []

    def test_create_splits_active_and_passive(self):
        yaw = _dof("yaw")
        x = _dof("x")
        ownership = DegreeOfFreedomOwnership.create(
            active={DegreeOfFreedomRole.YAW: yaw},
            passive={DegreeOfFreedomRole.X: x},
        )
        assert ownership.active == [OwnedDegreeOfFreedom(DegreeOfFreedomRole.YAW, yaw)]
        assert ownership.passive == [OwnedDegreeOfFreedom(DegreeOfFreedomRole.X, x)]

    def test_single_active_builds_one_active_main_dof(self):
        dof = _dof("dof")
        ownership = DegreeOfFreedomOwnership.single_active(dof)
        assert ownership.active_dofs() == [dof]
        assert ownership.passive == []
        assert ownership.dof_for(DegreeOfFreedomRole.MAIN) is dof

    def test_dof_for_finds_active_and_passive(self):
        active = _dof("yaw")
        passive = _dof("x")
        ownership = DegreeOfFreedomOwnership.create(
            active={DegreeOfFreedomRole.YAW: active},
            passive={DegreeOfFreedomRole.X: passive},
        )
        assert ownership.dof_for(DegreeOfFreedomRole.YAW) is active
        assert ownership.dof_for(DegreeOfFreedomRole.X) is passive

    def test_dof_for_raises_on_unknown_role(self):
        ownership = DegreeOfFreedomOwnership.create(
            active={DegreeOfFreedomRole.YAW: _dof("yaw")}
        )
        with pytest.raises(KeyError):
            ownership.dof_for(DegreeOfFreedomRole.PITCH)

    def test_dof_lists_preserve_declaration_order(self):
        x_velocity = _dof("x_vel")
        yaw = _dof("yaw")
        x = _dof("x")
        y = _dof("y")
        ownership = DegreeOfFreedomOwnership.create(
            active={
                DegreeOfFreedomRole.X_VELOCITY: x_velocity,
                DegreeOfFreedomRole.YAW: yaw,
            },
            passive={DegreeOfFreedomRole.X: x, DegreeOfFreedomRole.Y: y},
        )
        assert ownership.active_dofs() == [x_velocity, yaw]
        assert ownership.passive_dofs() == [x, y]

    def test_all_dofs_lists_active_before_passive(self):
        active = _dof("yaw")
        passive = _dof("x")
        ownership = DegreeOfFreedomOwnership.create(
            active={DegreeOfFreedomRole.YAW: active},
            passive={DegreeOfFreedomRole.X: passive},
        )
        assert ownership.all_dofs() == [active, passive]

    def test_for_world_re_resolves_each_dof_by_id(self):
        original = _dof("x")
        replacement = _dof("x")
        replacement.id = original.id
        ownership = DegreeOfFreedomOwnership.create(
            passive={DegreeOfFreedomRole.X: original}
        )

        class _FakeWorld:
            def get_degree_of_freedom_by_id(self, dof_id):
                assert dof_id == original.id
                return replacement

        resolved = ownership.copy_for_world(_FakeWorld())
        assert resolved.dof_for(DegreeOfFreedomRole.X) is replacement
        assert resolved.dof_for(DegreeOfFreedomRole.X) is not original

    def test_for_world_keeps_active_passive_split_and_roles(self):
        active = _dof("yaw")
        passive = _dof("x")
        ownership = DegreeOfFreedomOwnership.create(
            active={DegreeOfFreedomRole.YAW: active},
            passive={DegreeOfFreedomRole.X: passive},
        )

        class _FakeWorld:
            def get_degree_of_freedom_by_id(self, dof_id):
                return {active.id: active, passive.id: passive}[dof_id]

        resolved = ownership.copy_for_world(_FakeWorld())
        assert [owned.role for owned in resolved.active] == [DegreeOfFreedomRole.YAW]
        assert [owned.role for owned in resolved.passive] == [DegreeOfFreedomRole.X]


class TestConnectionRoundTrip:
    """
    Every connection type survives a JSON round trip with its ownership and references intact.

    These assertions deliberately go through the public interface of the rebuilt connection rather
    than the raw JSON, so a change to the on-disk field layout does not require touching the tests.
    """

    @pytest.mark.parametrize("factory", ALL_CONNECTION_FACTORIES)
    def test_round_trip_preserves_identity_and_type(self, factory):
        world, connection = _build_world(factory)
        rebuilt, _ = _round_trip(connection, world)
        assert type(rebuilt) is type(connection)
        assert rebuilt == connection

    @pytest.mark.parametrize("factory", ALL_CONNECTION_FACTORIES)
    def test_round_trip_preserves_parent_and_child(self, factory):
        world, connection = _build_world(factory)
        rebuilt, _ = _round_trip(connection, world)
        assert rebuilt.parent.id == connection.parent.id
        assert rebuilt.child.id == connection.child.id
        assert rebuilt.name == connection.name

    @pytest.mark.parametrize("factory", ALL_CONNECTION_FACTORIES)
    def test_round_trip_preserves_ownership_roles_and_ids(self, factory):
        world, connection = _build_world(factory)
        rebuilt, _ = _round_trip(connection, world)

        def role_id_pairs(ownership_list):
            return [(owned.role, owned.degree_of_fredom.id) for owned in ownership_list]

        assert role_id_pairs(rebuilt.degrees_of_freedom.active) == role_id_pairs(
            connection.degrees_of_freedom.active
        )
        assert role_id_pairs(rebuilt.degrees_of_freedom.passive) == role_id_pairs(
            connection.degrees_of_freedom.passive
        )

    @pytest.mark.parametrize("factory", ALL_CONNECTION_FACTORIES)
    def test_round_trip_resolves_dofs_into_target_world(self, factory):
        world, connection = _build_world(factory)
        rebuilt, target_world = _round_trip(connection, world)
        for owned in rebuilt.degrees_of_freedom.all_dofs():
            assert owned is target_world.get_degree_of_freedom_by_id(owned.id)

    @pytest.mark.parametrize("factory", ALL_CONNECTION_FACTORIES)
    def test_round_trip_preserves_origin_expressions(self, factory):
        world, connection = _build_world(factory)
        rebuilt, _ = _round_trip(connection, world)
        assert np.allclose(
            rebuilt.parent_T_connection_expression.to_np(),
            connection.parent_T_connection_expression.to_np(),
        )
        assert np.allclose(
            rebuilt.connection_T_child_expression.to_np(),
            connection.connection_T_child_expression.to_np(),
        )


class TestActiveConnection1DOFRoundTrip:
    """The 1-DOF joint fields (axis, multiplier, offset, dynamics) survive serialization."""

    def test_axis_multiplier_and_offset_round_trip(self):
        world, connection = _build_world(_make_prismatic)
        rebuilt, _ = _round_trip(connection, world)
        assert isinstance(rebuilt, ActiveConnection1DOF)
        assert np.allclose(rebuilt.axis.to_np(), connection.axis.to_np())
        assert rebuilt.multiplier == connection.multiplier
        assert rebuilt.offset == connection.offset

    def test_dynamics_round_trip(self):
        world, connection = _build_world(_make_revolute)
        connection.dynamics = JointDynamics(
            armature=1.5, dry_friction=0.2, damping=0.05
        )
        rebuilt, _ = _round_trip(connection, world)
        assert rebuilt.dynamics == connection.dynamics

    def test_parent_offset_round_trip(self):
        world = World()
        parent = Body(name=PrefixedName("parent", prefix="ser"))
        child = Body(name=PrefixedName("child", prefix="ser"))
        with world.modify_world():
            world.add_kinematic_structure_entity(parent)
            world.add_kinematic_structure_entity(child)
            connection = RevoluteConnection.create_with_dofs(
                world,
                parent,
                child,
                axis=Vector3.Z(reference_frame=parent),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1.0, y=2.0, yaw=0.5, reference_frame=parent
                ),
            )
            world.add_connection(connection)
        rebuilt, _ = _round_trip(connection, world)
        assert np.allclose(
            rebuilt.parent_T_connection_expression.to_np(),
            connection.parent_T_connection_expression.to_np(),
        )


class TestSharedDegreeOfFreedomRoundTrip:
    """Connections that share a degree of freedom keep sharing it after deserialization."""

    def test_mimic_connections_resolve_to_one_dof_object(self):
        world = World()
        root = Body(name=PrefixedName("root", prefix="mimic"))
        middle = Body(name=PrefixedName("middle", prefix="mimic"))
        tip = Body(name=PrefixedName("tip", prefix="mimic"))
        with world.modify_world():
            for body in (root, middle, tip):
                world.add_kinematic_structure_entity(body)
            shared_dof = DegreeOfFreedom(name=PrefixedName("shared"))
            world.add_degree_of_freedom(shared_dof)
            prismatic = PrismaticConnection(
                parent=root,
                child=middle,
                degrees_of_freedom=DegreeOfFreedomOwnership.single_active(shared_dof),
                axis=Vector3.X(reference_frame=root),
            )
            revolute = RevoluteConnection(
                parent=middle,
                child=tip,
                degrees_of_freedom=DegreeOfFreedomOwnership.single_active(shared_dof),
                axis=Vector3.Z(reference_frame=middle),
            )
            world.add_connection(prismatic)
            world.add_connection(revolute)

        target_world = deepcopy(world)
        tracker = WorldEntityWithIDKwargsTracker.from_world(target_world)
        kwargs = tracker.create_kwargs()
        rebuilt_prismatic = PrismaticConnection.from_json(prismatic.to_json(), **kwargs)
        rebuilt_revolute = RevoluteConnection.from_json(revolute.to_json(), **kwargs)

        assert rebuilt_prismatic.raw_dof is rebuilt_revolute.raw_dof
        assert rebuilt_prismatic.raw_dof.id == shared_dof.id
