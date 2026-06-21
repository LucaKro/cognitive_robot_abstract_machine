import pytest
from numpy.testing import assert_allclose

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.testing import world_setup
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connection_properties import JointDynamics
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    DifferentialDrive,
    FixedConnection,
    OmniDrive,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.world_entity import Body


def _two_body_world():
    """Builds a world and two unattached bodies; the caller adds them and a connection."""
    world = World()
    root = Body(name=PrefixedName("root", prefix="contract"))
    base = Body(name=PrefixedName("base", prefix="contract"))
    return world, root, base


def _world_with_drive(drive_type):
    """Builds a minimal world whose root is connected to a base via ``drive_type``."""
    world = World()
    root = Body(name=PrefixedName("root", prefix="contract"))
    base = Body(name=PrefixedName("base", prefix="contract"))
    with world.modify_world():
        world.add_kinematic_structure_entity(root)
        world.add_kinematic_structure_entity(base)
        drive = drive_type.create_with_dofs(world=world, parent=root, child=base)
        world.add_connection(drive)
    return world, drive


class TestUniformConstructionInterface:
    """Every connection is constructed through the same ``create_with_dofs(world, parent, child, ...)``."""

    @pytest.mark.parametrize(
        "connection_type, needs_axis",
        [
            (FixedConnection, False),
            (Connection6DoF, False),
            (OmniDrive, False),
            (DifferentialDrive, False),
            (RevoluteConnection, True),
            (PrismaticConnection, True),
        ],
    )
    def test_create_with_dofs_shares_positional_signature(self, connection_type, needs_axis):
        world, root, base = _two_body_world()
        specifics = {"axis": Vector3.Z(reference_frame=root)} if needs_axis else {}
        with world.modify_world():
            world.add_kinematic_structure_entity(root)
            world.add_kinematic_structure_entity(base)
            connection = connection_type.create_with_dofs(world, root, base, **specifics)
            world.add_connection(connection)
        assert isinstance(connection, connection_type)
        assert isinstance(connection.dofs, list)

    def test_axis_must_be_passed_by_keyword(self):
        world, root, base = _two_body_world()
        with pytest.raises(TypeError):
            RevoluteConnection.create_with_dofs(
                world, root, base, Vector3.Z(reference_frame=root)
            )


class TestDegreesOfFreedomOwnershipIsUniversal:
    """``degrees_of_freedom`` is the single source of truth for every connection, 1-DOF included."""

    def test_single_dof_joint_owns_its_dof_through_ownership(self):
        world, root, base = _two_body_world()
        with world.modify_world():
            world.add_kinematic_structure_entity(root)
            world.add_kinematic_structure_entity(base)
            revolute = RevoluteConnection.create_with_dofs(
                world, root, base, axis=Vector3.Z(reference_frame=root)
            )
            world.add_connection(revolute)
        owned = revolute.degrees_of_freedom.all_dofs()
        assert owned == [revolute.raw_dof]
        assert revolute.dof_id == revolute.raw_dof.id

    def test_mimic_joints_share_one_dof_object(self, world_setup):
        world = world_setup[0]
        prismatic = next(
            c for c in world.connections if isinstance(c, PrismaticConnection)
        )
        revolute = next(
            c for c in world.connections if isinstance(c, RevoluteConnection)
        )
        assert prismatic.raw_dof is revolute.raw_dof


class TestDegreesOfFreedomContract:
    """The ``dofs`` accessor must return the same container type for every connection."""

    def test_drive_dofs_is_a_list(self):
        _, drive = _world_with_drive(OmniDrive)
        assert isinstance(drive.dofs, list)

    def test_single_dof_connection_dofs_is_a_list(self, world_setup):
        world = world_setup[0]
        revolute = next(
            connection
            for connection in world.connections
            if isinstance(connection, RevoluteConnection)
        )
        assert isinstance(revolute.dofs, list)


class TestDriveOriginContract:
    """A planar drive projects an assigned origin onto its x, y, and yaw plane."""

    @pytest.mark.parametrize("drive_type", [OmniDrive, DifferentialDrive])
    def test_setting_representable_pose_succeeds(self, drive_type):
        world, drive = _world_with_drive(drive_type)
        drive.origin = HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0, y=2.0, yaw=0.3)
        assert world.state[drive.x.id].position == pytest.approx(1.0)
        assert world.state[drive.y.id].position == pytest.approx(2.0)
        assert world.state[drive.yaw.id].position == pytest.approx(0.3)

    @pytest.mark.parametrize("drive_type", [OmniDrive, DifferentialDrive])
    def test_out_of_plane_components_are_projected_away(self, drive_type):
        world, drive = _world_with_drive(drive_type)
        drive.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1.0, y=2.0, z=0.5, roll=0.4, pitch=0.6, yaw=0.3
        )
        assert world.state[drive.x.id].position == pytest.approx(1.0)
        assert world.state[drive.y.id].position == pytest.approx(2.0)
        assert world.state[drive.yaw.id].position == pytest.approx(0.3)
        assert world.state[drive.roll.id].position == pytest.approx(0.0)
        assert world.state[drive.pitch.id].position == pytest.approx(0.0)


class TestJointProperty:
    def test_default_values(self):
        joint_dynamics = JointDynamics()
        assert_allclose(joint_dynamics.armature, 0.0)
        assert_allclose(joint_dynamics.dry_friction, 0.0)
        assert_allclose(joint_dynamics.damping, 0.0)

    def test_custom_values(self):
        armature = 1.5
        dry_friction = 0.2
        damping = 0.05
        joint_dynamics = JointDynamics(
            armature=armature, dry_friction=dry_friction, damping=damping
        )
        assert_allclose(joint_dynamics.armature, armature)
        assert_allclose(joint_dynamics.dry_friction, dry_friction)
        assert_allclose(joint_dynamics.damping, damping)

        joint_prop_dict = joint_dynamics.__dict__
        assert_allclose(joint_prop_dict["armature"], armature)
        assert_allclose(joint_prop_dict["dry_friction"], dry_friction)
        assert_allclose(joint_prop_dict["damping"], damping)
