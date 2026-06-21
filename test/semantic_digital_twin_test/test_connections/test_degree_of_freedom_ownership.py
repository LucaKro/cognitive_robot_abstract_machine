from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.degree_of_freedom_ownership import (
    DegreeOfFreedomOwnership,
    DegreeOfFreedomRole,
    OwnedDegreeOfFreedom,
)


def _dof(name: str) -> DegreeOfFreedom:
    return DegreeOfFreedom(name=PrefixedName(name))


class TestDegreeOfFreedomOwnership:
    def test_dof_for_finds_active_and_passive(self):
        active = _dof("yaw")
        passive = _dof("x")
        ownership = DegreeOfFreedomOwnership.create(
            active={DegreeOfFreedomRole.YAW: active},
            passive={DegreeOfFreedomRole.X: passive},
        )
        assert ownership.dof_for(DegreeOfFreedomRole.YAW) is active
        assert ownership.dof_for(DegreeOfFreedomRole.X) is passive

    def test_dof_lists_split_active_and_passive(self):
        active = _dof("yaw")
        passive = _dof("x")
        ownership = DegreeOfFreedomOwnership.create(
            active={DegreeOfFreedomRole.YAW: active},
            passive={DegreeOfFreedomRole.X: passive},
        )
        assert ownership.active_dofs() == [active]
        assert ownership.passive_dofs() == [passive]
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

    def test_single_active_builds_one_active_main_dof(self):
        dof = _dof("dof")
        ownership = DegreeOfFreedomOwnership.single_active(dof)
        assert ownership.active_dofs() == [dof]
        assert ownership.passive == []
        assert ownership.dof_for(DegreeOfFreedomRole.MAIN) is dof

    def test_active_passive_held_as_owned_dofs_with_role(self):
        yaw = _dof("yaw")
        x = _dof("x")
        ownership = DegreeOfFreedomOwnership.create(
            active={DegreeOfFreedomRole.YAW: yaw},
            passive={DegreeOfFreedomRole.X: x},
        )
        assert ownership.active == [OwnedDegreeOfFreedom(DegreeOfFreedomRole.YAW, yaw)]
        assert ownership.passive == [OwnedDegreeOfFreedom(DegreeOfFreedomRole.X, x)]
