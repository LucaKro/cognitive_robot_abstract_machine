from uuid import uuid4

from semantic_digital_twin.world_description.degree_of_freedom_ownership import (
    DegreeOfFreedomOwnership,
    DegreeOfFreedomRole,
)


class TestDegreeOfFreedomOwnership:
    def test_id_for_finds_active_and_passive(self):
        active_id = uuid4()
        passive_id = uuid4()
        ownership = DegreeOfFreedomOwnership.create(
            active={DegreeOfFreedomRole.YAW: active_id},
            passive={DegreeOfFreedomRole.X: passive_id},
        )
        assert ownership.id_for(DegreeOfFreedomRole.YAW) == active_id
        assert ownership.id_for(DegreeOfFreedomRole.X) == passive_id

    def test_id_lists_split_active_and_passive(self):
        active_id = uuid4()
        passive_id = uuid4()
        ownership = DegreeOfFreedomOwnership.create(
            active={DegreeOfFreedomRole.YAW: active_id},
            passive={DegreeOfFreedomRole.X: passive_id},
        )
        assert ownership.active_ids() == [active_id]
        assert ownership.passive_ids() == [passive_id]
        assert ownership.all_ids() == [active_id, passive_id]

    def test_json_round_trip(self):
        ownership = DegreeOfFreedomOwnership.create(
            active={DegreeOfFreedomRole.X_VELOCITY: uuid4(), DegreeOfFreedomRole.YAW: uuid4()},
            passive={DegreeOfFreedomRole.X: uuid4(), DegreeOfFreedomRole.PITCH: uuid4()},
        )
        restored = DegreeOfFreedomOwnership.from_json(ownership.to_json())
        assert restored == ownership
