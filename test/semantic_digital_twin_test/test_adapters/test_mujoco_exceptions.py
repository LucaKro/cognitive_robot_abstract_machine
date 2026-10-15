"""
Tests for the exceptions the MuJoCo adapter raises.
"""

import mujoco

from semantic_digital_twin.exceptions import MujocoEntityNotFoundError

# %% missing entities


def test_a_missing_mujoco_entity_is_reported_with_its_name_and_type():
    entity_name, entity_type = "sink_drawer", mujoco.mjtObj.mjOBJ_BODY

    error = MujocoEntityNotFoundError(entity_name=entity_name, entity_type=entity_type)

    assert error.entity_name == entity_name
    assert error.entity_type is entity_type
