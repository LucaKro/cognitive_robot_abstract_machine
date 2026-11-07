from __future__ import annotations

from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body


# %% forward kinematics name both frames of the transform they return


def world_with_offset_body() -> tuple[World, Body, Body]:
    """
    :return: A world, its root, and a body fixed one meter above the root.
    """
    world = World()
    root = Body(name=PrefixedName("root"))
    body = Body(name=PrefixedName("body"))
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=root,
                child=body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=1.0
                ),
            )
        )
    return world, root, body


def test_forward_kinematics_name_the_tip_as_child_frame():
    world, root, body = world_with_offset_body()

    root_T_body = world.compute_forward_kinematics(root, body)

    assert root_T_body.child_frame == body


def test_inverse_of_a_global_transform_can_be_transformed():
    world, root, body = world_with_offset_body()

    body_T_root = body.global_transform.inverse()

    assert world.transform(body_T_root, body).reference_frame == body
