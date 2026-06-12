from robokudo.world_descriptor import BaseWorldDescriptor, PredefinedObject
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale, Color
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.specs import BodySpec
from semantic_digital_twin.world_description.world_entity import Body


def test_get_predefined_object_bodies():
    world_descriptor = BaseWorldDescriptor()
    test_world = world_descriptor.world
    root = test_world.root

    box_spec = BodySpec.box(
        PrefixedName(name="foobar", prefix="transform_example"),
        Scale(0.10, 0.06, 0.05),
        color=Color(0.1, 0.2, 0.8, 1.0),
    )
    with test_world.modify_world():
        foobar1_body = box_spec.spawn(test_world, connection_type=Connection6DoF)
        foobar2_body = box_spec.spawn(
            test_world,
            name=PrefixedName(name="foobar2", prefix="transform_example"),
            connection_type=Connection6DoF,
        )
        test_world.add_semantic_annotation(PredefinedObject(body=foobar1_body))
        test_world.add_semantic_annotation(PredefinedObject(body=foobar2_body))
    result_world_C_foobar1 = foobar1_body.parent_connection
    result_world_C_foobar2 = foobar2_body.parent_connection

    # Set origins in a separate modification block so FK is compiled first
    with test_world.modify_world():
        result_world_C_foobar1.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            z=0.5, reference_frame=root
        )
        result_world_C_foobar2.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            z=1.2, reference_frame=root
        )

    bodies = world_descriptor.get_predefined_object_bodies()
    assert len(bodies) == 2
    assert set(bodies) == {foobar1_body, foobar2_body}
