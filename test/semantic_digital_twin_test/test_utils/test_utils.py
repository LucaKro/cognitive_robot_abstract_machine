from krrood.utils import get_full_class_name
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle
from semantic_digital_twin.utils import (
    type_string_to_type,
)


def test_type_string_to_type_reads_back_the_class_it_names():
    original_class = Handle
    original_class_name = get_full_class_name(original_class)

    converted_class = type_string_to_type(original_class_name)

    assert converted_class == original_class
