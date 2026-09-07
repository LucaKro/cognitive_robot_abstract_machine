import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Self, Set, Tuple, TYPE_CHECKING

import numpy as np
import pytest

from krrood.adapters.exceptions import (
    MissingTypeError,
    InvalidTypeFormatError,
    UnknownModuleError,
    ClassNotFoundError,
    JSON_TYPE_NAME,
)
from krrood.adapters.json_serializer import (
    SubclassJSONSerializer,
    to_json,
    from_json,
    JSONAttributeDiff,
    shallow_diff_json,
    DataclassJSONSerializer,
    Rebuild,
)
from krrood.utils import get_full_class_name


@dataclass
class Animal(SubclassJSONSerializer):
    """
    Base animal used in tests.
    """

    name: str
    age: int
    owners: list[str] = field(default_factory=list)

    def to_json(self):
        data = super().to_json()
        data.update(
            {
                "name": self.name,
                "age": self.age,
                "owners": self.owners,
            }
        )
        return data

    @classmethod
    def _from_json(cls, data, **kwargs):
        return cls(
            name=(data["name"]),
            age=(data["age"]),
            owners=(data["owners"]),
        )


@dataclass
class Dog(Animal):
    """
    Dog subtype for tests.
    """

    breed: str = "mixed"

    def to_json(self):
        data = super().to_json()
        data.update(
            {
                "breed": self.breed,
            }
        )
        return data

    @classmethod
    def _from_json(cls, data, **kwargs):
        return cls(
            name=(data["name"]),
            age=(data["age"]),
            breed=(data["breed"]),
            owners=(data["owners"]),
        )


@dataclass
class Bulldog(Dog):
    """
    Deep subtype to ensure deep discovery works.
    """

    stubborn: bool = True

    def to_json(self):
        data = super().to_json()
        data.update(
            {
                "stubborn": (self.stubborn),
            }
        )
        return data

    @classmethod
    def _from_json(cls, data, **kwargs):
        return cls(
            name=(data["name"]),
            age=(data["age"]),
            breed=(data["breed"]),
            stubborn=(data["stubborn"]),
        )


@dataclass
class Cat(Animal):
    """
    Cat subtype for tests.
    """

    lives: int = 9

    def to_json(self):
        data = super().to_json()
        data.update(
            {
                "lives": (self.lives),
            }
        )
        return data

    @classmethod
    def _from_json(cls, data, **kwargs):
        return cls(
            name=(data["name"]),
            age=(data["age"]),
            lives=(data["lives"]),
        )


@dataclass
class ClassThatNeedsKWARGS(SubclassJSONSerializer):
    a: int
    b: float = 0

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "a": (self.a)}

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(a=(data["a"]), b=(kwargs["b"]))


@dataclass
class ClassThatNeedsKWARGSInList(SubclassJSONSerializer):
    a: int
    b: list[ClassThatNeedsKWARGS] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "a": (self.a), "b": to_json(self.b)}

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(a=(data["a"]), b=from_json(data["b"], **kwargs))


@dataclass
class ClassWithDict(DataclassJSONSerializer):
    a: Dict[str, int]


class CustomEnum(str, Enum):
    A = "a"
    B = "b"


def test_roundtrip_dog_and_cat():
    dog = Dog(name="Rex", age=5, breed="Shepherd")
    cat = Cat(name="Misty", age=3, lives=7)

    dog_json = dog.to_json()
    cat_json = cat.to_json()

    assert dog_json[JSON_TYPE_NAME] == get_full_class_name(Dog)
    assert cat_json[JSON_TYPE_NAME] == get_full_class_name(Cat)

    dog2 = SubclassJSONSerializer.from_json(dog_json)
    cat2 = SubclassJSONSerializer.from_json(cat_json)

    assert isinstance(dog2, Dog)
    assert isinstance(cat2, Cat)
    assert dog2 == dog
    assert cat2 == cat


def test_deep_subclass_discovery():
    b = Bulldog(name="Butch", age=4, breed="Bulldog", stubborn=True)
    b_json = b.to_json()

    assert b_json[JSON_TYPE_NAME] == get_full_class_name(Bulldog)

    b2 = SubclassJSONSerializer.from_json(b_json)
    assert isinstance(b2, Bulldog)
    assert b2 == b


def test_unknown_module_raises_unknown_module_error():
    with pytest.raises(UnknownModuleError):
        SubclassJSONSerializer.from_json({JSON_TYPE_NAME: "non.existent.Class"})


def test_missing_type_raises_missing_type_error():
    with pytest.raises(MissingTypeError):
        SubclassJSONSerializer.from_json({})


def test_invalid_type_format_raises_invalid_type_format_error():
    with pytest.raises(InvalidTypeFormatError):
        SubclassJSONSerializer.from_json({JSON_TYPE_NAME: "NotAQualifiedName"})


essential_existing_module = "krrood.utils"


def test_class_not_found_raises_class_not_found_error():
    with pytest.raises(ClassNotFoundError):
        SubclassJSONSerializer.from_json(
            {JSON_TYPE_NAME: f"{essential_existing_module}.DoesNotExist"}
        )


def test_uuid_encoding():
    u = uuid.uuid4()
    encoded = to_json(u)
    result = from_json(encoded)
    assert u == result

    us = [uuid.uuid4(), uuid.uuid4()]
    encoded = to_json(us)
    result = from_json(encoded)
    assert us == result


def test_with_kwargs():
    obj = ClassThatNeedsKWARGS(a=1, b=2.0)
    data = obj.to_json()
    result = from_json(data, b=2.0)
    assert obj == result


def test_with_kwargs_in_list():
    obj = ClassThatNeedsKWARGSInList(a=1, b=[ClassThatNeedsKWARGS(a=1, b=2.0)])
    data = obj.to_json()
    result = from_json(data, b=2.0)
    assert obj == result


def test_list_of_enums():
    obj = [CustomEnum.A, CustomEnum.B]
    data = to_json(obj)
    result = from_json(data)
    assert result == obj


def test_exception():
    e = ImportError("test")
    data = to_json(e)
    result = from_json(data)

    assert isinstance(result, ImportError)
    assert result.args == e.args


def test_classes():
    obj = [Dog("muh", 23, "cow"), Dog]
    data = to_json(obj)
    result = from_json(data)
    assert result == obj


def test_json_attribute_diff_roundtrip():
    diff = JSONAttributeDiff(
        attribute_name="test", added_values=[1, 2], removed_values=[3]
    )
    data = diff.to_json()
    result = from_json(data)
    assert isinstance(result, JSONAttributeDiff)
    assert diff == result


def test_json_attribute_diff_empty():
    diff = JSONAttributeDiff(attribute_name="test")
    assert diff.added_values == []
    assert diff.removed_values == []
    data = diff.to_json()
    result = from_json(data)
    assert diff == result


def test_shallow_diff_json():
    orig = {"a": 1, "b": [1, 2], "c": "foo"}
    new = {"a": 2, "b": [2, 3], "c": "bar"}
    diffs = shallow_diff_json(orig, new)

    diff_dict = {d.attribute_name: d for d in diffs}

    assert "a" in diff_dict
    assert diff_dict["a"].added_values == [2]

    assert "b" in diff_dict
    assert set(diff_dict["b"].added_values) == {3}
    assert set(diff_dict["b"].removed_values) == {1}

    assert "c" in diff_dict
    assert diff_dict["c"].added_values == ["bar"]


def test_update_from_json_diff():
    dog = Dog(
        name="Rex",
        age=5,
        breed="Shepherd",
        owners=["Alice", "Bob"],
    )
    orig_json = dog.to_json()
    new_json = orig_json.copy()
    new_json["name"] = "Max"
    new_json["age"] = 6
    new_json["owners"] = ["Alice", "Charlie"]

    diffs = shallow_diff_json(orig_json, new_json)

    dog.update_from_json_diff(diffs)

    assert dog.name == "Max"
    assert dog.age == 6
    assert dog.owners == ["Alice", "Charlie"]


def test_shallow_diff_json_nested():
    dog1 = Dog(name="Rex", age=5)
    dog2 = Dog(name="Max", age=6)

    orig = {"pet": dog1.to_json()}
    new = {"pet": dog2.to_json()}

    diffs = shallow_diff_json(orig, new)
    assert len(diffs) == 1
    assert diffs[0].attribute_name == "pet"
    added_values = from_json(diffs[0].added_values)
    assert isinstance(added_values[0], Dog)
    assert added_values[0].name == "Max"


def test_nparray():
    obj = np.array([1, 2, 3])
    data = to_json(obj)
    result = from_json(data)
    assert np.allclose(result, obj)

    obj = np.array([1, 2, 3], dtype=np.float64)
    data = to_json(obj)
    result = from_json(data)
    assert np.allclose(result, obj)

    obj = np.array([1.3, 2, 3], dtype=np.float64)
    data = to_json(obj)
    result = from_json(data)
    assert np.allclose(result, obj)


@dataclass
class Foo:
    bar: str = "baz"
    muh: int = field(default_factory=lambda: 42)


def test_dataclass_with_default_factory():
    foo = Foo()
    data = to_json(foo)
    result = from_json(data)
    assert result == foo


def test_dataclass_dict():
    cls = ClassWithDict({"foo": 1})
    data = to_json(cls)
    result = from_json(data)
    assert result == cls


@dataclass
class ClassWithContainers:
    """
    A class whose fields are the containers a JSON array can stand for.
    """

    claimants: Tuple[str, ...] = ("a", "b")
    """
    A tuple, which has to come back hashable.
    """

    tags: Set[str] = field(default_factory=lambda: {"x", "y"})
    """
    A set, which has to come back without an order.
    """

    listed: List[str] = field(default_factory=lambda: ["p", "q"])
    """
    A list, which is what a JSON array already is.
    """


def test_dataclass_containers_keep_their_type():
    """
    A JSON array is read back as whatever the field's annotation says it is.

    Every container is written as an array, so nothing in the file says which one it was.
    Reading them all back as lists means a tuple field returns unhashable, and the record
    no longer equals the one it was written from.
    """
    held = ClassWithContainers()
    result = from_json(to_json(held))
    assert result == held
    assert isinstance(result.claimants, tuple)
    assert isinstance(result.tags, set)
    assert isinstance(result.listed, list)


def test_a_tuple_field_comes_back_usable_as_a_key():
    """
    The point of keeping the tuple: it can still be put in a set or used as a key.
    """
    held = ClassWithContainers()
    result = from_json(to_json(held))
    assert {result.claimants} == {held.claimants}


@dataclass
class ClassWithPostponedAnnotations:
    """
    A class whose annotations are strings, as they are under postponed evaluation.
    """

    claimants: "Tuple[str, ...]" = ("a", "b")
    """
    The same tuple, written as a string annotation.
    """


def test_a_string_annotation_is_resolved_before_it_is_read():
    """
    A module with ``from __future__ import annotations`` hands over strings, not types.
    """
    assert DataclassJSONSerializer.rebuilds_by_field(ClassWithPostponedAnnotations) == {
        "claimants": Rebuild(container=tuple)
    }
    result = from_json(to_json(ClassWithPostponedAnnotations()))
    assert isinstance(result.claimants, tuple)


class Channel(str, Enum):
    """
    An enumeration whose members are strings, so JSON cannot tell them apart from one.
    """

    PART = "part"
    CONTAINS = "contains"


@dataclass
class ClassWithStringEnum:
    """
    A class holding a member of a string enumeration.
    """

    channel: Channel = Channel.PART
    """
    The member, which has to come back as the member and not as its value.
    """


def test_a_string_enum_comes_back_as_its_member():
    """
    A str-valued enum member is a string, so it is written as one and read back as one.

    It compares equal to its member either way; what is lost is that it *is* the member,
    which is the difference between ``channel == Channel.PART`` and
    ``channel is Channel.PART``.
    """
    result = from_json(to_json(ClassWithStringEnum()))
    assert result.channel is Channel.PART


# %% an enum member the annotation does not name at the top level


@dataclass
class ClassWithNestedStringEnum:
    """
    A class holding string enum members behind ``Optional`` and inside a container.
    """

    channel: Optional[Channel] = Channel.CONTAINS
    """
    A member behind ``Optional``, which is a ``Union`` and not the enum itself.
    """

    channels: Tuple[Channel, ...] = (Channel.PART, Channel.CONTAINS)
    """
    Members inside a tuple, which says what it holds only in its argument.
    """


def written_and_read_back(held: Any) -> Any:
    """
    Round-trip a record through JSON text, as writing it to a file does.

    :param held: The record to write.
    :return: The record read back from its text.
    """
    return from_json(json.loads(json.dumps(to_json(held))))


def test_an_optional_enum_comes_back_as_its_member():
    """
    ``Optional[Channel]`` is still an annotation naming an enum, one wrapper further out.
    """
    result = written_and_read_back(ClassWithNestedStringEnum())
    assert result.channel is Channel.CONTAINS


def test_enum_members_inside_a_container_come_back_as_members():
    """
    A tuple of members has to come back holding the members, not their values.
    """
    held = ClassWithNestedStringEnum()
    result = written_and_read_back(held)
    assert result.channels == held.channels
    assert all(
        member is expected for member, expected in zip(result.channels, held.channels)
    )
