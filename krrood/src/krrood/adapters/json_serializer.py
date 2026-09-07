from __future__ import annotations

import enum
import importlib
import inspect
import sys
import uuid
from abc import ABC
from dataclasses import dataclass, fields, is_dataclass
from dataclasses import field
from types import NoneType
from typing import List, Optional, TypeAlias, TYPE_CHECKING

import numpy as np
from typing_extensions import (
    Any,
    ClassVar,
    Dict,
    Self,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from krrood.adapters.exceptions import (
    MissingTypeError,
    InvalidTypeFormatError,
    UnknownModuleError,
    ClassNotFoundError,
    ClassNotSerializableError,
    JSON_TYPE_NAME,
)
from krrood.class_diagrams.attribute_introspector import DataclassOnlyIntrospector
from krrood.ormatic.data_access_objects.base import HasGeneric
from krrood.singleton import SingletonMeta
from krrood.utils import (
    get_full_class_name,
    recursive_subclasses,
)

list_like_classes = (
    list,
    tuple,
    set,
)  # classes that can be serialized by the built-in JSON module
leaf_types = (
    int,
    float,
    str,
    bool,
    NoneType,
)  # containers that can be serialized by the built-in JSON module

JSON_DICT_TYPE = Dict[str, Any]  # Commonly referred JSON dict
JSON_RETURN_TYPE = Union[
    JSON_DICT_TYPE, List[Any], *leaf_types
]  # Commonly referred JSON types
JSON_IS_CLASS = "__is_class__"
"""
We need to remember if something is a class, because the type of a class is often just
type.
"""

if TYPE_CHECKING:
    JSONData: TypeAlias = JSON_RETURN_TYPE
else:

    class JSONData:
        """
        Represents raw JSON data.

        Use this type for type hints when you want to tell KRROOD that something is JSON
        data that should not be further processed (e.g. by from_json()).
        """


@dataclass
class JSONSerializableTypeRegistry(metaclass=SingletonMeta):
    """
    Singleton registry for custom serializers and deserializers.

    Use this registry when you need to add custom JSON serialization/deserialization
    logic for a type where you cannot control its inheritance.
    """

    def get_external_serializer(self, clazz: Type) -> Type[ExternalClassJSONSerializer]:
        """
        Get the external serializer for the given class.

        This returns the serializer of the closest superclass if no direct match is
        found.

        :param clazz: The class to get the serializer for.
        :return: The serializer class.
        """
        # Imported lazily to avoid a circular import: inheritance_path_length pulls in the EQL
        # predicate/variable modules, which import back from json_serializer during package load.
        from krrood.inheritance_path_length import inheritance_path_length

        if issubclass(clazz, enum.Enum):
            return EnumJSONSerializer

        distances = {}  # mapping of subclasses to the distance to the clazz

        for subclass in recursive_subclasses(ExternalClassJSONSerializer):
            if subclass.matches_generic_type(clazz):
                return subclass
            else:
                distance = inheritance_path_length(clazz, subclass.original_class())
                if distance is not None:
                    distances[subclass] = distance

        if not distances:
            raise ClassNotSerializableError(clazz)
        else:
            return min(distances, key=distances.get)


class SubclassJSONSerializer:
    """
    Class for automatic (de)serialization of subclasses using importlib.

    Stores the fully qualified class name in `type` during serialization and imports
    that class during deserialization.
    """

    def to_json(self) -> Dict[str, Any]:
        return {JSON_TYPE_NAME: get_full_class_name(self.__class__)}

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        """
        Create an instance from a json dict.

        This method is called from the from_json method after the correct subclass is
        determined and should be overwritten by the subclass.

        :param data: The JSON dict
        :param kwargs: Additional keyword arguments to pass to the constructor of the
            subclass.
        :return: The deserialized object
        """
        raise NotImplementedError()

    @classmethod
    def from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        """
        Create the correct instanceof the subclass from a json dict.

        :param data: The json dict
        :param kwargs: Additional keyword arguments to pass to the constructor of the
            subclass.
        :return: The correct instance of the subclass
        """
        if isinstance(data, leaf_types):
            return data

        if isinstance(data, list_like_classes):
            return [from_json(d, **kwargs) for d in data]

        fully_qualified_class_name = data.get(JSON_TYPE_NAME)
        if not fully_qualified_class_name:
            raise MissingTypeError()

        try:
            module_name, class_name = fully_qualified_class_name.rsplit(".", 1)
        except ValueError as exc:
            raise InvalidTypeFormatError(fully_qualified_class_name) from exc

        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            raise UnknownModuleError(module_name) from exc

        try:
            target_cls = getattr(module, class_name)
        except AttributeError as exc:
            raise ClassNotFoundError(class_name, module_name) from exc

        if data.get(JSON_IS_CLASS, False):
            return ClassJSONSerializer.from_json(data, clazz=target_cls, **kwargs)

        if issubclass(target_cls, SubclassJSONSerializer):
            return target_cls._from_json(data, **kwargs)

        external_json_deserializer = (
            JSONSerializableTypeRegistry().get_external_serializer(target_cls)
        )

        return external_json_deserializer.from_json(data, clazz=target_cls, **kwargs)

    def update_from_json_diff(self, diffs: List[JSONAttributeDiff], **kwargs) -> None:
        """
        Update the current object from a list of shallow diffs.

        :param diffs: The shallow diffs to apply.
        :param kwargs: Additional keyword arguments to pass to the constructor of the
            subclass.
        """
        for diff in diffs:
            self._apply_diff(diff, **kwargs)

    def _apply_diff(self, diff: JSONAttributeDiff, **kwargs) -> None:
        """
        Apply a single diff to the current object.

        :param diff: The diff to apply.
        """
        current_value = getattr(self, diff.attribute_name)
        if isinstance(current_value, list):
            for item in diff.removed_values:
                current_value.remove(from_json(item, **kwargs))
            for item in diff.added_values:
                current_value.append(from_json(item, **kwargs))
        else:
            setattr(
                self,
                diff.attribute_name,
                from_json(diff.added_values[0], **kwargs),
            )


def from_json(data: Dict[str, Any], **kwargs) -> Union[SubclassJSONSerializer, Any]:
    """
    Deserialize a JSON dict to an object.

    :param data: The JSON string
    :return: The deserialized object
    """
    return SubclassJSONSerializer.from_json(data, **kwargs)


def to_json(obj: Union[SubclassJSONSerializer, Any]) -> JSON_RETURN_TYPE:
    """
    Serialize an object to a JSON dict.

    :param obj: The object to convert to json
    :return: The JSON string
    """
    if isinstance(obj, dict):
        json_type = obj.get(JSON_TYPE_NAME, None)
        if json_type is not None:
            return obj

    if isinstance(obj, (leaf_types)):
        return obj

    if isinstance(obj, list_like_classes):
        return [to_json(item) for item in obj]

    if isinstance(obj, SubclassJSONSerializer):
        return obj.to_json()

    if inspect.isclass(obj):
        return ClassJSONSerializer.to_json(obj)

    registered_json_serializer = JSONSerializableTypeRegistry().get_external_serializer(
        type(obj)
    )

    return registered_json_serializer.to_json(obj)


@dataclass
class JSONAttributeDiff(SubclassJSONSerializer):
    """
    A class representing a shallow diff for JSON-serializable keyword arguments.
    """

    attribute_name: str = field(kw_only=True)
    """
    The name of the attribute that has changed.
    """

    added_values: List[JSONData] = field(default_factory=list)
    """
    The items that have been added to the attribute.
    """

    removed_values: List[JSONData] = field(default_factory=list)
    """
    The items that have been removed from the attribute.
    """

    def to_json(self) -> Dict[str, Any]:
        super().to_json()
        return {
            JSON_TYPE_NAME: get_full_class_name(self.__class__),
            "attribute_name": self.attribute_name,
            "removed_values": self.removed_values,
            "added_values": self.added_values,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            attribute_name=data["attribute_name"],
            removed_values=data["removed_values"],
            added_values=data["added_values"],
        )


def shallow_diff_json(
    original_json: Dict[str, Any], new_json: Dict[str, Any], **kwargs
) -> List[JSONAttributeDiff]:
    """
    Create a shallow diff between two JSON dicts.

    Result describes the changes that need to be applied to first json to get second
    json.
    :param original_json: The original JSON dict.
    :param new_json: The new JSON dict.
    :return: List of JSONAttributeDiff describing the changes that need to be applied to
        first json to get second json.
    """
    all_keys = original_json.keys() | new_json.keys()
    diffs: List[JSONAttributeDiff] = [
        diff
        for key in all_keys
        if (diff := _compute_attribute_diff(original_json, new_json, key, **kwargs))
        is not None
    ]
    return diffs


def _compute_attribute_diff(
    original_json: Any, new_json: Any, key: str, **kwargs
) -> Optional[JSONAttributeDiff]:
    """
    Compute the attribute diff for a single key between two JSON dicts.

    :param original_json: The original JSON dict.
    :param new_json: The new JSON dict.
    :param key: The key to compute the diff for. :return JSONAttributeDiff describing
        the changes that need to be applied to first json to get second json for a
        specific key.
    """
    original_values = original_json.get(key)
    new_values = new_json.get(key)

    if not isinstance(original_values, list_like_classes):
        if original_values == new_values:
            return None
        return JSONAttributeDiff(
            attribute_name=key,
            added_values=[new_values],
            removed_values=[original_values],
        )

    add = [new_value for new_value in new_values if new_value not in original_values]
    remove = [
        original_value
        for original_value in original_values
        if original_value not in new_values
    ]
    if not (add or remove):
        return None
    return JSONAttributeDiff(
        attribute_name=key, added_values=add, removed_values=remove
    )


T = TypeVar("T")


@dataclass
class ExternalClassJSONSerializer(HasGeneric[T], ABC):
    """
    ABC for all added JSON de/serializers that are outside the control of your classes.

    Create a new subclass of this class pointing to your original class whenever you
    can't change its inheritance path to `SubclassJSONSerializer`.
    """

    @classmethod
    def to_json(cls, obj: Any) -> Dict[str, Any]:
        """
        Convert an object to a JSON serializable dictionary.

        :param obj: The object to convert.
        :return: The JSON serializable dictionary.
        """

    @classmethod
    def from_json(cls, data: Dict[str, Any], clazz: Type[T], **kwargs) -> Any:
        """
        Create a class instance from a JSON serializable dictionary.

        :param data: The JSON serializable dictionary.
        :param clazz: The class type to instantiate.
        :param kwargs: Additional keyword arguments for instantiation.
        :return: The instantiated class object.
        """

    @classmethod
    def matches_generic_type(cls, clazz: Type) -> bool:
        """
        Determines if the provided class type matches the original class type.

        :param clazz: The class type to compare against the original class type.
        :return: A boolean value indicating whether the provided class type matches the
            original class type.
        """
        return cls.original_class() == clazz


@dataclass
class UUIDJSONSerializer(ExternalClassJSONSerializer[uuid.UUID]):

    @classmethod
    def to_json(cls, obj: uuid.UUID) -> Dict[str, Any]:
        return {
            JSON_TYPE_NAME: get_full_class_name(type(obj)),
            "value": str(obj),
        }

    @classmethod
    def from_json(
        cls, data: Dict[str, Any], clazz: Type[uuid.UUID], **kwargs
    ) -> uuid.UUID:
        return clazz(data["value"])


@dataclass
class ClassJSONSerializer(ExternalClassJSONSerializer[None]):
    """
    A class that provides mechanisms for serializing and deserializing Python classes to
    and from JSON representations.
    """

    @classmethod
    def to_json(cls, obj: Type) -> Dict[str, Any]:
        """
        This is a special case because we need to remember that the type of the class is
        a class, not a type.

        .. note:: We can't do type(obj) because that often returns just `type`.
        """
        return {
            JSON_TYPE_NAME: get_full_class_name(obj),
            JSON_IS_CLASS: inspect.isclass(obj),
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any], clazz: Type, **kwargs) -> Type:
        return clazz


@dataclass
class EnumJSONSerializer(ExternalClassJSONSerializer[enum.Enum]):

    @classmethod
    def to_json(cls, obj: enum.Enum) -> Dict[str, Any]:
        return {
            JSON_TYPE_NAME: get_full_class_name(type(obj)),
            "name": obj.name,
        }

    @classmethod
    def from_json(
        cls, data: Dict[str, Any], clazz: Type[enum.Enum], **kwargs
    ) -> enum.Enum:
        return clazz[data["name"]]


@dataclass
class ExceptionJSONSerializer(ExternalClassJSONSerializer[Exception]):
    @classmethod
    def to_json(cls, obj: Exception) -> Dict[str, Any]:
        return {
            JSON_TYPE_NAME: get_full_class_name(type(obj)),
            "value": str(obj),
        }

    @classmethod
    def from_json(
        cls, data: Dict[str, Any], clazz: Type[Exception], **kwargs
    ) -> Exception:
        return clazz(data["value"])


@dataclass
class NumpyNDarrayJSONSerializer(ExternalClassJSONSerializer[np.ndarray]):
    """
    External JSON serializer for numpy ndarrays.
    """

    @classmethod
    def to_json(cls, obj: np.ndarray) -> Dict[str, Any]:
        return {
            JSON_TYPE_NAME: get_full_class_name(type(obj)),
            "type": str(obj.dtype),
            "data": obj.tolist(),
        }

    @classmethod
    def from_json(
        cls, data: Dict[str, Any], clazz: Type[np.ndarray], **kwargs
    ) -> np.ndarray:
        return np.array(data["data"], dtype=data["type"])


# %% rebuilding a value JSON cannot describe on its own

REBUILT_CONTAINERS = (tuple, set, frozenset)
"""
The containers a JSON array can stand for, beside the list it already is.
"""

REBUILT_CONTAINERS_BY_NAME = {
    container.__name__: container for container in REBUILT_CONTAINERS
}
"""
The same containers, under the name an annotation writes them as.
"""


@dataclass(frozen=True)
class Rebuild:
    """
    What a value read for one field is rebuilt into.

    JSON writes a tuple and a set as the same array, and a member of a string or integer
    enumeration as the plain string or integer it is. What the value was is recorded
    only in the field's annotation, which may name it behind ``Optional`` or inside a
    container.
    """

    container: Optional[type] = None
    """
    The container the read value is put back into, or None to leave it as it was read.
    """

    member: Optional[type] = None
    """
    The enumeration each read value is looked up in, or None to leave it as it was read.
    """

    def __bool__(self) -> bool:
        """
        :return: Whether this rebuilds anything at all.
        """
        return self.container is not None or self.member is not None

    def apply(self, value: Any) -> Any:
        """
        Rebuild one read value into what its annotation named.

        :param value: The value as it was read.
        :return: The value as the annotation describes it.
        """
        if self.member is not None:
            value = (
                [self.member(item) for item in value]
                if isinstance(value, (list, tuple, set, frozenset))
                else self.member(value)
            )
        if self.container is not None and not isinstance(value, self.container):
            value = self.container(value)
        return value


@dataclass
class DataclassJSONSerializer(ExternalClassJSONSerializer[None]):
    """
    Generic JSON serializer for dataclasses.

    It creates a dict where all fields are serialized using the to_json function. If
    this is not enough, you still need to implement a custom serializer.
    """

    rebuilds_by_class: ClassVar[Dict[Type, Dict[str, Rebuild]]] = {}
    """
    Per dataclass already read, the fields whose value is rebuilt from its annotation.

    Working this out means resolving the class's annotations, which is too slow to
    repeat for every record of a file, and the answer cannot change while the class is
    loaded.
    """

    @classmethod
    def to_json(cls, obj) -> Dict[str, Any]:
        result = {JSON_TYPE_NAME: get_full_class_name(type(obj))}
        introspector = DataclassOnlyIntrospector()
        for field_ in introspector.discover(obj.__class__):
            value = getattr(obj, field_.public_name)

            if isinstance(value, (list, set)):
                current_result = [to_json(item) for item in value]
            elif isinstance(value, dict):
                keys = [to_json(k) for k in value.keys()]
                values = [to_json(v) for v in value.values()]
                current_result = {"keys": keys, "values": values}
            else:
                current_result = to_json(value)
            result[field_.public_name] = current_result
        return result

    @classmethod
    def matches_generic_type(cls, clazz: Type) -> bool:
        return is_dataclass(clazz)

    @classmethod
    def rebuilds_by_field(cls, clazz: Type) -> Dict[str, Rebuild]:
        """
        Work out which of a dataclass's fields JSON alone does not say the type of.

        A tuple and a set are both written as an array, and a member of a string or
        integer enumeration is written as the string or integer it is, so the field's
        annotation is the only record of what it was.

        :param clazz: The dataclass being read.
        :return: Per field that needs it, what a read value is rebuilt into.
        """
        if clazz not in cls.rebuilds_by_class:
            rebuilt_by_field = {}
            for holder in reversed(clazz.__mro__):
                module = sys.modules.get(holder.__module__)
                namespace = dict(vars(module)) if module is not None else {}
                for name, hint in holder.__dict__.get("__annotations__", {}).items():
                    rebuilt = cls.rebuilt_with(hint, namespace)
                    if rebuilt:
                        rebuilt_by_field[name] = rebuilt
            cls.rebuilds_by_class[clazz] = rebuilt_by_field
        return cls.rebuilds_by_class[clazz]

    @classmethod
    def rebuilt_with(cls, hint: Any, namespace: Dict[str, Any]) -> Rebuild:
        """
        Read one annotation, whether it arrives as a type or as the text of one.

        A module using postponed evaluation hands over strings, and resolving those means
        binding every name they mention -- which a class annotating a field with a name it
        imports only for type checking does not allow. The text is read instead, so no
        annotation can stop a class being deserialized.

        An annotation names what it holds at any depth: ``Optional`` wraps it and a
        container says it only in its argument, so both are looked through.

        :param hint: What the field is annotated as.
        :param namespace: The names in scope where the annotation was written.
        :return: What a value read for the field is rebuilt into.
        """
        if isinstance(hint, str):
            return cls._rebuilt_from_text(hint, namespace)
        return cls._rebuilt_from_type(hint)

    @classmethod
    def _rebuilt_from_type(cls, hint: Any) -> Rebuild:
        """
        Read an annotation that arrived as a type.

        :param hint: What the field is annotated as.
        :return: What a value read for the field is rebuilt into.
        """
        origin = get_origin(hint)
        container = origin if origin in REBUILT_CONTAINERS else None
        if inspect.isclass(hint) and issubclass(hint, enum.Enum):
            return Rebuild(member=hint)
        for argument in cls._mentioned_in(hint):
            if inspect.isclass(argument) and issubclass(argument, enum.Enum):
                return Rebuild(container=container, member=argument)
        return Rebuild(container=container)

    @classmethod
    def _mentioned_in(cls, hint: Any) -> List[Any]:
        """
        Every type an annotation names inside itself, however deeply nested.

        :param hint: What the field is annotated as.
        :return: The types the annotation names within it.
        """
        found = []
        for argument in get_args(hint):
            if argument is NoneType or argument is Ellipsis:
                continue
            found.append(argument)
            found.extend(cls._mentioned_in(argument))
        return found

    @classmethod
    def _rebuilt_from_text(cls, hint: str, namespace: Dict[str, Any]) -> Rebuild:
        """
        Read an annotation that arrived as the text of a type.

        :param hint: The text the field is annotated as.
        :param namespace: The names in scope where the annotation was written.
        :return: What a value read for the field is rebuilt into.
        """
        written = hint.strip()
        head = written.split("[", 1)[0].rsplit(".", 1)[-1].lower()
        container = REBUILT_CONTAINERS_BY_NAME.get(head)

        named = namespace.get(written)
        if inspect.isclass(named) and issubclass(named, enum.Enum):
            return Rebuild(member=named)

        for name in cls._names_written_in(written):
            mentioned = namespace.get(name)
            if inspect.isclass(mentioned) and issubclass(mentioned, enum.Enum):
                return Rebuild(container=container, member=mentioned)
        return Rebuild(container=container)

    @staticmethod
    def _names_written_in(written: str) -> List[str]:
        """
        Every name a written annotation mentions between its brackets.

        :param written: The text the field is annotated as.
        :return: The names written inside it.
        """
        if "[" not in written or not written.endswith("]"):
            return []
        inside = written[written.index("[") + 1 : -1]
        separated = inside.replace("[", ",").replace("]", ",").split(",")
        return [part.strip() for part in separated if part.strip()]

    @classmethod
    def from_json(cls, data: Dict[str, Any], clazz: Type, **kwargs) -> Self:
        rebuilds = cls.rebuilds_by_field(clazz)
        introspector = DataclassOnlyIntrospector()
        discovered_attributes = {
            attr.field.name: attr.field for attr in introspector.discover(clazz)
        }

        init_args = {}
        post_init_args = {}

        for field_name, field_ in discovered_attributes.items():
            if field_name not in data.keys():
                continue

            current_data = data[field_name]

            if isinstance(current_data, list):
                current_result = [from_json(item, **kwargs) for item in current_data]
            elif (
                isinstance(current_data, dict)
                and "keys" in current_data.keys()
                and "values" in current_data.keys()
            ):
                keys = [from_json(item, **kwargs) for item in current_data["keys"]]
                values = [from_json(item, **kwargs) for item in current_data["values"]]
                current_result = dict(zip(keys, values))
            else:
                current_result = from_json(current_data, **kwargs)

            rebuild = rebuilds.get(field_name)
            if rebuild is not None:
                current_result = rebuild.apply(current_result)

            if field_.init:
                init_args[field_name] = current_result
            else:
                post_init_args[field_name] = current_result

        instance = clazz(**init_args)
        for field_name, field_value in post_init_args.items():
            setattr(instance, field_name, field_value)
        return instance


@dataclass
class NumpyFloatJSONSerializer(ExternalClassJSONSerializer[np.float32]):
    """
    External JSON serializer for numpy floats.
    """

    @classmethod
    def to_json(cls, obj: np.float32) -> Dict[str, Any]:
        return {JSON_TYPE_NAME: get_full_class_name(type(obj)), "value": float(obj)}

    @classmethod
    def from_json(cls, data: Dict[str, Any], clazz: Type, **kwargs) -> Self:
        return float(data["value"])
