"""
How a record is written to a run's files and read back from them.

Every step reads what the step before it wrote, so each field name is written in one
module and read in another. Mirroring each file in a dataclass writes those names once:
the reader and the writer are the same declaration, and a field that moves moves for
both.

A record is a dataclass and its fields say what it holds, so krrood's serializer is what
reads and writes it; the mapping is not written out anywhere.
"""

from __future__ import annotations

from krrood.adapters.exceptions import JSON_TYPE_NAME
from krrood.adapters.json_serializer import (
    DataclassJSONSerializer,
    SubclassJSONSerializer,
)
from krrood.utils import get_full_class_name
from typing_extensions import Any, Dict, Self


class JsonRecord(SubclassJSONSerializer):
    """
    Something a step writes into a run's directory, or reads out of a model's reply.

    A record reaches this class from two sides. A file the pipeline wrote carries the name
    of the class that wrote it, so reading it back needs nothing but the file. A model's
    reply carries no such name -- it was written by something that has never heard of these
    classes -- so the class it is read on is taken to be the class it is.
    """

    def to_json(self) -> Dict[str, Any]:
        """
        :return: The record as JSON-ready data, with the name of its class.
        """
        # SubclassJSONSerializer.to_json writes the class name and nothing else; the
        # fields are what DataclassJSONSerializer knows how to walk.
        return DataclassJSONSerializer.to_json(self)

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        """
        Build the record once the class to build has been settled.

        :param data: The record as JSON-ready data.
        :param kwargs: Passed on to the records held inside this one.
        :return: The record.
        """
        return DataclassJSONSerializer.from_json(data, clazz=cls, **kwargs)

    @classmethod
    def from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        """
        Read a record, from a file the pipeline wrote or from what a model answered.

        :param data: The record as JSON-ready data.
        :param kwargs: Passed on to the records held inside this one.
        :return: The record.
        """
        if isinstance(data, dict) and JSON_TYPE_NAME not in data:
            data = {**data, JSON_TYPE_NAME: get_full_class_name(cls)}
        return super().from_json(data, **kwargs)
