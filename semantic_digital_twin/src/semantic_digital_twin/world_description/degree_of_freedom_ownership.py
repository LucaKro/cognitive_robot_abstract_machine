from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID

from typing_extensions import Any, Dict, List, Optional

from krrood.adapters.json_serializer import from_json, to_json


class DegreeOfFreedomRole(Enum):
    """
    The role a degree of freedom plays within the connection that owns it.

    The role identifies a degree of freedom by its meaning (e.g. the yaw rotation of a drive)
    rather than by position in an argument list, so that a connection can expose its degrees of
    freedom uniformly.
    """

    MAIN = "main"
    """The single degree of freedom of a one-degree-of-freedom joint."""

    X = "x"
    Y = "y"
    Z = "z"
    ROLL = "roll"
    PITCH = "pitch"
    YAW = "yaw"
    QX = "qx"
    QY = "qy"
    QZ = "qz"
    QW = "qw"
    X_VELOCITY = "x_velocity"
    Y_VELOCITY = "y_velocity"


@dataclass
class OwnedDegreeOfFreedom:
    """
    A single degree of freedom owned by a connection, identified by its role.

    The degree of freedom itself lives in the world; only its id is stored here.
    """

    role: DegreeOfFreedomRole
    """The role this degree of freedom plays within the owning connection."""

    id: UUID
    """The id of the owned degree of freedom."""

    is_active: bool
    """Whether the degree of freedom can be actively controlled."""


@dataclass
class DegreeOfFreedomOwnership:
    """
    The degrees of freedom a connection owns.

    This is the single place a connection declares which degrees of freedom it owns and how they are
    classified into active and passive.
    """

    dofs: List[OwnedDegreeOfFreedom] = field(default_factory=list)
    """The owned degrees of freedom, in declaration order."""

    @classmethod
    def create(
        cls,
        active: Optional[Dict[DegreeOfFreedomRole, UUID]] = None,
        passive: Optional[Dict[DegreeOfFreedomRole, UUID]] = None,
    ) -> DegreeOfFreedomOwnership:
        """
        Build an ownership from role-to-id mappings for active and passive degrees of freedom.
        """
        owned = [
            OwnedDegreeOfFreedom(role=role, id=dof_id, is_active=True)
            for role, dof_id in (active or {}).items()
        ]
        owned += [
            OwnedDegreeOfFreedom(role=role, id=dof_id, is_active=False)
            for role, dof_id in (passive or {}).items()
        ]
        return cls(dofs=owned)

    def id_for(self, role: DegreeOfFreedomRole) -> UUID:
        """
        :return: The id of the owned degree of freedom with the given role.
        :raises KeyError: If no owned degree of freedom has the given role.
        """
        for owned in self.dofs:
            if owned.role == role:
                return owned.id
        raise KeyError(role)

    def active_ids(self) -> List[UUID]:
        """
        :return: The ids of the active degrees of freedom, in declaration order.
        """
        return [owned.id for owned in self.dofs if owned.is_active]

    def passive_ids(self) -> List[UUID]:
        """
        :return: The ids of the passive degrees of freedom, in declaration order.
        """
        return [owned.id for owned in self.dofs if not owned.is_active]

    def all_ids(self) -> List[UUID]:
        """
        :return: The ids of all owned degrees of freedom, in declaration order.
        """
        return [owned.id for owned in self.dofs]

    def to_json(self) -> Dict[str, Any]:
        return {
            "dofs": [
                {
                    "role": owned.role.name,
                    "id": to_json(owned.id),
                    "is_active": owned.is_active,
                }
                for owned in self.dofs
            ]
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> DegreeOfFreedomOwnership:
        return cls(
            dofs=[
                OwnedDegreeOfFreedom(
                    role=DegreeOfFreedomRole[owned["role"]],
                    id=from_json(owned["id"]),
                    is_active=owned["is_active"],
                )
                for owned in data["dofs"]
            ]
        )
