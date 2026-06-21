from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Self

from typing_extensions import Dict, List, Optional, TYPE_CHECKING

from krrood.adapters.json_serializer import to_json, from_json, SubclassJSONSerializer
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)

if TYPE_CHECKING:
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.degree_of_freedom import (
        DegreeOfFreedom,
    )


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
class OwnedDegreeOfFreedom(SubclassJSONSerializer):
    """
    A single degree of freedom owned by a connection, identified by its role.

    Whether it is active or passive is expressed by which list it lives in on
    :class:`DegreeOfFreedomOwnership`, not by a field here.
    """

    role: DegreeOfFreedomRole
    """The role this degree of freedom plays within the owning connection."""

    degree_of_fredom: DegreeOfFreedom
    """The owned degree of freedom."""

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "role": to_json(self.role),
            "degree_of_fredom": to_json(self.degree_of_fredom.id),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        tracker = WorldEntityWithIDKwargsTracker.from_kwargs(kwargs)
        return cls(
            role=from_json(data["role"]),
            degree_of_fredom=tracker.get_world_entity_with_id(
                from_json(data["degree_of_fredom"], **kwargs)
            ),
        )


@dataclass
class DegreeOfFreedomOwnership:
    """
    The degrees of freedom a connection owns, split into active (controllable) and passive ones.

    The degrees of freedom are stored as object references, just like a connection's parent and
    child.
    """

    active: List[OwnedDegreeOfFreedom] = field(default_factory=list)
    """The actively controllable degrees of freedom, in declaration order."""

    passive: List[OwnedDegreeOfFreedom] = field(default_factory=list)
    """The passive degrees of freedom, in declaration order."""

    @classmethod
    def create(
        cls,
        active: Optional[Dict[DegreeOfFreedomRole, DegreeOfFreedom]] = None,
        passive: Optional[Dict[DegreeOfFreedomRole, DegreeOfFreedom]] = None,
    ) -> DegreeOfFreedomOwnership:
        """
        Build an ownership from role-to-degree-of-freedom mappings for active and passive dofs.
        """
        return cls(
            active=[
                OwnedDegreeOfFreedom(role=role, degree_of_fredom=dof)
                for role, dof in (active or {}).items()
            ],
            passive=[
                OwnedDegreeOfFreedom(role=role, degree_of_fredom=dof)
                for role, dof in (passive or {}).items()
            ],
        )

    @classmethod
    def single_active(cls, dof: DegreeOfFreedom) -> DegreeOfFreedomOwnership:
        """
        Build an ownership for a one-degree-of-freedom joint whose single dof is the active ``MAIN``.
        """
        return cls.create(active={DegreeOfFreedomRole.MAIN: dof})

    def dof_for(self, role: DegreeOfFreedomRole) -> DegreeOfFreedom:
        """
        :return: The owned degree of freedom with the given role.
        :raises KeyError: If no owned degree of freedom has the given role.
        """
        for owned in self.active + self.passive:
            if owned.role == role:
                return owned.degree_of_fredom
        raise KeyError(role)

    def active_dofs(self) -> List[DegreeOfFreedom]:
        """
        :return: The active degrees of freedom, in declaration order.
        """
        return [owned.degree_of_fredom for owned in self.active]

    def passive_dofs(self) -> List[DegreeOfFreedom]:
        """
        :return: The passive degrees of freedom, in declaration order.
        """
        return [owned.degree_of_fredom for owned in self.passive]

    def all_dofs(self) -> List[DegreeOfFreedom]:
        """
        :return: All owned degrees of freedom, active ones first.
        """
        return self.active_dofs() + self.passive_dofs()

    def copy_for_world(self, world: World) -> DegreeOfFreedomOwnership:
        """
        Re-resolve every owned degree of freedom against ``world`` by id.

        Used when copying or merging a connection into another world, so the ownership points at the
        target world's degree-of-freedom instances rather than the source world's.
        """

        def resolved(
            owned_dofs: List[OwnedDegreeOfFreedom],
        ) -> List[OwnedDegreeOfFreedom]:
            return [
                OwnedDegreeOfFreedom(
                    role=owned.role,
                    degree_of_fredom=world.get_degree_of_freedom_by_id(
                        owned.degree_of_fredom.id
                    ),
                )
                for owned in owned_dofs
            ]

        return DegreeOfFreedomOwnership(
            active=resolved(self.active), passive=resolved(self.passive)
        )
