from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from typing_extensions import Dict, List, Optional, TYPE_CHECKING

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
class OwnedDegreeOfFreedom:
    """
    A single degree of freedom owned by a connection, identified by its role.

    Whether it is active or passive is expressed by which list it lives in on
    :class:`DegreeOfFreedomOwnership`, not by a field here.
    """

    role: DegreeOfFreedomRole
    """The role this degree of freedom plays within the owning connection."""

    dof: DegreeOfFreedom
    """The owned degree of freedom."""


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
                OwnedDegreeOfFreedom(role=role, dof=dof)
                for role, dof in (active or {}).items()
            ],
            passive=[
                OwnedDegreeOfFreedom(role=role, dof=dof)
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
                return owned.dof
        raise KeyError(role)

    def active_dofs(self) -> List[DegreeOfFreedom]:
        """
        :return: The active degrees of freedom, in declaration order.
        """
        return [owned.dof for owned in self.active]

    def passive_dofs(self) -> List[DegreeOfFreedom]:
        """
        :return: The passive degrees of freedom, in declaration order.
        """
        return [owned.dof for owned in self.passive]

    def all_dofs(self) -> List[DegreeOfFreedom]:
        """
        :return: All owned degrees of freedom, active ones first.
        """
        return self.active_dofs() + self.passive_dofs()

    def for_world(self, world: World) -> DegreeOfFreedomOwnership:
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
                    dof=world.get_degree_of_freedom_by_id(owned.dof.id),
                )
                for owned in owned_dofs
            ]

        return DegreeOfFreedomOwnership(
            active=resolved(self.active), passive=resolved(self.passive)
        )
