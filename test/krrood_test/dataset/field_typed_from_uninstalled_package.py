"""
Dataclasses whose fields are annotated with names only imported for type checking.

Mirrors a datastructure holding a handle of an optional dependency, such as a middleware
node, on a machine where that dependency's package is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from uninstalled_package.handles import Handle


# %% a type from a package that is not installed


@dataclass
class OwnerOfHandleFromUninstalledPackage:
    """
    Holds a handle whose type comes from a package that is not installed.
    """

    handle: Handle = field(kw_only=True)
    """
    The handle, whose type cannot be imported on this machine.
    """

    label: str = ""
    """
    A field of a type that is always available.
    """


# %% a name imported from nowhere


@dataclass
class OwnerOfNameImportedFromNowhere:
    """
    Holds a field annotated with a name that no import in this module binds.
    """

    handle: HandleNoModuleDefines = field(kw_only=True)  # noqa: F821
    """
    The field, whose annotation names nothing.
    """
