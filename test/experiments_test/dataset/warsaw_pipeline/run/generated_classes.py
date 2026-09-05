from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.semantic_annotations.mixins import HasDrawers, HasRootBody, HasRootRegion, HasSupportingSurface, IsStorageSpace
from semantic_digital_twin.semantic_annotations.semantic_annotations import Table

@dataclass(eq=False)
class Ceiling(HasRootRegion):
    pass


@dataclass(eq=False)
class Container(IsStorageSpace, HasRootBody):
    pass


@dataclass(eq=False)
class CuttingBoard(HasSupportingSurface, IsStorageSpace):
    pass


@dataclass(eq=False)
class Faucet(HasRootBody):
    pass


@dataclass(eq=False)
class KitchenIsland(Table, HasDrawers, HasSupportingSurface):
    pass
