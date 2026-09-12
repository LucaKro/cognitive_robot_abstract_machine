from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.semantic_annotations.mixins import HasDrawers, HasSupportingSurface
from semantic_digital_twin.semantic_annotations.semantic_annotations import Table

@dataclass(eq=False)
class KitchenIsland(Table, HasDrawers, HasSupportingSurface):
    pass
