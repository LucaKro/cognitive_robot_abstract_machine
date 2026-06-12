---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(regions)=
# Regions

Regions are entities in the world similar to bodies; they live in the same
kinematic tree but represent semantic areas rather than physical geometry.
For example, a region can represent the surface of a table that you can
place objects on, or the opening of a container you can insert items into.

This tutorial explores a region describing the supporting surface of a table-top.

Used Concepts:
- [](creating-custom-bodies)
- [](world-structure-manipulation)
- [](world-state-manipulation)

First, let's create a simple table with one leg. We describe the bodies with `BodySpec`s and spawn them into the
world, as introduced in [](creating-custom-bodies).

```{code-cell} ipython3
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.specs import BodySpec, RegionSpec
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.spatial_computations.raytracer import RayTracer

world = World()

root = Body(name=PrefixedName("root"))
with world.modify_world():
    world.add_body(root)

table_leg = BodySpec.box(PrefixedName("leg"), scale=Scale(0.1, 0.1, 0.6)).spawn(
    world, connection_type=Connection6DoF
)
table_top = BodySpec.box(PrefixedName("top"), scale=Scale(1, 1, 0.05)).spawn(
    world,
    parent=table_leg,
    pose=HomogeneousTransformationMatrix.from_xyz_rpy(z=0.3),
)
```

Next, we describe a region for the top of the table. We declare that the region is a very thin box that sits on
top of the table-top. Regions are described by a `RegionSpec`, the counterpart of `BodySpec` for semantic areas.

```{code-cell} ipython3
table_surface_spec = RegionSpec(
    name=PrefixedName("supporting surface of table"),
    shapes=[
        Box(
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(z=0.05 / 2),
            scale=Scale(1, 1, 0.001),
        )
    ],
)
```

Regions are connected the same way bodies are connected.
Hence, you can specify how the regions move w. r. t. to a body or even another region.
We will now say the the region moves exactly as the table top moves, by spawning it with the table top as its
parent (the default `FixedConnection` keeps it rigidly attached).

```{code-cell} ipython3
table_surface = table_surface_spec.spawn(world, parent=table_top)
print(world.regions)
```

We can now see that if we move the table, we also move the region.

```{code-cell} ipython3
print(table_surface.global_pose.to_position().to_np()[:3])

with world.modify_world():
    table_leg.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1.0, y=2.0, reference_frame=table_leg
    )

print(table_surface.global_pose.to_position().to_np()[:3])
```

Note that Regions are a relatively new concept that may change in the future.
