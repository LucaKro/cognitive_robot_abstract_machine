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
(creating-custom-bodies)=
# Creating Custom Bodies

The tutorial demonstrates the creation of a body and its visual and collision information.
First, let's create a world.

```{code-cell} ipython3
from pkg_resources import resource_filename

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import HomogeneousTransformationMatrix, RotationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

world = World()
```

Next, let's describe the visual and collision information for our body.

The collision describes the geometry to use when calculating collision relevant things, for instance if your robot is colliding with a table while moving.
The visual information is purely for esthetics.
Both of these are lists of shapes.

The recommended way to describe a body is the declarative {py:class}`semantic_digital_twin.world_description.specs.BodySpec`:
you state the name and shapes once, and the spec can then create as many bodies from that description as you like.
The `shapes` field is used for both collision and visual geometry; only when the visual geometry should differ
(for example, a detailed mesh for rendering but simple primitives for collision checking) do you additionally set
`visual_shapes`.

Supported Shapes are:
- Box
- Sphere
- Cylinder
- Mesh

Finally, in our kinematic structure, each entity has a name. For this we can use a simple datastructure called `PrefixedName`. You always need to provide a name, but the prefix is optional. This is for human readability and allows for easy identification of entities. For uniqueness constraints, a UUID is used and stored in the `id` field.

```{code-cell} ipython3
import os
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.world_description.specs import BodySpec
from semantic_digital_twin.world_description.geometry import Box, Scale, Sphere, Cylinder, Mesh, Color

box_origin = HomogeneousTransformationMatrix.from_xyz_rpy(x=0, y=0, z=0, roll=0, pitch=0, yaw=0)
box = Box(origin=box_origin, scale=Scale(1., 1., 0.5), color=Color(1., 0., 0., 1., ))

sphere_origin = HomogeneousTransformationMatrix.from_xyz_quaternion(pos_x=0, pos_y=1., pos_z=1., quat_x=0., quat_y=0., quat_z=0.,
                                                   quat_w=1.)
sphere = Sphere(origin=sphere_origin, radius=0.4)

cylinder_origin = HomogeneousTransformationMatrix.from_point_rotation_matrix(point=Point3.from_iterable([1, -1, 2]),
                                                                  rotation_matrix=RotationMatrix.from_axis_angle(
                                                                      Vector3.from_iterable([1., 0., 0.]), 0.8, ),)
cylinder = Cylinder(origin=cylinder_origin, width=0.05, height=0.5)

mesh = Mesh(origin=HomogeneousTransformationMatrix(),
            filename=os.path.join(resource_filename("semantic_digital_twin", "../../"), "resources", "stl", "milk.stl"))

body_spec = BodySpec(
    name=PrefixedName("my first body", "my first prefix"),
    shapes=[cylinder, sphere, box],  # collision geometry
    visual_shapes=[mesh],            # visual geometry, omit to reuse `shapes`
)
body = body_spec.to_body()
```

For bodies with a single primitive shape, `BodySpec` offers convenience constructors so you do not have to build
the shapes yourself: `BodySpec.box`, `BodySpec.sphere`, `BodySpec.cylinder` and `BodySpec.mesh`.

```{code-cell} ipython3
crate_spec = BodySpec.box(PrefixedName("crate"), scale=Scale(0.5, 0.5, 0.5))
```

A spec never becomes part of a world itself — every call to `to_body()` produces a fresh, independent body with
its own copies of the shapes. This makes specs reusable: create the same description several times under
different names with `to_body(name=...)`.

When modifying your world, keep in mind that you need to open a `world.modify_world()` whenever you want to add or remove things to/from your world

```{code-cell} ipython3
with world.modify_world():
    world.add_body(body)

from semantic_digital_twin.spatial_computations.raytracer import RayTracer
rt = RayTracer(world)
rt.update_scene()
rt.scene.show("jupyter")
```

Once the world has a root body, additional bodies are best added with `spawn`, which creates a body from the
spec and connects it to a parent — the world root by default — in a single step:

```{code-cell} ipython3
crate = crate_spec.spawn(world, pose=HomogeneousTransformationMatrix.from_xyz_rpy(x=2.0))

rt = RayTracer(world)
rt.update_scene()
rt.scene.show("jupyter")
```

By default `spawn` attaches the new body with a `FixedConnection`. Pass `connection_type=Connection6DoF` for a
body that can move freely, `parent=...` to attach it to a specific body, and `name=...` to spawn the same spec
multiple times under different names.

## Under the hood

A `Body` can also be constructed directly from shape collections — this is essentially what `to_body()` does for
you (plus copying the shapes, so the spec stays reusable). The direct construction is occasionally useful, for
example inside file format parsers:

```{code-cell} ipython3
from semantic_digital_twin.world_description.shape_collection import ShapeCollection

manual_body = Body(
    name=PrefixedName("manual body"),
    visual=ShapeCollection([mesh]),
    collision=ShapeCollection([cylinder, sphere, box]),
)
```

If you think you have understood everything in this tutorial, you may try out 
[our self-assessment quiz for this user guide](creating-custom-bodies-quiz)

```{warning}
Using the above method to visualize your world only really makes sense in a notebook setting like this.
If you want learn how to properly visualize your worlds, check out the [](visualizing-worlds) tutorial.
```

```{warning}
If you are trying to create multiple bodies without connecting them,
you will run into trouble with the world validation.
If you want to see how to create multiple bodies, 
check out the [](world-structure-manipulation) tutorial.
```