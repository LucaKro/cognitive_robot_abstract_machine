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

# Estimating connections from observed poses

A robot that opens a door, a drawer or a jar it has never seen does not know in advance whether the
part turns on a hinge, slides or screws, nor where its axis lies. It can observe the part's pose while
the part moves, for example with a camera tracking a marker on it. The `articulation_estimation`
package turns such observations into a [screw axis connection](screw_axis_connections.md).

It fits four kinds of motion to the observed poses: rigid, prismatic, revolute and helical. It then
selects the kind that explains the poses best without being needlessly rich, as
{cite:t}`sturm2011probabilistic` do.

## Observing a door

The poses below are those of a door, observed fifteen times while it swings open by 70 degrees. Each
observed position is off by about a millimeter and each orientation by about a quarter of a degree, as
with a good motion tracker:

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import plotly.graph_objects as go
from IPython.display import HTML
from scipy.spatial.transform import Rotation

from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Twist,
    Vector3,
)
from semantic_digital_twin.articulation_estimation.estimator import ArticulationEstimator
from semantic_digital_twin.articulation_estimation.fits import (
    HelicalFit,
    PrismaticFit,
    RevoluteFit,
    RigidFit,
    RotatingFit,
)
from semantic_digital_twin.articulation_estimation.observations import (
    ObservationNoise,
    PoseObservations,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ScrewAxisConnection
from semantic_digital_twin.world_description.world_entity import Body


def observed_poses(
    *,
    screw_axis: Twist,
    positions: np.ndarray,
    parent_T_connection: HomogeneousTransformationMatrix,
    connection_T_child: HomogeneousTransformationMatrix,
    noise: ObservationNoise,
    seed: int,
) -> np.ndarray:
    world = World()
    parent = Body(name=PrefixedName("parent"))
    child = Body(name=PrefixedName("child"))
    with world.modify_world():
        connection = ScrewAxisConnection.create_with_dofs(
            world=world,
            parent=parent,
            child=child,
            parent_T_connection_expression=parent_T_connection,
            connection_T_child_expression=connection_T_child,
            screw_axis=screw_axis,
        )
        world.add_connection(connection)
    poses = []
    for position in positions:
        connection.position = position
        poses.append(world.compute_forward_kinematics_np(root=parent, tip=child))
    poses = np.stack(poses)
    random_numbers = np.random.default_rng(seed)
    poses[:, :3, 3] += random_numbers.normal(
        scale=noise.position_standard_deviation, size=(len(poses), 3)
    )
    poses[:, :3, :3] = (
        Rotation.from_matrix(poses[:, :3, :3])
        * Rotation.from_rotvec(
            random_numbers.normal(
                scale=noise.orientation_standard_deviation, size=(len(poses), 3)
            )
        )
    ).as_matrix()
    return poses


tracker_noise = ObservationNoise(
    position_standard_deviation=0.001, orientation_standard_deviation=0.005
)
door_poses = observed_poses(
    screw_axis=ScrewAxisConnection.create_revolute_screw_axis(axis=Vector3.Z()),
    positions=np.linspace(0.0, np.radians(70), 15),
    parent_T_connection=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.3, y=-0.4, z=0.9),
    connection_T_child=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.4, y=0.02),
    noise=tracker_noise,
    seed=0,
)
```

`PoseObservations` holds the observed poses of the door relative to its cabinet, in the order they were
observed. `ObservationNoise` states how far an observed pose may lie from the true one, as standard
deviations. The estimator has to be told the noise: without it, a millimeter of wobble could be taken
as a motion, or a real motion as wobble.

```{code-cell} ipython3
door_observations = PoseObservations(parent_T_child=door_poses)
estimator = ArticulationEstimator(noise=tracker_noise)
door_fit = estimator.most_likely_fit(observations=door_observations)
print(type(door_fit).__name__)
```

The fit says where the hinge lies and how far the door turned at each observation:

```{code-cell} ipython3
print("axis direction:", door_fit.axis_direction.round(3))
print("axis point:", door_fit.axis_point.round(3))
print("opening angle in degrees:", np.degrees(door_fit.highest_observed_position).round(1))
```

The axis point is the point of the axis closest to the cabinet's origin, so the hinge is the vertical
line through $(0.3, -0.4)$. The figure shows the observed origins of the door, the hinge the fit found,
and the path the fit predicts between the observations. Drag it to look at it from another side:

```{code-cell} ipython3
:tags: [hide-input]


def fit_figure(fit: RotatingFit, title: str) -> go.Figure:
    observed = fit.observations.parent_T_child[:, :3, 3]
    swept_positions = np.linspace(
        fit.lowest_observed_position, fit.highest_observed_position, 100
    )
    swept = (
        fit.displacements(positions=swept_positions) @ fit.reference_parent_T_child
    )[:, :3, 3]
    origin = fit.connection_origin
    axis_ends = np.stack(
        [origin - 0.3 * fit.axis_direction, origin + 0.3 * fit.axis_direction]
    )
    figure = go.Figure(
        [
            go.Scatter3d(
                x=observed[:, 0],
                y=observed[:, 1],
                z=observed[:, 2],
                mode="markers",
                marker={"size": 4},
                name="observed origin of the child",
            ),
            go.Scatter3d(
                x=swept[:, 0],
                y=swept[:, 1],
                z=swept[:, 2],
                mode="lines",
                line={"width": 4},
                name="path the fit predicts",
            ),
            go.Scatter3d(
                x=axis_ends[:, 0],
                y=axis_ends[:, 1],
                z=axis_ends[:, 2],
                mode="lines",
                line={"width": 8},
                name="fitted screw axis",
            ),
        ]
    )
    figure.update_layout(
        title=title,
        scene={"aspectmode": "data"},
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        legend={"x": 0, "y": 1},
    )
    return figure


def embedded(figure: go.Figure) -> HTML:
    return HTML(
        figure.to_html(full_html=False, include_plotlyjs="cdn", include_mathjax=False)
    )


embedded(fit_figure(door_fit, "A door swinging open"))
```

## Choosing the kind of motion

Every kind of motion is fit to the observations, and each fit predicts a pose for every observation.
How likely the observations are under a fit depends on its errors: each position error is divided by
the position's standard deviation, and each orientation error by the orientation's. Under normally
distributed noise, the observations are the more likely, the smaller the sum $E$ of these squared,
standardized errors.

A richer kind of motion never fits worse: a helical fit can always set its pitch to zero and become
revolute. So the fits are compared by the
[Bayesian information criterion](https://en.wikipedia.org/wiki/Bayesian_information_criterion), which
adds a penalty for each number $k$ the fit chooses:

$$
\mathrm{BIC} = E + k \ln m,
$$

where $m$ is the number of observed numbers, six per pose. The positions at the observations count
among the chosen numbers, so a moving fit pays for each observation as well as for its axis: two
numbers for the direction of a prismatic axis, four for the line of a revolute axis, and one more for
the pitch of a helical one. The fit with the lowest criterion wins.

The table compares the four fits for the door, a drawer pulled out by 40 centimeters and a jar lid
unscrewed by one and a half turns of a 4 millimeter thread:

```{code-cell} ipython3
:tags: [hide-input]

drawer_poses = observed_poses(
    screw_axis=ScrewAxisConnection.create_prismatic_screw_axis(axis=Vector3.X()),
    positions=np.linspace(0.0, 0.4, 15),
    parent_T_connection=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.3, z=0.7),
    connection_T_child=HomogeneousTransformationMatrix(),
    noise=tracker_noise,
    seed=1,
)
lid_poses = observed_poses(
    screw_axis=ScrewAxisConnection.create_helical_screw_axis(
        axis=Vector3.Z(), pitch=0.004
    ),
    positions=np.linspace(0.0, 3 * np.pi, 15),
    parent_T_connection=HomogeneousTransformationMatrix.from_xyz_rpy(z=0.12),
    connection_T_child=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.03),
    noise=tracker_noise,
    seed=2,
)
fit_types = [RigidFit, PrismaticFit, RevoluteFit, HelicalFit]
rows = []
for name, poses in [("Door", door_poses), ("Drawer", drawer_poses), ("Jar lid", lid_poses)]:
    observations = PoseObservations(parent_T_child=poses)
    criteria = [
        fit_type.fit(
            observations=observations, noise=tracker_noise
        ).bayesian_information_criterion
        for fit_type in fit_types
    ]
    cells = "".join(
        f"<td><b>{criterion:,.0f}</b></td>"
        if criterion == min(criteria)
        else f"<td>{criterion:,.0f}</td>"
        for criterion in criteria
    )
    rows.append(f"<tr><th>{name}</th>{cells}</tr>")
header = "".join(f"<th>{fit_type.__name__}</th>" for fit_type in fit_types)
HTML(
    f"<table><thead><tr><th>Observed</th>{header}</tr></thead>"
    f"<tbody>{''.join(rows)}</tbody></table>"
)
```

For the door, the helical fit explains the observations as well as the revolute one, since the best
pitch it finds is almost zero, but it pays for the pitch, so the revolute fit wins. The rigid fit
explains none of the moving parts, and its criterion is larger by orders of magnitude.

## Turning a fit into a connection

`create_connection` turns a fit into a connection that moves the child as the fit does. It creates the
connection's degree of freedom in the given world, and the connection is then added like any other:

```{code-cell} ipython3
cabinet_world = World()
cabinet = Body(name=PrefixedName("cabinet"))
door = Body(name=PrefixedName("door"))
with cabinet_world.modify_world():
    door_connection = door_fit.create_connection(
        world=cabinet_world, parent=cabinet, child=door
    )
    cabinet_world.add_connection(door_connection)

door_connection.position = door_fit.positions[-1]
print(cabinet_world.compute_forward_kinematics_np(root=cabinet, tip=door).round(3))
```

At each fitted position, the connection puts the child where the fit predicts it for that observation.
A rigid fit creates a `FixedConnection` instead.

The fit fixes the connection frame, which the observations alone leave open: the frame can turn about
the axis and slide along it without changing the motion. The frame is placed on the axis at the point
closest to the child's origin, with its z axis along the screw axis and its x axis pointing toward the
child. For a prismatic fit, which has no axis position, the frame sits at the child's origin. Each fit
orients its axis so that the position grows along the observed motion.

## How the fit works

Each moving fit starts from an approximation in closed form, which takes the first observation as exact
and describes every other one by the
[rigid transformation](https://en.wikipedia.org/wiki/Rigid_transformation) from the first:

- A prismatic fit takes the direction along which these transformations translate the most, by a
  [singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition).
- A rotating fit takes the axis direction from the rotations the same way. A rotation $R$ by the angle
  $\theta$ about an axis through the point $q$, with the pitch $h$ per radian, translates by
  $(I - R)\,q + h\,\theta\,\hat{s}$ ({cite:t}`lynch2017modern`,
  [Proposition 3.25](https://hades.mech.northwestern.edu/images/7/7f/MR.pdf#page=123)), which is linear
  in $q$ and $h$, so a linear least-squares problem gives both.

The approximation then is refined by [nonlinear least squares](https://en.wikipedia.org/wiki/Non-linear_least_squares)
over the pose of the child at position zero, the positions after the first, and the axis. This spreads
the error over all observations instead of trusting the first.

```{note}
The angle of a rotation is only known up to whole revolutions, so consecutive observations have to be
less than half a revolution apart. The fit then follows a rotation past half a revolution, as the jar
lid's one and a half turns show.
```

The range of positions a fit observed, `lowest_observed_position` and `highest_observed_position`, is
not a limit of the connection: the door may well open further than it was seen to.

```{code-cell} ipython3
:tags: [remove-cell]

lid_fit = estimator.most_likely_fit(observations=PoseObservations(parent_T_child=lid_poses))
drawer_fit = estimator.most_likely_fit(
    observations=PoseObservations(parent_T_child=drawer_poses)
)
assert type(door_fit) is RevoluteFit
assert type(drawer_fit) is PrismaticFit
assert type(lid_fit) is HelicalFit
assert np.allclose(door_fit.axis_direction, [0, 0, 1], atol=0.02)
assert np.allclose(door_fit.axis_point[:2], [0.3, -0.4], atol=0.01)
assert np.isclose(door_fit.highest_observed_position, np.radians(70), atol=0.01)
assert np.isclose(2 * np.pi * lid_fit.translation_per_radian, 0.004, atol=0.0005)
assert np.isclose(lid_fit.highest_observed_position, 3 * np.pi, atol=0.02)
assert np.allclose(
    cabinet_world.compute_forward_kinematics_np(root=cabinet, tip=door),
    door_fit.predicted_parent_T_child[-1],
)
```

## References

```{bibliography}
:filter: docname in docnames
```
