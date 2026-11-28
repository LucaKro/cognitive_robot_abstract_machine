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

{cite:t}`sturm2011probabilistic` fit each kind of joint separately and then choose among them. This
package fits one motion instead: the child moves along a screw axis, turning about a line and sliding
along it in proportion. A hinge turns without sliding, a drawer slides without turning and a lid does
both, so one fit describes all three, and deciding which kind of connection the motion is is left to
a later step.

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

from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.articulation_estimation.fits import (
    ScrewMotionFit,
    ScrewMotionParameters,
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
    screw_axis,
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
deviations. The fit has to be told the noise: without it, a millimeter of wobble could be taken as a
motion, or a real motion as wobble.

```{code-cell} ipython3
door_fit = ScrewMotionFit.fit(
    observations=PoseObservations(parent_T_child_stack=door_poses), noise=tracker_noise
)
```

The fit holds the motion as a `ScrewAxis` in the cabinet's frame, and the position of the door at each
observation. The screw axis gives the line the door turns about and how far the door advances along it
per radian, as the page
[Rates of motion](https://cram2.github.io/cognitive_robot_abstract_machine/krrood_spatial_math/rates_of_motion.html)
explains:

```{code-cell} ipython3
screw_axis = door_fit.screw_axis
print("axis direction:", np.round(screw_axis.unit_direction[:3].to_list(), 3))
print("axis point:", np.round(screw_axis.axis_point_closest_to_origin[:3].to_list(), 3))
print("advance per radian in mm:", round(1000 * float(screw_axis.translation_per_radian), 2))
print("opening angle in degrees:", np.degrees(door_fit.positions[-1]).round(1))
```

The axis point is the point of the axis closest to the cabinet's origin, so the hinge is the vertical
line through $(0.3, -0.4)$. The door advances along it by a fraction of a millimeter per radian. That is
noise: the fit describes the motion it sees and does not round it to a hinge.

The figure shows the observed origins of the door, the hinge the fit found, and the path the fit
predicts between the observations. Drag it to look at it from another side:

```{code-cell} ipython3
:tags: [hide-input]


def fit_figure(fit: ScrewMotionFit, title: str) -> go.Figure:
    observed = fit.observations.parent_T_child_stack[:, :3, 3]
    swept = ScrewMotionParameters(
        reference_parent_T_child=fit.reference_parent_T_child,
        positions=np.linspace(fit.positions[0], fit.positions[-1], 100),
        screw_axis_entries=fit.screw_axis.to_np(),
    ).predicted_parent_T_child_stack[:, :3, 3]
    direction = np.array(fit.screw_axis.unit_direction[:3].to_list())
    axis_point = (
        np.array(fit.screw_axis.axis_point_closest_to_origin[:3].to_list())
        if fit.screw_axis.turns_body()
        else observed[0]
    )
    axis_point = axis_point + direction * np.dot(observed.mean(axis=0) - axis_point, direction)
    axis_ends = np.stack([axis_point - 0.3 * direction, axis_point + 0.3 * direction])
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

## A drawer and a jar lid

The same fit describes a drawer pulled out by 40 centimeters and a jar lid unscrewed by one and a half
turns of a 4 millimeter thread:

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
fits = {
    name: ScrewMotionFit.fit(
        observations=PoseObservations(parent_T_child_stack=poses), noise=tracker_noise
    )
    for name, poses in [("Door", door_poses), ("Drawer", drawer_poses), ("Jar lid", lid_poses)]
}


def written_out(numbers: list[float], *, decimals: int) -> str:
    written = [f"{round(number, decimals) + 0.0:.{decimals}f}" for number in numbers]
    return f"({', '.join(written)})"


def table_row(name: str, fit: ScrewMotionFit) -> str:
    screw_axis = fit.screw_axis
    predicted = fit.predicted_parent_T_child_stack
    cells = [
        f"{np.degrees(fit.positions[-1]) if screw_axis.turns_body() else 0:.2f}°",
        f"{100 * np.linalg.norm(predicted[-1, :3, 3] - predicted[0, :3, 3]):.1f} cm",
        written_out(screw_axis.unit_direction[:3].to_list(), decimals=3),
        written_out(screw_axis.axis_point_closest_to_origin[:3].to_list(), decimals=2),
        f"{float(screw_axis.translation_per_radian):.4g} m",
    ]
    return f"<tr><th>{name}</th>{''.join(f'<td>{cell}</td>' for cell in cells)}</tr>"


header = "".join(
    f"<th>{column}</th>"
    for column in [
        "Observed",
        "turned by",
        "moved its origin by",
        "axis direction",
        "axis point",
        "advance per radian",
    ]
)
HTML(
    f"<table><thead><tr>{header}</tr></thead><tbody>"
    f"{''.join(table_row(name, fit) for name, fit in fits.items())}</tbody></table>"
)
```

The lid advances by about 0.64 mm per radian, which is 4 mm per turn, and its position follows it past
half a revolution to one and a half turns.

The drawer's row shows what noise does to a slide. A screw axis that does not turn at all fits only
poses without any noise in their orientation, so the fit lets the drawer turn by less than a tenth of a
degree, about a line more than a hundred meters away, along which it advances hundreds of meters per
radian. Over the observed motion, this is a slide of 40 cm along x, and the line and the turn are
within the noise. The fit does not decide that the drawer slides; that is up to the later step that
chooses the kind of connection.

```{code-cell} ipython3
:tags: [hide-input]

embedded(fit_figure(fits["Jar lid"], "A jar lid unscrewed"))
```

## When the tracker jumps

A tracker sometimes loses a marker for a moment and reports a pose far from the true one. Below, two of
the door's observations jumped by 5 centimeters and 0.2 radians:

```{code-cell} ipython3
jumped_door_poses = door_poses.copy()
jumped_door_poses[[4, 9], :3, 3] += [[0.0, 0.03, 0.04], [0.05, 0.0, 0.0]]
jumped_door_poses[[4, 9], :3, :3] = (
    Rotation.from_rotvec([[0.2, 0.0, 0.0], [0.0, 0.0, -0.2]])
    * Rotation.from_matrix(jumped_door_poses[[4, 9], :3, :3])
).as_matrix()

jumped_door_fit = ScrewMotionFit.fit(
    observations=PoseObservations(parent_T_child_stack=jumped_door_poses),
    noise=tracker_noise,
)
print("axis direction:", np.round(jumped_door_fit.screw_axis.unit_direction[:3].to_list(), 3))
print(
    "axis point:",
    np.round(jumped_door_fit.screw_axis.axis_point_closest_to_origin[:3].to_list(), 3),
)
```

The fit still finds the hinge. It treats errors of more than three standard deviations as outliers, as
the next section explains, so the two jumps do not pull it away. The figure shows each observation's
errors, in standard deviations, as the root mean square of its three position and three orientation
errors. The two jumps stand far out, and the other observations stay near one:

```{code-cell} ipython3
:tags: [hide-input]

errors_per_observation = np.sqrt(
    np.mean(jumped_door_fit.standardized_errors.reshape(-1, 6) ** 2, axis=1)
)
figure = go.Figure(go.Bar(x=np.arange(len(errors_per_observation)), y=errors_per_observation))
figure.add_hline(
    y=ScrewMotionParameters.outlier_threshold, line_dash="dash", annotation_text="outlier threshold"
)
figure.update_layout(
    xaxis_title="observation",
    yaxis_title="error in standard deviations",
    yaxis_type="log",
    margin={"l": 0, "r": 0, "t": 20, "b": 0},
)
embedded(figure)
```

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

A screw axis connection needs its axis to pass through the origin of its connection frame. The
connection frame therefore sits at the point of the screw axis closest to the parent's origin, along
the parent's axes, and the connection's screw axis is the fitted one, moved there. For a child that
does not turn, that point is the parent's origin. The observations leave the frame's orientation open,
since turning the frame about the axis does not change the motion, so this placement is a choice.

The positions the fit observed are not limits of the connection: the door may well open further than it
was seen to.

[Estimating connections from recorded motions](articulation_estimation_rbo.md) measures how well the
fit recovers real joints recorded by motion capture, and compares it with other estimators.

## How the fit works

The fit explains the observed poses $P_i$ by a pose $P_0$ of the child at position zero, a screw axis
$S$ in the parent frame and a position $\theta_i$ at each observation, which predict the poses

$$
\hat{P}_i = e^{[S]\,\theta_i}\,P_0,
$$

where $e^{[S]\,\theta}$ is the displacement by the position $\theta$ along $S$. The displacement and its
inverse, the logarithm, are those of
[`scipy.spatial.transform.RigidTransform`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.RigidTransform.html),
`from_exp_coords` and `as_exp_coords`, which compute the same exponential as
`ScrewAxis.to_displacement`.

- **The first guess** comes from pairs of observations, as {cite:t}`sturm2011probabilistic` start from
  minimal sets of observations. The displacement from observation $i$ to a later one $j$ is a move by
  $\theta_j - \theta_i$ along $S$, so its logarithm is a guess for $S$. The positions follow by
  projecting the logarithm of each step between consecutive observations onto it. The fit tries every
  pair, or 30 of them for many observations, and keeps the guess whose errors, each capped at three
  standard deviations, are smallest. A pair that includes an outlier gives a poor guess and loses.
- **The refinement** is [nonlinear least squares](https://en.wikipedia.org/wiki/Non-linear_least_squares)
  over $P_0$, the positions and $S$, by
  [Levenberg–Marquardt](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm) steps.
  Each position error is divided by the position's standard deviation, and each orientation error by
  the orientation's. The loss of an error $r$ is $c(r) = s^2 (\sqrt{1 + (r/s)^2} - 1)$ with $s$ three
  standard deviations: about $r^2/2$ for small errors, and growing only linearly beyond $s$, so that a
  few outliers do not dominate the sum. Each step weights an error by $c''(r)$ and takes the exact
  derivatives of the errors, from the left Jacobians of the exponentials of rotations and rigid
  transformations {cite:p}`barfoot2017state`.
- **The structure.** Only the errors of an observation depend on its position. A step therefore first
  solves for the twelve numbers all observations share, the change of $P_0$ and of $S$, through the
  [Schur complement](https://en.wikipedia.org/wiki/Schur_complement), and then for each position on
  its own, so that it takes time linear in the number of observations.
- **Several starts.** Recorded motions with many outliers can have several minima, and the best first
  guess does not always lead to the lowest. The fit refines the three best guesses for ten steps each
  and goes on with the one that is then lowest.
- **The scale.** Scaling $S$ and dividing the positions by the same factor does not change the
  predicted poses, and neither does moving $P_0$ along $S$ while shifting every position back. The
  refinement leaves every position free and, after each step, expresses the motion again with the
  first position zero and $S$, divided by the standard deviations, of length one. Pinning the first
  and the last position instead would tie these choices to two observations, which may be outliers.
  At the end the screw axis is normalized as {cite:t}`lynch2017modern` define it, the rotation part a
  unit vector if the child turns and the translation part otherwise, which makes the positions radians
  or meters.

```{note}
The angle of a rotation is only known up to whole revolutions, so consecutive observations have to be
less than half a revolution apart. The fit then follows a rotation past half a revolution, as the jar
lid's one and a half turns show.
```

```{code-cell} ipython3
:tags: [remove-cell]


def direction_of(fit: ScrewMotionFit) -> np.ndarray:
    return np.array(fit.screw_axis.unit_direction[:3].to_list())


def axis_point_of(fit: ScrewMotionFit) -> np.ndarray:
    return np.array(fit.screw_axis.axis_point_closest_to_origin[:3].to_list())


drawer_fit, lid_fit = fits["Drawer"], fits["Jar lid"]
assert np.degrees(drawer_fit.positions[-1]) < 0.1
assert np.linalg.norm(axis_point_of(drawer_fit)) > 100
assert float(drawer_fit.screw_axis.translation_per_radian) > 100
assert np.allclose(direction_of(door_fit), [0, 0, 1], atol=0.02)
assert np.allclose(axis_point_of(door_fit)[:2], [0.3, -0.4], atol=0.01)
assert abs(float(door_fit.screw_axis.translation_per_radian)) < 0.001
assert np.isclose(door_fit.positions[-1], np.radians(70), atol=0.01)
drawer_travel = (
    drawer_fit.predicted_parent_T_child_stack[-1, :3, 3]
    - drawer_fit.predicted_parent_T_child_stack[0, :3, 3]
)
assert np.allclose(drawer_travel, [0.4, 0, 0], atol=0.005)
assert np.isclose(
    2 * np.pi * float(lid_fit.screw_axis.translation_per_radian), 0.004, atol=0.0005
)
assert np.isclose(lid_fit.positions[-1], 3 * np.pi, atol=0.02)
assert np.allclose(direction_of(jumped_door_fit), [0, 0, 1], atol=0.02)
assert np.allclose(axis_point_of(jumped_door_fit)[:2], [0.3, -0.4], atol=0.01)
assert set(np.flatnonzero(errors_per_observation > 3)) == {4, 9}
assert np.allclose(
    cabinet_world.compute_forward_kinematics_np(root=cabinet, tip=door),
    door_fit.predicted_parent_T_child_stack[-1],
)
```

## References

```{bibliography}
:filter: docname in docnames
```
