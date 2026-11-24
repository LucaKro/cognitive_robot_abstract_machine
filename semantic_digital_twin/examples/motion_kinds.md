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

# Telling revolute, helical and sliding motions apart

The [screw motion fit](articulation_estimation.md) describes any observed motion as one screw axis and
never decides whether the part turns on a hinge, screws or slides. This page is about that decision.
For observed poses of a part, `MotionKindMixture` gives the probability of each kind of motion,
revolute, helical or sliding, and a fit of each kind that is likely enough to matter. The fit of the
most probable kind is what becomes a connection.

The first half of the page follows the data through the code, one step at a time, and checks each step
against what the code does in one call. The second half explains the theory behind each step.

```{code-cell} ipython3
:tags: [hide-input]

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from IPython.display import HTML
from plotly.subplots import make_subplots
from probabilistic_model.learning.region_graph.region_graph import RegionGraph
from random_events.variable import Continuous
from scipy.spatial.transform import RigidTransform, Rotation
from sortedcontainers import SortedSet

from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.articulation_estimation.motion_kind_classification import (
    MotionKindCircuit,
    MotionSummary,
    MotionSummaryClassifier,
    ProbabilityCalibration,
)
from semantic_digital_twin.articulation_estimation.motion_kinds import (
    MotionKind,
    MotionKindMixture,
    MotionKindPriors,
    MotionKindProbabilities,
)
from semantic_digital_twin.articulation_estimation.observations import (
    ObservationNoise,
    PoseObservations,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ScrewAxisConnection
from semantic_digital_twin.world_description.world_entity import Body

logging.getLogger("semantic_digital_twin.articulation_estimation").setLevel(
    logging.ERROR
)


def observed_poses(
    *,
    screw_axis,
    positions: np.ndarray,
    parent_T_connection: HomogeneousTransformationMatrix,
    connection_T_child: HomogeneousTransformationMatrix,
    noise: ObservationNoise,
    seed: int,
) -> PoseObservations:
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
    return PoseObservations(parent_T_child_stack=poses)


def with_noise(stack: np.ndarray, *, noise: ObservationNoise, seed: int) -> PoseObservations:
    random_numbers = np.random.default_rng(seed)
    stack = stack.copy()
    stack[:, :3, 3] += random_numbers.normal(
        scale=noise.position_standard_deviation, size=(len(stack), 3)
    )
    stack[:, :3, :3] = (
        Rotation.from_matrix(stack[:, :3, :3])
        * Rotation.from_rotvec(
            random_numbers.normal(
                scale=noise.orientation_standard_deviation, size=(len(stack), 3)
            )
        )
    ).as_matrix()
    return PoseObservations(parent_T_child_stack=stack)


def show(figure: go.Figure) -> HTML:
    return HTML(
        figure.to_html(
            full_html=False,
            include_plotlyjs="cdn",
            include_mathjax=False,
            config=dict(displayModeBar=False),
        )
    )


def stepped(figure: go.Figure, *, traces_per_step: int, labels: list[str], prefix: str, active: int):
    """
    Adds a slider to ``figure`` whose steps each show the next ``traces_per_step`` of its
    last ``len(labels) * traces_per_step`` traces; the traces before them always show.
    """
    step_count = len(labels)
    fixed = len(figure.data) - step_count * traces_per_step
    for index, trace in enumerate(figure.data[fixed:]):
        trace.visible = index // traces_per_step == active
    steps = []
    for step in range(step_count):
        visible = [True] * fixed + [
            index // traces_per_step == step
            for index in range(step_count * traces_per_step)
        ]
        steps.append(dict(label=labels[step], method="restyle", args=[{"visible": visible}]))
    figure.update_layout(
        sliders=[dict(active=active, currentvalue=dict(prefix=prefix), pad=dict(t=40), steps=steps)]
    )


kind_colors = {
    MotionKind.REVOLUTE: "#1f77b4",
    MotionKind.HELICAL: "#e0a800",
    MotionKind.SLIDING: "#2ca02c",
}
tracker_noise = ObservationNoise(
    position_standard_deviation=0.001, orientation_standard_deviation=0.005
)
```

## From observed poses to a connection

### The observations

The examples are those of the [previous page](articulation_estimation.md): a door that swings open by
70 degrees, a drawer pulled out by 40 centimeters and a jar lid unscrewed by one and a half turns of a
4 millimeter thread, each observed fifteen times with a millimeter and a quarter of a degree of noise.

```{code-cell} ipython3
:tags: [hide-input]

door = observed_poses(
    screw_axis=ScrewAxisConnection.create_revolute_screw_axis(axis=Vector3.Z()),
    positions=np.linspace(0.0, np.radians(70), 15),
    parent_T_connection=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.3, y=-0.4, z=0.9),
    connection_T_child=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.4, y=0.02),
    noise=tracker_noise,
    seed=0,
)
drawer = observed_poses(
    screw_axis=ScrewAxisConnection.create_prismatic_screw_axis(axis=Vector3.X()),
    positions=np.linspace(0.0, 0.4, 15),
    parent_T_connection=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.3, z=0.7),
    connection_T_child=HomogeneousTransformationMatrix(),
    noise=tracker_noise,
    seed=1,
)
lid = observed_poses(
    screw_axis=ScrewAxisConnection.create_helical_screw_axis(axis=Vector3.Z(), pitch=0.004),
    positions=np.linspace(0.0, 3 * np.pi, 15),
    parent_T_connection=HomogeneousTransformationMatrix.from_xyz_rpy(z=0.12),
    connection_T_child=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.03),
    noise=tracker_noise,
    seed=2,
)
observed = {"door": door, "drawer": drawer, "lid": lid}
```

### In one call

`MotionKindMixture.fit` takes the observed poses and their noise, like the screw motion fit, and
returns a mixture: the probability of each kind, and a `KindFit` of each kind it fitted. Unless given
others, it uses the default `MotionKindPriors`: how probable each kind is before any observation, and
what the fits expect of the axis and the pitch. They are explained {ref}`further down <theory-of-the-evidence>`:

```{code-cell} ipython3
print(MotionKindPriors())
```

```{code-cell} ipython3
door_mixture = MotionKindMixture.fit(observations=door, noise=tracker_noise)
print(door_mixture.kind_probabilities)
print("fitted kinds:", [fit.chart.kind.value for fit in door_mixture.fits])
print("most probable:", door_mixture.most_probable.chart.kind.value)
print("explains the observations:", door_mixture.explains_observations)
```

Without a classifier, as here, the mixture fits every kind and weighs the fits against each other.
For all three examples:

```{code-cell} ipython3
:tags: [hide-input]

mixtures = {
    name: MotionKindMixture.fit(observations=observations, noise=tracker_noise)
    for name, observations in observed.items()
}
rows = "".join(
    f"<tr><th>{name}</th>"
    + "".join(
        f"<td>{mixture.kind_probabilities.probability_of(kind=kind):.3f}</td>"
        for kind in MotionKind
    )
    + f"<td>{mixture.most_probable.chart.kind.value}</td>"
    + f"<td>{mixture.most_probable.deviation_in_noise_units:.2f}</td></tr>"
    for name, mixture in mixtures.items()
)
HTML(
    "<table><thead><tr><th>Observed</th><th>P(revolute)</th><th>P(helical)</th>"
    "<th>P(sliding)</th><th>most probable</th><th>deviation in noise units</th></tr></thead>"
    f"<tbody>{rows}</tbody></table>"
)
```

```{code-cell} ipython3
:tags: [remove-cell]

assert mixtures["door"].most_probable.chart.kind == MotionKind.REVOLUTE
assert mixtures["drawer"].most_probable.chart.kind == MotionKind.SLIDING
assert mixtures["lid"].most_probable.chart.kind == MotionKind.HELICAL
for mixture in mixtures.values():
    assert mixture.explains_observations
    assert mixture.most_probable.deviation_in_noise_units < 1.5
```

### The same, step by step

Without a classifier, the call does two things. `MotionKindMixture.fit_kinds` fits each kind, and the
probability of each kind is its evidence times its prior, normalized. The evidence and the priors are
explained {ref}`further down <theory-of-the-evidence>`:

```{code-cell} ipython3
priors = MotionKindPriors()
fits = MotionKindMixture.fit_kinds(
    kinds=list(MotionKind), observations=door, noise=tracker_noise, priors=priors
)
by_hand = MotionKindProbabilities.from_log_weights(
    log_weights=np.array(
        [fit.log_evidence + np.log(priors.probability_of(kind=fit.chart.kind)) for fit in fits]
    )
)
assert by_hand == door_mixture.kind_probabilities
```

With a classifier, the probabilities come from the classifier instead, and only the likely kinds are
fitted. A classifier is anything that implements `MotionKindClassifier`: it gives the probability of
each kind from the observed poses, without fitting them, and says which share of each kind it assumed
before seeing them, its `training_probabilities`. The rule below is not a trained classifier, but it
shows the interface. It calls a motion turning when the part turned by more than a few times the
orientation noise, and cannot tell revolute from helical:

```{code-cell} ipython3
@dataclass(kw_only=True)
class TurnedAngleRule(MotionSummaryClassifier):
    """
    Calls a motion turning when its turned angle is significant, and gives revolute and
    helical the same probability.
    """

    def log_probabilities(self, *, standardized_summary: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        significance = standardized_summary[MotionSummary.field_names.index("turned_angle_significance")]
        turning = 4.0 * (significance - 2.0)
        return np.array([turning, turning, 0.0])


field_count = len(MotionSummary.field_names)
rule = TurnedAngleRule(
    summary_mean=np.zeros(field_count),
    summary_standard_deviation=np.ones(field_count),
    calibration=ProbabilityCalibration(),
)
```

`MotionSummaryClassifier` is the base of classifiers that look at a `MotionSummary` of the poses:
eleven numbers that describe the motion, explained {ref}`further down <theory-of-the-summary>`. It summarizes
and standardizes the poses, asks its model for log probabilities and calibrates them. A model only
implements `log_probabilities`. Its training share defaults to a third of each kind.

The mixture then does four things:

1. It asks the classifier for the probability of each kind.
2. It replaces the classifier's training share by the priors, explained
   {ref}`further down <theory-of-the-priors>`.
3. It keeps the kinds at least as probable as `minimum_probability`, 5% unless given, and always the
   most probable one.
4. It fits only those kinds.

```{code-cell} ipython3
summary = MotionSummary.from_observations(observations=door, noise=tracker_noise)
print("turned angle significance:", round(summary.turned_angle_significance, 2))

classified = rule.kind_probabilities(observations=door, noise=tracker_noise)
print("classifier:", classified)
with_priors = classified.with_prior(
    prior=priors.kind_probabilities, replacing=rule.training_probabilities
)
print("with the priors:", with_priors)
likely_kinds = [
    kind
    for kind in MotionKind
    if kind == with_priors.most_probable_kind or with_priors.probability_of(kind=kind) >= 0.05
]
print("kinds to fit:", [kind.value for kind in likely_kinds])
likely_fits = MotionKindMixture.fit_kinds(
    kinds=likely_kinds, observations=door, noise=tracker_noise, priors=priors
)

classified_mixture = MotionKindMixture.fit(observations=door, noise=tracker_noise, classifier=rule)
assert classified_mixture.kind_probabilities == with_priors
assert [fit.chart.kind for fit in classified_mixture.fits] == likely_kinds
assert [fit.log_evidence for fit in classified_mixture.fits] == [fit.log_evidence for fit in likely_fits]
```

The rule gives revolute and helical the same probability, and the default priors, which make helical
motions rare, turn that into revolute. Only the revolute kind is fitted, which saves the helical fit.

### Into the world

The fit of the most probable kind holds a `ScrewMotionFit`, which creates the connection as on the
[previous page](articulation_estimation.md):

```{code-cell} ipython3
cabinet_world = World()
cabinet = Body(name=PrefixedName("cabinet"))
door_body = Body(name=PrefixedName("door"))
door_fit = classified_mixture.most_probable.screw_motion
with cabinet_world.modify_world():
    door_connection = door_fit.create_connection(world=cabinet_world, parent=cabinet, child=door_body)
    cabinet_world.add_connection(door_connection)

door_connection.position = door_fit.positions[-1]
assert np.allclose(
    cabinet_world.compute_forward_kinematics_np(root=cabinet, tip=door_body),
    door_fit.predicted_parent_T_child_stack[-1],
)
print("advance per radian:", float(door_fit.screw_axis.translation_per_radian))
```

Unlike the screw motion fit on the previous page, which let the door advance a fraction of a millimeter
per radian, the revolute fit's door does not advance at all: the kind is a constraint of the fit, and
the connection holds it in its parameters.

```{code-cell} ipython3
:tags: [remove-cell]

assert abs(float(door_fit.screw_axis.translation_per_radian)) < 1e-12
```

### Reading a classifier from a file

A trained classifier is a file. `to_file` writes a `MotionSummaryClassifier` as JSON, and
`MotionSummaryClassifier.from_file` reads it back as the model it was written with, so code that loads
a classifier does not change when the model does. Below, an untrained circuit makes the round trip;
its probabilities mean nothing until it is trained:

```{code-cell} ipython3
circuit = (
    RegionGraph(
        variables=SortedSet([Continuous(name) for name in MotionSummary.field_names]),
        partitions=2,
        depth=2,
        repetitions=2,
        seed=0,
    )
    .create_random_region_graph()
    .to_probabilistic_circuit(
        input_units=3, sum_units=3, predicted_variable=MotionKindProbabilities.kind_variable
    )
)
untrained = MotionKindCircuit(
    circuit=circuit,
    summary_mean=np.zeros(field_count),
    summary_standard_deviation=np.ones(field_count),
    calibration=ProbabilityCalibration(),
)
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "motion_kind_circuit.json"
    untrained.to_file(path=path)
    read = MotionSummaryClassifier.from_file(path=path)

assert isinstance(read, MotionKindCircuit)
assert read.kind_probabilities(observations=door, noise=tracker_noise) == untrained.kind_probabilities(
    observations=door, noise=tracker_noise
)
```

The script `semantic_digital_twin/scripts/train_motion_kind_circuit.py` trains such a circuit on
simulated motions and writes it to the file it is given, as
{ref}`explained below <theory-of-the-circuit>`. Trained files are not part of the repository.

### When no kind explains the observations

Each `KindFit` says how far the observations stray from it, in units of their stated noise:
`deviation_in_noise_units`. It is about one when the kind explains the observations and the noise is
stated right. `explains_observations` is false when the most probable kind's deviation is larger than
`MotionKindMixture.largest_explained_deviation`, three, and the mixture then logs a warning. Below, a
part turns by 90 degrees and then slides 10 centimeters along its axis, which no single kind does:

```{code-cell} ipython3
def turned_then_slid(slide: float) -> np.ndarray:
    start = HomogeneousTransformationMatrix.from_xyz_rpy(x=0.4, y=0.02).to_np()
    turning = np.concatenate([[0.0, 0.0, 1.0], -np.cross([0.0, 0.0, 1.0], [0.3, 0.0, 0.0])])
    turned = RigidTransform.from_exp_coords(np.outer(np.linspace(0.0, np.pi / 2, 15), turning)).as_matrix() @ start
    slid = (
        RigidTransform.from_exp_coords(np.outer(np.linspace(0.0, slide, 8)[1:], [0, 0, 0, 0, 0, 1.0])).as_matrix()
        @ turned[-1]
    )
    return np.concatenate([turned, slid])


unexplained = MotionKindMixture.fit(
    observations=with_noise(turned_then_slid(0.1), noise=tracker_noise, seed=3), noise=tracker_noise
)
print("most probable:", unexplained.most_probable.chart.kind.value)
print("deviation in noise units:", round(unexplained.most_probable.deviation_in_noise_units, 1))
print("explains the observations:", unexplained.explains_observations)
assert not unexplained.explains_observations
```

The mixture still names a most probable kind, and still creates a connection from it. Whether to act on
a motion the mixture does not explain, or to observe more, is up to the code that uses it.

### The pipeline in short

| Step | Code | Gives |
|---|---|---|
| Observe | `PoseObservations`, `ObservationNoise` | the poses of the child relative to the parent, and their noise |
| Summarize | `MotionSummary.from_observations` | eleven numbers that describe the motion, independent of the parent frame |
| Classify | `MotionKindClassifier.kind_probabilities`, for example a `MotionKindCircuit` read with `MotionSummaryClassifier.from_file` | the probability of each kind, assuming the classifier's training share |
| Take the priors | `MotionKindProbabilities.with_prior`, with `MotionKindPriors.kind_probabilities` | the probability of each kind under the priors |
| Fit the likely kinds | `MotionKindMixture.fit_kinds` | a `KindFit` of each kind: its chart, covariance, evidence, deviation and `ScrewMotionFit` |
| Check | `MotionKindMixture.explains_observations` | whether the most probable kind explains the observations |
| Into the world | `ScrewMotionFit.create_connection` | a screw axis connection |

`MotionKindMixture.fit` does all but the first and the last. Without a classifier, it skips the
summary and the classifier, fits every kind, and takes the probabilities from the evidence.

## Theory

(theory-of-the-kinds)=
### Why kinds at all

A screw axis describes every motion of one degree of freedom, so why decide on a kind? Because the free
fit describes the noise too. For the door, it finds a fraction of a millimeter of advance per radian;
for the drawer, a turn of less than a tenth of a degree about a line a hundred meters away. Both are
within the noise over the observed motion, but they are wrong in kind, and they grow when the
connection moves the part further than it was observed. A kind is a constraint: a revolute fit has no
advance, a sliding fit no turn, and only a helical fit has both. Fewer free numbers also make the fit
less uncertain.

```{code-cell} ipython3
:tags: [hide-input]

free_door = {fit.chart.kind: fit for fit in mixtures["door"].fits}
rows = "".join(
    f"<tr><th>{name}</th><td>{fit.chart.kind.value}</td><td>{len(fit.chart.names)}</td>"
    f"<td>{', '.join(fit.chart.axis_names)}</td></tr>"
    for name, fit in [(kind.value, free_door[kind]) for kind in MotionKind]
)
HTML(
    "<table><thead><tr><th>Kind</th><th>chart</th><th>numbers</th><th>numbers of the axis</th></tr></thead>"
    f"<tbody>{rows}</tbody></table>"
)
```

Each kind is fitted in its own *chart*, `ScrewMotionChart`: plain numbers that are all zero at the
fitted motion. Two tilts move the axis's direction; for a turning kind, two offsets move the axis
sideways; for a helical one, the pitch is the advance per radian. Six more numbers move the child's
pose at position zero, three for its rotation and three for its position. Besides the chart's numbers,
the fit has one position per observation, zero at the first.

(theory-of-the-evidence)=
### The evidence of a kind

Which kind explains the observations $D$ best? A kind with more free numbers always fits at least as
well, since a helical motion with zero pitch is a revolute one. Comparing how well each kind fits would
therefore always choose the helical kind. The *evidence* compares the kinds fairly
{cite}`mackay2003information` (section 28.1). It is the probability of the observations under the kind,
averaged over the numbers $w$ of the kind's chart and its positions, with their prior:

$$
p(D \mid k) = \int p(D \mid w, k)\, p(w \mid k)\, dw .
$$

A kind with more numbers spreads its prior over more possible motions, so it gives each of them less
probability. If the observations do not need the extra number, the evidence of the simpler kind is
larger. This is the *Occam factor*.

The integral has no closed form. The *Laplace approximation* replaces the integrand by a Gaussian
around its peak, the fitted motion $\hat w$ {cite}`mackay2003information` (chapter 27, equation
(27.8)), which makes the evidence the best fit's likelihood times an Occam factor (section 28.1,
equation (28.10)). The fit
minimizes the sum of squared standardized residuals $r(w)$, the pose errors divided by the noise and the
prior's terms, so with $J$ the Jacobian of $r$ at $\hat w$ and $n$ the number of fitted numbers,

$$
\log p(D \mid k) \approx -\tfrac{1}{2}\,\lvert r(\hat w)\rvert^2
- \tfrac{1}{2} \log \det J^\top J + \tfrac{n}{2} \log 2\pi + \text{the priors' normalizers}.
$$

The first term is how well the kind fits. The second and third are the Occam factor: $J^\top J$ is the
precision of the fitted numbers, the Gauss–Newton approximation of the curvature of
$\tfrac{1}{2}\lvert r(w)\rvert^2$ at $\hat w$, which is accurate when the residuals there are small.
Every number that the observations pin down narrowly costs the
evidence the logarithm of how narrowly. `KindFit.log_evidence` holds the result, and `KindFit.covariance`
the inverse of $J^\top J$ for the chart's numbers, with the positions marginalized out.

The priors of the numbers live in `MotionKindPriors`:

- A turning axis is expected within `axis_distance_standard_deviation`, 2 meters, of the child's origin:
  a Gaussian in the two offsets.
- A helical motion's pitch is expected within `pitch_standard_deviation`, 1 centimeter per radian, of
  zero.
- Positions are in radians for a turning kind and in meters for a sliding one. To compare the kinds,
  each turning position is counted by how far it moves the child: a radian moves the child's origin by
  $\sqrt{d^2 + h^2 + \ell^2}$, where $d$ is the distance from the axis, $h$ the pitch and $\ell$ the
  `travel_length`, 5 centimeters, which keeps a child turning about its own origin from travelling
  nowhere.

The priors on the axis direction and on the child's pose at position zero are the same for every kind,
so they cancel and are left out.

The figure shows the Occam factor at work. A lid three centimeters from its axis turns once, with a
thread whose pitch the slider sets. The bars give the probability of each kind from the evidence, under
equal priors and under the default priors. Up to about two millimeters per turn, the observations
cannot tell the advance from noise, and the evidence prefers the revolute kind even under equal
priors:

```{code-cell} ipython3
:tags: [hide-input]

pitches_per_turn = [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004]
equal_priors = MotionKindPriors(
    revolute_probability=1 / 3, helical_probability=1 / 3, sliding_probability=1 / 3
)
turning_kinds = [MotionKind.REVOLUTE, MotionKind.HELICAL]
occam_figure = make_subplots(rows=1, cols=2, subplot_titles=["equal priors", "default priors"])
occam_results = []
for pitch in pitches_per_turn:
    lid_turned_once = observed_poses(
        screw_axis=ScrewAxisConnection.create_helical_screw_axis(axis=Vector3.Z(), pitch=pitch)
        if pitch
        else ScrewAxisConnection.create_revolute_screw_axis(axis=Vector3.Z()),
        positions=np.linspace(0.0, 2 * np.pi, 15),
        parent_T_connection=HomogeneousTransformationMatrix.from_xyz_rpy(z=0.12),
        connection_T_child=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.03),
        noise=tracker_noise,
        seed=5,
    )
    revolute_fit, helical_fit = MotionKindMixture.fit_kinds(
        kinds=turning_kinds, observations=lid_turned_once, noise=tracker_noise, priors=priors
    )
    log_bayes_factor = helical_fit.log_evidence - revolute_fit.log_evidence
    occam_results.append(log_bayes_factor)
    for column, kind_priors in enumerate([equal_priors, priors]):
        weights = np.array(
            [
                fit.log_evidence + np.log(kind_priors.probability_of(kind=fit.chart.kind))
                for fit in (revolute_fit, helical_fit)
            ]
        )
        probabilities = np.exp(weights - weights.max())
        probabilities /= probabilities.sum()
        occam_figure.add_trace(
            go.Bar(
                x=[kind.value for kind in turning_kinds],
                y=probabilities,
                marker_color=[kind_colors[kind] for kind in turning_kinds],
                text=[f"{value:.3f}" for value in probabilities],
                textposition="outside",
                showlegend=False,
                hovertemplate="%{y:.4f}<extra></extra>",
            ),
            row=1,
            col=column + 1,
        )
occam_figure.update_yaxes(range=[0, 1.15], title="probability", row=1, col=1)
occam_figure.update_yaxes(range=[0, 1.15], row=1, col=2)
occam_figure.update_layout(template="plotly_white", height=420, margin=dict(l=50, r=10, t=40, b=0))
stepped(
    occam_figure,
    traces_per_step=2,
    labels=[f"{1000 * pitch:.1f}" for pitch in pitches_per_turn],
    prefix="the thread's pitch (mm per turn): ",
    active=len(pitches_per_turn) - 1,
)
show(occam_figure)
```

```{code-cell} ipython3
:tags: [remove-cell]

assert occam_results[0] < 0
assert occam_results[-1] > 5
```

(theory-of-the-priors)=
### The priors of the kinds and replacing them

Before any observation, `MotionKindPriors` expects a revolute and a sliding motion with 49% each and a
helical one with 2%, since doors and drawers are far more common than screwed parts. The probability of
a kind given the observations follows from its evidence and its prior $\pi(k)$ by
[Bayes' theorem](https://cram2.github.io/cognitive_robot_abstract_machine/probabilistic_model/probability_theory.html#bayes-theorem). A classifier learns $p(k \mid D)$ from training motions in which each kind had a
share $\pi_\text{train}(k)$, and that share is built into its probabilities. Dividing it out and
multiplying by other priors $\pi'$ gives what the classifier would have said had it been trained with
those {cite}`saerens2002adjusting` (section 2.2, equation (4)):

$$
p'(k \mid D) = \frac{p(k \mid D)\,\pi'(k) / \pi_\text{train}(k)}{\sum_{k'} p(k' \mid D)\,\pi'(k') / \pi_\text{train}(k')} .
$$

`MotionKindProbabilities.with_prior` computes this, and the mixture applies it with the classifier's
`training_probabilities` and the priors' `kind_probabilities`. Priors can therefore come from what is
known about the object, such as that it is a jar lid, without training the classifier again.

The figure takes a classifier that cannot tell a turning motion's kind, 52% revolute and 48% helical,
trained on a third of each kind, and replaces its training share by priors that give helical motions
the share the slider sets, and revolute and sliding motions half of the rest each:

```{code-cell} ipython3
:tags: [hide-input]

undecided = MotionKindProbabilities(revolute=0.52, helical=0.48, sliding=0.0)
thirds = MotionKindProbabilities(revolute=1 / 3, helical=1 / 3, sliding=1 / 3)
helical_shares = [0.01, 0.02, 0.05, 0.1, 0.2, 1 / 3, 0.5, 0.8]
prior_figure = go.Figure()
swapped_results = []
for share in helical_shares:
    prior = MotionKindProbabilities(revolute=(1 - share) / 2, helical=share, sliding=(1 - share) / 2)
    swapped = undecided.with_prior(prior=prior, replacing=thirds)
    swapped_results.append(swapped)
    values = [swapped.probability_of(kind=kind) for kind in MotionKind]
    prior_figure.add_trace(
        go.Bar(
            x=[kind.value for kind in MotionKind],
            y=values,
            marker_color=[kind_colors[kind] for kind in MotionKind],
            text=[f"{value:.3f}" for value in values],
            textposition="outside",
            showlegend=False,
        )
    )
prior_figure.update_layout(
    template="plotly_white",
    height=400,
    margin=dict(l=50, r=10, t=10, b=0),
    yaxis=dict(range=[0, 1.15], title="probability with the priors"),
)
stepped(
    prior_figure,
    traces_per_step=1,
    labels=[f"{share:.2f}" for share in helical_shares],
    prefix="prior of a helical motion: ",
    active=1,
)
show(prior_figure)
```

```{code-cell} ipython3
:tags: [remove-cell]

default_swap = swapped_results[1]
assert np.isclose(default_swap.revolute, 0.52 * 0.49 / (0.52 * 0.49 + 0.48 * 0.02))
assert swapped_results[-1].most_probable_kind == MotionKind.HELICAL
```

(theory-of-the-deviation)=
### When the evidence is too sure

The evidence is the probability of the observations *if* one of the kinds produced them, with the
stated noise. When neither holds, it can still be certain, and wrongly so. The squared residuals grow
with the number of observations and with the inverse square of the noise, so the differences between
the kinds' log evidences grow with them too: noise stated half as large makes them roughly four times
larger.

The figure turns a part by 90 degrees about an axis 30 centimeters away and then slides it along the
axis, by the distance on the horizontal axis. Hover over it to read the values at a distance. A few
millimeters of slide are enough for the evidence to call the motion helical, with a probability near
one, because a helix is the kind closest to it. The deviation of that fit,
$\sqrt{\lvert r_\text{pose}\rvert^2 / (6N - n)}$ with the pose errors $r_\text{pose}$ of the $N$
observations, grows much more slowly and passes the threshold of three only at about 25 millimeters.
In between, shaded, the evidence is sure of a kind that does not describe the motion, and nothing
flags it:

```{code-cell} ipython3
:tags: [hide-input]

slides = np.concatenate([np.arange(0.0, 0.0061, 0.0005), np.arange(0.0075, 0.0501, 0.0025)])
slide_mixtures = [
    MotionKindMixture.fit(
        observations=with_noise(turned_then_slid(slide), noise=tracker_noise, seed=3), noise=tracker_noise
    )
    for slide in slides
]
deviations = np.array([mixture.most_probable.deviation_in_noise_units for mixture in slide_mixtures])
helical_probabilities = np.array([mixture.kind_probabilities.helical for mixture in slide_mixtures])
sure_from = 1000 * slides[np.argmax(helical_probabilities > 0.99)]
flagged_from = 1000 * slides[np.argmax(deviations > MotionKindMixture.largest_explained_deviation)]
deviation_figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
for kind in MotionKind:
    deviation_figure.add_trace(
        go.Scatter(
            x=1000 * slides,
            y=[mixture.kind_probabilities.probability_of(kind=kind) for mixture in slide_mixtures],
            mode="lines+markers",
            line=dict(color=kind_colors[kind]),
            name=f"P({kind.value})",
            hovertemplate="%{y:.3f}",
        ),
        row=1,
        col=1,
    )
deviation_figure.add_trace(
    go.Scatter(
        x=1000 * slides,
        y=deviations,
        mode="lines+markers",
        line=dict(color="#555555"),
        name="deviation of the most probable kind",
        hovertemplate="%{y:.2f}",
    ),
    row=2,
    col=1,
)
deviation_figure.add_trace(
    go.Scatter(
        x=[0, 1000 * slides[-1]],
        y=[MotionKindMixture.largest_explained_deviation] * 2,
        mode="lines",
        line=dict(color="#d62728", dash="dash"),
        name="largest explained deviation",
        hoverinfo="skip",
    ),
    row=2,
    col=1,
)
for row in (1, 2):
    deviation_figure.add_vrect(
        x0=sure_from,
        x1=flagged_from,
        fillcolor="rgba(214, 39, 40, 0.08)",
        line_width=0,
        row=row,
        col=1,
    )
deviation_figure.update_yaxes(range=[-0.05, 1.05], title="probability from the evidence", row=1, col=1)
deviation_figure.update_yaxes(range=[0, 6.5], title="deviation in noise units", row=2, col=1)
deviation_figure.update_xaxes(title="slide after the turn (mm)", row=2, col=1)
deviation_figure.update_layout(
    template="plotly_white",
    height=620,
    hovermode="x unified",
    margin=dict(l=60, r=10, t=10, b=0),
    legend=dict(x=0.6, y=0.75, bgcolor="rgba(255, 255, 255, 0.85)"),
)
show(deviation_figure)
```

```{code-cell} ipython3
:tags: [remove-cell]

assert 2 < sure_from < 8
assert 20 < flagged_from < 30
assert slide_mixtures[0].most_probable.chart.kind == MotionKind.REVOLUTE
```

The deviation measures whether a kind reproduces what was observed, within a few times the noise. It
tolerates noise stated up to about three times too small, and it does not notice a mechanism that a
single kind imitates within the noise, such as a hinge whose axis drifts by a few centimeters over the
observed opening. Over the observed range, such a fit is still a good description; beyond it, it may
not be.

A classifier trained on motions under many conditions, including noise stated wrongly, is less
overconfident than the evidence, which is why the mixture can take its probabilities instead.

(theory-of-the-summary)=
### The summary a classifier sees

A `MotionSummary` describes observed poses with eleven numbers that do not change when the parent frame
moves, since every observed pose moves alike. Most are significances, $\log(1 + s)$ of a test statistic
$s$ in units of the noise, so that zero means the observations show nothing:

| Number | What it measures |
|---|---|
| `turned_angle_significance` | how far the child turned between the first and the last pose, in orientation noise |
| `log_turned_angle` | the logarithm of that angle, in radians |
| `chord_significance` | how far the child's origin moved between the first and the last pose, in position noise |
| `straight_line_deviation_significance` | how far the origin strays from the straight line that fits its path best, in position noise |
| `axial_advance_significance` | how significantly the origin advances along the turning axis as the child turns |
| `log_implied_radius` | the logarithm of the radius of the circle the chord and the turned angle imply |
| `log_position_noise`, `log_orientation_noise` | the logarithms of the stated noise |
| `log_observation_count` | the logarithm of the number of observed poses |
| `orientation_change_significance` | how significantly the orientation changes with the progress along the path, which does not assume a constant speed |
| `path_bending_significance` | how significantly the path bends away from its main direction |

The figure shows the numbers for the door, the drawer and the lid. The door and the lid turn, so their
turned angle and the change of their orientation along the path are significant, and the drawer's are
not. The axial advance, however, is about as significant for the door as for the lid. The axis it is
measured along is estimated from the noisy orientations, and a tilt of a fraction of a degree, times
the half meter the door's origin travels, looks like millimeters of advance. The summary therefore
separates turning from sliding much better than helical from revolute, and the priors weigh more in
the second decision:

```{code-cell} ipython3
:tags: [hide-input]

summaries = {
    name: MotionSummary.from_observations(observations=observations, noise=tracker_noise)
    for name, observations in observed.items()
}
summary_figure = go.Figure(
    [
        go.Bar(
            x=MotionSummary.field_names,
            y=summary.values,
            name=name,
            marker_color=kind_colors[mixtures[name].most_probable.chart.kind],
        )
        for name, summary in summaries.items()
    ]
)
summary_figure.update_layout(
    template="plotly_white",
    barmode="group",
    height=460,
    margin=dict(l=50, r=10, t=10, b=0),
    xaxis=dict(tickangle=-35),
    legend=dict(x=0.01, y=0.99),
)
show(summary_figure)
```

```{code-cell} ipython3
:tags: [remove-cell]

moved_parent = RigidTransform.from_components(
    [0.3, -1.2, 0.7], Rotation.from_rotvec([0.4, -0.2, 1.1])
).as_matrix()
moved_lid = PoseObservations(parent_T_child_stack=moved_parent @ lid.parent_T_child_stack)
assert np.allclose(
    MotionSummary.from_observations(observations=moved_lid, noise=tracker_noise).values,
    summaries["lid"].values,
)
assert summaries["drawer"].turned_angle_significance < summaries["door"].turned_angle_significance
assert summaries["drawer"].orientation_change_significance < summaries["door"].orientation_change_significance
assert abs(summaries["lid"].axial_advance_significance - summaries["door"].axial_advance_significance) < 0.5
```

(theory-of-the-circuit)=
### The circuit

`MotionKindCircuit` classifies with a
[probabilistic circuit](https://cram2.github.io/cognitive_robot_abstract_machine/probabilistic_model/circuits.html) over the standardized summary and the kind. The
probability of each kind given the summary is the circuit's
[conditional probability](https://cram2.github.io/cognitive_robot_abstract_machine/probabilistic_model/probability_theory.html#conditional-probability), which the circuit
computes exactly. The circuit is a randomly structured RAT-SPN of depth 2 with 3 repetitions and 4 units
per region, built and trained as
[Array Backends and Training](https://cram2.github.io/cognitive_robot_abstract_machine/probabilistic_model/layered_circuit.html#array-backends-and-training) describes:
discriminatively, on the cross entropy of the kind, so that it learns the conditional rather than the
distribution of the summaries.

The training motions are simulated by `train_motion_kind_circuit.py`: revolute, helical and sliding
motions of random size, with axes 5 centimeters to a meter from the child, under randomized conditions.
The noise varies, is correlated over time in half of the motions and stated too small in a third of
them, so that the classifier does not rely on the noise being stated right. The circuit kept is the one
with the lowest cross entropy on validation motions, which include small motions, larger noise,
correlated noise and understated noise.

Last, a calibration fitted on the validation motions corrects the probabilities, `ProbabilityCalibration`:
the log probabilities are divided by a temperature and shifted per kind, and normalized again
{cite}`guo2017calibration`:

$$
p_\text{calibrated}(k \mid s) \propto \exp\!\left(\frac{\log p(k \mid s)}{T} + b_k\right).
$$

A temperature above one moves the probabilities towards each other, which makes a classifier that is
too sure less so.

## Known limits and future work

- The trained classifier knows only the motions it was simulated with: a single kind, turning by up to
  270 degrees, with axes 5 centimeters to a meter from the child. It has not been tested on recorded
  motions yet, and simulating motions of several revolutions, such as unscrewing a lid, would let it
  learn what a full turn shows.
- Telling a helical motion from a revolute one needs the advance to stand out of the noise, which for a
  fine thread takes about a full turn. Below that, the priors decide. The summary's axial advance is
  measured along an axis estimated from the orientations alone, which mistakes a slight tilt for
  advance, {ref}`as shown above <theory-of-the-summary>`; measuring it along the fitted axis would
  separate the two kinds better.
- Consecutive observations have to be less than half a revolution apart, as for the screw motion fit.
- The deviation does not notice a mechanism that a single kind imitates within the noise, and the
  evidence can be sure of the wrong kind well before the deviation flags it. Noise stated far too small
  makes every motion look unexplained.

## References

```{bibliography}
:filter: docname in docnames
```
