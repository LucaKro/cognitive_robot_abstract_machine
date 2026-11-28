# Estimating connections from recorded motions

The [screw motion fit](articulation_estimation.md) recovers the axis of a real joint, recorded by
motion capture, to a median of 1.35° in direction and 2.25 mm in position for hinges, and 1.41° in
direction for slides. With a tenth of the motion seen, its median error is 10.9° for hinges and 18.9°
for slides. This page describes the experiment behind these numbers, how to run it, and the other
estimators measured on the same recordings.

## The recordings

The RBO dataset {cite:p}`martin2018rbo,martin2017rbodata` holds 358 recordings of people moving 14
articulated objects, among them a microwave, a cabinet, a laptop, a book, pliers and a folding rule.
Motion capture tracks a set of markers on each part at 100 Hz. Each object comes with a model naming
its joints, each joint by the marker sets on its two parts and by its kind, revolute or prismatic.

The experiment uses only the poses of the marker sets and the object models. For each joint of each
recording it takes the poses at which both marker sets were seen, and the pose of the child part
relative to the parent part at each of them is one observation. A recording therefore yields one
joint motion per joint that moved, with a median of 869 observations for revolute joints and 1,038
for prismatic ones, and up to 6,911.

## What the fit is compared with

The dataset gives no axes for its joints. Each joint motion is therefore compared with a *reference*
made from the fits of the other recordings of the same joint: the same object, the same recording
date (which fixes where the markers sat) and the same joint. The reference takes the mean direction
of those fits and, for a revolute joint, the median of their axis points and of their pitches, so
that one poor fit does not move it.

- A joint is evaluated only with at least three recordings, so that every reference combines at
  least two fits.
- A recording counts only if the joint moved at least 15° or 5 cm in it, since a part that barely
  moves shows little of its axis.
- The reference is not ground truth. A bias all recordings of a joint share, such as a marker set
  placed off the part's true frame, does not show in it.

The comparison reports:
- the angle between the fitted direction and the reference's;
- for a revolute joint, the distance between the two axes next to the child part, and the
  difference in pitch;
- for a prismatic joint, how much the fit turns the child per meter it slides.

For a prismatic joint the fitted direction is the direction in which the child's origin moves,
$v + \omega \times p$ for a screw axis $(\omega, v)$ and the child's position $p$: $v$ alone is the
velocity of the parent's origin, which depends on where that origin lies once the fit turns at all.

## The noise

Each fit needs the noise of its observations. The experiment fits each joint motion three times,
the first time assuming 1 mm and 5 mrad and each further time with the noise estimated from the
errors of the fit before. The estimate is the median absolute error scaled to a standard deviation,
so that outliers barely change it. It is biased low for orientations: the fit absorbs the noise about
the axis into each observation's position, which makes one of the three orientation errors small. On
a simulated door with 2 mrad of orientation noise the estimate was 1.15 mrad.

## Results

| Joint kind | Motions | Objects | Direction, median [90th percentile] | Axis distance | Pitch |
| --- | --- | --- | --- | --- | --- |
| Revolute | 275 | 11 | 1.35° [4.91°] | 2.25 mm [10.4 mm] | 0.56 mm/rad |
| Prismatic | 138 | 5 | 1.41° [3.77°] | | |

Six revolute and five prismatic motions are more than 10° off their reference. The prismatic ones are
slides that the fit describes as a tight turn, up to 8,565° per meter, which is a question of which
kind of joint the motion is rather than of its axis.

The error by object, as a median and the largest:

| Object | Kind | Motions | Direction, median | Largest |
| --- | --- | --- | --- | --- |
| ikea | revolute | 14 | 0.41° | 4.2° |
| treasurebox | revolute | 26 | 0.57° | 2.0° |
| cardboardbox | revolute | 24 | 0.72° | 2.1° |
| globe | revolute | 25 | 0.75° | 4.3° |
| microwave | revolute | 24 | 1.21° | 7.5° |
| rubikscube | revolute | 26 | 1.24° | 5.0° |
| laptop | revolute | 24 | 1.30° | 4.8° |
| tripod | revolute | 18 | 1.43° | 7.8° |
| book | revolute | 20 | 3.14° | 52.1° |
| pliers | revolute | 25 | 3.45° | 12.7° |
| foldingrule | revolute | 49 | 3.72° | 77.1° |
| tripod | prismatic | 9 | 0.70° | 2.4° |
| clamp | prismatic | 24 | 0.88° | 5.3° |
| ikeasmall | prismatic | 46 | 1.19° | 71.1° |
| cabinet | prismatic | 47 | 2.15° | 78.6° |
| ikea | prismatic | 12 | 2.88° | 34.0° |

### Early estimates

The fit of the observations up to a share of a motion's *excursion*, the farthest the joint has
moved, shows how well the axis is known while the motion is still going on. The excursion is the
fitted angle for a revolute joint and the distance the child's origin travelled for a prismatic one,
each a running median over 25 observations, so that an outlier, which the fit may place whole
revolutions away, does not count.

| Share of the excursion | Revolute direction, median [90th percentile] | Prismatic direction |
| --- | --- | --- |
| 10 % | 10.9° [79.6°] | 18.9° [82.8°] |
| 25 % | 4.3° [35.7°] | 4.5° [75.0°] |
| 50 % | 2.5° [10.0°] | 2.6° [64.9°] |
| all | 1.35° [4.91°] | 1.41° [3.77°] |

### Time

A fit took a median of 1.2 s, and 18.8 s for the longest motion, 6,911 observations, with six fits
running at once on a 12-core machine.

## Running it

The experiment lives in the `experiments` package, in `experiments.articulation_estimation`. It
downloads what it needs from the dataset's Zenodo record into the given directory on first use and
keeps only the poses of each recording and the object models, about 470 MB for all of them:

```bash
python -m experiments.articulation_estimation.runner \
    --directory <data directory> --output results.json --processes 6
```

`--objects microwave cabinet` limits it to some objects. The results hold, for every joint motion,
its errors against the reference, the noise estimated for it, and the fits of the observations up to
10 %, 25 % and 50 % of its excursion.

## Other estimators measured

These were measured to choose the estimator; only the fit above is part of the experiment.

### Solvers for the fit

The fit minimizes the summed `soft_l1` loss over the pose at position zero, the screw axis and one
position per observation. Seven slow recordings of the microwave, from 183 to 2,114 observations:

| Solver | Time per fit | Result |
| --- | --- | --- |
| `scipy.optimize.least_squares`, dense Jacobian (the earlier fit) | 6–285 s | the reference minimum |
| the same, with the Jacobian's sparsity given | stopped at its limit of 3,000 evaluations on 3 of 4 recordings | the same axis when it finished |
| the same, with tolerances of $10^{-12}$ for its inner solver | up to 7 s on partial motions, 85–414 s on whole ones | the same minimum |
| `least_squares` with `method="dogbox"` | 0.2–4.4 s | stops early: cost up to 0.7 % above |
| the structured Levenberg–Marquardt solver, finite differences | 0.1–12 s, two of the seven at its limit of 200 steps | axes within about $10^{-4}$ |
| the same, with all positions free | 0.2–2.4 s | the same axes, 21–32 steps |
| the same, with exact derivatives (the fit above) | 0.1–0.9 s | the same axes |

Pinning the first and the last position, as the earlier fit did to fix the scale of the screw axis
and where position zero lies, ties both to two observations. When those are outliers, the solver
crept along the nearly flat direction this leaves for 160 of its 200 steps while the axis no longer
changed.

On 354 motions of 10 objects, each cut to its first 600 observations, the fit above reaches the
same cost as `least_squares` to 0.01 % on 339, a cost up to 0.29 % lower on 12 and a cost 1.1–1.6 %
higher on 3. It took a median of 0.56 s and at most 2.16 s, against 1.21 s and 177 s. Starting from
the best initial guess alone, 5 of those motions ended in a minimum with a cost 25–75 % higher;
refining the three best guesses for ten steps each and going on with the lowest brought four of them to the same cost as `least_squares`, and the fifth to 1.4 % above it.

### Loss functions

On a simulated drawer whose first pose jumped, over 10 seeds, the Cauchy loss left the direction a
median of 18.6° off (39.5° at most), against 0.23° for `soft_l1`.

### Estimators that follow the motion

Estimators that update with each new observation, on the same recordings cut to their first 3,000
observations, against references from the other recordings' fits. The direction error of revolute
joints, median [90th percentile], over 271 motions:

| Estimator | 10 % | 25 % | 50 % | all | Time per update |
| --- | --- | --- | --- | --- | --- |
| Refit at each new keyframe | 9.24° [61.2°] | 3.70° [15.7°] | 2.07° [7.78°] | 1.55° [5.14°] | 225 ms, 4.8 s at most |
| Extended Kalman filter from a first fit | 27.5° [86.7°] | 10.1° [84.6°] | 8.83° [84.0°] | 8.69° [84.2°] | 2.3 ms |
| The same, restarted from a refit after rejected observations | 14.7° [86.9°] | 4.78° [72.7°] | 2.46° [10.7°] | 1.62° [5.18°] | 2.5 ms, 312 s per recording |
| The same filter assuming 10 times the noise | 18.4° [84.5°] | 6.19° [74.2°] | 3.36° [62.3°] | 2.08° [52.1°] | 2.3 ms |

For the 138 prismatic motions the refit was 54.9°, 11.7°, 4.6° and 3.47° off at the same shares, and
the filter assuming 10 times the noise 69.5°, 47.7°, 13.5° and 8.53°.

- **The filters believe the recordings too much.** The errors of a fit stay correlated over 40–100
  consecutive observations, so 100 Hz carries far fewer independent observations than it seems. The
  normalized error of the axis against the reference, about 4.4 for a consistent estimator, had a
  median of 164 for the refit and 11.5 for a filter assuming 30 times the noise. On simulated motions
  with independent noise, the mean was between 3.8 and 5.7 for every estimator, against 5 expected.
- **Starting late helps the filters.** Waiting until the joint has moved 10° or 5 cm before the first
  estimate lowered the share of motions more than 10° off at the end from 20 % to 4 % for the filter
  assuming 10 times the noise, on 100 joints of the pliers, laptop, globe and treasure box.
- **A refit handing over to a filter** once three refits in a row agree within 1° left none of those
  motions more than 10° off at the end, at about 2.6 ms per update and one fit of about 0.5 s at the
  hand-over.

## References

```{bibliography}
:filter: docname in docnames
```
