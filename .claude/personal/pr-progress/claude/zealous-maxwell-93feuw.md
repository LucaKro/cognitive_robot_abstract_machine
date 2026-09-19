# belief-pickup-experiment — PR #17 (draft), base #16

Plan item `belief-pickup-experiment` (renamed from `belief-drawer-experiment`) of
`aicon-belief-integration`. **The experiment is done, on the cube, in physics, and
the deciding cell answers yes.**

## The result

Four conditions x 3 block placements, everything read from MuJoCo. mm.

| Condition | Samples | Lifted | Block moved | Hand from target | p | Finished |
|---|---|---|---|---|---|---|
| Full weight, on cube | 2 | 3/3 | 246.0 | 2.12 | 0.59 | 3/3 |
| Believed, on cube | 2 | 3/3 | 276.2 | 97.5 | 0.59 | 1/3 |
| Believed, on cube | 20 | 3/3 | 248.5 | 88.7 | 0.73 | 2/3 |
| Believed, on cube | 200 | 3/3 | 245.97 | 2.09 | 0.85 | 3/3 |
| Believed, on cube | 2000 | 3/3 | 245.95 | 2.11 | 0.92 | 3/3 |
| Full weight, on nothing | 2 | 0/3 | 0.06 | 2.12 | 0.26 | 3/3 |
| Believed, on nothing | 2 | 0/3 | 0.06 | 238.4 | 0.26 | 0/3 |
| Believed, on nothing | 20 | 0/3 | 0.06 | 252.5 | 0.07 | 0/3 |
| Believed, on nothing | 200 | 0/3 | 0.06 | 254.7 | 0.02 | 0/3 |
| Believed, on nothing | 2000 | 0/3 | 0.06 | 255.0 | 0.00 | 0/3 |

- Believed + nothing gives way at every strength; unconditional + nothing carries air
  the whole way and reports success.
- Believed + cube == unconditional + cube from 200 samples up; below that a *good*
  carry is degraded too. That band is the honest cost.
- Ordering flips at p = 0.26, not the drawer's p = 4e-4: a weight ratio is not what
  decides a QP (7 posture rows vs 3 carry rows).

## Design calls that cost time

- **A weight needs an antagonist.** Return-to-ready at WEIGHT_BELOW_COLLISION_AVOIDANCE.
- **Its window cannot end on a step it competes with** — deadlock. Both narrower windows
  hung before this was understood. It ends at ABOVE_THE_TARGET.
- **A contested carry settles ~14 mm short at 2500:1**, so carrying steps take a 30 mm
  threshold; the placement step keeps the default.
- **Block distances must stay in 0.48-0.58 m** — outside that the baseline itself fails,
  which would make a give-way the experiment's own doing.

## Done

- Drawer code deleted (`belief_drawer_experiment/` + its test file).
- `contact_likelihood.py`, `belief_experiment.py`, 24 tests.
- `PhysicalGrasp` seams; baseline numbers bit-identical to the confirmed run.
- Manifest: item renamed, optional `belief-drawer-experiment` added (`deferred`, nothing
  depends on it), `epistemic-action-demo` repointed. Saved; commented on #7.
- Roadmap section appended.
- 23 mutations; two exposed real gaps, both closed.

## Next

- Video + observation guide to the author (in flight).
- Commit, push, update PR description, keep it a draft.
- CI has not run.
- Three bugs still unfixed, each wanting its own PR off main: `is_body_in_gripper`
  deduplicates before counting rays; `MJCFParser.parse_actuator` cannot follow a tendon
  transmission; a mimic coupling miswires both finger servos.
- Dashboard is stale (`/plan-dashboard aicon-belief-integration`).
