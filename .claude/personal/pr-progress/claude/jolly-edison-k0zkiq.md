# odometry-covariance-capture (PR #9)

Plan item `odometry-covariance-capture` of `aicon-belief-integration`, wave 1,
track *uncertainty plumbing*. No dependencies. Base: `main`.
Roadmap section: `.claude/personal/plans/aicon-belief-integration/roadmap.md`.

## Plan

`OdometrySynchronizer.apply_message` drops all 36 numbers of
`message.pose.covariance`. Capture them, and publish the total variance as a
registered `FloatVariable` so the statechart can condition on base uncertainty.

Four pieces, tests first per `AGENTS.md`:

1. `PoseCovariance` + `PoseAxis` - the 36 row-major floats as a 6x6 with named
   axes; `total_variance` (trace), position and rotation halves.
2. `PoseCovarianceSource` - one read-only property, declared in the statechart
   layer so the node never imports the ROS middleware (middleware imports
   motion_statechart here, never the reverse).
3. `OdometrySynchronizer` implements it - keeps the covariance it applied.
4. `BasePoseUncertainty` - a node registering the variable and writing it each
   tick, primed to infinity so an unread covariance is not read as certainty.

## Done

- Branch opened, draft PR #9, manifest recorded `in_progress`, roadmap section
  written.

## Next

- Step 1, then 2-4 in order.
- Re-check the PR description against what actually landed before finishing.

## Notes

- Nothing runs locally: no numpy, casadi or rclpy in this container. Every test
  here is CI-verified only, same as #8 reported.
- Overlaps #8 on `motion_statechart/exceptions.py` only (each appends its own
  read-before-built exception). No dependency on #8 otherwise - deliberately not
  using its `trinary_logic_from_continuous`.
- Subscribing to tracking issue #7 was denied by the permission classifier this
  session.
