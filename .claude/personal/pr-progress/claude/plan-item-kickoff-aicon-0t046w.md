# estimator-node-base — PR #12 (draft)

Plan item `estimator-node-base`, plan `aicon-belief-integration`, track *Belief core*.
Branch `claude/plan-item-kickoff-aicon-0t046w`, based on #10's branch
`claude/belief-integration-gaussian-zi1o82` (the beliefs package does not exist on main).
Kickoff ran in `auto` mode. Full reasoning is in the plan's roadmap.md section.

## Plan

1. `beliefs/estimator.py`: `Prediction` (transition + process noise + offset) and
   `EstimatorNode(MotionStatechartNode, ABC)`.
   - abstract: `create_initial_belief`, `create_prediction`, `measure`.
   - `build_artifacts`: build the prior, register it in the BeliefContext, register one
     estimate and one uncertainty `FloatVariable` per quantity.
   - `on_start`: publish the prior. `on_tick`: predict → measure → update → publish,
     observing whether anything was measured this cycle.
   - readers: `estimate_variable_of` / `uncertainty_variable_of`, raising
     `NodeNotBuiltError` before build and `VariableNotInBeliefError` for a foreign quantity.
     No new exception class.
2. `beliefs/context.py`: `BeliefContext.of(statechart_context)` — find or add the
   extension. The only edit to a file #10 owns.
3. `test/giskardpy_test/test_motion_statechart/test_estimator.py`, TDD-first, driving the
   base through a mimic estimator whose readings the test decides. Expected values derived
   by running `GaussianBelief` itself rather than hardcoded.

## Done

- Setup check + `/setup-personal-notes` (markdown/nh3 installed; all three labels present).
- Context gathered: plan.yaml, full roadmap, dependency readiness (#10 `open_ready`),
  scope-overlap check, the two sibling nodes (#8 `GraspLikelihood`, #9 `PoseUncertainty`)
  and segmind's `AbstractDetector` as the precedents.
- Branch created off #10, pushed; draft PR #12 opened; manifest `in_progress`; roadmap
  section appended.
- Test environment built in this container for the first time at kickoff rather than at
  the end: numpy/casadi/scipy/sqlalchemy/rustworkx/trimesh/mujoco/pytest from PyPI plus the
  five workspace packages `-e --no-deps`. `urdf_parser_py` is still absent, so the root
  `test/conftest.py` cannot be collected — run the test file from a copy outside `test/`.

## Next

- Write the failing tests, then `estimator.py` and the `BeliefContext.of` edit.
- Run them locally; report honestly which ones this container could not execute.
- Update PR #12's description to match what actually landed; keep it a draft.
