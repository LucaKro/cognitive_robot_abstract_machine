# estimator-node-base — PR #12 (draft)

Plan item `estimator-node-base`, plan `aicon-belief-integration`, track *Belief core*.
Branch `claude/plan-item-kickoff-aicon-0t046w`, based on #10's branch
`claude/belief-integration-gaussian-zi1o82`. Kickoff ran in `auto` mode.
Full reasoning is in the plan's roadmap.md, in this item's two sections.

## Done — the item is implemented and pushed

- `beliefs/estimator.py`: `PublishedValue`, `Prediction`, `EstimatorNode`.
  Abstract: `create_initial_belief`, `create_prediction`, `measure`. The base owns
  predict → measure → update → publish, registers the belief in the `BeliefContext` at
  build, publishes one estimate and one uncertainty `FloatVariable` per quantity, and
  observes whether it measured this cycle. `NodeNotBuiltError` and
  `VariableNotInBeliefError` are reused, so no exception class was added.
- `beliefs/context.py`: `BeliefContext.of(statechart_context)` — the find-or-add wiring
  #10 left to this item. The only edit to a file #10 owns; #11 does not touch that file.
- `test/giskardpy_test/test_motion_statechart/test_estimator.py`: 14 tests through the
  `RecordedReadingEstimator` mimic, expectations derived by running `GaussianBelief`.
- Verified locally — 58 pass (14 new + #10's 44 unchanged). Each behaviour confirmed
  load-bearing by mutating the implementation. The wider directory's 4 failures are
  present on the clean base too.
- Commit `eb88a5d1` pushed; PR #12 description rewritten to match; still a draft.
- Roadmap carries both the kickoff plan and an implementation section.

## Test environment (worth reusing on the next item)

PyPI: pytest numpy casadi scipy sqlalchemy rustworkx trimesh mujoco matplotlib pandas
pydot lxml piqp daqp plotly tqdm inflect lemminflect ordered_set platformdirs pillow
plyfile psutil giskardpy_bullet_bindings. Workspace: `pip install --no-deps -e` for
random_events, probabilistic_model, krrood, semantic_digital_twin, giskardpy.
`urdf_parser_py` and `xacro` fail to build a wheel here (Debian setuptools,
`install_layout`) but are pure Python — `pip download --no-binary :all:`, untar, copy the
package dir into site-packages. Run tests with `--noconftest`.

## Outstanding — for the user, not for this session

- **Dashboard not republished.** `Artifact` treats the plan dashboard as a public
  third-party artifact: `read` returns a summary, not source, so `publish` refuses with
  "you haven't viewed the latest version". Needs `force: true`, which the user has to
  ask for. #11's session hit the same wall.
- **Tracking issue #7 not subscribed** — the call was refused by this session's
  permission mode.
- Nothing reviewed on #12; CI's first run on `eb88a5d1` not yet seen.
