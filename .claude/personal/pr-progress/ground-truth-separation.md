## ground-truth-separation (draft #28, stacked on #24; stack #29)

Done (96073ff3, aa1b1708, pushed; PR description current):
- `MujocoSynchronizer.unobserved_connections` (empty by default): skipped
  in read and write; any connection sharing one of their DOFs (a mimic)
  too, via `_is_unobserved`, which also feeds `_is_moved_only_by_physics`.
- `divergence_log: list[DivergenceRecord(simulation_time, [DegreeOfFreedomDivergence])]`.
- `_read_6dof/_read_1dof_from_qpos` return positions (names kept: patched by
  name in test_multi_sim).
- `CabinetScene.environment_connections`; tests in test_mujoco_ground_truth.py
  and test_cabinet_scene.py (shared `drive` helper).

Local runs: `CI=true .venv/bin/python -m pytest --orm-build=never ...`
(`uv sync` with /usr/local/bin/uv; no ROS, so giskard/Tracy tests are CI-only).

Next: watch CI on the Tracy cabinet tests (never run locally); user review.
