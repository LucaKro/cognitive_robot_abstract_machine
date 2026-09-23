## ground-truth-separation (draft #28, stacked on #24; stack #29)

Review round 1 done (44b9ca07, pushed; threads replied + resolved; PR
description, item notes and roadmap current):
- `World.uncontrolled_connections` (dofs, no hardware interface) replaces the
  caller-supplied set; stepped sim sets
  `MujocoSynchronizer.physics_alone_moves_uncontrolled_connections` (was #24's
  `physics_moves_uncommanded_joints`): not written, not read back, logged.
- `DivergenceRecord(simulation_time: timedelta, divergences)`, one per DOF per read.
- `CabinetScene.environment_connections` removed.
- `_read_6dof/_read_1dof_from_qpos` return positions (names kept: patched by
  name in test_multi_sim).

Local runs: `CI=true .venv/bin/python -m pytest --orm-build=never ...`
(`uv sync` with /usr/local/bin/uv; no ROS, so giskard/Tracy tests are CI-only).

Next: CI on 44b9ca07 (Tracy cabinet tests never run locally); user review.
