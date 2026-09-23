## ground-truth-separation (draft #28, stacked on #24; stack #29)

Plan (roadmap section, 2026-09-23): `MujocoSynchronizer.unobserved_connections`
(set, empty by default) - read skips them, write skips them too (joined into
`_is_moved_only_by_physics`); 1-DOF and 6-DoF. `divergence_log`:
`DivergenceRecord(simulation_time, [DegreeOfFreedomDivergence(dof, world, physics)])`
appended on every read, physics values from the shared qpos->state conversion.
`CabinetScene.environment_connections` = cabinet connections with DOFs
(incl. its parent connection). Evaluator reads MuJoCo directly.

Done: branch, draft PR, stack registration, manifest `open`/`record`.

Next (TDD): failing tests in test/semantic_digital_twin_test/test_adapters/
(new file, reuse `_pendulum_world` from test_mujoco_servos via relative import)
+ experiments cabinet test; then implement in multi_sim.py and cabinet_scene.py.
MuJoCo tests only run in CI (runs_in_continuous_integration).
