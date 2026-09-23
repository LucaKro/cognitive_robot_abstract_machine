## sim-drawer-scene (draft PR #24, plan articulated-manipulation-under-uncertainty)

Plan (roadmap section "sim-drawer-scene" has the reasons):
1. TDD in test/semantic_digital_twin_test: in stepped MuJoCo simulation, a
   world-side change of an uncontrolled 1-DOF joint (no hardware interface)
   must not move the physical joint. Fails on main (_write_1dof_to_qpos
   teleports it). Fix in MujocoSynchronizer's write path, stepped mode only.
2. Scene builder in experiments: Tracy + cabinet with drawer (slider + handle),
   door (hinge + handle) variant, from existing semantic-annotation factories.
3. Scene tests (CI only, Tracy): giskard commanding the drawer joint leaves the
   physical drawer put; Tracy's hand pushing the front moves it, and the world
   reads it back. Same for the door.

Done: branch + draft PR + manifest/roadmap recorded.
Next: step 1 (local env: mujoco + workspace install, Tracy likely CI-only).
