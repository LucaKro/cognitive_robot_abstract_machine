## sim-drawer-scene (draft PR #24, plan articulated-manipulation-under-uncertainty)

Done (pushed, 553b7a8b):
- multi_sim: MujocoSynchronizer.physics_moves_uncommanded_joints (on in
  start_stepped_simulation) stops writing 1-DOF joints with no hardware
  interface and no actuator. Tests in test_mujoco_servos.py (TDD: failed first).
- experiments/articulated_manipulation/cabinet_scene.py: CabinetSceneBuilder
  (ArticulatedPart.DRAWER | DOOR) -> CabinetScene (world, robot, cabinet, part,
  handle, mechanism, set_opening). Cabinet fixed on Tracy's table, front 0.85 m
  from the arms' edge, 0.35 m left.
- test/experiments_test/articulated_manipulation/test_cabinet_scene.py: joint
  command does not move part (fails on main: drawer 0.248 m, door 1.569 rad);
  hand push closes it (drawer 0.15->0, door 0.6->0).

Local Tracy setup (CI-free runs): /tmp/claude-0/rosws with a hand-made ament
index; AMENT_PREFIX_PATH + PYTHONPATH=ament_index_python; CI=true; --orm-build never.

Next: check CI on PR #24; fix anything red.
