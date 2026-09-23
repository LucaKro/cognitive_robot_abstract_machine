## sim-drawer-scene (draft PR #24, plan articulated-manipulation-under-uncertainty)

Done (pushed, 131d75ca):
- multi_sim: MujocoSynchronizer.physics_moves_uncommanded_joints (on in
  start_stepped_simulation) stops writing 1-DOF joints with no hardware
  interface and no actuator. Tests in test_mujoco_servos.py.
- experiments/articulated_manipulation/cabinet_scene.py:
  CabinetSceneSpecification (ArticulatedPart.DRAWER | DOOR) ->
  world_specification() (WorldSpecification: Tracy + cabinet with nested
  part specs) and to_domain_object() -> CabinetScene. Cabinet placed by
  table_T_cabinet_front: Pose2D (front centre on the table, yaw turns it).
  Root renamed "floor" (Tracy has its own "map" link).
- test_cabinet_scene.py: 15 pass with CI=true.
- Review threads (WorldSpecification; 2D pose) answered and resolved.

Local Tracy setup (CI-free runs): /tmp/claude-0/rosws with a hand-made ament
index; AMENT_PREFIX_PATH + PYTHONPATH=ament_index_python; CI=true; --orm-build never.

Next: CI on 131d75ca; nothing else open.
