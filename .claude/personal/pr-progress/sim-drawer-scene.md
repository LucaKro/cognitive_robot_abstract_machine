## sim-drawer-scene (draft PR #24, plan articulated-manipulation-under-uncertainty)

Done (pushed, 459de055):
- multi_sim: MujocoSynchronizer.physics_moves_uncommanded_joints (on in
  start_stepped_simulation) stops writing 1-DOF joints with no hardware
  interface and no actuator. Tests in test_mujoco_servos.py.
- experiments/articulated_manipulation/cabinet_scene.py:
  CabinetSceneSpecification (ArticulatedPart.DRAWER | DOOR) ->
  world_specification() (WorldSpecification: Tracy + cabinet with nested
  part specs) and to_domain_object() -> CabinetScene. Root renamed to
  "floor" (Tracy has its own "map" link; MuJoCo needs unique names).
- test_cabinet_scene.py: 14 pass with CI=true.
- Review thread "use a WorldSpecification" answered and resolved.
- Video of the scene sent in chat (scratch scripts, not committed).

Local Tracy setup (CI-free runs): /tmp/claude-0/rosws with a hand-made ament
index; AMENT_PREFIX_PATH + PYTHONPATH=ament_index_python; CI=true; --orm-build never.

Next: CI on 459de055; nothing else open.
