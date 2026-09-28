## PR #588 (cram2) - rip_grasp_descriptions

Committed by the user (up to a932a1dcf): lazy locations, Move-and-X transport, stall rule,
one ErrorSignal, ViewManager/Arms enum removed (actions take `arm: Arm`), CarryAction and
coraplex/utils.py deleted, pre-grasp as a grasp-frame step (proved identical poses).

Uncommitted 2026-09-25 (user: "dont commit"):
- FaceAtAction(target) is a core action (core/navigation.py) that only turns: TurnMotion =
  Parallel(Pointing base forward axis at target lowered to base height, CartesianPosition
  hold at the base's own origin, bound on start). FaceAndLookAtAction(face_at, look_at) in
  composite/facing.py keeps turn-then-look.
- Move-and-X steps expose sub-actions: navigate/face_and_look_at + pick_up/place/
  open_container, each with from_standing_position(...). Underspecified callers write nested
  a(...) matches (standing pose = inner NavigateAction variable). TransportAction lost its
  MoveTorsoAction(HIGH).
- PlaceAction(object_designator, target_location): no arm; `_arm` (init=False) = arm holding
  the object, else the preceding PickUpAction of that object (user: "for now"), else
  ObjectIsNotHeld (replaced NothingToPlace). Conditions check every arm. Conditions are
  built but never evaluated at runtime (_add_condition_monitors unused).
User committed the renames as 2503e37dc, but test_grasp_candidates.py (plain mv) is still untracked.
Nearest-grasp: user refuses krrood changes. Generative backend only treats Variable instances as
variables (match.py:195); FlatVariable, InstantiatedVariable, Attribute are not, so neither a derived
grasp domain nor a pair domain's attributes work. Open: pick a krrood-free option (see chat 2026-09-28).
ProbabilisticBackend: one truncation per domain object, cost ~n^3 (2000 location samples ~hours).
Uncommitted isolated test: test_backends.py::test_generating_from_more_objects_takes_less_than_quadratically_longer
(100 vs 200 KRROODPositions, min of 3 timings, ratio ~7 vs limit 4); fails here and on cram2/main 6645c1892.
Uncommitted (user adds/commits himself): facing targets are Pose(reference_frame=<body>) in from_grasp,
MoveAndPickUp.from_standing_position, _make_open_container_actions, MoveAndOpen.from_standing_position;
4 tests in test_transporting.py, old world-frame assertion removed (user approved). 33 tests pass.
Nearest grasp (uncommitted, user reviews): instance methods on MoveAndPickUpAction -
grasp_faces_its_standing_position() (used as step.where(step.variable.<method>())), misalignment_of(grasp),
ClassVar facing_grasps_per_standing_position=3 - plus classmethod from_graspable only. Predicate class and
extra factories removed at user's request. 39 transport/open tests pass. User removed the krrood scaling
test from test_backends.py; its imports (time, Match) are left over - asked.
Open: tool_paths._local_bounding_box removal (user asked; cutting's duration scale uses the
collision box while its path uses the visual box - ask which to keep).
Tests: --orm-build never, systemd MemoryMax cap, <= 8 workers, RViz via scratchpad
rviz_debug_plugin.py. Nothing committed/pushed by Claude.
