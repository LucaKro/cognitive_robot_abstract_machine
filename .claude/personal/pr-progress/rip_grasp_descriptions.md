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
Nearest grasp (uncommitted, user reviews): predicate IsAmongTheClosestGraspsTo(grasp, standing_position,
grasps, number_of_grasps=3) in coraplex/querying/predicates.py - distance first, angle tie-break,
plain per-grasp numpy loop (vectorization and np.isclose removed 2026-10-02 for readability; +0.9 ms/pose), no state. MoveAndPickUpAction.from_graspable applies it;
the instance methods are gone. Benchmark 2026-09-30: ~5.4 ms/standing pose (was ~457), ~14 ms end-to-end.
Verified 2026-10-02: full coraplex suite 554 passed/14 skipped; bullet demo + action_designator/orm_example
notebooks pass. Timing vs HEAD (single runs): transport tests 1317->1154 s, bullet demo 1126->1061 s,
notebooks 47->42 s and 73->59 s.
Option B done: TransportAction.from_grasp replaced by from_graspable (13 call sites: demo, 2 examples,
querying.py, 5 test files). test_bowl_grasping now reads the grasp from the first grounded pick-up.
71 transport-using tests pass. Examples (.md) not executed. User removed the krrood scaling test;
leftover imports time/Match in test_backends.py - asked.
Open: tool_paths._local_bounding_box removal (user asked; cutting's duration scale uses the
collision box while its path uses the visual box - ask which to keep).
Tests: --orm-build never, systemd MemoryMax cap, <= 8 workers, RViz via scratchpad
rviz_debug_plugin.py. Nothing committed/pushed by Claude.
