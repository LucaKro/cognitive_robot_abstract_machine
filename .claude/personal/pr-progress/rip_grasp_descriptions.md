## PR #588 (cram2) - rip_grasp_descriptions

Committed by the user: d01fa026c (reachability validation removed, Move-and-X transport,
FaceAt standing position, trial RViz publishing, rebind memo, at-goal stall rule).

Uncommitted on top (2026-09-25), all reviewed with the user:
- Lazy location classes (CostmapLocation/ReachabilityLocation/VisibilityLocation) in
  locations/locations.py; never call them "deferred". keep_joint_states gone everywhere.
- TransportAction(pick_up: MoveAndPickUpAction, place: MoveAndPlaceAction) + from_grasp();
  pick-up step opens drawers; place step places what the arm holds (NothingToPlace).
  PickAndPlaceAction(pick_up, place) likewise. BoundsItsCandidates caps each step at 50.
  Step fields carry JSONMetadata(serialize=False) until EQL queries serialize.
- CollisionViolatedError -> MotionViolatedCollisionAvoidance (PlanFailure); previous_nodes
  depth-first; helpers deleted; docstrings never mention underspecified queries.

Fixed 2026-09-25 (TDD, uncommitted):
- giskardpy NotApproachingGoal: approaching = error keeps setting a new low at the minimum
  rate (was instantaneous rate; jitter reset the stall timer forever - TIAGo base in the
  opened drawer). Stretch "hang" was only its 0.0067 rad/s finger limit.
- MoveAndPlaceAction._placed_object takes any annotation of the held body (test_detect
  from main leaked a second Milk into the session apartment world; test_detect now uses
  the mutable fixture and no longer adds a duplicate Milk).
Green: giskardpy statechart/executor + coraplex plan/failure (675), open_container x4,
designator/transport/locations/ORM/demo sweep, detect+transport x4.
Done: create_rate_expression + Symbolic/Sampled split removed (one concrete ErrorSignal);
time_derivative_from_joint_motion stays (cartesian_tasks uses it). ORM regenerated.

All verification green (experiments 184, notebooks 3/3)
(action_designator.md, orm_example.md, location_designator.md).
2026-09-25 (uncommitted, user asked "dont commit"): ViewManager deleted and coraplex's
Arms enum removed. Actions/motions take `arm: Arm` (gripper-only ones `EndEffector`),
ParkArmsAction(arms: List[Arm]), PouringAction.pour_side: PouringSide enum, cramera has its
own ArmSide StrEnum (payload/EQL label). Tool actions take HasGraspPoses (container,
object_to_cut, surface, target_container). Tests use context.robot arms; multi-robot tests
use left_or_only_arm/right_or_only_arm from test/coraplex_test/conftest.py. EQL generative
and probabilistic backends cannot enumerate `arm=...` any more (not an enum) - tests give
`variable_from(robot.get_arms())`. Flaky mark on the collision timing test.
Deleted (user): CarryAction; coraplex/utils.py entirely (unused helpers + split_list_by_type and its tests); translate_pose_along_local_axis became HasApproachesGraspPoses._pre_grasp_pose.
Debugging: scratchpad rviz_debug_plugin.py (-p, HUNG_TICKS). Tests: --orm-build never,
systemd MemoryMax cap, <= 8 workers. Nothing committed/pushed by Claude.
