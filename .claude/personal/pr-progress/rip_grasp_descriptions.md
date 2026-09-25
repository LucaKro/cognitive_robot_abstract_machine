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

test_transport_open_container diagnosis (HSR->Stretch->TIAGo order):
- Stretch: not a hang - finger velocity limit 0.0067 rad/s (stretch.py, from main) makes
  each gripper close ~25 s sim. The 3000-tick debug cutoff was too short.
- TIAGo: real hang in the pick-up grasp. Base penetrates the opened drawer (-1.7 cm);
  MoveTCP error flat but its rate flips sign every tick, so NotApproachingGoal flips and
  StillProgressing's timer resets forever. Proposed fix (awaiting user): judge progress by
  net improvement over the best error so far, not the instantaneous rate.

Still to verify: designator sweep, bullet demo, experiment tests, notebooks
(action_designator.md, orm_example.md, location_designator.md).
Debugging: scratchpad rviz_debug_plugin.py (-p, HUNG_TICKS). Tests: --orm-build never,
systemd MemoryMax cap, <= 8 workers. Nothing committed/pushed by Claude.
