## PR #588 (cram2) - rip_grasp_descriptions

Committed by the user (up to f12e14c94): lazy locations, Move-and-X transport (navigate/
face_and_look_at + pick_up/place/open_container sub-actions), FaceAtAction core action, PlaceAction
without arm, actions take `arm: Arm`, GraspCandidate rename, nearest-grasp predicate
IsAmongTheClosestGraspsTo (2D distance, angle tie-break) with
MoveAndPickUp/TransportAction.from_graspable_by_closest_grasps(number_of_grasps=...).

2026-10-02: `git merge cram2/main` in progress, all conflicts resolved and staged, NOT committed
(user: report first, commit only when told). Followed main's plan transformations: the built-in
perceive_before_grasp and MoveAndPickUp drawer opening are removed; plan_transformations.py,
its tests and examples/plan_transformations.md ported to grasp/Arm/ReachabilityLocation;
OpenDrawerBeforeTransport -> OpenDrawerBeforeMoveAndPickUp (opening = MoveAndOpenAction);
PerceptionTargetMissing dropped; ParkArmsBeforeFirstAction has no arm field. Branch tests moved
to main's snapshot fixtures (seeded contexts kept). Two multi-robot tests lost their
VizMarkerPublisher (leaked onto the shared snapshot world; main had none).
Verified: full coraplex suite green after that fix (568 + the 92 multi-robot tests, 14 skipped).
Still open: unused `time`/`Match` imports in test_backends.py (asked); tool_paths._local_bounding_box
(collision vs visual box - ask). Tests: --orm-build never, MemoryMax scope, -n 4 when RAM is tight.
Nothing pushed by Claude.
