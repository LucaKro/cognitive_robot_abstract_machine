## PR #588 (cram2) - rip_grasp_descriptions

Committed by the user: d01fa026c "first batch on changes to merge costmaps and
locations" (reachability validation removed, Move-and-X transport, FaceAt standing
position, trial RViz publishing with TF prefix, rebind memo, stall rule for tasks at
their goal). Plan: ~/.claude/plans/okay-my-coworkers-and-agile-starfish.md

Uncommitted on top (2026-09-24, user reviewed the round):
- keep_joint_states removed everywhere (it was dead: MoveMotion never read it).
- Location classes in coraplex/locations/locations.py: CostmapLocation (context, seeds
  draw from context.sampling_seed, abstract costmap(), costmap built when drawn from,
  targets resolved via world.transform so a body-frame Pose follows the body),
  ReachabilityLocation(target_pose, arm, reach_fraction), VisibilityLocation(target_pose)
  Pose only. factories.py, DeferredLocation, accessing_location, occupancy_location,
  Context.candidate_draw deleted. Do not describe them as "deferred" (user).
- TransportAction.standing_poses_to_try = 50, applied as .limit() on each Move-and-X.
- WorldEntityRebinding.rebind: list/dict branches collapsed; ORM ignores it.
Tests: fast affected set green; transport set 7/8 - test_transport_open_container[stretch]
hung once (gripper close juddering against the handle: CloseGripper not at goal,
LocalMinimumReached FAILED, NotApproachingGoal reads approaching). Passed when run alone.
User said "nvm" - not pursuing for now.

Still to run: transport-plan + demo tests (filtered out by a -k mistake).
Open for the user: delete the unused helpers (explained why unused, awaiting decision).

Debugging: publish robot-behaviour runs (scratchpad rviz_debug_plugin.py, -p, HUNG_TICKS).
Tests: --orm-build never, systemd MemoryMax cap, <= 8 workers.
