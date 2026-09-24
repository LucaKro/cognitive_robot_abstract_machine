## PR #588 (cram2) - rip_grasp_descriptions

State: review threads addressed and cram2/main merged (both committed, a2834ad92).
Now: team-agreed overhaul (2026-09-24), local edits only, NOT committed - user
has not asked for a commit/push/PR-description update yet.

Overhaul - no reachability validation anywhere (plan:
~/.claude/plans/okay-my-coworkers-and-agile-starfish.md):
- Preconditions are cheap state checks: PickUp/Grasping/Open = `GripperIsFree`
  only; ReachAction has no precondition. A single-argument `and_` collapses to its
  predicate, so the conditions return the bare predicate.
- `pose_validator.py`, `backends.py` (GiskardLocationBackend),
  `TipLinkDoesNotMatchAnyArm`, `ReachableGrasps`, `grasping_location`,
  `giskard_reachability_location` deleted.
- `Location` (base.py) is now the abstract costmap base (`draw`, abstract
  `candidates`, generator `__iter__`, `ground`); `PoseGeneratorBackend` is gone and
  `Costmap(Location)`. No per-candidate collision check / world copy. Seed comes
  from the new `Context.candidate_draw`; `Costmap.merge` carries `self.draw`.
- `reachability_location(target_pose, context, arm, reach_fraction)`.
- Bullet demo: bowl grasp domain = `bowl.grasp_poses()`; ActionTrial picks.
- ActionTrial (main) untouched - user chose that.

Waiting on the user: whether to remove the helpers left unused
(`ViewManager.get_arm_by_tool_frame`, `Context.for_world`,
`GraspPose.copy_for_world`, `GraspPose.world_T_grasp`, and
`EndEffector.grasp_poses_by_distance`, which only tests use now).

Known local-only failure: `test_every_robot_states_its_axes_in_the_frame_they_belong_to`
(needs `iai_daisy_description`).

Deliberately not done (earlier review): recursive `add_semantic_annotation`
(own PR), robot_parts.py heuristic (no alternative offered), mixins.py axes /
strategy pattern (deferred by both reviewers). PickAndPlaceAction still takes
`graspable_object` (no callers).
