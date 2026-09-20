## PR #588 (cram2) - rip_grasp_descriptions

Addressing the 4 unresolved review threads that ask for a code change. Local
edits only for now; no commits, no GitHub replies, no thread resolutions.

Plan:
1. `GraspPose` (frozen, validates its frame in `__post_init__`) replaces the bare
   `Pose` used for grasps; `HasGraspPoses.grasp_poses` returns `List[GraspPose]`.
   Closes Tigul's "enforce body frame" (backends.py:72, pose_validator.py:198) and
   "move this computation somewhere central" (factories.py:86).
2. Move `approach_clearance`, `retreat_distance`, `reach_fraction`,
   `accessing_reach_fraction` out of `ActionConfig` onto the classes that use them.
   Tigul, action_conf.py:4 - already agreed to.
3. `MotionMadeNoProgress(PlanFailure)` wrapping `NoProgressError` at the giskard
   boundary; `RecoverableFailure`/`RECOVERABLE_FAILURES` deleted. tomsch420,
   failures.py:36.
4. Template method on `GraspReachabilityValidator` so `IsObjectReachableBy` and
   `IsGraspReachableBy` share one reaching algorithm. Tigul, pose_validator.py:530.

Deliberately not done:
- `add_semantic_annotation` recursive by default (Tigul, action_designator.md:295):
  own PR. The replay path (`AddSemanticAnnotationModification.apply`,
  `RemoveSemanticAnnotationModification.revert`) inserts one annotation per
  modification entry and `is_semantic_annotation_in_world` only recognises an
  annotation whose `_world` is set - so it is not "a few lines".
- robot_parts.py:774 heuristic (tomsch420): asked twice for an alternative, none given.
- mixins.py:432 axes / :1109 strategy pattern: both reviewers agreed to defer.

Done so far: nothing committed; implementation starting.
Next: items 2 and 3 (small, independent), then 1, then 4.

Full plan: ~/.claude/plans/please-have-a-look-composed-clock.md
