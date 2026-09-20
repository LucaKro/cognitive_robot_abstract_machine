## PR #588 (cram2) - rip_grasp_descriptions

Addressing the 4 unresolved review threads that ask for a code change. Local
edits only; no commits, no GitHub replies, no thread resolutions.

Done:
1. `GraspPose` (frozen dataclass in `semantic_annotations/mixins.py`, pairs a
   `HasGraspPoses` with a `Pose` and refuses one that is not in the object's own
   frame). `grasp_poses()` returns `List[GraspPose]`. Threaded through
   `EndEffector.grasp_poses_by_distance`, `HasApproachesGraspPoses`, the
   validators, `GiskardLocationBackend`, the location factories and the actions.
   `reachability_location`/`giskard_reachability_location` now take a `GraspPose`
   instead of body + pose, and `GraspPose.global_pose` /
   `pose_at_destination` replace the `target @ body_T_grasp` composition that
   was written out in four places. `PlaceAction.object_designator` and
   `MoveAndPlaceAction.object_designator` now take the annotation, not the body.
   `ReachAction.grasp_pose` renamed to `target_pose` (it is a plain pose, and a
   reach onto a bare pose with no object is still supported);
   `MoveToReach.grasp_pose` likewise.
2. The four action parameters moved out of `ActionConfig`:
   `approach_clearance`/`retreat_distance` onto `HasApproachesGraspPoses`,
   `reach_fraction` onto `RingCostmap`, `accessing_reach_fraction` into
   `factories.ACCESSING_REACH_FRACTION`. `from_arm_reach_distance` now requires
   the fraction rather than defaulting it.
3. `MotionMadeNoProgress(PlanFailure)` wraps `NoProgressError` in
   `GiskardExecutable.execute`; `RecoverableFailure`/`RECOVERABLE_FAILURES`
   deleted, five `except` sites now catch `PlanFailure`, `tool_based.py` catches
   the new type.
4. `GraspReachabilityValidator` owns `__call__`, the loop and the
   `reachable_grasp` bookkeeping; subclasses only supply `grasps_to_try`.
   `IsGraspReachableBy` gained `reachable_grasp` and `copy_for_world`.

ORM regenerated. Docstrings formatted.

Deliberately not done:
- `add_semantic_annotation` recursive by default (Tigul,
  action_designator.md:295): own PR. The replay path
  (`AddSemanticAnnotationModification.apply`,
  `RemoveSemanticAnnotationModification.revert`) inserts one annotation per
  modification entry and `is_semantic_annotation_in_world` only recognises an
  annotation whose `_world` is set - so it is not "a few lines".
- robot_parts.py:774 heuristic (tomsch420): asked twice for an alternative, none
  given.
- mixins.py:432 axes / :1109 strategy pattern: both reviewers agreed to defer.

Next: finish the test runs (coraplex grasp/placing/plan/designator suites) and
report. Nothing committed.

Full plan: ~/.claude/plans/please-have-a-look-composed-clock.md
