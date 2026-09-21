## PR #588 (cram2) - rip_grasp_descriptions

Addressing the unresolved review threads. Local edits only; nothing committed,
no GitHub replies, no thread resolutions.

Done:
1. `GraspPose` in `semantic_annotations/mixins.py`: pairs a `HasGraspPoses` with a
   `Pose` and refuses in `__post_init__` anything not in the object's own frame.
   `grasp_poses()` returns `List[GraspPose]`. Threaded through
   `grasp_poses_by_distance`, `HasApproachesGraspPoses`, the validators, the
   giskard backend, the location factories and the actions.
   `GraspPose.global_pose` / `pose_at_destination` replace the
   `target @ body_T_grasp` composition that was written out in four places.
   - `@dataclass(eq=False)`, NOT frozen and NOT compared by value. Two hard
     constraints found by the tests: `World.rebind_world_entities` assigns to a
     designator's dataclass fields to re-bind it to a world copy (frozen raises
     `FrozenInstanceError`), and krrood's parameterizer hashes domain objects
     (a plain `@dataclass` sets `__hash__ = None`). Grasps therefore compare by
     identity, like every semantic annotation in that module.
2. The four action parameters moved out of `ActionConfig`: clearances onto
   `HasApproachesGraspPoses`, `reach_fraction` onto `RingCostmap`,
   `accessing_reach_fraction` into `factories.ACCESSING_REACH_FRACTION`.
3. `MotionMadeNoProgress(PlanFailure)` wraps `NoProgressError` in
   `GiskardExecutable.execute`; `RecoverableFailure`/`RECOVERABLE_FAILURES` gone.
4. `GraspReachabilityValidator` owns `__call__`, the loop and the
   `reachable_grasp` bookkeeping; subclasses supply only `grasps_to_try`.
5. No duplicated object: `HasGraspChoice` holds one `grasp: GraspPose` field with
   `graspable_object` as a property over `self.grasp.graspable`.
   `resolve_grasp_pose` and `OffersNoGrasp` are deleted - the action no longer
   picks a grasp, so it has nothing to refuse. Call sites pass
   `<annotation>.grasp_poses()[0]`. Post-conditions read
   `kwargs["grasp"].graspable.root`.
6. Naming: `grasp_pose_sequence(target_pose, end_effector, grasp, reverse)`;
   `GiskardLocationBackend.grasp_pose` deleted as derivable from `grasp` +
   `target_pose`; `ReachAction.grasp_pose` -> `target_pose` (it is a plain pose,
   and a reach onto a bare pose with no object is still supported);
   `MoveToReach.grasp_pose` -> `target_pose`.

Tests: 383 passed across the affected coraplex and semantic_digital_twin suites.
Two failures remain, neither from this work:
  - `test_every_robot_states_its_axes_in_the_frame_they_belong_to`: the local
    install has no `iai_daisy_description` ROS package.
  - (resolved) `test_grasping[Tracy]` was a missing import, fixed.
New tests: foreign-frame and frameless refusal, `from_body_origin`, the stall
wrapper carrying its `NoProgressError`. Deleted: the two tests whose behaviour no
longer exists (a pick-up resolving a default grasp, and refusing an object that
offers none) plus the `GraspableOfferingNoGrasp` mimic.

ORM regenerated. Docstrings formatted.

Deliberately not done:
- `add_semantic_annotation` recursive by default (Tigul): own PR. The replay path
  inserts one annotation per modification entry and
  `is_semantic_annotation_in_world` only recognises an annotation whose `_world`
  is set - so it is not "a few lines".
- robot_parts.py heuristic (tomsch420): asked twice for an alternative, none given.
- mixins.py axes / strategy pattern: both reviewers agreed to defer.

Full plan: ~/.claude/plans/please-have-a-look-composed-clock.md
