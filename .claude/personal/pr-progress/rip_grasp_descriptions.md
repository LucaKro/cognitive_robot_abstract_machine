## PR #588 (cram2) - rip_grasp_descriptions

Addressing the unresolved review threads. Local edits only; nothing committed,
no GitHub replies, no thread resolutions.

Done:
1. `GraspPose` in `semantic_annotations/mixins.py`: pairs a `HasGraspPoses` with
   `root_T_grasp` and refuses in `__post_init__` anything not in the object's own
   frame. `grasp_poses()` returns `List[GraspPose]`. `world_T_grasp` and
   `moved_to(reference_T_object)` replace the `target @ body_T_grasp` composition
   that was written out in four places.
   - `@dataclass(eq=False)`, NOT frozen and NOT compared by value. Two hard
     constraints found by the tests: `World.rebind_world_entities` assigns to a
     designator's dataclass fields to re-bind it to a world copy (frozen raises
     `FrozenInstanceError`), and krrood's parameterizer hashes domain objects
     (a plain `@dataclass` sets `__hash__ = None`). Grasps compare by identity.
   - An annotation used with the giskard backend must be REGISTERED in the world:
     the backend re-binds the grasp into a world copy and looks the annotation up
     by id there.
2. Action parameters out of `ActionConfig`: clearances are dataclass field
   defaults on `HasApproachesGraspPoses`; the reach fractions are
   `ReachFraction(float, Enum)` in `datastructures/enums.py` (GRASPING 0.5,
   ACCESSING 0.66). No ClassVars, no module-level constants - the user rejected
   both.
3. `MotionMadeNoProgress(PlanFailure)` wraps `NoProgressError` in
   `GiskardExecutable.execute`; `RecoverableFailure`/`RECOVERABLE_FAILURES` gone.
4. `GraspReachabilityValidator` owns `__call__`, the loop and the
   `reachable_grasp` bookkeeping; subclasses supply only `grasps_to_try`.
5. No duplicated object anywhere: `HasGraspChoice` and `ReachAction` each hold one
   `grasp: GraspPose`. No `graspable_object` wrapper properties - call sites read
   `self.grasp.graspable`. `resolve_grasp_pose`, `OffersNoGrasp`,
   `GraspPoseMissing` and `PerceptionTargetMissing` are deleted along with the
   tests whose behaviour no longer exists.
6. Naming per `semantic_digital_twin/doc/style_guide.md` (`root_T_tip`):
   `grasp_pose_sequence(reference_T_grasp, end_effector, grasp, reverse)`;
   `MoveToReach.target_pose` -> `reference_T_tool_frame`;
   `GiskardLocationBackend.grasp_pose` deleted as derivable from grasp + target.

Tests: 378 passed / 5 failed on the full affected set, then the 5 fixed and
re-verified (76 passed). The only remaining failure is
`test_every_robot_states_its_axes_in_the_frame_they_belong_to`, which needs the
`iai_daisy_description` ROS package that is not installed locally.

ORM regenerated. Docstrings formatted. Examples and docs updated
(quickstart, action_designator, orm_example, location_designator, conditions).

CI fix for 203b2d0 (2026-09-21, local only - user said do not commit/push yet):
- `ReachableGrasps.__iter__` re-addresses the grasp into `self.context.world`;
  `Location.__iter__` points the validator at a deep-copied test world, so the grasp
  named the copy's annotation -> MismatchingWorld on attach in the bullet demo.
  New test `test_reachable_grasps_are_on_the_annotation_the_caller_named`; stand-ins
  of `test_reachable_grasps_sees_the_world_as_it_is_when_consumed` updated.
- Stretch demo `a(PickUpAction)(graspable_object=...)` -> `grasp=cereal.grasp_poses()[0]`.
- `coraplex/examples/location_designator.md`: missing `Milk` import.

Deliberately not done:
- `add_semantic_annotation` recursive by default (Tigul): own PR. The replay path
  inserts one annotation per modification entry and
  `is_semantic_annotation_in_world` only recognises an annotation whose `_world`
  is set - so it is not "a few lines".
- robot_parts.py heuristic (tomsch420): asked twice for an alternative, none given.
- mixins.py axes / strategy pattern: both reviewers agreed to defer.

Full plan: ~/.claude/plans/please-have-a-look-composed-clock.md
