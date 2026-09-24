## PR #588 (cram2) - rip_grasp_descriptions

State: review threads addressed and cram2/main merged (both committed, a2834ad92).
Overhaul of 2026-09-24 is local only, NOT committed - no commit/push/PR-description
update until the user asks. Plan: ~/.claude/plans/okay-my-coworkers-and-agile-starfish.md

Done (all tested):
- No reachability validation anywhere: preconditions are cheap state checks
  (PickUp/Grasping/Open = GripperIsFree; ReachAction none). pose_validator.py,
  backends.py (GiskardLocationBackend), TipLinkDoesNotMatchAnyArm, ReachableGrasps,
  grasping_location, giskard_reachability_location deleted.
- Location (locations/base.py) = abstract costmap base (draw, candidates, generator
  __iter__, ground); PoseGeneratorBackend gone; Costmap(Location). Seed via
  Context.candidate_draw; Costmap.merge carries self.draw.
  reachability_location(target_pose, context, arm, reach_fraction).
- TransportAction grounds a(MoveAndPickUpAction)/a(MoveAndPlaceAction)/a(MoveAndOpenAction)
  (new) so ActionTrial tries the move and the act together. MoveAndPickUp takes clearances.
- FaceAtAction.standing_position: Move-and-X face the target from their standing pose.
  Bug (pre-existing on main): plans are built before they run, so FaceAt read the start
  pose and navigated the robot back there before acting.
- ActionTrial publishes its copy while context.debug: RvizVisualization with
  frame_prefix/marker_topic from ActionTrialVisualization (StrEnum), copy_marker_alpha 0.9.
  TfFrameNames.prefix + TFPublisher joins a prefixed tree to the unprefixed root via a
  latched tf_static identity (TfTopic.STATIC).
- World.rebind_world_entities -> WorldEntityRebinding with memo (cycles/shared refs,
  honours __deepcopy__). Context.__deepcopy__ returns self (holds the ROS node).
- Demo: bowl grasp domain = bowl.grasp_poses().

Formerly failing transport tests all pass (pr2/stretch, open container, parse, replay,
memory leak). Regression sweep was running when this note was written.

Open, waiting on the user:
- Cap on trial candidates (unbounded; a place that never succeeds runs forever).
- Unused helpers: ViewManager.get_arm_by_tool_frame, Context.for_world,
  GraspPose.copy_for_world, GraspPose.world_T_grasp, EndEffector.grasp_poses_by_distance.
- ORM now maps WorldEntityRebinding (transient helper) - ignore it?

Debugging: always publish robot-behaviour runs to RViz; scratchpad pytest plugin
rviz_debug_plugin.py (-p, PYTHONPATH). Known local-only failures: iai_daisy_description,
cramera.live missing.

Deliberately not done (earlier review): recursive add_semantic_annotation (own PR),
robot_parts.py heuristic, mixins.py axes/strategy pattern. PickAndPlaceAction still
takes graspable_object.
