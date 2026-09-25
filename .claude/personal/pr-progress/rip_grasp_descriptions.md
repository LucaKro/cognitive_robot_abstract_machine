## PR #588 (cram2) - rip_grasp_descriptions

Committed by the user (up to a932a1dcf): lazy locations, Move-and-X transport, stall rule,
one ErrorSignal, ViewManager/Arms enum removed (actions take `arm: Arm`), CarryAction and
coraplex/utils.py deleted, pre-grasp as a grasp-frame step (proved identical poses).

Uncommitted 2026-09-25 (user: "dont commit"):
- FaceAtAction(target) is a core action (core/navigation.py) that only turns: TurnMotion =
  Parallel(Pointing base forward axis at target lowered to base height, CartesianPosition
  hold at the base's own origin, bound on start). FaceAndLookAtAction(face_at, look_at) in
  composite/facing.py keeps turn-then-look.
- Move-and-X steps expose sub-actions: navigate/face_and_look_at + pick_up/place/
  open_container, each with from_standing_position(...). Underspecified callers write nested
  a(...) matches (standing pose = inner NavigateAction variable). TransportAction lost its
  MoveTorsoAction(HIGH).
- PlaceAction(object_designator, target_location): no arm; `_arm` (init=False) = arm holding
  the object, else the preceding PickUpAction of that object (user: "for now"), else
  ObjectIsNotHeld (replaced NothingToPlace). Conditions check every arm. Conditions are
  built but never evaluated at runtime (_add_condition_monitors unused).
Open: tool_paths._local_bounding_box removal (user asked; cutting's duration scale uses the
collision box while its path uses the visual box - ask which to keep).
Tests: --orm-build never, systemd MemoryMax cap, <= 8 workers, RViz via scratchpad
rviz_debug_plugin.py. Nothing committed/pushed by Claude.
