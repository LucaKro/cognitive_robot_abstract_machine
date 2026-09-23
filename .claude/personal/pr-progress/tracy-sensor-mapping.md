## tracy-sensor-mapping (PR #27, draft, base main)

Plan item of articulated-manipulation-under-uncertainty. Review rounds 1 and 2
resolved (auto mode); record in roadmap.md "review round" sections.

State (pushed b9d733e1):
- semdt: ForceTorqueSensor + Tracy wrist sensors at <side>_tool0 (only).
- krrood: PintUnitJSONSerializer (df308b97); krrood declares pint.
- experiments: signals keyed by semantic parts, ObjectDetectionStatus here,
  pint units + stamp conversion, Gaussian/Dirac statistics, no ClassVars,
  Tracy built via WorldSpecification/RobotSpecification.
- AGENTS.md: npt typing; no constants, module-level or ClassVar (c11d417c).
- Open threads for the user: krrood aggregations, segmind, actuators
  (round 1), "ros stuff may move to semdt later" (round 2, no action).

Next:
- Check CI on b9d733e1 (Tracy/ROS tests only run there; 9b87f33a was green).
- #24 adds articulated_manipulation/__init__.py with a docstring; here empty.
- Run the tool on the real Tracy; commit the report. Only then done.
Plan tooling: worktree of basstler_experiments in the scratchpad.
