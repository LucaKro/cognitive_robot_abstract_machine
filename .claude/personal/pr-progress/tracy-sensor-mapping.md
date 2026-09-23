## tracy-sensor-mapping (PR #27, draft, base main)

Plan item of articulated-manipulation-under-uncertainty. Review round 1
resolved (auto mode); record in roadmap.md, "first review round" section.

Done (pushed 9b87f33a): semdt ForceTorqueSensor + Tracy wrist sensors at
<side>_tool0 + Robotiq ObjectDetectionStatus (84a590bb); experiments signals
keyed by semantic parts, pint units, Gaussian/Dirac statistics, ClassVars,
npt typing (9b87f33a); AGENTS.md rules (eabf5059). 12 threads resolved;
krrood / segmind / actuator threads answered and left open for the user.
simulated-sensors notes updated; tracking issue #23 commented.

Next:
- Check CI on 9b87f33a: semdt Tracy tests and the experiments tests only run
  there (Tracy description + ROS). test_tracy_semantic_annotation's sensor
  count was changed on purpose (1 camera + 1 F/T per arm).
- User's call on the three open threads.
- #24 still adds articulated_manipulation/__init__.py with a docstring; this
  PR empties it - reconcile whichever lands second.
- Run the tool on the real Tracy; commit the report. Only then done.
Plan tooling: worktree of basstler_experiments in the scratchpad.
