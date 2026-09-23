## tracy-sensor-mapping (PR #27, draft, base main)

Plan item of articulated-manipulation-under-uncertainty. Auto mode; plan and
config findings are in roadmap.md under `tracy-sensor-mapping`.

Done (90e846e5, a730ab8c, pushed):
- tracy_signals.py: inventory per side - wrist wrench, finger position,
  gripper motor current, object-detection flag (dynamic_joint_states), arm
  joint effort; each signal reads its values from its ROS message.
- signal_statistics.py (numpy only): rate, interval spread, longest gap;
  per-channel mean, std and smallest step.
- tracy_signal_measurement.py: rclpy recorder + JSON report (krrood to_json);
  fewer than 2 samples = not exposed.
- Decisions (in PR body): camera out; no F/T sensor in the robot model
  (simulated-sensors owns that); arm joint effort included; queue depth 1000.

Next:
- Check CI on #27: ROS-message tests were only run locally against stand-ins.
- User runs the tool on the real Tracy (arms still, grippers open and empty),
  commits the JSON report, and the roadmap section gets the measured values.
  Only then is the item done.
Plan tooling lives on basstler_experiments: run it from a worktree of it.
