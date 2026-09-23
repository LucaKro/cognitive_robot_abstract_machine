## tracy-sensor-mapping (PR #27, draft, base main)

Plan item of articulated-manipulation-under-uncertainty. Auto mode; plan is in
roadmap.md under `tracy-sensor-mapping`.

Plan (tests first):
1. signal_statistics.py (numpy only): SignalRecording -> SignalStatistics
   (rate, interval jitter, longest gap; per channel mean, std, smallest step).
2. tracy_signals.py: inventory per side - wrist wrench, finger position,
   gripper motor current, object-detection flag (dynamic_joint_states),
   arm joint effort; each reads its values from its ROS message.
3. tracy_signal_measurement.py: rclpy recorder + JSON report (krrood to_json);
   a signal that sends no messages is reported as not exposed.
Package: experiments/articulated_manipulation (same __init__ as #24).

Done: branch, draft PR, manifest open/record, roadmap section.
Next: step 1.
Open: the tool must be run on the real Tracy (needs someone at the robot).
Plan tooling lives on basstler_experiments: run it from a worktree of it.
