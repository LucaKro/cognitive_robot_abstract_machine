Branch robosimstudio_test_demo (from upstream main be2ac4f9), push to origin only, no PR unless asked.
Plan: genuine-bug fix commits (each with a failing-first test), then the demo in
experiments/src/experiments/robosimstudio_tiago_demo (+ README, requirements.txt, tests in
test/experiments_test/robosimstudio_tiago_demo_test).
Done: 7 fix commits (MuJoCo 3.12, dataclass exception, rclpy import, TIAGo Robotiq gripper
states, inertiafromgeom auto, balanceinertia, non-PSD inertia). Uncommitted: round-off
snap in to_principal_moments_and_axes + tiny-tensor test (verify inertial/multi_sim tests,
then commit). Demo modules written; plan probe running (fixed stands, closest grasp).
Next: demo end to end green, run demo tests in cienv, format, review diff, push.
Report: URDF 1 kg default kept (upstream test pins it); drive/servos/collisions demo-local;
RSS ShapeWriter CPU limitation; RoboKudo transformers crash environmental.
