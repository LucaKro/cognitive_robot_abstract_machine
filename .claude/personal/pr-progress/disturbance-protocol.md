## disturbance-protocol (PR #30, stack 29: #24 -> #28 -> #30)

Done (pushed c66ef840): protocol, CabinetPhysics, Episode, metrics, tests.
Roadmap `## \`disturbance-protocol\` — what the implementation settled` has
the decisions. Supporting changes: MujocoSimulator.set_fixed_body_pose /
set_joint_applied_force; MultiSim(ground_truth=...); cabinet
mechanism_axis_deviation + public world_T_cabinet; profiler
EXECUTOR_CONTROL_CYCLE_PHASES. All numbers are placeholders (user decision).

Local runs: scratchpad/ros/env.sh (fake ament index with Tracy + ROS stub),
`--orm-build=never`. Lost with the container; see roadmap note to rebuild.

Next: watch CI on c66ef840; open points for baseline-stock-cram are in the
roadmap (task adapter, recovery-transition counting, cancelled motions).
