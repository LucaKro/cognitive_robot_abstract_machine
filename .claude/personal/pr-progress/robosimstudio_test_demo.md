Branch robosimstudio_test_demo (from upstream main be2ac4f9), push to origin only, no PR unless asked.
Done (10 commits, unpushed history rewritten for the new AGENTS.md rules): 9 bug-fix commits
(dataclass exception, MuJoCo 3.12, rclpy import, TIAGo Robotiq states, inertiafromgeom auto,
balanceinertia, non-PSD inertia, zero-moment round-off + InertiaTensor exceptions, TiagoMoveSim
in physics) + demo commit (experiments/src/experiments/robosimstudio_tiago_demo, README,
requirements.txt force-added since *.txt is ignored). Demo end to end: milk 2.4 cm off target.
Demo tests 15/15 incl. slow in cienv; physics_simulators 26/26.
Next: finish SDT + coraplex regression runs, review, push to origin.
Report: URDF/STL 1 kg default kept (upstream test pins it; demo sets grocery inertials);
InertiaTensor.to_values tuple API and multi_sim literal strings left (API/scope - ask user);
RSS needs even substeps; RSS ShapeWriter CPU limitation; general actuators via ctrl don't
track in RSS CPU; tiago_apartment_world fixture broken upstream; RoboKudo crash env.
