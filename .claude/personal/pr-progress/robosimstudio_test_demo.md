Branch robosimstudio_test_demo pushed to origin (head f058ae76), no PR opened (not asked).
12 commits on upstream main be2ac4f9: 10 bug fixes with failing-first tests, demo commit,
demo relative-path fix. README command verified end to end with video (milk 2.4 cm off).
Tests green in cienv: demo 16 (incl. slow), physics_simulators 26, SDT adapters/inertial/robots
392 (test_gazebo.py skipped: very slow, unrelated), coraplex motion/stationary/import 61,
TIAGo multi-robot 23.
Open questions for the user: InertiaTensor.to_values tuple -> dataclass (API change);
multi_sim literal strings cleanup; robosimstudio/imageio/pillow as new demo dependencies.
