Draft PR #120 (fork), stacked on #121 (displacement_and_twist): steps 1+2 of plan #117
(discussion #116). Pushed 3217aa6 (713b49a + merge of main).
Done: ScrewAxisConnection.screw_axis: Twist (normalized screw axis through the connection origin)
replaces axis + NormalizedTwist; factories create_revolute/prismatic/helical_screw_axis;
behaves_like_* on the connection; Twist.screw_axis_direction; TwistMapping; MJCF/Gazebo axes
scaled to unit length. Tested: semdt+ksm 3091, giskardpy+others 1645 passed (env failures only:
DAiSy, griplink_interfaces, Tracy RealSense mesh); characterization vs main: only intended diffs.
After the main merge only collection was checked (ORM interfaces regenerated).
Labels api-break + new-feature missing on the fork: Luca to create/set.
Next (#117 step 3, later): symbolic params for estimated connections (option 2 / PoE form then),
commit flag, Giskard k-scaling, estimator.
