Draft PR #120 (fork): steps 1+2 of plan #117 (discussion #116). Pushed 4a77261.
Done: ScrewAxisConnection + NormalizedTwist (create_revolute/create_prismatic/create_screw,
behaves_like_*), old Revolute/Prismatic/Screw classes and specs removed, keyword-only
create_with_dofs/add_to_world, MuJoCo hinge/slide converter, test.srdf ignored (#118).
Labels api-break + new-feature missing on the fork: Luca to create/set.
Tested on final code: semdt 3054, giskardpy+others 1674 passed (only local env failures);
characterization vs main c6014b6 identical (script in session scratchpad, results in PR).
Open: verbalization reads "has behaves_like_a_..." (Luca to decide on rephrasing).
Uncommitted: NormalizedTwist docstring now calls it a normalized krrood Twist; only
correct once draft #121 (Displacement + real Twist, branch displacement_and_twist) merges.
Next (#117 step 3, later): symbolic params for estimated connections, commit flag,
Giskard k-scaling, estimator; decide Twist type then (#117 comments, #119 rename).
