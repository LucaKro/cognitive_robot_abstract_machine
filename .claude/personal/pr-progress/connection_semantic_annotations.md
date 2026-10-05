Replace MechanicalJoint (Hinge/Slider/ScrewMechanism bodies) with plain active connections.
Plan: ~/.claude/plans/okay-please-write-a-golden-parrot.md (approved 2026-10-05).
Decisions: annotations lose `mechanical_joint`; `HasJoint.joint` forwards root.parent_connection;
hinge_T_door -> ConnectionSpecification.connection_T_child; parent_T_self stays child pose at q=0;
ConnectionSpecification.replace(world, child) for pivot-after-spawn builders (procthor, SAGE).
Done: nothing yet. Next: baseline pose dump in worktree, then API + tests (TDD).
