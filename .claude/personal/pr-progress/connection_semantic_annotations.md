Replace MechanicalJoint (Hinge/Slider/ScrewMechanism bodies) with plain active connections.
Plan: ~/.claude/plans/okay-please-write-a-golden-parrot.md (approved 2026-10-05). UNCOMMITTED, user asked: no commits.
Done 2026-10-05: ConnectionSpecification.connection_T_child + .replace(); HasJoint.joint; Door.hang_on_hinge_opposite_handle;
all MechanicalJoint code/tests/docs removed or ported; ORM regenerated. Pose dump of 28 doors/drawers (kitchen, apartment,
procthor, pipeline) identical to baseline. Targeted tests green (specs, factories, ORM, annotations, reasoning, procthor,
giskard open/close, SAGE door, elevator nav); 5 edited docs execute.
Next: user review; open questions in the session report (oven_side drawers lack position limits; parent_connection_specification hook now unused).
