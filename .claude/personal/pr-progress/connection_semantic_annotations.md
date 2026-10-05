Replace MechanicalJoint (Hinge/Slider/ScrewMechanism bodies) with plain active connections. UNCOMMITTED (user: no commits).
Plan: ~/.claude/plans/okay-please-write-a-golden-parrot.md. Done 2026-10-05:
- ConnectionSpecification.connection_T_child + .replace(); MechanicalJoint & co removed; ORM regenerated.
- HasMovableJoint(ABC): movable_joint (ActiveConnection1DOF | None, user's design), abstract
  calculate_self_T_movable_joint(axis), mount_on_movable_joint(spec). Door = edge opposite handle; Drawer/Elevator/BottleCap identity.
- parent_connection_specification() hook removed; MissingMovableJointError guards (opening_ratio, Elevator, navigation).
- kitchen: cupboard handle_pose -> hinge_pose; oven_side drawers got 0..0.25 m travel.
Pose dump matches baseline (except the 2 oven side drawers' new sample points; equal at q=0). Targeted tests + 5 docs green.
Open: test_specifications ORM tests need the interface imported (user asked why DAOs are used).
