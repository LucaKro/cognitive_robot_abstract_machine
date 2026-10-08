PR cram2#699: replace MechanicalJoint bodies with plain active connections (HasMovableJoint, connection_T_child, reconnect). Pushed up to 2eb20a4d10 (merge with cram2/main). User: no commits by Claude.
2026-10-08, UNCOMMITTED locally (no GitHub replies made):
- sunava threads: example uses .position (not .to_position()); reconnect keeps the connection name (test_reconnect_keeps_the_name_of_the_specification).
- CI fixes after the main merge: semantic_digital_twin.api imports -> specifications.{base,worlds} (shelf_schema.py, coraplex place-setting test; also fixed missing experiments DAOs); restored DerivativeMap import in giskardpy conftest; dropped stray giskardpy.utils.math import.
Next: user commits/pushes and answers the two threads.
