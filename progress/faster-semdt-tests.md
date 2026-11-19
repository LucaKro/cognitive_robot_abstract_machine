Plan: /home/itsme/work/plans/2026-11-19-spatial-types-orm.md, three stacked PRs.
Decided (Luca, 2026-11-19): implementers always get a link column (never inherit the interface table);
only interfaces a stored field is typed with and no alternative mapping covers get a table;
rotations stored as quaternion x,y,z,w; composite spatial types keep nested spatial-type fields (Pose.position: Point3, ...).
Step 1 done: branch interface-tables, draft #139 (labels api-break + new-feature named to Luca; fork lacks them).
Next: step 2 on a branch stacked on #139 (krrood_spatial_math parameter fields + ORM package), then step 3 (semdt drops mappings).
Tests run with .venv (uv sync --extra dev); warsaw_env lacks krrood_compute_resources.
