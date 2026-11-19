Plan: /home/itsme/work/plans/2026-11-19-ormatic-multiple-inheritance.md (one PR, branch multiple-inheritance,
replaces draft #139 and tom's #1013 fix; Tom agreed). Then re-stack step 2 (branch spatial-types-orm, committed,
not pushed: spatial types as fields, ksm ORM package, NumberLike, bottom-up from_dao) and do step 3 (semdt drops
spatial mappings) per /home/itsme/work/plans/2026-11-19-spatial-types-orm.md.
Decided: universal root StoredObjectDAO, one id per object, base fields in base tables, part objects for secondary
bases, relationships target the root with FK to the declared table, interfaces = tables without fields.
Luca wants a query-speed check vs main at the end. Tests run with .venv (uv sync --extra dev).
Step 3 open questions (not yet asked): Scalar type mapping inherited by cramph/giskardpy symbolic fields;
test_query_access_paths axis-angle paths.
