Draft LucaKro#144 (replaces #139; plan /home/itsme/work/plans/2026-11-19-ormatic-multiple-inheritance.md).
Design: StoredObjectDAO root, bulk root rows before flush, parts outside relationships (deferred PK FKs),
part loading batched per class via load event. Also: KeepHeldBodyUpright numeric tip_normal; semdt
spatial mappings store expressions evaluated at storing time. All package ORM tests pass locally.
Tom's review addressed in 7887bbe: StoredObjectDAO in krrood/ormatic/stored_object.py on base
StoredObjectRow (template generation breaks with 2 root interfaces on one Base). Open threads awaiting
Tom: 4640762437 (why not generated), 4640766235 (global before_flush; options asked).
Merged main in 0245536 (krrood split into krrood-*; giskardpy tip_link -> kinematic_chain, method now
express_tip_normal_in_tip). Answered Tom's Any + speed comments; asked about a krrood sessionmaker
for the flush hook (thread open). Speed rerun vs current main (busy machine) showed branch slower
storing/loading kitchen: rerun quietly with more rounds (scratchpad bench_orm.py, main_now export).
Next: decide closed-field gap; re-stack
step 2 (spatial-types-orm) onto #144 and update pr2-body; labels api-break/new-feature/bug-fix missing on fork.
