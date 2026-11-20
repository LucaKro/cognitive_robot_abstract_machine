Draft LucaKro#144 (replaces #139; plan /home/itsme/work/plans/2026-11-19-ormatic-multiple-inheritance.md).
Design: StoredObjectDAO root, bulk root rows before flush, parts outside relationships (deferred PK FKs),
part loading batched per class via load event. Also: KeepHeldBodyUpright numeric tip_normal; semdt
spatial mappings store expressions evaluated at storing time. All package ORM tests pass locally.
Tom's review addressed in 7887bbe: StoredObjectDAO in krrood/ormatic/stored_object.py on base
StoredObjectRow (template generation breaks with 2 root interfaces on one Base). Open threads awaiting
Tom: 4640762437 (why not generated), 4640766235 (global before_flush; options asked).
Next: sync with main (95 behind, conflicts; only when asked); decide closed-field gap; re-stack
step 2 (spatial-types-orm) onto #144 and update pr2-body; labels api-break/new-feature/bug-fix missing on fork.
