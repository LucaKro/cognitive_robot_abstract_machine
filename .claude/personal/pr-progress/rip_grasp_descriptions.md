## PR #588 (cram2) - rip_grasp_descriptions

User committed/pushed up to 9bf252ae9 (merges of cram2/main + krrood_match_bug PR #696, Match API
renames ported). Coraplex CI green there; Examples job: demo slow (~25 min Run Script).

2026-10-03 (uncommitted, user: NEVER commit, no git ops on checkout, judgement calls on code only,
recorded in scratchpad DECISIONS_FOR_REVIEW.md of session 1536c00c):
- Drawer.opening_ratio reads the mechanical joint's connection (crashed on reasoned drawers).
- OpenDrawerBeforeTransport[TransportAction] + TransportAction.transported_object/carrying_arm
  (user chose: read from steps, _kwargs_ for matches); demo registers it; opening runs once
  instead of per pick-up candidate (was ~100 s per candidate).
- Demo bowl uses from_graspable_by_closest_grasps. Demo PASSES in ~7 min locally.
- Reach cap on RingCostmap (arm length) reverted: broke
  test_the_opening_joins_the_sequence_an_underspecified_pick_up_runs; patch in scratchpad.
Next: investigate that, tighter place reach, then full suite + demo + notebooks.
Profiler: scratchpad/demo_profile.py (DEMO_STEPS=0,1,3 isolates the bowl, 0,1,4 the spoon).
Tests: --orm-build never, MemoryMax scope, -n 4 when RAM is tight.
