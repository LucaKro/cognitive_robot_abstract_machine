## PR #588 (cram2) - rip_grasp_descriptions

User committed/pushed up to 9bf252ae9 (merges of cram2/main + krrood_match_bug PR #696, Match API
renames ported).

2026-10-03 demo speed-up, uncommitted, awaiting user review (user: NEVER commit, no git ops on the
checkout; decisions recorded in scratchpad DECISIONS_FOR_REVIEW.md of session 1536c00c):
- Drawer.opening_ratio reads the mechanical joint's connection (crashed on reasoned drawers; also on main).
- OpenDrawerBeforeTransport[TransportAction] + TransportAction.transported_object/carrying_arm
  (user chose reading from the steps, _kwargs_ for matches); demo registers it; multi-robot
  test_transport_open_container back to main's OpenDrawerBeforeTransport line.
- Demo bowl uses from_graspable_by_closest_grasps.
- Verified: demo 331 s alone (CI was ~25 min), coraplex suite 675 passed/14 skipped, 9 notebooks pass.
- Rejected: narrower ring (no gain), reach cap at arm length (test depends on sampled pose; patch kept).
Remaining demo cost: bowl pick-up (19 candidates, 140 s); places succeed first try; each step ~30 s min (trial + real run).
