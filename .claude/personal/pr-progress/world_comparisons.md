Goal: library-level comparison of two SemDT `World`s (ground truth vs reconstruction) in
`semantic_digital_twin/world_comparison/`, one PR to cram2 via origin, built step by step, test-first.
Design: briefing https://claude.ai/artifact/AtuBV7wXS8iVEUzQLkrCHQ; walkthrough https://claude.ai/artifact/LF9jNC2sH5uYA5gj73cCMr

Steps:
1. DONE (fb0f422b67): body matching. Each reconstructed sample goes to the GT body with the nearest
   sample within tau (WorldSurfaces, one k-d tree); OverlapTable; Hungarian with theta; merged_bodies /
   split_bodies via minimum_partial_overlap. 25 tests; ORM regenerated, test_orm passes.
   Known limit: GT bodies with coincident surfaces (kitchen oven panel/main/knobs) flip at ~1 cm misalignment.
2. DONE (01eec1c018): geometry_scores.py: GeometryScorer(tau, ObservedRegion) -> GeometryEvaluation of GeometryScore
   (precision, recall over observed GT surface, F, mean / 90th pct distance, observed_share); WholeSurface,
   SurfaceNearReconstruction; BodyCorrespondence.recognition_quality + overlap_table; PQ. 35 tests, test_orm ok.
   Exact distances via Open3D RaycastingScene (DistanceToSurface, ~1 um single precision): kitchen 35 s -> 0.5 s.
   open3d added to semantic_digital_twin/pyproject.toml; uv.lock NOT relocked (already stale: robokudo pins >=0.20, lock 0.19).
3. DONE (3cef0bc579): joint_scores.py: RigidGroups (union over FixedConnections), GroupMatch (Hungarian on shared
   matched samples), JointMotion (world-frame axis, travel from current position; multiplier/offset/sign folded),
   JointScore (same_type, axis_angle, axis_distance at GT part for revolute/screw, travel_error), JointEvaluation
   (welded / unmatched_ground_truth / extra joints, type_accuracy). BodyCorrespondence now holds both worlds + alignment.
   Zero-axis joints (kitchen oven_area_area_right_drawer_joint) scored by type only. 50 tests, test_orm ok.
   Also done: numpy.typing (npt.NDArray[...]) hints throughout world_comparison.
   Note: scan pipeline on semdt-creation-from-video creates no movable joints -> all GT joints show as welded today.
4. DONE (uncommitted): semantic_scores.py: SemanticTaxonomy.of_module (classes defined in a module, so mixins
   don't count; classes list not persisted by ORM), annotations paired via BodyCorrespondence.matched_samples_between
   on their root body (Hungarian, pairs_sharing_the_most, now also used by JointScorer); SemanticScore (same_class,
   ancestor counts -> hierarchical P/R); SemanticEvaluation.exact / .hierarchical (micro, unpaired count).
   structure_scores.py: PartWholeRelation triplets from IsPartWholeRelationship fields; found/missed/extra,
   relation recall/precision + recall between paired annotations. Region-only annotations (EntryWay) take no part.
   Test scene AnnotatedCabinetScene in worlds.py. 69 tests, test_orm ok.
5. Result object, JSON, docs page. 6. (maybe) EQL competency-query battery.
Later, on semdt-creation-from-video: replace the class+size matcher in experiments/warsaw/evaluation.

Run tests with .venv/bin/python (no xdist there, run serially). User commits; Claude never commits. Draft PR not opened yet (ask before pushing).
