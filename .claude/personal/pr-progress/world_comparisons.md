Goal: library-level comparison of two SemDT `World`s (ground truth vs reconstruction) in
`semantic_digital_twin/world_comparison/`, one PR to cram2 via origin, built step by step, test-first.
Design: briefing https://claude.ai/artifact/AtuBV7wXS8iVEUzQLkrCHQ; walkthrough https://claude.ai/artifact/LF9jNC2sH5uYA5gj73cCMr

Steps:
1. DONE (uncommitted): body matching. Each reconstructed sample goes to the GT body with the nearest
   sample within tau (WorldSurfaces, one k-d tree); OverlapTable; Hungarian with theta; merged_bodies /
   split_bodies via minimum_partial_overlap. 25 tests; ORM regenerated, test_orm passes.
   Known limit: GT bodies with coincident surfaces (kitchen oven panel/main/knobs) flip at ~1 cm misalignment.
2. DONE (uncommitted): geometry_scores.py: GeometryScorer(tau, ObservedRegion) -> GeometryEvaluation of GeometryScore
   (precision, recall over observed GT surface, F, mean / 90th pct distance, observed_share); WholeSurface,
   SurfaceNearReconstruction; BodyCorrespondence.recognition_quality + overlap_table; PQ. 35 tests, test_orm ok.
   OPEN: exact trimesh distances take ~35 s on kitchen (match 1.8 s); options asked: keep / KD-tree approx / Open3D (robokudo dep only).
3. <- next: rigid groups + kinematics
   (world-frame axis lines, sign-folded angle, line distance, range, type, weld/split; merges of moving parts = welds).
4. Semantics (taxonomy over semantic classes, exact + hF) and structure (annotation relation recall).
5. Result object, JSON, docs page. 6. (maybe) EQL competency-query battery.
Later, on semdt-creation-from-video: replace the class+size matcher in experiments/warsaw/evaluation.

Run tests with .venv/bin/python (no xdist there, run serially). User commits; Claude never commits. Draft PR after the user's first commit.
