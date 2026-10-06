Goal: library-level comparison of two SemDT `World`s (reconstruction vs ground truth)
in `semantic_digital_twin`, one PR to cram2 via origin, built step by step, test-first.
Design source: field briefing https://claude.ai/artifact/AtuBV7wXS8iVEUzQLkrCHQ (rev. 2026-10-06).

Steps:
1. DONE (uncommitted): world_comparison/{surface_samples,matching}.py, 2 exceptions, 17 tests in test/semantic_digital_twin_test/test_world_comparison; ORM regenerated, test_orm passes.
2. <- next: Geometry scores on matches (F-score at tau, one-sided accuracy, coverage-based mask: PLY scans have no camera poses).
3. Rigid groups + kinematics (world-frame axis lines, sign-folded angle, line distance, range, type, weld/split).
4. Semantics (taxonomy over semantic classes, exact + hF) and structure (annotation relation recall).
5. Result object, JSON, docs page. 6. (maybe) EQL competency-query battery.
Later, on semdt-creation-from-video: replace the class+size matcher in experiments/warsaw/evaluation.

Run tests with .venv/bin/python (no xdist there, run serially). User commits; Claude never commits. Draft PR to open after the user's first commit.
