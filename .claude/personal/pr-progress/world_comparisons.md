Goal: library-level comparison of two SemDT `World`s (reconstruction vs ground truth)
in `semantic_digital_twin`, one PR to cram2 via origin, built step by step, test-first.
Design source: field briefing https://claude.ai/artifact/AtuBV7wXS8iVEUzQLkrCHQ (rev. 2026-10-06).

Steps:
1. Body matching on geometry only (visual-mesh surface samples, overlap at tau, Hungarian, theta cutoff). <- next
2. Geometry scores on matches (F-score at tau, one-sided accuracy, coverage-based mask: PLY scans have no camera poses).
3. Rigid groups + kinematics (world-frame axis lines, sign-folded angle, line distance, range, type, weld/split).
4. Semantics (taxonomy over semantic classes, exact + hF) and structure (annotation relation recall).
5. Result object, JSON, docs page. 6. (maybe) EQL competency-query battery.
Later, on semdt-creation-from-video: replace the class+size matcher in experiments/warsaw/evaluation.

Done: nothing yet (branch == cram2/main f38d633f59). User commits; Claude never commits.
