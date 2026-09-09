# Warsaw SDT evaluation

The evaluation treats a pipeline run as an immutable experimental unit. New runs keep
the following files in their run directory:

| Path | Purpose |
| --- | --- |
| `provenance.json` | Pipeline settings, input hashes, Python identity, and Git state |
| `python_environment.txt` | Exact installed distribution versions |
| `source.patch` | Local source changes relative to the recorded commit |
| `model_calls/<stage>/<question>/attempt_N.json` | Every prompt attempt, raw response, validation result, model, and latency |
| `evaluation_graph.json` | Database-independent final nodes and accepted or refused semantic edges |

The existing answer directories remain available for replaying a run. They contain the
latest answer per question; `model_calls` is the audit trail and does not overwrite a
failed attempt when a correction succeeds.

## Coordinate alignment

Camera poses are not required. Select corresponding physical points in the
reconstruction and GT viewers and write them as:

```json
{
  "schema_version": 1,
  "reconstruction_frame": "pipeline_mesh",
  "ground_truth_frame": "iai_apartment",
  "estimate_scale": false,
  "landmarks": [
    {
      "name": "northwest floor corner",
      "reconstruction": [0.0, 0.0, 0.0],
      "ground_truth": [1.0, 2.0, 0.0]
    },
    {
      "name": "counter left corner",
      "reconstruction": [1.0, 0.0, 0.9],
      "ground_truth": [1.0, 3.0, 0.9]
    },
    {
      "name": "counter right corner",
      "reconstruction": [1.0, 2.0, 0.9],
      "ground_truth": [-1.0, 3.0, 0.9]
    }
  ]
}
```

Use at least six points spread over the scanned volume in practice. Avoid placing all
points on one line or on one small object. Set `estimate_scale` to `true` only when the
coordinate frames may use different units. Fit the transform with:

```bash
python -m experiments.warsaw.evaluation.alignment landmarks.json \
  --output alignment.json
```

The output contains the reconstruction-to-GT homogeneous matrix, scale, RMSE, and the
residual of every landmark. A large initial rotation or translation does not affect the
closed-form fit. ICP can refine this result after the landmark residuals have been
checked.

## Evaluation scope

For the apartment experiments, only fixed apartment structure belongs in the primary
comparison. Movable objects, people, and robot bodies are excluded before matching.
Coverage is defined from the aligned reconstructed surface: a GT entity is eligible
only when enough of its surface lies within a chosen distance of the reconstructed
surface. Report the threshold and minimum covered fraction with every result, and report
results at several reasonable thresholds as a sensitivity analysis.

The count-based label metric in `label_metrics.py` reproduces the earlier HM3D metric.
It deliberately ignores object identity and is therefore a compatibility result. The
primary evaluator should first establish instance correspondences and then report
instance detection, class correctness, geometry overlap, and typed hierarchy metrics on
the same eligible set.
