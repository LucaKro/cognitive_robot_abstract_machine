"""Reproducibility information written before an expensive pipeline run."""

from __future__ import annotations

import json
import subprocess
from hashlib import sha256

from experiments.warsaw.pipeline.provenance import inspect_source, record_run_provenance
from experiments.warsaw.pipeline.run import Run, RunFile
from experiments.warsaw.pipeline.settings import Model, PipelineSettings

# %% settings, input, software, and source state


def test_run_provenance_records_reproducible_inputs(tmp_path) -> None:
    """A run explains its configuration and exact input without credentials."""
    scene = tmp_path / "scene"
    scene.mkdir()
    mesh = scene / "apartment.glb"
    mesh.write_bytes(b"reconstructed mesh")
    run = Run(directory=tmp_path / "run")
    run.directory.mkdir()
    settings = PipelineSettings(
        scene_directory=scene,
        model=Model.GPT_5_6_LUNA,
        group_size=4,
        runs_directory=tmp_path / "runs",
    )

    record_run_provenance(
        settings=settings,
        run=run,
        repository=tmp_path / "not-a-repository",
    )

    manifest = json.loads(run.path(RunFile.PROVENANCE).read_text())
    assert manifest["schema_version"] == 1
    assert manifest["settings"]["model"] == "openai/gpt-5.6-luna"
    assert manifest["settings"]["group_size"] == 4
    assert manifest["settings"]["render_sizes"] == {
        "kept": [1024, 768],
        "deciding": None,
    }
    assert manifest["inputs"] == [
        {
            "path": "apartment.glb",
            "bytes": len(b"reconstructed mesh"),
            "sha256": sha256(b"reconstructed mesh").hexdigest(),
        }
    ]
    assert manifest["source"]["commit"] is None
    assert manifest["source"]["dirty"] is None
    assert manifest["python"]["executable"]
    assert run.path(RunFile.PYTHON_ENVIRONMENT).read_text()
    assert run.path(RunFile.SOURCE_PATCH).read_bytes() == b""


def test_source_patch_includes_tracked_and_untracked_files(tmp_path) -> None:
    """An uncommitted evaluator remains reproducible before its first commit."""
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True)
    tracked = repository / "tracked.py"
    tracked.write_text("before = True\n")
    subprocess.run(["git", "add", "tracked.py"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Evaluation Test",
            "-c",
            "user.email=evaluation@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "base",
        ],
        cwd=repository,
        check=True,
    )
    tracked.write_text("after = True\n")
    (repository / "untracked.py").write_text("new = True\n")

    source, patch = inspect_source(repository)

    assert source.commit
    assert source.dirty is True
    assert " M tracked.py" in source.status
    assert "?? untracked.py" in source.status
    assert b"after = True" in patch
    assert b"untracked.py" in patch
    assert b"new = True" in patch
