"""Record everything needed to interpret and reproduce an expensive run."""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

from experiments.warsaw.pipeline.run import Run, RunFile
from experiments.warsaw.pipeline.settings import PipelineSettings

# %% exact input files


@dataclass(frozen=True)
class InputArtifact:
    """One file consumed from the reconstructed scene directory."""

    path: str
    """The path relative to the configured scene directory."""

    bytes: int
    """The file size in bytes."""

    sha256: str
    """A digest of the exact file content."""

    def to_json(self) -> dict[str, Any]:
        """Return this input as a JSON-ready record."""
        return {"path": self.path, "bytes": self.bytes, "sha256": self.sha256}

    @classmethod
    def inspect(cls, root: Path, path: Path) -> InputArtifact:
        """Read an input file's stable identity.

        :param root: The directory against which the path is recorded.
        :param path: The file to inspect.
        :return: Its relative path, size, and digest.
        """
        digest = sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return cls(
            path=str(path.relative_to(root)),
            bytes=path.stat().st_size,
            sha256=digest.hexdigest(),
        )


# %% the source tree used


@dataclass(frozen=True)
class SourceState:
    """The Git revision and local modifications used by a run."""

    repository: str
    """The source repository inspected for this run."""

    commit: str | None
    """The checked-out commit, when the directory is a Git work tree."""

    dirty: bool | None
    """Whether tracked or untracked local changes were present."""

    status: list[str]
    """Git porcelain status lines identifying local changes."""

    patch_sha256: str | None
    """A digest of the tracked source patch written beside the manifest."""

    def to_json(self) -> dict[str, Any]:
        """Return this source state as a JSON-ready record."""
        return {
            "repository": self.repository,
            "commit": self.commit,
            "dirty": self.dirty,
            "status": self.status,
            "patch_sha256": self.patch_sha256,
        }


def inspect_source(repository: Path) -> tuple[SourceState, bytes]:
    """Inspect a Git work tree without changing it.

    :param repository: The expected repository root.
    :return: Its recorded state and a binary-capable patch of tracked changes.
    """
    repository = Path(repository).resolve()
    if not repository.is_dir():
        return (
            SourceState(
                repository=str(repository),
                commit=None,
                dirty=None,
                status=[],
                patch_sha256=None,
            ),
            b"",
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if commit.returncode != 0:
        return (
            SourceState(
                repository=str(repository),
                commit=None,
                dirty=None,
                status=[],
                patch_sha256=None,
            ),
            b"",
        )
    status_result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    )
    patch_result = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repository,
        capture_output=True,
        check=True,
    )
    patch = patch_result.stdout
    untracked_result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=repository,
        capture_output=True,
        check=True,
    )
    untracked_paths = [
        path.decode("utf-8") for path in untracked_result.stdout.split(b"\0") if path
    ]
    for untracked_path in untracked_paths:
        untracked_patch = subprocess.run(
            [
                "git",
                "diff",
                "--no-index",
                "--binary",
                "--",
                "/dev/null",
                untracked_path,
            ],
            cwd=repository,
            capture_output=True,
            check=False,
        )
        if untracked_patch.returncode not in (0, 1):
            continue
        patch += untracked_patch.stdout
    status = status_result.stdout.splitlines()
    return (
        SourceState(
            repository=str(repository),
            commit=commit.stdout.strip(),
            dirty=bool(status),
            status=status,
            patch_sha256=sha256(patch).hexdigest() if patch else None,
        ),
        patch,
    )


# %% the complete run manifest


@dataclass(frozen=True)
class RunProvenance:
    """Configuration, data, interpreter, and source identity for one run."""

    started_at: str
    """The UTC time at which provenance was captured."""

    command: list[str]
    """The interpreter arguments that launched the process."""

    settings: dict[str, Any]
    """Every pipeline setting in JSON-ready form."""

    inputs: list[InputArtifact]
    """Every file below the configured scene directory."""

    python: dict[str, str]
    """The interpreter and operating-system identity."""

    source: SourceState
    """The Git revision and local source modifications."""

    def to_json(self) -> dict[str, Any]:
        """Return the versioned manifest as JSON-ready data."""
        return {
            "schema_version": 1,
            "started_at": self.started_at,
            "command": self.command,
            "settings": self.settings,
            "inputs": [artifact.to_json() for artifact in self.inputs],
            "python": self.python,
            "source": self.source.to_json(),
        }


def settings_to_json(settings: PipelineSettings) -> dict[str, Any]:
    """Write every pipeline setting without serializing process-specific objects."""
    return {
        "scene_directory": str(settings.scene_directory.resolve()),
        "model": settings.model.value,
        "render_sizes": {
            "kept": list(settings.render_sizes.kept),
            "deciding": (
                list(settings.render_sizes.deciding)
                if settings.render_sizes.deciding is not None
                else None
            ),
        },
        "viewpoint_choice": settings.viewpoint_choice.value,
        "kept_viewpoints": [viewpoint.value for viewpoint in settings.kept_viewpoints],
        "group_size": settings.group_size,
        "nearest": settings.nearest,
        "corrections": settings.corrections,
        "headless": settings.headless,
        "persist": settings.persist,
        "ask_about_the_ontology": settings.ask_about_the_ontology,
        "amend_the_ontology": settings.amend_the_ontology,
        "ignore_amendments": settings.ignore_amendments,
        "reuse_answers": settings.reuse_answers,
        "runs_directory": str(settings.runs_directory.resolve()),
    }


def python_environment() -> str:
    """Return installed distribution versions in a stable text format."""
    installed = {
        distribution.metadata["Name"]: distribution.version
        for distribution in distributions()
        if distribution.metadata["Name"]
    }
    return "".join(
        f"{name}=={version}\n"
        for name, version in sorted(installed.items(), key=lambda item: item[0].lower())
    )


def record_run_provenance(
    *, settings: PipelineSettings, run: Run, repository: Path
) -> None:
    """Write reproducibility evidence before the expensive work starts.

    :param settings: The exact configuration of the run.
    :param run: The newly created run directory.
    :param repository: The source work tree executing the pipeline.
    """
    scene = settings.scene_directory.resolve()
    inputs = (
        [
            InputArtifact.inspect(scene, path)
            for path in sorted(scene.rglob("*"))
            if path.is_file()
        ]
        if scene.is_dir()
        else []
    )
    source, patch = inspect_source(repository)
    provenance = RunProvenance(
        started_at=datetime.now(UTC).isoformat(),
        command=list(sys.argv),
        settings=settings_to_json(settings),
        inputs=inputs,
        python={
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        source=source,
    )
    run.write_json(RunFile.PROVENANCE, provenance.to_json())
    run.path(RunFile.PYTHON_ENVIRONMENT).write_text(python_environment())
    run.path(RunFile.SOURCE_PATCH).write_bytes(patch)
