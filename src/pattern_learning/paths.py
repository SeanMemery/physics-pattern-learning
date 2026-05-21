from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactLayout:
    project_root: Path
    config_root: Path
    environment_manifest_root: Path
    data_root: Path
    evaluations_root: Path
    runs_root: Path
    reports_root: Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_artifact_layout() -> ArtifactLayout:
    project_root = get_project_root()
    config_root = project_root / "configs"
    return ArtifactLayout(
        project_root=project_root,
        config_root=config_root,
        environment_manifest_root=config_root,
        data_root=project_root / "data",
        evaluations_root=project_root / "evaluations",
        runs_root=project_root / "runs",
        reports_root=project_root / "reports",
    )
