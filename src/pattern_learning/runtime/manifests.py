from __future__ import annotations

from pathlib import Path

import yaml

from runtime.models import EnvironmentManifest


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def manifest_root() -> Path:
    return project_root() / "configs"


def list_manifest_paths(root: Path | None = None) -> list[Path]:
    selected_root = root or manifest_root()
    if selected_root.is_file():
        return [selected_root]
    return sorted(selected_root.glob("*/environment.yaml"))


def load_manifest(path: Path) -> EnvironmentManifest:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return EnvironmentManifest.model_validate(data)


def load_all_manifests(root: Path | None = None) -> list[EnvironmentManifest]:
    return [load_manifest(path) for path in list_manifest_paths(root=root)]


def load_manifest_map(root: Path | None = None) -> dict[str, EnvironmentManifest]:
    return {manifest.env_id: manifest for manifest in load_all_manifests(root=root)}
