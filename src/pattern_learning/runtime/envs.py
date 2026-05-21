from __future__ import annotations

from pathlib import Path

from runtime.models import EnvironmentManifest, RequirementKind, RuntimeRequirement


VENV_ROOT_RELPATH = Path("venv")


def default_interpreter_relpath(env_id: str) -> Path:
    return VENV_ROOT_RELPATH / env_id / "bin" / "python"


def get_python_requirement(manifest: EnvironmentManifest) -> RuntimeRequirement | None:
    for requirement in manifest.runtime:
        if requirement.kind is RequirementKind.PYTHON:
            return requirement
    return None


def resolve_manifest_interpreter(
    manifest: EnvironmentManifest,
    *,
    project_root: Path,
) -> Path | None:
    requirement = get_python_requirement(manifest)
    if requirement is None:
        return None
    if requirement.interpreter_path:
        return project_root / requirement.interpreter_path
    if manifest.execution_mode.value == "simulator":
        return project_root / default_interpreter_relpath(manifest.env_id)
    return None
