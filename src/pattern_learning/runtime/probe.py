from __future__ import annotations

import contextlib
import importlib
import io
import os
import platform
import subprocess
import sys
from pathlib import Path

from runtime.envs import resolve_manifest_interpreter
from runtime.isolation import isolated_pooltool_home, pooltool_home_root, pooltool_source_root
from runtime.models import (
    AvailabilityStatus,
    EnvironmentManifest,
    EnvironmentProbe,
    ProbeStatus,
    RequirementKind,
    RequirementProbe,
    RuntimeRequirement,
)


def _matches_python_requirement(required_value: str, actual_value: str) -> bool:
    required = tuple(int(part) for part in required_value.split(".")[:2])
    actual = tuple(int(part) for part in actual_value.split(".")[: len(required)])
    return actual == required


def _read_interpreter_version(interpreter: Path) -> str:
    result = subprocess.run(
        [str(interpreter), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip() or "failed to inspect interpreter")
    return (result.stdout or "").strip()


def _module_probe_env(root: Path, *, module_name: str) -> dict[str, str]:
    env = dict(os.environ)
    if module_name == "pooltool":
        home_root = pooltool_home_root(root)
        config_root = home_root / ".config"
        cache_root = home_root / ".cache"
        config_root.mkdir(parents=True, exist_ok=True)
        cache_root.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(home_root)
        env["XDG_CONFIG_HOME"] = str(config_root)
        env["XDG_CACHE_HOME"] = str(cache_root)
        source_root = pooltool_source_root(root)
        python_paths = [str(root / "src")]
        if source_root.is_dir():
            python_paths.append(str(source_root))
        existing = env.get("PYTHONPATH")
        if existing:
            python_paths.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
    return env


def probe_requirement(
    requirement: RuntimeRequirement,
    *,
    root: Path,
    interpreter: Path | None = None,
) -> RequirementProbe:
    if requirement.kind is RequirementKind.PYTHON and requirement.value:
        if interpreter is None:
            current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            if _matches_python_requirement(requirement.value, current_version):
                return RequirementProbe(
                    name=requirement.name,
                    kind=requirement.kind,
                    status=ProbeStatus.MET,
                    detail=f"Current interpreter matches Python {requirement.value}.",
                    required=requirement.required,
                )
            return RequirementProbe(
                name=requirement.name,
                kind=requirement.kind,
                status=ProbeStatus.UNMET,
                detail=f"Current interpreter is Python {current_version}; requirement is {requirement.value}.",
                required=requirement.required,
            )

        if not interpreter.is_file():
            return RequirementProbe(
                name=requirement.name,
                kind=requirement.kind,
                status=ProbeStatus.UNMET,
                detail=f"Expected interpreter at {interpreter}, but it does not exist.",
                required=requirement.required,
            )

        try:
            actual_version = _read_interpreter_version(interpreter)
        except Exception as error:
            return RequirementProbe(
                name=requirement.name,
                kind=requirement.kind,
                status=ProbeStatus.UNMET,
                detail=f"Failed to inspect interpreter at {interpreter}: {type(error).__name__}: {error}",
                required=requirement.required,
            )

        if _matches_python_requirement(requirement.value, actual_version):
            return RequirementProbe(
                name=requirement.name,
                kind=requirement.kind,
                status=ProbeStatus.MET,
                detail=f"Interpreter {interpreter} matches Python {requirement.value}.",
                required=requirement.required,
            )
        return RequirementProbe(
            name=requirement.name,
            kind=requirement.kind,
            status=ProbeStatus.UNMET,
            detail=f"Interpreter {interpreter} is Python {actual_version}; requirement is {requirement.value}.",
            required=requirement.required,
        )

    if requirement.kind is RequirementKind.PLATFORM and requirement.value:
        current_platform = platform.system().lower()
        if current_platform == requirement.value.lower():
            return RequirementProbe(
                name=requirement.name,
                kind=requirement.kind,
                status=ProbeStatus.MET,
                detail=f"Current platform is {current_platform}.",
                required=requirement.required,
            )
        return RequirementProbe(
            name=requirement.name,
            kind=requirement.kind,
            status=ProbeStatus.UNMET,
            detail=f"Current platform is {current_platform}; requirement is {requirement.value}.",
            required=requirement.required,
        )

    if requirement.kind is RequirementKind.MODULE and requirement.value:
        module_name = requirement.value
        try:
            if interpreter is not None:
                result = subprocess.run(
                    [str(interpreter), "-c", f"import importlib; importlib.import_module({module_name!r})"],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=_module_probe_env(root, module_name=module_name),
                )
                if result.returncode != 0:
                    raise RuntimeError((result.stderr or result.stdout).strip() or "module import failed")
            else:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    if module_name == "pooltool":
                        with isolated_pooltool_home(root):
                            importlib.import_module(module_name)
                    else:
                        importlib.import_module(module_name)
        except Exception as error:
            return RequirementProbe(
                name=requirement.name,
                kind=requirement.kind,
                status=ProbeStatus.UNMET,
                detail=(
                    f"Python module '{module_name}' is not importable in "
                    f"{interpreter if interpreter is not None else 'the current interpreter'}: "
                    f"{type(error).__name__}: {error}"
                ),
                required=requirement.required,
            )
        return RequirementProbe(
            name=requirement.name,
            kind=requirement.kind,
            status=ProbeStatus.MET,
            detail=(
                f"Python module '{module_name}' is importable in {interpreter}."
                if interpreter is not None
                else f"Python module '{module_name}' is importable."
            ),
            required=requirement.required,
        )

    if requirement.kind in {RequirementKind.FILE, RequirementKind.DIRECTORY} and requirement.path:
        target = root / requirement.path
        if requirement.kind is RequirementKind.FILE:
            exists = target.is_file()
            kind_label = "file"
        else:
            exists = target.is_dir()
            kind_label = "directory"
        return RequirementProbe(
            name=requirement.name,
            kind=requirement.kind,
            status=ProbeStatus.MET if exists else ProbeStatus.UNMET,
            detail=f"Checked {kind_label} at {target}.",
            required=requirement.required,
        )

    if requirement.kind is RequirementKind.APPLICATION:
        return RequirementProbe(
            name=requirement.name,
            kind=requirement.kind,
            status=ProbeStatus.EXTERNAL,
            detail=f"External application '{requirement.value or requirement.name}' must be verified manually.",
            required=requirement.required,
        )

    return RequirementProbe(
        name=requirement.name,
        kind=requirement.kind,
        status=ProbeStatus.MANUAL,
        detail=requirement.notes or "Manual verification required.",
        required=requirement.required,
    )


def summarize_availability(requirements: list[RequirementProbe]) -> AvailabilityStatus:
    required_requirements = [requirement for requirement in requirements if requirement.required]
    if any(requirement.status is ProbeStatus.UNMET for requirement in required_requirements):
        return AvailabilityStatus.BLOCKED
    if any(
        requirement.status in {ProbeStatus.EXTERNAL, ProbeStatus.MANUAL}
        for requirement in required_requirements
    ):
        return AvailabilityStatus.NEEDS_EXTERNAL
    return AvailabilityStatus.READY


def probe_environment(manifest: EnvironmentManifest, *, root: Path) -> EnvironmentProbe:
    interpreter = resolve_manifest_interpreter(manifest, project_root=root)
    requirement_probes = [
        probe_requirement(requirement, root=root, interpreter=interpreter)
        for requirement in manifest.runtime
    ]
    return EnvironmentProbe(
        env_id=manifest.env_id,
        display_name=manifest.display_name,
        execution_mode=manifest.execution_mode,
        native_metric=manifest.native_metric.name,
        availability=summarize_availability(requirement_probes),
        requirements=requirement_probes,
        capabilities=manifest.capabilities,
    )


def probe_environments(manifests: list[EnvironmentManifest], *, root: Path) -> list[EnvironmentProbe]:
    return [probe_environment(manifest, root=root) for manifest in manifests]
