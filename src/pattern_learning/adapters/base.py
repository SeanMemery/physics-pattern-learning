from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from runtime.models import EnvironmentManifest, EnvironmentProbe
from runtime.probe import probe_environment
from schema import CanonicalTrace, TaskSpec


class EnvironmentAdapter(ABC):
    def __init__(self, manifest: EnvironmentManifest, *, project_root: Path) -> None:
        self._manifest = manifest
        self._project_root = project_root

    @property
    def env_id(self) -> str:
        return self._manifest.env_id

    def get_manifest(self) -> EnvironmentManifest:
        return self._manifest

    def resolve_project_path(self, relative_path: str) -> Path:
        return self._project_root / relative_path

    def load_upstream_config(self) -> dict[str, Any]:
        config_path = self.resolve_project_path(self._manifest.upstream.config)
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def build_task_spec(self) -> TaskSpec:
        action_space = self._manifest.task_contract.get("action_space", {})
        observation = self._manifest.task_contract.get("observation", {})
        trace_contract = self._manifest.task_contract.get("trace", {})
        return TaskSpec(
            env_id=self._manifest.env_id,
            display_name=self._manifest.display_name,
            execution_mode=self._manifest.execution_mode.value,
            action_space=action_space,
            observation=observation,
            native_metric=self._manifest.native_metric.model_dump(),
            trace_step_unit=trace_contract.get("step_unit"),
            attachment_kinds=trace_contract.get("attachments", []),
            notes=self._manifest.notes,
        )

    def probe(self) -> EnvironmentProbe:
        return probe_environment(self._manifest, root=self._project_root)

    @abstractmethod
    def list_tasks(self, **kwargs: object) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def get_task_spec(self) -> TaskSpec:
        raise NotImplementedError

    @abstractmethod
    def generate_trace(self, task_id: str, **kwargs: object) -> CanonicalTrace:
        raise NotImplementedError

    @abstractmethod
    def evaluate_trace(self, trace: CanonicalTrace) -> dict[str, object]:
        raise NotImplementedError
