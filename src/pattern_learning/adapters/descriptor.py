from __future__ import annotations

from adapters.base import EnvironmentAdapter
from runtime.models import EnvironmentManifest
from schema import CanonicalTrace, TaskSpec


class ManifestDescriptorAdapter(EnvironmentAdapter):
    def __init__(self, manifest: EnvironmentManifest, *, project_root):
        super().__init__(manifest, project_root=project_root)

    def list_tasks(self, **kwargs: object) -> list[str]:
        raise NotImplementedError(
            f"{self._manifest.env_id} task inventory is scheduled for a later extraction phase."
        )

    def get_task_spec(self) -> TaskSpec:
        return self.build_task_spec()

    def generate_trace(self, task_id: str, **kwargs: object) -> CanonicalTrace:
        raise NotImplementedError(
            f"{self._manifest.env_id} trace generation is scheduled for later extraction phases."
        )

    def evaluate_trace(self, trace: CanonicalTrace) -> dict[str, object]:
        raise NotImplementedError(
            f"{self._manifest.env_id} evaluation execution is scheduled for later extraction phases."
        )
