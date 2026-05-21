from __future__ import annotations

from pathlib import Path

from adapters.base import EnvironmentAdapter
from adapters.descriptor import ManifestDescriptorAdapter
from adapters.iphyre import IPHYREAdapter
from adapters.kinetix import KinetixAdapter
from adapters.phyre import PhyreAdapter
from adapters.pooltool import PooltoolAdapter
from paths import get_project_root
from runtime.manifests import load_all_manifests
from runtime.models import EnvironmentManifest, EnvironmentProbe


_CONCRETE_ADAPTERS: dict[str, type[EnvironmentAdapter]] = {
    "phyre": PhyreAdapter,
    "iphyre": IPHYREAdapter,
    "kinetix": KinetixAdapter,
    "pooltool": PooltoolAdapter,
}


class AdapterRegistry:
    def __init__(
        self,
        *,
        project_root: Path | None = None,
        manifests: list[EnvironmentManifest] | None = None,
    ) -> None:
        self._project_root = project_root or get_project_root()
        loaded_manifests = manifests or load_all_manifests()
        self._manifests = {manifest.env_id: manifest for manifest in loaded_manifests}

    def list_env_ids(self) -> list[str]:
        return sorted(self._manifests)

    def get_manifest(self, env_id: str) -> EnvironmentManifest:
        try:
            return self._manifests[env_id]
        except KeyError as error:
            raise KeyError(f"Unknown environment id: {env_id}") from error

    def load(self, env_id: str) -> EnvironmentAdapter:
        manifest = self.get_manifest(env_id)
        adapter_cls = _CONCRETE_ADAPTERS.get(env_id, ManifestDescriptorAdapter)
        return adapter_cls(manifest, project_root=self._project_root)

    def load_all(self) -> list[EnvironmentAdapter]:
        return [self.load(env_id) for env_id in self.list_env_ids()]

    def probe_all(self) -> list[EnvironmentProbe]:
        return [adapter.probe() for adapter in self.load_all()]
