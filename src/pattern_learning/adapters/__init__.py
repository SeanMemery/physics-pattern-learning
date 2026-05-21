from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from adapters.base import EnvironmentAdapter

if TYPE_CHECKING:
    from adapters.descriptor import ManifestDescriptorAdapter
    from adapters.iphyre import IPHYREAdapter
    from adapters.kinetix import KinetixAdapter
    from adapters.phyre import PhyreAdapter
    from adapters.pooltool import PooltoolAdapter
    from adapters.registry import AdapterRegistry


_LAZY_EXPORTS = {
    "AdapterRegistry": "adapters.registry",
    "IPHYREAdapter": "adapters.iphyre",
    "KinetixAdapter": "adapters.kinetix",
    "ManifestDescriptorAdapter": "adapters.descriptor",
    "PhyreAdapter": "adapters.phyre",
    "PooltoolAdapter": "adapters.pooltool",
}

__all__ = ["AdapterRegistry", "EnvironmentAdapter", *sorted(_LAZY_EXPORTS)]


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
