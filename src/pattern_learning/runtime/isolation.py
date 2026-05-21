from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def pooltool_home_root(project_root: Path) -> Path:
    source_root = pooltool_source_root(project_root)
    if source_root.is_dir():
        return project_root / ".runtime-home" / "pooltool-home-source"
    return project_root / ".runtime-home" / "pooltool-home"


def pooltool_source_root(project_root: Path) -> Path:
    return project_root / "deps" / "pooltool"


@contextmanager
def isolated_pooltool_home(project_root: Path) -> Iterator[Path]:
    home_root = pooltool_home_root(project_root)
    config_root = home_root / ".config"
    cache_root = home_root / ".cache"
    config_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    original_values = {
        "HOME": os.environ.get("HOME"),
        "XDG_CONFIG_HOME": os.environ.get("XDG_CONFIG_HOME"),
        "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME"),
    }
    os.environ["HOME"] = str(home_root)
    os.environ["XDG_CONFIG_HOME"] = str(config_root)
    os.environ["XDG_CACHE_HOME"] = str(cache_root)
    try:
        yield home_root
    finally:
        for key, value in original_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
