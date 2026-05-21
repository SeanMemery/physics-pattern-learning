from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENV_ROOT = PROJECT_ROOT / "venv"

COMMON_PACKAGES = [
    "fastapi",
    "uvicorn",
    "httpx",
    "petname",
    "pytest",
    "Pillow",
    "imageio",
    "imageio-ffmpeg",
    "matplotlib",
    "PyYAML",
    "typer",
    "weave",
    "wandb",
]

ROOT_REQUIREMENTS = ["-r", "requirements.txt"]

ENV_SPECS = {
    "learning": {
        "python": "3.10",
        "install_project": True,
        "install_evo_lib": True,
        "packages": [
            "numpy",
            "pygame",
            "pygame-widgets",
            "pymunk",
            "iphyre",
            "hydra-core",
            "omegaconf",
        ],
    },
    "phyre": {
        "python": "3.9",
        "install_project": False,
        "install_evo_lib": False,
        "packages": [
            "numpy<1.24",
            "phyre",
        ],
    },
    "iphyre": {
        "python": "3.10",
        "install_project": True,
        "install_evo_lib": True,
        "packages": [
            "numpy",
            "pygame",
            "pygame-widgets",
            "pymunk",
            "iphyre",
            "hydra-core",
            "omegaconf",
        ],
    },
    "kinetix": {
        "python": "3.10",
        "install_project": True,
        "install_evo_lib": True,
        "no_build_isolation": True,
        "pre_install_commands": [
            [
                "{python}",
                "-m",
                "pip",
                "install",
                "setuptools<81",
            ]
        ],
        "packages": [
            "numpy",
            "gymnax",
            "jax==0.6.2",
            "jax2d",
            "jaxgl",
            "distrax",
            "optax",
            "chex",
            "flax",
            "safetensors",
            "hydra-core",
            "omegaconf",
        ],
    },
    "pooltool": {
        "python": "3.10",
        "install_project": True,
        "install_evo_lib": True,
        "packages": ["numpy"],
        "post_install_commands": [
            [
                "{python}",
                "-m",
                "pip",
                "install",
                "pooltool-billiards",
                "--extra-index-url",
                "https://archive.panda3d.org/simple/",
            ]
        ],
    },
}


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def _python_command(version: str) -> str:
    return f"python{version}"


def setup_env(env_id: str) -> None:
    spec = ENV_SPECS[env_id]
    env_root = VENV_ROOT / env_id
    python_cmd = _python_command(str(spec["python"]))
    run([python_cmd, "-m", "venv", str(env_root), "--clear"])
    pip_cmd = [str(env_root / "bin" / "python"), "-m", "pip"]
    run([*pip_cmd, "install", "--upgrade", "pip", "setuptools", "wheel"])
    for pre_install_cmd in spec.get("pre_install_commands", []):
        run([part.format(python=str(env_root / "bin" / "python")) for part in pre_install_cmd])
    if spec.get("install_project", True):
        run([*pip_cmd, "install", *ROOT_REQUIREMENTS])
    else:
        run([*pip_cmd, "install", *COMMON_PACKAGES])
    extra_install_cmd = [*pip_cmd, "install"]
    if spec.get("no_build_isolation", False):
        extra_install_cmd.append("--no-build-isolation")
    extra_install_cmd.extend(spec["packages"])
    run(extra_install_cmd)
    for post_install_cmd in spec.get("post_install_commands", []):
        run([part.format(python=str(env_root / "bin" / "python")) for part in post_install_cmd])


def main() -> None:
    parser = argparse.ArgumentParser(description="Create per-environment virtualenvs under venv/.")
    parser.add_argument("env", nargs="*", choices=sorted(ENV_SPECS), help="Environment ids to provision")
    args = parser.parse_args()
    env_ids = args.env or sorted(ENV_SPECS)
    VENV_ROOT.mkdir(parents=True, exist_ok=True)
    for env_id in env_ids:
        setup_env(env_id)


if __name__ == "__main__":
    main()
