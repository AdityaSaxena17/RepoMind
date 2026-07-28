from __future__ import annotations

import os
import sys
from pathlib import Path


def api_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def uv_python_executable() -> str:
    base_executable = getattr(sys, "_base_executable", None)

    if isinstance(base_executable, str) and base_executable:
        return base_executable

    return sys.executable


def uv_environment() -> dict[str, str]:
    env = os.environ.copy()
    root = repo_root()
    api = api_root()
    env.setdefault("UV_CACHE_DIR", str(root / ".uv-cache"))
    env.setdefault("UV_PROJECT_ENVIRONMENT", str(root / ".venv-api"))
    env.setdefault("MYPY_CACHE_DIR", str(root / ".mypy-cache"))
    env.setdefault("PYTHONPYCACHEPREFIX", str(root / ".python-cache"))
    env.setdefault("RUFF_CACHE_DIR", str(root / ".ruff-cache"))
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(api)
        if existing_pythonpath is None
        else f"{api}{os.pathsep}{existing_pythonpath}"
    )
    return env
