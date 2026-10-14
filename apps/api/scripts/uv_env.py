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


def load_env_file(env: dict[str, str], path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key:
            env[key] = value.replace("\\n", "\n")


def uv_environment() -> dict[str, str]:
    """Load local defaults, preserving explicitly supplied process variables."""
    env: dict[str, str] = {}
    root = repo_root()
    api = api_root()
    load_env_file(env, root / ".env")
    load_env_file(env, root / "apps" / "web" / ".env")
    load_env_file(env, api / ".env")
    # Shell/CI overrides win; API-local defaults win over shared web/root files.
    env.update(os.environ)
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
