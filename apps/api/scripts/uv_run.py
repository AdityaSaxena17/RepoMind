from __future__ import annotations

import subprocess
import sys

from uv_env import api_root, uv_environment, uv_python_executable


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/uv_run.py <command> [args...]", file=sys.stderr)
        return 2

    process = subprocess.run(
        [uv_python_executable(), "-m", "uv", "run", *sys.argv[1:]],
        check=False,
        cwd=api_root(),
        env=uv_environment(),
    )

    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
