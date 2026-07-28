from __future__ import annotations

import subprocess
import sys

from uv_env import api_root, uv_environment, uv_python_executable


def main() -> int:
    process = subprocess.run(
        [uv_python_executable(), "-m", "uv", "lock"],
        check=False,
        cwd=api_root(),
        env=uv_environment(),
    )

    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
