"""Offline regressions for the environment used by API commands and workers."""

import os
from pathlib import Path

import pytest

from scripts import uv_env


@pytest.mark.parametrize("shell_provider", [None, "openai"])
def test_environment_file_precedence_and_explicit_process_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shell_provider: str | None,
) -> None:
    api = tmp_path / "apps/api"
    web = tmp_path / "apps/web"
    api.mkdir(parents=True)
    web.mkdir(parents=True)
    (tmp_path / ".env").write_text(
        "EMBEDDINGS_PROVIDER=openai\nHIVE_CHAT_MODEL=root-model\nREDIS_PORT=6380\n",
        encoding="utf-8",
    )
    (web / ".env").write_text(
        "HIVE_CHAT_MODEL=web-model\nGITHUB_TOKEN=web-token\n",
        encoding="utf-8",
    )
    (api / ".env").write_text(
        "EMBEDDINGS_PROVIDER=ollama\nHIVE_CHAT_MODEL=api-model\nHIVE_API_KEY=file-key\n",
        encoding="utf-8",
    )
    for key in ("EMBEDDINGS_PROVIDER", "HIVE_CHAT_MODEL", "REDIS_PORT", "GITHUB_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HIVE_API_KEY", "process-key")
    monkeypatch.setenv("PYTHONPATH", "existing-path")
    if shell_provider is not None:
        monkeypatch.setenv("EMBEDDINGS_PROVIDER", shell_provider)
    monkeypatch.setattr(uv_env, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(uv_env, "api_root", lambda: api)

    environment = uv_env.uv_environment()

    assert environment["EMBEDDINGS_PROVIDER"] == (shell_provider or "ollama")
    assert environment["HIVE_CHAT_MODEL"] == "api-model"
    assert environment["HIVE_API_KEY"] == "process-key"
    assert environment["GITHUB_TOKEN"] == "web-token"
    assert environment["REDIS_PORT"] == "6380"
    assert environment["PYTHONPATH"] == f"{api}{os.pathsep}existing-path"
    assert os.environ["HIVE_API_KEY"] == "process-key"


def test_env_reader_handles_comments_quotes_and_escaped_newlines(
    tmp_path: Path,
) -> None:
    path = tmp_path / ".env"
    path.write_text(
        '# HIVE_API_KEY=commented\n\nHIVE_API_KEY="offline-key"\n'
        "INTERNAL_API_CREDENTIAL=\nGITHUB_APP_PRIVATE_KEY='first\\nsecond'\n",
        encoding="utf-8",
    )
    environment: dict[str, str] = {}
    uv_env.load_env_file(environment, path)
    assert environment == {
        "HIVE_API_KEY": "offline-key",
        "INTERNAL_API_CREDENTIAL": "",
        "GITHUB_APP_PRIVATE_KEY": "first\nsecond",
    }
