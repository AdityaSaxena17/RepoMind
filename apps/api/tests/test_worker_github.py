import pytest

from app import worker
from app.config import get_settings


def test_worker_github_headers_use_configured_token(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("GITHUB_TOKEN", "github_pat_real")
  get_settings.cache_clear()

  assert worker.github_headers()["Authorization"] == "Bearer github_pat_real"

  get_settings.cache_clear()


def test_worker_github_headers_omit_placeholder_token(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("GITHUB_TOKEN", "github_pat_optional_public_read_token")
  get_settings.cache_clear()

  assert "Authorization" not in worker.github_headers()

  get_settings.cache_clear()
