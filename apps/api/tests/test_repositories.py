from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.queue import get_redis
from app.routers.repositories import router


class FakeJob:
  def __init__(self, job_id: str) -> None:
    self.job_id = job_id


class FakeRedis:
  def __init__(self) -> None:
    self.calls: list[dict[str, Any]] = []

  async def enqueue_job(
    self,
    function: str,
    *args: Any,
    _job_id: str | None = None,
    _queue_name: str | None = None,
    **kwargs: Any,
  ) -> FakeJob:
    self.calls.append(
      {
        "function": function,
        "args": args,
        "job_id": _job_id,
        "queue_name": _queue_name,
        "kwargs": kwargs,
      },
    )

    if _job_id is None:
      raise AssertionError("Expected a deterministic job id.")

    return FakeJob(_job_id)


def create_test_client(fake_redis: FakeRedis) -> TestClient:
  app = FastAPI()
  app.include_router(router, prefix="/api/v1")

  async def override_redis() -> FakeRedis:
    return fake_redis

  app.dependency_overrides[get_redis] = override_redis

  return TestClient(app)


def test_connect_repository_enqueues_index_job() -> None:
  fake_redis = FakeRedis()
  client = create_test_client(fake_redis)

  response = client.post(
    "/api/v1/repositories/connect",
    headers={"X-RepoMind-User-Id": "user_123"},
    json={
      "repository_id": "repo_cuid",
      "job_id": "job_cuid",
    },
  )

  assert response.status_code == 200
  assert response.json() == {
    "job_id": "repo-index:repo_cuid:job_cuid",
    "enqueued": True,
    "queue_name": "repomind:indexing",
    "repository_id": "repo_cuid",
    "status": "queued",
  }
  assert fake_redis.calls == [
    {
      "function": "index_repository",
      "args": (
        {
          "user_id": "user_123",
          "repository_id": "repo_cuid",
          "job_id": "job_cuid",
        },
      ),
      "job_id": "repo-index:repo_cuid:job_cuid",
      "queue_name": "repomind:indexing",
      "kwargs": {},
    },
  ]


def test_connect_repository_requires_user_header() -> None:
  fake_redis = FakeRedis()
  client = create_test_client(fake_redis)

  response = client.post(
    "/api/v1/repositories/connect",
    json={"owner": "vercel", "repo": "commerce"},
  )

  assert response.status_code == 422
  assert fake_redis.calls == []
