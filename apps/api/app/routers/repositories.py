from typing import Annotated

from arq.connections import ArqRedis
from fastapi import APIRouter, Depends, Header, HTTPException, status

from ..config import get_settings
from ..queue import get_redis
from ..schemas import ConnectRepositoryRequest, ConnectRepositoryResponse

router = APIRouter(prefix="/repositories", tags=["repositories"])


@router.post("/connect")
async def connect_repository(
    payload: ConnectRepositoryRequest,
    redis: Annotated[ArqRedis, Depends(get_redis)],
    user_id: Annotated[
        str,
        Header(alias="X-RepoMind-User-Id", min_length=1),
    ],
    internal_credential: Annotated[
        str | None,
        Header(alias="X-RepoMind-Internal-Credential"),
    ] = None,
) -> ConnectRepositoryResponse:
    settings = get_settings()

    if (
        settings.internal_api_credential
        and internal_credential != settings.internal_api_credential
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal API credential.",
        )

    job_id = f"repo-index:{payload.repository_id}:{payload.job_id}".lower()
    job_payload = {
        "user_id": user_id,
        "repository_id": payload.repository_id,
        "job_id": payload.job_id,
    }

    job = await redis.enqueue_job(
        "index_repository",
        job_payload,
        _job_id=job_id,
        _queue_name=settings.arq_queue_name,
    )

    return ConnectRepositoryResponse(
        job_id=job.job_id if job else job_id,
        enqueued=job is not None,
        queue_name=settings.arq_queue_name,
        repository_id=payload.repository_id,
        status="queued" if job else "already_queued",
    )
