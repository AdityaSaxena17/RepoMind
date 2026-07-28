from collections.abc import AsyncIterator
from typing import Annotated

import asyncpg
from fastapi import APIRouter, Header, HTTPException, status
from fastapi.responses import StreamingResponse

from ..chat import stream_repository_chat
from ..config import get_settings
from ..schemas import ChatStreamRequest

router = APIRouter(prefix="/repositories/chat", tags=["repository-chat"])


@router.post("/stream")
async def stream_chat(
    payload: ChatStreamRequest,
    internal_credential: Annotated[
        str | None,
        Header(alias="X-RepoMind-Internal-Credential"),
    ] = None,
) -> StreamingResponse:
    settings = get_settings()

    if (
        settings.internal_api_credential
        and internal_credential != settings.internal_api_credential
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal API credential.",
        )

    connection = await asyncpg.connect(settings.database_url)

    async def events() -> AsyncIterator[str]:
        try:
            async for event in stream_repository_chat(
                connection,
                payload.repository_id,
                payload.conversation_id,
                payload.user_id,
                payload.message_id,
            ):
                yield event
        finally:
            await connection.close()

    return StreamingResponse(events(), media_type="text/event-stream")
