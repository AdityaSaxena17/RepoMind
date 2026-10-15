from collections.abc import AsyncGenerator, Mapping
from contextlib import aclosing
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from ..chat import stream_repository_chat
from ..config import get_settings
from ..schemas import ChatStreamRequest

router = APIRouter(prefix="/repositories/chat", tags=["repository-chat"])


class ChatStreamingResponse(StreamingResponse):
    """Close the iterator even when an ASGI 2.4 send fails after yielding a chunk."""

    def __init__(
        self, content: AsyncGenerator[str, None], *, headers: Mapping[str, str]
    ) -> None:
        super().__init__(content, media_type="text/event-stream", headers=headers)
        self.chat_iterator = content

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async with aclosing(self.chat_iterator):
            await super().__call__(scope, receive, send)


@router.post("/stream")
async def stream_chat(
    payload: ChatStreamRequest,
    request: Request,
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

    async def events() -> AsyncGenerator[str, None]:
        async with aclosing(
            stream_repository_chat(
                request.app.state.database_pool,
                payload.repository_id,
                payload.conversation_id,
                payload.user_id,
                payload.message_id,
                redis=request.app.state.redis,
            )
        ) as stream:
            async for event in stream:
                yield event

    return ChatStreamingResponse(
        events(),
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )
