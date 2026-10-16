"""Request-correlated progress and timing without prompts, arguments or results."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.messages import ToolMessage
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)
request_id_context: ContextVar[str] = ContextVar("repomind_request_id", default="-")


@contextmanager
def trace_operation(stage: str, operation: str) -> Iterator[None]:
    """Time an operation without recording its inputs, outputs or error details."""
    started = time.monotonic()
    log_event(stage, operation, "started")
    try:
        yield
    except BaseException as error:
        log_event(
            stage,
            operation,
            "cancelled" if isinstance(error, asyncio.CancelledError) else "failed",
            duration_ms=round((time.monotonic() - started) * 1000, 1),
            error_type=type(error).__name__,
        )
        raise
    else:
        log_event(
            stage,
            operation,
            "completed",
            duration_ms=round((time.monotonic() - started) * 1000, 1),
        )


def log_event(stage: str, operation: str, status: str, **fields: Any) -> None:
    """Log only caller-supplied operational metadata, never model/tool payloads."""
    logger.info(
        "%s",
        json.dumps(
            {
                "request_id": request_id_context.get(),
                "stage": stage,
                "operation": operation,
                "status": status,
                **fields,
            },
            ensure_ascii=True,
        ),
    )


async def progress(
    stage: str,
    operation: str,
    status: str,
    config: RunnableConfig | None,
    **fields: Any,
) -> None:
    """Publish safe progress separately from answer tokens and retain a server log."""
    log_event(stage, operation, status, **fields)
    if config is not None:
        try:
            await adispatch_custom_event(
                "repomind_progress",
                {
                    "request_id": request_id_context.get(),
                    "stage": stage,
                    "operation": operation,
                    "status": status,
                    **fields,
                },
                config=config,
            )
        except Exception as error:
            # Observability must not turn a successful investigation into a failure.
            logger.warning("Progress delivery failed (%s).", type(error).__name__)


class RequestLoggingCallback(AsyncCallbackHandler):
    """Record native model/tool durations, including failures and cancellation."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.starts: dict[UUID, tuple[float, str, str, str]] = {}

    def _start(self, run_id: UUID, stage: str, operation: str, role: str) -> None:
        self.starts[run_id] = (time.monotonic(), stage, operation, role)
        log_event(
            stage,
            operation,
            "started",
            request_id=self.request_id,
            role=role,
            run_id=str(run_id),
        )

    def _finish(self, run_id: UUID, status: str) -> None:
        started = self.starts.pop(run_id, None)
        if started is not None:
            timestamp, stage, operation, role = started
            log_event(
                stage,
                operation,
                status,
                request_id=self.request_id,
                role=role,
                run_id=str(run_id),
                duration_ms=round((time.monotonic() - timestamp) * 1000, 1),
            )

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        metadata = kwargs.get("metadata") or {}
        self._start(
            run_id, "model", "generation", metadata.get("repomind_role", "ROUTER")
        )

    async def on_llm_end(
        self, response: LLMResult, *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._finish(run_id, "completed")

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._finish(
            run_id,
            "cancelled" if isinstance(error, asyncio.CancelledError) else "failed",
        )

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        metadata = kwargs.get("metadata") or {}
        self._start(
            run_id,
            "tool",
            serialized.get("name", "tool"),
            metadata.get("repomind_role", "UNKNOWN"),
        )

    async def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish(
            run_id,
            "failed"
            if isinstance(output, ToolMessage) and output.status == "error"
            else "completed",
        )

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._finish(
            run_id,
            "cancelled" if isinstance(error, asyncio.CancelledError) else "failed",
        )
