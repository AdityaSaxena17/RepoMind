"""Operational traces must correlate requests without retaining model/tool payloads."""

import asyncio
import json
import logging
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import ToolMessage

from app import observability
from app.observability import (
    RequestLoggingCallback,
    log_event,
    request_id_context,
    trace_operation,
)


def test_native_callbacks_record_roles_outcomes_and_durations_without_payloads(
    caplog, monkeypatch
):
    caplog.set_level(logging.INFO, logger="app.observability")
    tick = iter([10.0, 10.5, 11.0, 11.25, 12.0, 12.75])
    monkeypatch.setattr(
        observability, "time", SimpleNamespace(monotonic=lambda: next(tick))
    )
    callback = RequestLoggingCallback("request-a")

    async def run():
        model, tool, cancelled = uuid4(), uuid4(), uuid4()
        await callback.on_chat_model_start(
            {}, [["SECRET QUESTION"]], run_id=model, metadata={"repomind_role": "CODE"}
        )
        await callback.on_llm_error(ValueError("SECRET RESULT"), run_id=model)
        await callback.on_tool_start(
            {"name": "get_file_context"},
            "SECRET ARGUMENTS",
            run_id=tool,
            metadata={"repomind_role": "CODE"},
        )
        await callback.on_tool_end(
            ToolMessage(content="SECRET EVIDENCE", tool_call_id="call", status="error"),
            run_id=tool,
        )
        await callback.on_tool_start(
            {"name": "search_issues"},
            "SECRET",
            run_id=cancelled,
            metadata={"repomind_role": "GITHUB"},
        )
        await callback.on_tool_error(asyncio.CancelledError(), run_id=cancelled)

    asyncio.run(run())
    entries = [json.loads(record.message) for record in caplog.records]
    assert [entry["status"] for entry in entries] == [
        "started",
        "failed",
        "started",
        "failed",
        "started",
        "cancelled",
    ]
    assert [entry["duration_ms"] for entry in entries if "duration_ms" in entry] == [
        500,
        250,
        750,
    ]
    assert all(entry["request_id"] == "request-a" for entry in entries)
    assert callback.starts == {}
    assert "SECRET" not in caplog.text


def test_concurrent_request_contexts_and_database_outcomes_are_isolated(caplog):
    caplog.set_level(logging.INFO, logger="app.observability")

    async def request(identifier, fail):
        token = request_id_context.set(identifier)
        try:
            with trace_operation("database", "file_context"):
                await asyncio.sleep(0)
                if fail:
                    raise ValueError("SECRET DATABASE DETAIL")
        except ValueError:
            log_event("request", "chat", "failed")
        finally:
            request_id_context.reset(token)

    async def run():
        await asyncio.gather(request("a", False), request("b", True))

    asyncio.run(run())
    entries = [json.loads(record.message) for record in caplog.records]
    assert [(entry["request_id"], entry["status"]) for entry in entries] == [
        ("a", "started"),
        ("b", "started"),
        ("a", "completed"),
        ("b", "failed"),
        ("b", "failed"),
    ]
    assert request_id_context.get() == "-"
    assert "SECRET" not in caplog.text


def test_failed_progress_delivery_does_not_fail_investigation(monkeypatch, caplog):
    async def unavailable(*args, **kwargs):
        raise ValueError("SECRET CALLBACK DETAIL")

    monkeypatch.setattr(observability, "adispatch_custom_event", unavailable)
    asyncio.run(observability.progress("redis", "context_get", "hit", {}))
    assert "ValueError" in caplog.text and "SECRET" not in caplog.text


def test_progress_cancellation_propagates(monkeypatch):
    async def cancelled(*args, **kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(observability, "adispatch_custom_event", cancelled)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(observability.progress("agent", "investigation", "started", {}))
