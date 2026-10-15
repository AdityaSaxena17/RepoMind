"""Opt-in production SSE persistence and context reuse against isolated SQL/Redis."""

import asyncio
import json
import os
import uuid
from typing import Any

import asyncpg
import httpx
import pytest
from langchain_openai import ChatOpenAI
from test_chat_streaming import Wire, wire_chunks
from test_evals import FakeEmbeddings

from app import chat, worker
from app.code_tools import CodeTools
from app.config import get_settings, normalize_asyncpg_database_url
from app.evals.execution import Benchmark, connect_evaluation_database, prepare_index
from app.evals.models import Dataset, Fixtures, Question
from app.queue import create_redis_pool
from app.tool_registry import RequestScope, ToolRegistry

pytestmark = pytest.mark.skipif(
    os.getenv("REPOMIND_EVAL_INTEGRATION") != "1",
    reason="Opt in with evaluation-only database credentials and Redis.",
)


def test_real_cache_authorization_and_streamed_message_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_EMBEDDINGS_MODEL", "eval-context-cache-1536")
    get_settings.cache_clear()
    embeddings = FakeEmbeddings()
    monkeypatch.setattr(worker, "create_embedding_model", lambda: embeddings)
    user_id = "eval_chat_" + uuid.uuid4().hex
    sha, content = "a" * 40, "def cache_symbol():\n    return 7\n"
    benchmark = Benchmark(
        Dataset(
            version=user_id,
            repository="fixture/cache",
            corpus_commit_sha=sha,
            annotation_policy="Integration mechanics only.",
            questions=[
                Question(
                    id="cache",
                    category="single_file",
                    evidence_scope="CODE",
                    question="Read cache_symbol.",
                    review_note="Synthetic integration mechanics.",
                )
            ],
        ),
        Fixtures(version="test", provenance="synthetic", scenarios={"none": []}),
        {"app.py": content},
        {"app.py": sha},
    )

    async def run() -> None:
        connection = await connect_evaluation_database()
        redis = pool = None
        repository_id = ""
        keys: set[str] = set()
        try:
            repository_id = await prepare_index(connection, benchmark, embeddings)
            await connection.execute(
                'INSERT INTO "User" ("id","clerkId","updatedAt") VALUES ($1,$1,now())',
                user_id,
            )
            await connection.execute(
                'INSERT INTO "UserRepository" '
                '("id","userId","repositoryId","updatedAt") VALUES ($1,$1,$2,now())',
                user_id,
                repository_id,
            )
            pool = await asyncpg.create_pool(
                normalize_asyncpg_database_url(os.environ["EVAL_DATABASE_URL"]),
                min_size=1,
                max_size=3,
                timeout=10,
                command_timeout=10,
            )
            redis = await create_redis_pool()

            class Cache:
                gets = 0

                async def get(self, key: str) -> Any:
                    self.gets += 1
                    return await redis.get(key)

                async def set(self, key: str, value: bytes, **kwargs: Any) -> Any:
                    keys.add(key)
                    return await redis.set(key, value, **kwargs)

            cache = Cache()
            stamp = await connection.fetchval(
                'SELECT "lastIndexedAt" FROM "Repository" WHERE "id"=$1', repository_id
            )
            for warm in (False, True):
                registry = ToolRegistry(
                    RequestScope(
                        user_id=user_id,
                        repository_id=repository_id,
                        capabilities=frozenset({"code.read"}),
                    )
                )
                tools_context = CodeTools(
                    registry,
                    embeddings if not warm else None,
                    sha,
                    pool=pool,
                    redis=cache,
                    last_indexed_at=stamp,
                    cache_database_url=os.environ["EVAL_DATABASE_URL"],
                    rerank=False,
                )
                async with registry.scoped_tools(frozenset({"code.read"})) as tools:
                    searched = json.loads(
                        await tools[0].ainvoke({"query": "cache_symbol"})
                    )
                    read = json.loads(await tools[1].ainvoke({"path": "app.py"}))
                assert searched["sources"] == read["sources"]
                assert tools_context.cache_stats["hits"] == (2 if warm else 0)
                assert (
                    tools_context.embeddings is None
                    if warm
                    else tools_context.embeddings is embeddings
                )
            assert len(keys) == 2
            ttls = await asyncio.gather(*(redis.ttl(key) for key in keys))
            assert all(0 < ttl <= 900 for ttl in ttls)

            conversation_id = "eval_conversation_" + uuid.uuid4().hex
            await connection.execute(
                'INSERT INTO "ChatConversation" '
                '("id","userId","repositoryId","title","updatedAt") '
                "VALUES ($1,$2,$3,'Integration',now())",
                conversation_id,
                user_id,
                repository_id,
            )
            for canceled in (False, True):
                message_id = uuid.uuid4().hex
                await connection.execute(
                    'INSERT INTO "ChatMessage" '
                    '("id","conversationId","role","content","status","updatedAt") '
                    "VALUES ($1,$2,'USER','Read code','SUCCEEDED',now())",
                    message_id,
                    conversation_id,
                )
                pause = asyncio.Event()

                def respond(
                    request: httpx.Request, pause: asyncio.Event = pause
                ) -> httpx.Response:
                    payload = json.loads(request.content)
                    if not any(m.get("role") == "tool" for m in payload["messages"]):
                        deltas = [
                            (
                                {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "read",
                                            "type": "function",
                                            "function": {
                                                "name": "get_file_context",
                                                "arguments": '{"path":"app.py"}',
                                            },
                                        }
                                    ]
                                },
                                None,
                            ),
                            ({}, "tool_calls"),
                        ]
                        wire = Wire(wire_chunks(deltas), None)
                    else:
                        wire = Wire(
                            wire_chunks(
                                [
                                    ({"content": "<fi"}, None),
                                    ({"content": "nal_answer>"}, None),
                                    ({"content": "Value "}, None),
                                    ({"content": "is 7 "}, None),
                                    ({"content": "[1]."}, None),
                                    ({}, "stop"),
                                ]
                            ),
                            pause,
                        )
                    return httpx.Response(
                        200, headers={"content-type": "text/event-stream"}, stream=wire
                    )

                from langchain_core.messages import AIMessage
                from test_repository_graph import RouterModel

                router = RouterModel(
                    responses=[
                        AIMessage(
                            content=json.dumps({"scope": "CODE", "clarification": None})
                        )
                    ]
                )
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(respond)
                ) as client:
                    model = ChatOpenAI(
                        model="offline",
                        api_key="offline",
                        http_async_client=client,
                        use_responses_api=False,
                        streaming=True,
                        stream_usage=False,
                    )
                    monkeypatch.setattr(
                        chat,
                        "create_chat_model",
                        lambda *, purpose="answer", router=router, model=model: (
                            router if purpose == "router" else model
                        ),
                    )
                    stream = chat.stream_repository_chat(
                        pool, repository_id, conversation_id, user_id, message_id
                    )
                    events = []
                    try:
                        async for event in stream:
                            events.append(event)
                            if event.startswith("event: delta"):
                                if canceled:
                                    break
                                pause.set()
                    finally:
                        await stream.aclose()
                assistant_id = json.loads(events[0].split("data: ")[1])["message_id"]
                row = await connection.fetchrow(
                    'SELECT "status","content" FROM "ChatMessage" WHERE "id"=$1',
                    assistant_id,
                )
                deltas = "".join(
                    json.loads(event.split("data: ")[1])["text"]
                    for event in events
                    if event.startswith("event: delta")
                )
                assert row["content"] == deltas
                assert row["status"] == ("FAILED" if canceled else "SUCCEEDED")
                citations = await connection.fetchval(
                    'SELECT count(*) FROM "ChatMessageCitation" WHERE "messageId"=$1',
                    assistant_id,
                )
                assert citations == (0 if canceled else 1)
                assert pool.get_idle_size() == pool.get_size()

            await connection.execute(
                'UPDATE "UserRepository" SET "accessStatus"=\'REVOKED\' '
                'WHERE "userId"=$1',
                user_id,
            )
            registry = ToolRegistry(
                RequestScope(
                    user_id=user_id,
                    repository_id=repository_id,
                    capabilities=frozenset({"code.read"}),
                )
            )
            CodeTools(
                registry,
                None,
                sha,
                pool=pool,
                redis=cache,
                last_indexed_at=stamp,
                cache_database_url=os.environ["EVAL_DATABASE_URL"],
                rerank=False,
            )
            gets = cache.gets
            async with registry.scoped_tools(frozenset({"code.read"})) as tools:
                result = await tools[1].ainvoke(
                    {
                        "name": "get_file_context",
                        "args": {"path": "app.py"},
                        "id": "revoked",
                        "type": "tool_call",
                    }
                )
                assert result.status == "error" and "not authorized" in result.text
            assert cache.gets == gets
        finally:
            if pool is not None:
                await pool.close()
            if redis is not None:
                if keys:
                    await redis.delete(*keys)
                await redis.aclose()
            if repository_id:
                await connection.execute(
                    'DELETE FROM "Repository" WHERE "id"=$1', repository_id
                )
            await connection.execute('DELETE FROM "User" WHERE "id"=$1', user_id)
            await connection.close()

    try:
        asyncio.run(run())
    finally:
        get_settings.cache_clear()
