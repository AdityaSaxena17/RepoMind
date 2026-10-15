from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from test_code_tools import Database, scripted_retriever, setup_tools
from test_tool_registry import connect_offline, provider, request

from app import agents, github_mcp_server
from app.agents import AgentLimits, investigate_code, investigate_github
from app.tool_registry import ToolRegistry


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedModel:
        return self


def call(name: str, arguments: dict[str, Any], call_id: str = "call") -> AIMessage:
    return AIMessage(
        content="Private intermediate research.",
        tool_calls=[
            {
                "name": name,
                "args": arguments,
                "id": call_id,
            }
        ],
    )


def summary(
    answer: str = "Supported answer [1].",
    sources: list[str] | None = None,
    missing: list[str] | None = None,
) -> AIMessage:
    return AIMessage(
        content=json.dumps(
            {
                "answer": answer,
                "source_ids": sources if sources is not None else ["1"],
                "missing_information": missing or [],
            }
        ),
    )


@pytest.mark.parametrize("github", [False, True])
@pytest.mark.parametrize("final_answer", [False, True])
def test_few_shot_output_matches_the_active_protocol(
    monkeypatch: pytest.MonkeyPatch, github: bool, final_answer: bool
) -> None:
    prompts: list[str] = []
    create_agent = agents.create_agent

    def capture(*args: Any, **kwargs: Any) -> Any:
        prompts.append(kwargs["system_prompt"])
        return create_agent(*args, **kwargs)

    monkeypatch.setattr(agents, "create_agent", capture)
    factory = agents.create_github_agent if github else agents.create_code_agent
    factory(ScriptedModel(responses=[summary()]), [], final_answer=final_answer)
    prompt = prompts[0]
    assert "Fictional examples of tool use" in prompt
    output = prompt.split("using only the example evidence above:\n", 1)[1]
    if final_answer:
        assert output.startswith(agents.FINAL_ANSWER_MARKER)
        assert "</final_answer>" not in output
        assert "return only a JSON object" not in prompt
    else:
        value = json.loads(output)
        assert set(value) == {"answer", "source_ids", "missing_information"}
        assert value["missing_information"] == []
    if github:
        assert "https://github.com/fictional-lab/atlas/" in output
        assert "[7]" not in output
    else:
        assert "[7]" in output


@pytest.mark.parametrize("github", [False, True])
def test_few_shot_evidence_is_not_admitted_to_observed_catalog(
    monkeypatch: pytest.MonkeyPatch, github: bool
) -> None:
    messages_seen: list[BaseMessage] = []

    class RecordingModel(ScriptedModel):
        async def _agenerate(self, messages: list[BaseMessage], **kwargs: Any) -> Any:
            messages_seen.extend(messages)
            return await super()._agenerate(messages, **kwargs)

    database = Database()
    scripted_retriever(monkeypatch, database)
    github_scope(monkeypatch)
    observed_url = "https://github.com/example/project/issues/9"

    async def get(path: str) -> Any:
        return {
            "number": 9,
            "title": "Observed",
            "state": "open",
            "html_url": observed_url,
        }

    monkeypatch.setattr(github_mcp_server, "github_json", get)
    fictional_source = (
        "https://github.com/fictional-lab/atlas/issues/73#issuecomment-900"
        if github
        else "7"
    )

    async def run() -> None:
        model = RecordingModel(
            responses=[
                call("issue_read", {"issue_number": 9})
                if github
                else call("get_file_context", {"path": "app.py"}),
                summary("Claim copied from a fictional example.", [fictional_source]),
            ]
        )
        if github:
            findings = await investigate_github(
                model, ToolRegistry(request(), (provider(),)), "Read issue #9."
            )
        else:
            registry, code = setup_tools(database)
            findings = await investigate_code(model, registry, code, "Read app.py.")
        assert findings.status == "failed"
        assert findings.errors[-1].code == "unobserved_sources"
        assert findings.citation_catalog
        assert fictional_source not in findings.citation_catalog
        assert len(findings.tool_executions) == 1
        assert any(
            isinstance(message, SystemMessage)
            and "Fictional examples of tool use" in message.text
            for message in messages_seen
        )
        assert all(
            "fictional-lab" not in message.text
            and "codec/packet.py" not in message.text
            for message in messages_seen
            if isinstance(message, ToolMessage)
        )

    asyncio.run(run())


def test_mcp_argument_errors_are_bounded_and_redact_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    github_scope(monkeypatch)

    async def run() -> None:
        registry = ToolRegistry(request(), (provider(),))
        async with registry.scoped_tools(frozenset({"github.read"})) as tools:
            descriptions = {
                tool.name: tool.description
                for tool in await github_mcp_server.mcp.list_tools()
            }
            assert {tool.name: tool.description for tool in tools} == descriptions
            tool = next(tool for tool in tools if tool.name == "issue_read")
            result = await tool.ainvoke(
                {
                    "name": "issue_read",
                    "args": {"issue_number": "never_echo_this_input"},
                    "id": "invalid",
                    "type": "tool_call",
                }
            )
        assert result.status == "error"
        assert result.additional_kwargs["repomind_error_kind"] == "validation"
        assert "issue_number" in result.text and "schema" in result.text
        assert "never_echo_this_input" not in result.text
        assert "Traceback" not in result.text and len(result.text) <= 500

    asyncio.run(run())


def test_known_resource_read_precedes_batched_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    github_scope(monkeypatch)
    requests = []
    url = "https://github.com/example/project/issues/42"

    async def get(path: str) -> Any:
        requests.append(path)
        return {
            "number": 42,
            "title": "Import failure",
            "state": "open",
            "html_url": url,
        }

    monkeypatch.setattr(github_mcp_server, "github_json", get)
    batch = AIMessage(
        content="",
        tool_calls=[
            call("issue_read", {"issue_number": 42}, "details").tool_calls[0],
            call(
                "search_pull_requests",
                {"query": "import loader", "state": "all"},
                "search",
            ).tool_calls[0],
        ],
    )

    async def run() -> None:
        registry = ToolRegistry(request(), (provider(),))
        findings = await investigate_github(
            ScriptedModel(responses=[batch, summary(url, [url])]),
            registry,
            "Read issue #42 and explain the indexed import loader.",
        )
        assert findings.status == "complete", findings.errors
        assert len(findings.tool_executions) == 2
        assert all(error.recovered for error in findings.errors)
        assert requests == ["/repos/example/project/issues/42"]

    asyncio.run(run())


@pytest.mark.parametrize("retry_start", [False, True])
def test_github_agent_can_search_and_read_comments_with_request_scoped_tools(
    monkeypatch: pytest.MonkeyPatch, retry_start: bool
) -> None:
    github_scope(monkeypatch)
    requests: list[str] = []
    url = "https://github.com/example/project/issues/9"
    comment_url = url + "#issuecomment-19"

    async def get(path: str) -> Any:
        requests.append(path)
        if "/comments?" in path:
            return [
                {
                    "id": 19,
                    "html_url": comment_url,
                    "body": "Observed discussion.",
                    "user": {"login": "reviewer"},
                }
            ]
        if "/search/issues?" in path:
            return {"total_count": 0, "items": []}
        return {
            "number": 9,
            "title": "Observed proposal",
            "state": "open",
            "html_url": url,
        }

    class Model(ScriptedModel):
        failed_once: bool = False

        async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
            if retry_start and not self.failed_once:
                self.failed_once = True
                raise httpx.RemoteProtocolError("Controlled empty-start failure")
            return await super()._agenerate(*args, **kwargs)

    monkeypatch.setattr(github_mcp_server, "github_json", get)

    async def run() -> None:
        registry = ToolRegistry(request(), (provider(),))
        findings = await investigate_github(
            Model(
                responses=[
                    call("issue_read", {"issue_number": 9}, "details"),
                    call(
                        "search_issues", {"query": "proposal", "state": "all"}, "search"
                    ),
                    call(
                        "issue_read",
                        {"issue_number": 9, "method": "get_comments"},
                        "comments",
                    ),
                    summary(url, [url]),
                ]
            ),
            registry,
            "Read the assigned live facts.",
        )
        assert findings.status == "complete", findings.errors
        assert len(findings.tool_executions) == 3
        comments = findings.tool_executions[2].message
        assert len(requests) == 3 and comments.status == "success"
        assert comments.tool_call_id == "comments"
        assert comment_url in findings.citation_catalog
        assert all(error.recovered for error in findings.errors)

    asyncio.run(run())


def test_unavailable_resource_does_not_stop_other_requested_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    github_scope(monkeypatch)
    requests = []
    url = "https://github.com/example/project/issues/2"

    async def get(path: str) -> Any:
        requests.append(path)
        if path.endswith("/1"):
            raise RuntimeError("GitHub resource was not found.")
        return {"number": 2, "title": "Available", "state": "open", "html_url": url}

    monkeypatch.setattr(github_mcp_server, "github_json", get)

    async def run() -> None:
        registry = ToolRegistry(request(), (provider(),))
        findings = await investigate_github(
            ScriptedModel(
                responses=[
                    call("issue_read", {"issue_number": 1}, "missing"),
                    call("issue_read", {"issue_number": 2}, "available"),
                    summary(url, [url], ["Issue #1 is unavailable."]),
                ]
            ),
            registry,
            "Compare issues #1 and #2.",
        )
        assert findings.status == "incomplete", findings.errors
        assert url in findings.citation_catalog
        assert len(requests) == 2 and requests[-1].endswith("/2")

    asyncio.run(run())


def test_code_agent_suppresses_unchanged_retrieval_and_retains_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = Database()
    queries = scripted_retriever(monkeypatch, database)
    callbacks: list[Any] = []

    class Callback(AsyncCallbackHandler):
        async def on_tool_end(self, output: Any, **kwargs: Any) -> None:
            callbacks.append((output, kwargs.get("tags")))

    async def run() -> None:
        registry, code = setup_tools(database)
        model = ScriptedModel(
            responses=[
                call("search_repository_code", {"query": "initial"}, "search-1"),
                call(
                    "get_file_context",
                    {"path": "app.py", "start_line": 3, "end_line": 4},
                    "file",
                ),
                call("search_repository_code", {"query": "initial"}, "search-2"),
                summary("One [1], then three [2].", ["1", "2"]),
            ]
        )
        findings = await investigate_code(
            model,
            registry,
            code,
            "Explain code.",
            config={
                "callbacks": [Callback()],
                "tags": ["parent"],
            },
        )
        assert findings.status == "complete", findings.errors
        assert findings.answer == "One [1], then three [2]."
        assert "Private" not in findings.answer
        assert len(findings.evidence) == 2
        assert [e.source_index for e in findings.evidence] == [1, 2]
        assert len(code.first_pass) == 1
        assert findings.model_calls == 4
        assert [r.call_id for r in findings.tool_executions] == [
            "search-1",
            "file",
            "search-2",
        ]
        assert (
            findings.tool_executions[1].message.artifact["sources"][0]["start_line"]
            == 3
        )
        assert findings.citation_catalog["1"] == findings.evidence[0].id
        assert findings.citation_catalog["2"] == findings.evidence[1].id
        assert len(callbacks) == 2
        assert findings.tool_executions[-1].message.status == "error"
        assert findings.errors[-1].recovered
        assert all("parent" in tags for _, tags in callbacks)
        assert database.active == 0

    asyncio.run(run())
    assert queries == ["initial"]


@pytest.mark.parametrize(
    "fault", ["unknown_source", "plaintext", "model_limit", "missing"]
)
def test_code_incomplete_and_failed_results_preserve_evidence(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    database = Database()
    scripted_retriever(monkeypatch, database)

    async def run() -> None:
        registry, code = setup_tools(database)
        completion = (
            AIMessage(content="Unvalidated answer.")
            if fault == "plaintext"
            else summary(
                sources=["invented"] if fault == "unknown_source" else ["1"],
                missing=["The related implementation was not found."]
                if fault == "missing"
                else [],
            )
        )
        findings = await investigate_code(
            ScriptedModel(
                responses=[
                    call("search_repository_code", {"query": "initial"}),
                    *(
                        [call("get_file_context", {"path": "app.py"}, "late")]
                        if fault == "model_limit"
                        else []
                    ),
                    completion,
                ]
            ),
            registry,
            code,
            "question",
            limits=AgentLimits(model_calls=2 if fault == "model_limit" else 8),
        )
        assert len(findings.evidence) == 1
        assert len(findings.tool_executions) == 1
        assert findings.missing_information
        if fault == "missing":
            assert findings.status == "incomplete"
            assert findings.answer
        else:
            assert findings.status == "failed"
            assert findings.answer == ""
            assert findings.errors
        assert database.active == 0

    asyncio.run(run())


def test_tool_budget_blocks_parallel_excess_and_allows_structured_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = Database()
    queries = scripted_retriever(monkeypatch, database)

    async def run() -> None:
        registry, code = setup_tools(database)
        batch = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_repository_code",
                    "args": {"query": f"q{index}"},
                    "id": f"call-{index}",
                }
                for index in range(7)
            ],
        )
        findings = await investigate_code(
            ScriptedModel(responses=[batch, summary()]),
            registry,
            code,
            "question",
            limits=AgentLimits(tool_calls=6, model_calls=2),
        )
        assert findings.status == "incomplete", findings.errors
        assert findings.answer == "Supported answer [1]."
        assert findings.model_calls == 2
        assert len(queries) == 6
        assert "q6" not in queries
        assert len(findings.tool_executions) == 7
        assert len(findings.errors) == 1

    asyncio.run(run())


def test_deadline_retains_evidence_from_completed_tools() -> None:
    class DelayedModel(ScriptedModel):
        async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
            if self.i == 1:
                await asyncio.sleep(10)
            return await super()._agenerate(*args, **kwargs)

    async def run() -> None:
        database = Database()
        registry, code = setup_tools(database)
        findings = await investigate_code(
            DelayedModel(
                responses=[call("get_file_context", {"path": "app.py"}), summary()]
            ),
            registry,
            code,
            "Read app.py.",
            limits=AgentLimits(timeout_seconds=0.25),
        )
        assert findings.status == "failed"
        assert findings.errors[-1].message == "Investigation timed out."
        assert len(findings.evidence) == len(findings.tool_executions) == 1
        assert findings.evidence[0].content == database.files["app.py"]
        assert database.active == 0

    asyncio.run(run())


@pytest.mark.parametrize(
    "invalid",
    [
        {"answer": "Missing required fields."},
        {"answer": " ", "source_ids": [], "missing_information": []},
        {"answer": "Invalid types.", "source_ids": [1], "missing_information": []},
        {
            "answer": "Unexpected field.",
            "source_ids": [],
            "missing_information": [],
            "extra": True,
        },
    ],
)
def test_invalid_completion_fails_validation(
    monkeypatch: pytest.MonkeyPatch,
    invalid: dict[str, Any],
) -> None:
    database = Database()
    scripted_retriever(monkeypatch, database)

    async def run() -> None:
        registry, code = setup_tools(database)
        findings = await investigate_code(
            ScriptedModel(
                responses=[
                    AIMessage(content=json.dumps(invalid)),
                ]
            ),
            registry,
            code,
            "question",
            limits=AgentLimits(tool_calls=1, model_calls=2),
        )
        assert findings.status == "failed", findings.errors
        assert findings.errors[-1].stage == "validation"
        assert findings.model_calls == 1
        assert findings.tool_executions == []
        assert findings.evidence == []

    asyncio.run(run())


def github_scope(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, object]]:
    async def scope() -> dict[str, str]:
        return {
            "owner": "example",
            "repo": "project",
            "html_url": "https://github.com/example/project",
        }

    async def get(path: str) -> dict[str, Any]:
        return {
            "number": 1,
            "title": "Streaming",
            "state": "open",
            "html_url": "https://github.com/example/project/issues/1",
            "user": {"login": "fixture"},
            "body": "Observed issue.",
        }

    monkeypatch.setattr(github_mcp_server, "repository_scope", scope)
    monkeypatch.setattr(github_mcp_server, "github_json", get)
    return connect_offline(monkeypatch, {"github": github_mcp_server.mcp})


@pytest.mark.parametrize("construction_failure", [False, True])
def test_github_native_mcp_artifacts_and_construction_cleanup(
    monkeypatch: pytest.MonkeyPatch, construction_failure: bool
) -> None:
    events = github_scope(monkeypatch)
    url = "https://github.com/example/project/issues/1"
    if construction_failure:

        def fail(*args: Any, **kwargs: Any) -> None:
            raise ValueError("Construction failed.")

        monkeypatch.setattr(agents, "create_github_agent", fail)

    async def run() -> None:
        registry = ToolRegistry(request(), (provider(),))
        findings = await investigate_github(
            ScriptedModel(
                responses=[
                    call("issue_read", {"issue_number": 1}),
                    summary(f"Observed issue: {url}", [url]),
                ]
            ),
            registry,
            "Read issue 1.",
        )
        if construction_failure:
            assert findings.status == "failed"
            assert findings.tool_executions == []
        else:
            assert findings.status == "complete", findings.errors
            assert findings.evidence == []
            assert (
                findings.tool_executions[0].message.artifact["structured_content"][
                    "issue"
                ]["url"]
                == url
            )
            assert findings.citation_catalog[url] == "github:call"

    asyncio.run(run())
    assert [e[:2] for e in events] == [("open", "github"), ("close", "github")]


@pytest.mark.parametrize("cancel", [False, True])
def test_github_deadline_and_cancellation_close_session(
    monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    events = github_scope(monkeypatch)
    started = asyncio.Event()

    async def delayed(path: str) -> dict[str, Any]:
        started.set()
        await asyncio.sleep(10)
        return {}

    monkeypatch.setattr(github_mcp_server, "github_json", delayed)

    async def run() -> None:
        registry = ToolRegistry(request(), (provider(),))
        task = asyncio.create_task(
            investigate_github(
                ScriptedModel(responses=[call("issue_read", {"issue_number": 1})]),
                registry,
                "question",
                limits=AgentLimits(timeout_seconds=0.1),
            )
        )
        await asyncio.wait_for(started.wait(), 2)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            findings = await task
            assert findings.status == "failed"
            assert "timed out" in findings.errors[-1].message

    asyncio.run(run())
    assert [e[:2] for e in events] == [("open", "github"), ("close", "github")]


def test_chat_completions_tool_schema_and_structured_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = github_scope(monkeypatch)
    requests: list[dict[str, Any]] = []
    url = "https://github.com/example/project/issues/1"

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload.get("tool_choice", "auto") == "auto"
        definitions = {t["function"]["name"]: t["function"] for t in payload["tools"]}
        assert "InvestigationSummary" not in definitions
        assert "owner" not in definitions["issue_read"]["parameters"]["properties"]
        message = (
            call("issue_read", {"issue_number": 1})
            if len(requests) == 1
            else summary(f"Observed issue: {url}", [url])
        )
        tool = message.tool_calls[0] if message.tool_calls else None
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "object": "chat.completion",
                "created": 0,
                "model": "offline",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls" if tool else "stop",
                        "message": {
                            "role": "assistant",
                            "content": None if tool else message.content,
                            "tool_calls": [
                                {
                                    "id": tool["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tool["name"],
                                        "arguments": json.dumps(tool["args"]),
                                    },
                                }
                            ]
                            if tool
                            else [],
                        },
                    }
                ],
            },
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            model = ChatOpenAI(
                api_key="offline-test",
                model="offline",
                streaming=False,
                use_responses_api=False,
                max_retries=0,
                http_async_client=client,
            )
            findings = await investigate_github(
                model, ToolRegistry(request(), (provider(),)), "Read issue 1."
            )
            assert findings.status == "complete", findings.errors

    asyncio.run(run())
    assert len(requests) == 2
    assert [e[:2] for e in events] == [("open", "github"), ("close", "github")]
