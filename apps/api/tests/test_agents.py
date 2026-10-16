from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

import httpx
import pytest
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from test_code_tools import Database, scripted_retriever, setup_tools
from test_tool_registry import connect_offline, provider, request

from app import agents, github_mcp_server
from app.agents import AgentLimits, investigate_code, investigate_github
from app.tool_registry import ToolRegistry


def test_context_projection_preserves_lossless_lookup_and_original_records() -> None:
    source = agents.CodeEvidence(
        id="code:one",
        source_index=1,
        corpus_commit_sha="a" * 40,
        path="app.py",
        start_line=1,
        end_line=2,
        content="one\ntwo",
        url="https://github.com/example/project/blob/commit/app.py#L1-L2",
    )
    other = source.model_copy(
        update={
            "id": "code:two",
            "source_index": 2,
            "start_line": 2,
            "end_line": 3,
            "content": "two\nthree",
            "url": None,
        }
    )
    result = {
        "sources": [
            source.model_dump(),
            other.model_dump(),
            {**source.model_dump(), "id": "unknown"},
            {**source.model_dump(), "content": "different\ncontent"},
        ],
        "missing_information": ["Requested branch unavailable."],
        "context_notes": ["Optional surrounding context unread."],
    }
    records = [
        agents.ToolExecutionRecord(
            id=f"code:{status}",
            call_id=status,
            name="get_file_context",
            arguments={"path": "app.py", "start_line": 1, "end_line": 2},
            result=result,
            message=ToolMessage(
                content=json.dumps(result),
                artifact=result,
                tool_call_id=status,
                status=status,
            ),
        )
        for status in ("success", "error")
    ]
    findings = agents.AgentFindings(
        role="CODE",
        status="incomplete",
        answer="Observed [1,2].",
        source_ids=[source.id, other.id],
        evidence=[source, other],
        citation_catalog={"1": source.id, "2": other.id},
        tool_executions=records,
        missing_information=result["missing_information"],
        context_notes=result["context_notes"],
        errors=[agents.InvestigationError(stage="tool", message="Read failed.")],
    )
    original = copy.deepcopy(findings.model_dump())
    payload = findings.to_context_payload()
    projected = payload["tool_executions"][0]
    assert projected["status"] == "success" and "message" not in projected
    canonical = {s["id"]: s for s in payload["evidence"]}
    for index, item in enumerate(projected["result"]["sources"]):
        expected = result["sources"][index]
        if index < 2:
            assert "content" not in item
            assert {**item, "content": canonical[item["id"]]["content"]} == expected
        else:
            assert item == expected
    assert {k: v for k, v in projected.items() if k not in {"status", "result"}} == {
        "id": "code:success",
        "call_id": "success",
        "name": "get_file_context",
        "arguments": records[0].arguments,
    }
    assert payload["tool_executions"][1]["result"] == result
    assert payload["tool_executions"][1]["status"] == "error"
    for field in ("citation_catalog", "missing_information", "context_notes", "errors"):
        assert payload[field] == findings.model_dump(mode="json")[field]
    assert findings.model_dump() == original


def test_context_projection_keeps_github_evidence_and_errors() -> None:
    result = {
        "files": [{"filename": "app.py", "patch": "-old\n+new"}],
        "issue": {
            "url": "https://github.com/example/project/issues/1",
            "body": "Live.",
        },
        "sources": [{"id": "github:one", "content": "Unmatched live evidence."}],
    }
    findings = agents.AgentFindings(
        role="GITHUB",
        status="incomplete",
        citation_catalog={result["issue"]["url"]: "github:one"},
        tool_executions=[
            agents.ToolExecutionRecord(
                id="github:one",
                call_id="one",
                name="issue_read",
                arguments={"issue_number": 1},
                result=result,
                message=ToolMessage(content="Live.", tool_call_id="one"),
            )
        ],
        missing_information=["Comments unavailable."],
        context_notes=["Live proposal."],
        errors=[agents.InvestigationError(stage="tool", message="Comments failed.")],
    )
    original = copy.deepcopy(findings.model_dump())
    payload = findings.to_context_payload()
    assert payload["tool_executions"][0]["result"] == result
    assert payload["errors"][0]["message"] == "Comments failed."
    assert payload["missing_information"] == ["Comments unavailable."]
    assert payload["context_notes"] == ["Live proposal."]
    assert findings.model_dump() == original


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedModel:
        return self


@pytest.mark.parametrize("role", ["CODE", "GITHUB", "SYNTHESIS"])
@pytest.mark.parametrize("marked", [False, True])
def test_tool_free_final_generation_does_not_require_a_marker(
    role: Any, marked: bool
) -> None:
    answer = "Observed final answer."
    model = ScriptedModel(
        responses=[
            AIMessage(
                content=(agents.FINAL_ANSWER_MARKER if marked else "") + answer,
                response_metadata={"finish_reason": "stop"},
            )
        ]
    )
    result = asyncio.run(
        agents.stream_final_answer(
            model,
            [],
            role=role,
            config=None,
            timeout=1,
            partial=False,
        )
    )
    assert result == answer


def test_specialist_context_deduplicates_exact_sources_without_changing_records() -> (
    None
):
    source = {"id": "code:one", "content": "one\ntwo", "path": "app.py"}
    distinct = {"id": "code:two", "content": "two\nthree", "path": "app.py"}
    results = [
        {"sources": [source]},
        {"sources": [source, distinct], "context_notes": ["Optional context."]},
        {"sources": [{**source, "content": "conflicting text"}]},
        {"sources": [source], "missing_information": ["Requested read failed."]},
        {"sources": [source]},
    ]
    messages = [
        ToolMessage(
            content=json.dumps(result),
            artifact=copy.deepcopy(result),
            tool_call_id=str(index),
            name="get_file_context",
            status="error" if index == 3 else "success",
        )
        for index, result in enumerate(results)
    ]
    # A native body that differs from its structured artifact must be retained.
    messages[-1].content = "Additional native content, not the artifact."
    before = [message.model_dump() for message in messages]
    received = []

    async def run() -> None:
        async def handler(request: Any) -> Any:
            received.append(request)
            return agents.ModelResponse(result=[AIMessage(content="Done.")])

        await agents._InvestigationControl(
            AgentLimits(), False, False
        ).awrap_model_call(
            agents.ModelRequest(model=ScriptedModel(responses=[]), messages=messages),
            handler,
        )

    asyncio.run(run())
    projected = received[0].messages
    assert json.loads(projected[0].content) == results[0]
    second = json.loads(projected[1].content)
    assert second["sources"][0] == {"id": "code:one", "path": "app.py"}
    assert second["sources"][1] == distinct
    assert second["context_notes"] == ["Optional context."]
    for index in (2, 3, 4):
        assert projected[index].model_dump() == before[index]
    for original, compact in zip(messages, projected, strict=True):
        assert compact.artifact == original.artifact
        assert compact.tool_call_id == original.tool_call_id
        assert compact.status == original.status
    assert [message.model_dump() for message in messages] == before
    assert "earlier tool result" in received[0].system_message.text


@pytest.mark.parametrize("final_answer", [False, True])
@pytest.mark.parametrize("github", [False, True])
def test_completion_checkpoint_keeps_necessary_tools_available(
    final_answer: bool, github: bool
) -> None:
    tool = (
        call("issue_read", {"issue_number": 9})
        if github
        else call("search_repository_code", {"query": "requested behavior"})
    )
    observation = ToolMessage(
        content=json.dumps({"sources": [{"id": "code:one", "content": "one"}]}),
        tool_call_id="call",
        name=tool.tool_calls[0]["name"],
    )
    supplied_tools = [{"name": "necessary_read"}]

    async def run() -> None:
        async def handler(request: Any) -> Any:
            assert request.tools == supplied_tools
            assert (
                "finish now without confirmation reads" in request.system_message.text
            )
            assert "Read comments, callers or callees" in request.system_message.text
            return agents.ModelResponse(result=[AIMessage(content="Done.")])

        await agents._InvestigationControl(
            AgentLimits(), final_answer, github
        ).awrap_model_call(
            agents.ModelRequest(
                model=ScriptedModel(responses=[]),
                messages=[tool, observation],
                tools=supplied_tools,
            ),
            handler,
        )

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["execution", "timeout", "validation"])
def test_github_failed_read_checkpoint_preserves_tools_and_native_errors(
    kind: str,
) -> None:
    tool = call("pull_request_read", {"pull_number": 7, "method": "get_diff"})
    observation = ToolMessage(
        content="Controlled read failure.",
        tool_call_id="call",
        name="pull_request_read",
        status="error",
        additional_kwargs={"repomind_error_kind": kind},
    )
    supplied_tools = [{"name": "pull_request_read"}, {"name": "issue_read"}]

    async def run() -> None:
        async def handler(request: Any) -> Any:
            assert request.tools == supplied_tools
            assert request.messages[-1] is observation
            assert observation.content == "Controlled read failure."
            assert (
                "including transient server errors" in request.system_message.text
            ) == (kind != "validation")
            assert (
                "Correct invalid arguments before retrying"
                in request.system_message.text
            )
            return agents.ModelResponse(result=[AIMessage(content="Done.")])

        await agents._InvestigationControl(AgentLimits(), False, True).awrap_model_call(
            agents.ModelRequest(
                model=ScriptedModel(responses=[]),
                messages=[tool, observation],
                tools=supplied_tools,
            ),
            handler,
        )

    asyncio.run(run())


def test_specialist_compaction_reaches_model_and_keeps_native_tool_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = Database()
    scripted_retriever(monkeypatch, database)
    batches: list[list[BaseMessage]] = []

    class RecordingModel(ScriptedModel):
        async def _agenerate(self, messages: list[BaseMessage], **kwargs: Any) -> Any:
            batches.append(messages)
            return await super()._agenerate(messages, **kwargs)

    async def run() -> None:
        registry, code = setup_tools(database)
        findings = await investigate_code(
            RecordingModel(
                responses=[
                    call("search_repository_code", {"query": "initial"}, "search"),
                    call("get_file_context", {"path": "app.py", "end_line": 2}, "read"),
                    summary(),
                ]
            ),
            registry,
            code,
            "Inspect the requested behavior.",
        )
        assert findings.status == "complete", findings.errors
        observations = [m for m in batches[-1] if isinstance(m, ToolMessage)]
        first, second = [json.loads(message.content) for message in observations]
        assert first["sources"][0]["content"] == findings.evidence[0].content
        assert "content" not in second["sources"][0]
        for record in findings.tool_executions:
            assert (
                record.result["sources"][0]["content"] == findings.evidence[0].content
            )
            assert json.loads(record.message.content) == record.result
            assert record.message.artifact == record.result
        assert findings.model_calls == 3 and len(findings.tool_executions) == 2

    asyncio.run(run())


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


def test_snapshot_evidence_does_not_erase_an_unsuccessful_code_search() -> None:
    missing, unresolved = agents._tool_limitations(
        [
            ToolMessage(
                name="search_repository_code",
                tool_call_id="search",
                content=json.dumps(
                    {
                        "sources": [],
                        "missing_information": ["No indexed code matched this query."],
                    }
                ),
            ),
            ToolMessage(
                name="get_repository_snapshot",
                tool_call_id="snapshot",
                content=json.dumps(
                    {
                        "indexed_commit_sha": "a" * 40,
                        "url": "https://github.com/example/project/tree/" + "a" * 40,
                    }
                ),
            ),
        ]
    )
    assert missing == ["No indexed code matched this query."]
    assert unresolved == set()


def summary(
    answer: str = "Supported answer [1].",
    sources: list[str] | None = None,
    missing: list[str] | None = None,
    notes: list[str] | None = None,
) -> AIMessage:
    return AIMessage(
        content=json.dumps(
            {
                "answer": answer,
                "source_ids": sources if sources is not None else ["1"],
                "missing_information": missing or [],
                "context_notes": notes or [],
            }
        ),
    )


@pytest.mark.parametrize("split", range(len(agents.FINAL_ANSWER_CLOSER) + 1))
def test_final_closer_is_buffered_across_native_chunks(
    monkeypatch: pytest.MonkeyPatch, split: int
) -> None:
    deltas: list[str] = []

    async def publish(name: str, data: Any, **kwargs: Any) -> None:
        assert name == "repomind_answer_delta" and data["native"]
        deltas.append(data["text"])

    monkeypatch.setattr(agents, "adispatch_custom_event", publish)

    async def run() -> None:
        tokens = agents._FinalAnswerTokens("CODE", {})
        await tokens.feed(AIMessageChunk(content="<final_answer>Answer [1]."))
        assert deltas == ["Answer [1]."]
        for part in (
            agents.FINAL_ANSWER_CLOSER[:split],
            agents.FINAL_ANSWER_CLOSER[split:],
            "\n" * 300,
        ):
            await tokens.feed(AIMessageChunk(content=part))
            assert len(tokens.pending) <= len(agents.FINAL_ANSWER_CLOSER)
        assert await tokens.finish() == "Answer [1]."
        assert "".join(deltas) == "Answer [1]."

    asyncio.run(run())


@pytest.mark.parametrize(
    "body",
    [
        "Use `</final_answer>` as an example.",
        "```xml\n</final_answer>\n```",
        "A literal </final_answer>   appears here.",
        "A partial literal </final_ans",
    ],
)
def test_literal_closers_in_answer_content_are_preserved(
    monkeypatch: pytest.MonkeyPatch, body: str
) -> None:
    deltas: list[str] = []

    async def publish(name: str, data: Any, **kwargs: Any) -> None:
        deltas.append(data["text"])

    monkeypatch.setattr(agents, "adispatch_custom_event", publish)

    async def run() -> None:
        tokens = agents._FinalAnswerTokens("CODE", {})
        for character in "<final_answer>" + body:
            await tokens.feed(AIMessageChunk(content=character))
        assert await tokens.finish() == body
        assert "".join(deltas) == body

    asyncio.run(run())


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
        assert set(value) == {
            "answer",
            "source_ids",
            "missing_information",
            "context_notes",
        }
        assert value["missing_information"] == []
        assert value["context_notes"] == []
    if github:
        assert "https://github.com/fictional-lab/atlas/" in output
        assert "[7]" not in output
    else:
        assert "[7]" in output


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("followup", [False, True])
def test_partial_status_tracks_requested_ranges_not_optional_context(
    explicit: bool,
    followup: bool,
) -> None:
    database = Database()
    database.files["large.py"] = "x" * 8000 + "\n" + "y" * 8000 + "\ntail"

    async def run() -> None:
        registry, code = setup_tools(database)
        arguments: dict[str, Any] = {"path": "large.py"}
        if explicit:
            arguments["end_line"] = 2
        responses = [call("get_file_context", arguments, "first")]
        if followup:
            responses.append(
                call(
                    "get_file_context",
                    {"path": "large.py", "start_line": 2, "end_line": 2},
                    "second",
                )
            )
        responses.append(
            summary(
                "Observed content [1].", ["1"], notes=["Other context is optional."]
            )
        )
        findings = await investigate_code(
            ScriptedModel(responses=responses),
            registry,
            code,
            "Read requested context.",
        )
        assert findings.status == (
            "incomplete" if explicit and not followup else "complete"
        )
        assert bool(findings.missing_information) is (explicit and not followup)

    asyncio.run(run())


@pytest.mark.parametrize("github", [False, True])
@pytest.mark.parametrize("required_gap", [False, True])
def test_handoff_notes_do_not_qualify_answers_or_erase_required_gaps(
    monkeypatch: pytest.MonkeyPatch, github: bool, required_gap: bool
) -> None:
    github_scope(monkeypatch)
    note = "The other agent owns the remaining portion of the question."
    gap = "A requested fact within this role is unavailable."
    url = "https://github.com/example/project/issues/1"

    async def run() -> None:
        model = ScriptedModel(
            responses=[
                call("issue_read", {"issue_number": 1})
                if github
                else call("get_file_context", {"path": "app.py"}),
                summary(
                    url if github else "Observed [1].",
                    [url if github else "1"],
                    [gap] if required_gap else [],
                    [note],
                ),
            ]
        )
        if github:
            findings = await investigate_github(
                model, ToolRegistry(request(), (provider(),)), "Compare code and issue."
            )
        else:
            registry, code = setup_tools(Database())
            findings = await investigate_code(
                model, registry, code, "Compare code and issue."
            )
        assert findings.context_notes == [note]
        assert findings.missing_information == ([gap] if required_gap else [])
        assert findings.status == ("incomplete" if required_gap else "complete")
        assert findings.evidence or findings.citation_catalog

    asyncio.run(run())


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
        assert findings.errors[-1].code == "summary_schema"
        assert findings.model_calls == 1
        assert findings.tool_executions == []
        assert findings.evidence == []

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["json", "field", "extra", "many"])
def test_summary_diagnostics_are_actionable_bounded_and_redacted(failure: str) -> None:
    secret = "sk-private-output-do-not-retain"
    invalid: dict[str, Any] = {
        "answer": "Observed evidence.",
        "source_ids": [],
        "missing_information": [],
    }
    if failure in {"field", "many"}:
        invalid["source_ids"] = [{"token": secret}] * (20 if failure == "many" else 1)
    if failure == "extra":
        invalid[secret] = secret
    content = "Malformed JSON: " + secret if failure == "json" else json.dumps(invalid)

    async def run() -> None:
        registry, code = setup_tools(Database())
        findings = await investigate_code(
            ScriptedModel(responses=[AIMessage(content=content)]),
            registry,
            code,
            "Question",
        )
        error = findings.errors[-1]
        assert findings.status == "failed" and error.code == "summary_schema"
        assert secret not in error.message and len(error.message) < 300
        assert (
            "summary: json_invalid"
            if failure == "json"
            else "unexpected_field: extra_forbidden"
            if failure == "extra"
            else "source_ids[0]: string_type"
        ) in error.message
        if failure == "many":
            assert (
                "source_ids[4]" in error.message
                and "source_ids[5]" not in error.message
            )
        assert findings.model_calls == 1

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
