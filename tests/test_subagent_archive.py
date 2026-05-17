"""Tests for sub-agent transcript archiving (v1.2 schema)."""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cc_session_toolkit.archive import (
    _classify_agent_kind,
    _collect_parent_tool_uses,
    _duration_seconds,
    _existing_uncompressed_sha,
    _extract_subagent_info,
    _file_sha256,
    _find_subagent_jsonls,
    _first_user_prompt_preview,
    _match_subagents_to_parent_tool_uses,
    _resolve_nested_depth,
    archive_subagent_transcripts,
    capture_code_state,
    create_session_metadata,
    get_session_files,
    summarise_subagents,
)
from cc_session_toolkit.config import (
    DEFAULT_LICENCE,
    EXTRACTOR_MODEL_ID,
    SCHEMA_VERSION,
)
from cc_session_toolkit.extraction import extract_session_stats


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

def _write_subagent_jsonl(
    path: Path,
    agent_id: str,
    session_id: str,
    *,
    first_prompt: str = "Go and do the thing.",
    parent_uuid: str | None = None,
    extra_turns: int = 0,
    base_time: datetime | None = None,
) -> Path:
    """
    Write a minimal but realistic sub-agent JSONL file.

    One user message (the invocation prompt) plus *extra_turns* assistant
    messages with a Read tool-use each.  Records carry their own UUIDs
    so nested-depth detection can use them as linkage keys.
    """
    t0 = base_time or datetime(2026, 4, 22, 12, 0, 0, tzinfo=timezone.utc)
    lines: list[str] = []

    user_uuid = f"msg-{agent_id}-user"
    lines.append(json.dumps({
        "parentUuid": parent_uuid,
        "isSidechain": True,
        "agentId": agent_id,
        "sessionId": session_id,
        "type": "user",
        "uuid": user_uuid,
        "timestamp": t0.isoformat(),
        "message": {"role": "user", "content": first_prompt},
    }))

    for i in range(extra_turns):
        t = t0 + timedelta(seconds=5 * (i + 1))
        lines.append(json.dumps({
            "parentUuid": user_uuid if i == 0 else f"msg-{agent_id}-{i-1}",
            "isSidechain": True,
            "agentId": agent_id,
            "sessionId": session_id,
            "type": "assistant",
            "uuid": f"msg-{agent_id}-{i}",
            "timestamp": t.isoformat(),
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-6-20251001",
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"toolu_sub_{agent_id}_{i}",
                        "name": "Read",
                        "input": {"file_path": "/tmp/foo.txt"},
                    }
                ],
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 40,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
        }))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_parent_jsonl(
    path: Path,
    session_id: str,
    agent_invocations: list[dict[str, str]],
    *,
    base_time: datetime | None = None,
) -> Path:
    """
    Write a parent JSONL that records the given sub-agent invocations as
    Agent tool_use blocks, in order.
    """
    t0 = base_time or datetime(2026, 4, 22, 11, 59, 0, tzinfo=timezone.utc)
    lines: list[str] = []
    for i, inv in enumerate(agent_invocations):
        lines.append(json.dumps({
            "parentUuid": None,
            "isSidechain": False,
            "sessionId": session_id,
            "type": "user",
            "uuid": f"parent-user-{i}",
            "timestamp": (t0 + timedelta(seconds=i * 60)).isoformat(),
            "message": {
                "role": "user",
                "content": f"please spawn agent {i}",
            },
        }))
        lines.append(json.dumps({
            "parentUuid": f"parent-user-{i}",
            "isSidechain": False,
            "sessionId": session_id,
            "type": "assistant",
            "uuid": f"parent-asst-{i}",
            "timestamp": (t0 + timedelta(seconds=i * 60 + 1)).isoformat(),
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-7",
                "content": [
                    {
                        "type": "tool_use",
                        "id": inv["tool_use_id"],
                        "name": inv.get("tool_name", "Agent"),
                        "input": {
                            "description": inv.get(
                                "description", "work"
                            ),
                            "subagent_type": inv.get("subagent_type", "Explore"),
                            "prompt": inv.get("prompt", "Go and do the thing."),
                        },
                    }
                ],
                "usage": {
                    "input_tokens": 500,
                    "output_tokens": 50,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
        }))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def current_layout_session(tmp_path: Path) -> dict[str, Path]:
    """
    Build a fake ``~/.claude/projects/<proj>/`` layout matching the
    current (post-2026-01-12) CC shape.

    Produces a parent JSONL and two sub-agent transcripts under
    ``<session>/subagents/``.
    """
    session_id = "abc12345-1234-5678-9abc-def012345678"
    proj_dir = tmp_path / "-home-shawn-test-project"
    proj_dir.mkdir()
    parent = proj_dir / f"{session_id}.jsonl"
    _write_parent_jsonl(
        parent,
        session_id,
        [
            {
                "tool_use_id": "toolu_01AAAA",
                "subagent_type": "Explore",
                "prompt": "Audit the daily-sync script.",
            },
            {
                "tool_use_id": "toolu_01BBBB",
                "subagent_type": "Plan",
                "prompt": "Plan the fix for the audit findings.",
            },
        ],
    )

    sub_dir = proj_dir / session_id / "subagents"
    sub_dir.mkdir(parents=True)
    agent_a = sub_dir / "agent-a1a1a1a1a1.jsonl"
    _write_subagent_jsonl(
        agent_a,
        "a1a1a1a1a1",
        session_id,
        first_prompt="Audit the daily-sync script.",
        parent_uuid="parent-user-0",
        extra_turns=2,
        base_time=datetime(2026, 4, 22, 12, 0, 0, tzinfo=timezone.utc),
    )
    agent_b = sub_dir / "agent-b2b2b2b2b2.jsonl"
    _write_subagent_jsonl(
        agent_b,
        "b2b2b2b2b2",
        session_id,
        first_prompt="Plan the fix for the audit findings.",
        parent_uuid="parent-user-1",
        extra_turns=3,
        base_time=datetime(2026, 4, 22, 12, 5, 0, tzinfo=timezone.utc),
    )

    return {
        "parent": parent,
        "session_id": session_id,
        "agent_a": agent_a,
        "agent_b": agent_b,
        "proj_dir": proj_dir,
        "sub_dir": sub_dir,
    }


@pytest.fixture()
def legacy_layout_session(tmp_path: Path) -> dict[str, Path]:
    """
    Build a fake pre-2026-01-12 CC layout: flat ``agent-*.jsonl`` files
    next to the parent, filtered by sessionId.
    """
    session_id = "legacy5-1234-5678-9abc-def012345678"
    other_id = "other55-1234-5678-9abc-def012345678"
    proj_dir = tmp_path / "-home-shawn-legacy-project"
    proj_dir.mkdir()
    parent = proj_dir / f"{session_id}.jsonl"
    _write_parent_jsonl(
        parent,
        session_id,
        [{"tool_use_id": "toolu_01ZZZZ", "prompt": "Legacy work."}],
    )
    ours = proj_dir / "agent-l1e9acy.jsonl"
    _write_subagent_jsonl(
        ours,
        "l1e9acy",
        session_id,
        first_prompt="Legacy work.",
        extra_turns=1,
    )
    theirs = proj_dir / "agent-o1t7her.jsonl"
    _write_subagent_jsonl(
        theirs,
        "o1t7her",
        other_id,
        first_prompt="Not ours.",
        extra_turns=1,
    )
    return {
        "parent": parent,
        "session_id": session_id,
        "ours": ours,
        "theirs": theirs,
        "proj_dir": proj_dir,
    }


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

class TestClassifyAgentKind:
    """Tests for :func:`_classify_agent_kind`."""

    def test_user_invoked_by_default(self) -> None:
        assert _classify_agent_kind("a1a1a1a1a1") == "user_invoked"

    def test_acompact_prefix(self) -> None:
        assert _classify_agent_kind("acompact-edeaa0") == "acompact"

    def test_prompt_suggestion_prefix(self) -> None:
        assert (
            _classify_agent_kind("aprompt_suggestion-20f424")
            == "aprompt_suggestion"
        )

    def test_empty_id_is_unknown(self) -> None:
        assert _classify_agent_kind("") == "unknown"


class TestDurationSeconds:
    """Tests for :func:`_duration_seconds`."""

    def test_basic_duration(self) -> None:
        s = "2026-04-22T12:00:00+00:00"
        e = "2026-04-22T12:00:30+00:00"
        assert _duration_seconds(s, e) == 30.0

    def test_z_suffix_accepted(self) -> None:
        s = "2026-04-22T12:00:00Z"
        e = "2026-04-22T12:00:10Z"
        assert _duration_seconds(s, e) == 10.0

    def test_missing_returns_none(self) -> None:
        assert _duration_seconds(None, "x") is None
        assert _duration_seconds("x", None) is None


# -------------------------------------------------------------------------
# Discovery
# -------------------------------------------------------------------------

class TestFindSubagentJsonls:
    """Tests for :func:`_find_subagent_jsonls`."""

    def test_current_layout_discovers_both(
        self, current_layout_session: dict[str, Path]
    ) -> None:
        found = _find_subagent_jsonls(
            current_layout_session["parent"],
            current_layout_session["session_id"],
        )
        assert len(found) == 2
        assert current_layout_session["agent_a"] in found
        assert current_layout_session["agent_b"] in found

    def test_current_layout_sorted_by_timestamp(
        self, current_layout_session: dict[str, Path]
    ) -> None:
        found = _find_subagent_jsonls(
            current_layout_session["parent"],
            current_layout_session["session_id"],
        )
        # agent_a started at 12:00:00, agent_b at 12:05:00 — a first
        assert found[0] == current_layout_session["agent_a"]
        assert found[1] == current_layout_session["agent_b"]

    def test_legacy_layout_filters_by_session_id(
        self, legacy_layout_session: dict[str, Path]
    ) -> None:
        found = _find_subagent_jsonls(
            legacy_layout_session["parent"],
            legacy_layout_session["session_id"],
        )
        assert legacy_layout_session["ours"] in found
        assert legacy_layout_session["theirs"] not in found

    def test_empty_when_no_subagents(self, tmp_path: Path) -> None:
        parent = tmp_path / "session.jsonl"
        parent.write_text("", encoding="utf-8")
        assert _find_subagent_jsonls(parent, "session") == []


# -------------------------------------------------------------------------
# Extraction
# -------------------------------------------------------------------------

class TestExtractSubagentInfo:
    """Tests for :func:`_extract_subagent_info`."""

    def test_captures_identity_and_parent(
        self, current_layout_session: dict[str, Path]
    ) -> None:
        info = _extract_subagent_info(current_layout_session["agent_a"])
        assert info is not None
        assert info["agent_id"] == "a1a1a1a1a1"
        assert info["agent_kind"] == "user_invoked"
        assert info["parent_session_id"] == current_layout_session["session_id"]
        assert info["parent_session_message_uuid"] == "parent-user-0"

    def test_counts_records_and_messages(
        self, current_layout_session: dict[str, Path]
    ) -> None:
        info = _extract_subagent_info(current_layout_session["agent_a"])
        assert info["records"] == 3  # 1 user + 2 assistant turns
        assert info["user_messages"] == 1
        assert info["assistant_messages"] == 2

    def test_first_prompt_preview_captured(
        self, current_layout_session: dict[str, Path]
    ) -> None:
        info = _extract_subagent_info(current_layout_session["agent_a"])
        assert info["first_prompt_preview"].startswith(
            "Audit the daily-sync script"
        )

    def test_sha_matches_file_bytes(
        self, current_layout_session: dict[str, Path]
    ) -> None:
        info = _extract_subagent_info(current_layout_session["agent_a"])
        assert (
            info["jsonl_sha256_uncompressed"]
            == _file_sha256(current_layout_session["agent_a"])
        )

    def test_returns_none_on_unreadable_first_line(
        self, tmp_path: Path
    ) -> None:
        bad = tmp_path / "agent-bad.jsonl"
        bad.write_text("not json\n", encoding="utf-8")
        assert _extract_subagent_info(bad) is None


class TestFirstUserPromptPreview:
    """Tests for :func:`_first_user_prompt_preview`."""

    def test_string_content(
        self, current_layout_session: dict[str, Path]
    ) -> None:
        got = _first_user_prompt_preview(current_layout_session["agent_a"])
        assert got.startswith("Audit the daily-sync")

    def test_limit_applied(self, tmp_path: Path) -> None:
        path = tmp_path / "agent-x.jsonl"
        _write_subagent_jsonl(
            path, "x1x1x1x1x1", "sess",
            first_prompt="x" * 500,
        )
        got = _first_user_prompt_preview(path, limit=50)
        assert len(got) == 50


# -------------------------------------------------------------------------
# Linkage
# -------------------------------------------------------------------------

class TestCollectParentToolUses:
    """Tests for :func:`_collect_parent_tool_uses`."""

    def test_returns_in_order(
        self, current_layout_session: dict[str, Path]
    ) -> None:
        calls = _collect_parent_tool_uses(current_layout_session["parent"])
        assert [c["id"] for c in calls] == ["toolu_01AAAA", "toolu_01BBBB"]

    def test_accepts_both_agent_and_task_names(self, tmp_path: Path) -> None:
        parent = tmp_path / "session.jsonl"
        _write_parent_jsonl(
            parent, "s",
            [
                {"tool_use_id": "a", "tool_name": "Agent", "prompt": "p1"},
                {"tool_use_id": "b", "tool_name": "Task", "prompt": "p2"},
            ],
        )
        calls = _collect_parent_tool_uses(parent)
        assert {c["id"] for c in calls} == {"a", "b"}


class TestMatchSubagentsToParentToolUses:
    """Tests for :func:`_match_subagents_to_parent_tool_uses`."""

    def test_ordinal_matches_confirmed_by_prompt(
        self, current_layout_session: dict[str, Path]
    ) -> None:
        subs = [
            _extract_subagent_info(current_layout_session["agent_a"]),
            _extract_subagent_info(current_layout_session["agent_b"]),
        ]
        _match_subagents_to_parent_tool_uses(
            current_layout_session["parent"], subs
        )
        assert subs[0]["parent_tool_use_id"] == "toolu_01AAAA"
        assert subs[0]["parent_tool_use_index"] == 1
        assert subs[0]["subagent_type"] == "Explore"
        assert subs[0]["parent_tool_use_linkage"] == "confirmed"
        assert subs[1]["parent_tool_use_id"] == "toolu_01BBBB"
        assert subs[1]["parent_tool_use_linkage"] == "confirmed"

    def test_inferred_when_prompts_dont_match(self, tmp_path: Path) -> None:
        parent = tmp_path / "session.jsonl"
        _write_parent_jsonl(
            parent, "s",
            [{"tool_use_id": "toolu_XX", "prompt": "X" * 100}],
        )
        agent = tmp_path / "agent-q.jsonl"
        _write_subagent_jsonl(
            agent, "q1q1q1q1q1", "s",
            first_prompt="Y" * 100, parent_uuid="parent-user-0",
        )
        sub = _extract_subagent_info(agent)
        _match_subagents_to_parent_tool_uses(parent, [sub])
        assert sub["parent_tool_use_linkage"] == "inferred"

    def test_internal_agents_not_matched(self, tmp_path: Path) -> None:
        parent = tmp_path / "session.jsonl"
        _write_parent_jsonl(parent, "s", [])
        agent = tmp_path / "agent-acompact-abc.jsonl"
        _write_subagent_jsonl(
            agent, "acompact-abcdef", "s",
            first_prompt="internal compact task",
        )
        sub = _extract_subagent_info(agent)
        _match_subagents_to_parent_tool_uses(parent, [sub])
        assert sub["parent_tool_use_linkage"] == "not_applicable"
        assert sub["parent_tool_use_id"] is None

    def test_unknown_when_more_subagents_than_tool_uses(
        self, tmp_path: Path
    ) -> None:
        parent = tmp_path / "session.jsonl"
        _write_parent_jsonl(parent, "s", [])
        agent = tmp_path / "agent-orphan.jsonl"
        _write_subagent_jsonl(agent, "orphan0000", "s")
        sub = _extract_subagent_info(agent)
        _match_subagents_to_parent_tool_uses(parent, [sub])
        assert sub["parent_tool_use_linkage"] == "unknown"


class TestResolveNestedDepth:
    """Tests for :func:`_resolve_nested_depth`."""

    def test_flat_layout_all_depth_zero(
        self, current_layout_session: dict[str, Path]
    ) -> None:
        subs = [
            _extract_subagent_info(current_layout_session["agent_a"]),
            _extract_subagent_info(current_layout_session["agent_b"]),
        ]
        _resolve_nested_depth(subs)
        for sub in subs:
            assert sub["depth"] == 0
            assert sub["parent_agent_id"] is None


# -------------------------------------------------------------------------
# End-to-end archive
# -------------------------------------------------------------------------

class TestArchiveSubagentTranscripts:
    """Tests for :func:`archive_subagent_transcripts`."""

    def test_copies_to_subagents_subdirectory(
        self,
        current_layout_session: dict[str, Path],
        tmp_path: Path,
    ) -> None:
        dest = tmp_path / "archive"
        result = archive_subagent_transcripts(
            current_layout_session["parent"],
            current_layout_session["session_id"],
            dest,
            use_gzip=True,
        )
        assert len(result) == 2
        assert (dest / "subagents" / "agent-a1a1a1a1a1.jsonl.gz").is_file()
        assert (dest / "subagents" / "agent-b2b2b2b2b2.jsonl.gz").is_file()

    def test_gzip_preserves_uncompressed_sha(
        self,
        current_layout_session: dict[str, Path],
        tmp_path: Path,
    ) -> None:
        dest = tmp_path / "archive"
        result = archive_subagent_transcripts(
            current_layout_session["parent"],
            current_layout_session["session_id"],
            dest,
            use_gzip=True,
        )
        for sub in result:
            archived = dest / sub["archive_path"]
            recovered_sha = _existing_uncompressed_sha(archived, True)
            assert recovered_sha == sub["jsonl_sha256_uncompressed"]

    def test_idempotent_when_source_unchanged(
        self,
        current_layout_session: dict[str, Path],
        tmp_path: Path,
    ) -> None:
        dest = tmp_path / "archive"
        archive_subagent_transcripts(
            current_layout_session["parent"],
            current_layout_session["session_id"],
            dest, use_gzip=True,
        )
        second = archive_subagent_transcripts(
            current_layout_session["parent"],
            current_layout_session["session_id"],
            dest, use_gzip=True,
        )
        for sub in second:
            assert "idempotent: target unchanged" in sub["capture_notes"]

    def test_overwrites_when_source_grows(
        self,
        current_layout_session: dict[str, Path],
        tmp_path: Path,
    ) -> None:
        dest = tmp_path / "archive"
        archive_subagent_transcripts(
            current_layout_session["parent"],
            current_layout_session["session_id"],
            dest, use_gzip=True,
        )
        # Append a new turn to agent_a's source file
        src = current_layout_session["agent_a"]
        with open(src, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "parentUuid": "msg-a1a1a1a1a1-1",
                "isSidechain": True,
                "agentId": "a1a1a1a1a1",
                "sessionId": current_layout_session["session_id"],
                "type": "assistant",
                "uuid": "msg-a1a1a1a1a1-grown",
                "timestamp": "2026-04-22T12:00:30+00:00",
                "message": {
                    "role": "assistant",
                    "model": "claude-sonnet-4-6-20251001",
                    "content": [{"type": "text", "text": "extra"}],
                    "usage": {
                        "input_tokens": 1, "output_tokens": 1,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            }) + "\n")
        second = archive_subagent_transcripts(
            current_layout_session["parent"],
            current_layout_session["session_id"],
            dest, use_gzip=True,
        )
        grown = next(s for s in second if s["agent_id"] == "a1a1a1a1a1")
        assert any(
            "superseded" in note for note in grown["capture_notes"]
        )

    def test_no_subagents_returns_empty(self, tmp_path: Path) -> None:
        parent = tmp_path / "session.jsonl"
        parent.write_text("", encoding="utf-8")
        assert archive_subagent_transcripts(
            parent, "sid", tmp_path / "archive"
        ) == []

    def test_legacy_layout_archived(
        self,
        legacy_layout_session: dict[str, Path],
        tmp_path: Path,
    ) -> None:
        dest = tmp_path / "archive"
        result = archive_subagent_transcripts(
            legacy_layout_session["parent"],
            legacy_layout_session["session_id"],
            dest,
            use_gzip=True,
        )
        assert len(result) == 1
        assert result[0]["agent_id"] == "l1e9acy"


# -------------------------------------------------------------------------
# Schema v1.2 integration
# -------------------------------------------------------------------------

class TestSessionMetaV12:
    """Tests for v1.2 ``session.meta.json`` shape."""

    def test_schema_version_is_1_2(
        self, sample_session_jsonl: Path
    ) -> None:
        stats = extract_session_stats(sample_session_jsonl)
        meta = create_session_metadata(
            session_id="abc12345",
            session_path=sample_session_jsonl,
            stats=stats,
        )
        assert meta["schema_version"] == "1.2"
        assert SCHEMA_VERSION == "1.2"

    def test_subagents_empty_list_by_default(
        self, sample_session_jsonl: Path
    ) -> None:
        stats = extract_session_stats(sample_session_jsonl)
        meta = create_session_metadata(
            session_id="abc12345",
            session_path=sample_session_jsonl,
            stats=stats,
        )
        assert meta["subagents"] == []
        assert meta["statistics"]["subagents_summary"]["count"] == 0

    def test_subagents_list_populated(
        self, sample_session_jsonl: Path
    ) -> None:
        stats = extract_session_stats(sample_session_jsonl)
        fake_sub = {
            "agent_id": "x1x1x1x1x1",
            "agent_kind": "user_invoked",
            "subagent_type": "Explore",
            "tokens": {"input": 100, "output": 50},
            "estimated_cost_usd": 0.01,
            "records": 5,
            "capture_status": "complete",
        }
        meta = create_session_metadata(
            session_id="abc12345",
            session_path=sample_session_jsonl,
            stats=stats,
            subagents=[fake_sub],
        )
        assert len(meta["subagents"]) == 1
        summary = meta["statistics"]["subagents_summary"]
        assert summary["count"] == 1
        assert summary["by_type"] == {"Explore": 1}
        assert summary["estimated_cost_usd"] == 0.01


class TestSummariseSubagents:
    """Tests for :func:`summarise_subagents`."""

    def test_aggregates_counts_and_costs(self) -> None:
        subs = [
            {
                "agent_kind": "user_invoked",
                "subagent_type": "Explore",
                "tokens": {"input": 10, "output": 5},
                "estimated_cost_usd": 0.05,
                "records": 4,
                "capture_status": "complete",
            },
            {
                "agent_kind": "user_invoked",
                "subagent_type": "Plan",
                "tokens": {"input": 20, "output": 10},
                "estimated_cost_usd": 0.07,
                "records": 6,
                "capture_status": "complete",
            },
            {
                "agent_kind": "acompact",
                "subagent_type": None,
                "tokens": {"input": 1, "output": 1},
                "estimated_cost_usd": 0.0,
                "records": 1,
                "capture_status": "complete",
            },
        ]
        summary = summarise_subagents(subs)
        assert summary["count"] == 3
        assert summary["total_records"] == 11
        assert summary["total_tokens"] == {"input": 31, "output": 16}
        assert summary["estimated_cost_usd"] == 0.12
        assert summary["by_type"] == {
            "Explore": 1, "Plan": 1, "unspecified": 1,
        }
        assert summary["by_kind"] == {"user_invoked": 2, "acompact": 1}

    def test_empty_input(self) -> None:
        summary = summarise_subagents([])
        assert summary["count"] == 0
        assert summary["total_records"] == 0
        assert summary["estimated_cost_usd"] == 0.0


# -------------------------------------------------------------------------
# get_session_files — latent-bug fix
# -------------------------------------------------------------------------

class TestGetSessionFilesFiltering:
    """Sub-agent JSONLs must not be swept up as top-level sessions."""

    def test_agent_files_excluded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from cc_session_toolkit import archive as archive_mod
        fake_home = tmp_path / "home"
        cc_dir = fake_home / ".claude" / "projects"
        cc_dir.mkdir(parents=True)
        monkeypatch.setattr(
            archive_mod, "CLAUDE_PROJECTS_DIR", cc_dir,
        )
        # Project root under the same synthetic home
        project_root = fake_home / "test-project"
        project_root.mkdir()
        (project_root / ".git").mkdir()
        (project_root / "CLAUDE.md").write_text("# Project: test-project\n")
        # Populate the CC-side project directory with mixed files
        encoded = str(project_root.resolve()).replace("/", "-")
        proj_cc_dir = cc_dir / encoded
        proj_cc_dir.mkdir()
        (proj_cc_dir / "abc-main.jsonl").write_text("")
        (proj_cc_dir / "agent-subagent01.jsonl").write_text("")
        (proj_cc_dir / "xyz-main.jsonl").write_text("")

        files = get_session_files(project_root)
        names = {p.name for p in files}
        assert "abc-main.jsonl" in names
        assert "xyz-main.jsonl" in names
        assert "agent-subagent01.jsonl" not in names


# -------------------------------------------------------------------------
# Provenance audit Gap 2 — code_state capture
# -------------------------------------------------------------------------


def _init_git_repo(path: Path) -> str:
    """
    Initialise a minimal git repo at ``path`` and return the HEAD sha.

    Helper for code_state tests. Uses ``git init -q`` + a single empty
    commit so we have a deterministic HEAD to assert against. Each
    test runs in an isolated tmp_path so there is no cross-contamination.
    """
    import subprocess as sp
    env = {
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
        # Suppress global gitconfig interference (e.g. signing keys)
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "HOME": str(path),
    }
    sp.run(["git", "init", "-q"], cwd=path, check=True, env=env)
    sp.run(
        ["git", "commit", "--allow-empty", "-q", "-m", "init"],
        cwd=path, check=True, env=env,
    )
    return sp.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path, check=True, capture_output=True, text=True, env=env,
    ).stdout.strip()


class TestCaptureCodeState:
    """Tests for :func:`capture_code_state` (provenance audit Gap 2)."""

    def test_returns_none_for_missing_project_root(self) -> None:
        """A *None* project root yields all-None fields."""
        state = capture_code_state(None)
        assert state == {
            "commit_at_start": None,
            "commit_at_end": None,
            "dirty_at_end": None,
        }

    def test_returns_none_for_nonexistent_path(
        self, tmp_path: Path,
    ) -> None:
        """Non-existent project root path yields all-None fields."""
        state = capture_code_state(tmp_path / "does-not-exist")
        assert state["commit_at_end"] is None
        assert state["dirty_at_end"] is None

    def test_returns_none_for_non_git_directory(
        self, tmp_path: Path,
    ) -> None:
        """A real directory that is not a git repo yields all-None."""
        bare = tmp_path / "bare-dir"
        bare.mkdir()
        state = capture_code_state(bare)
        assert state["commit_at_end"] is None
        assert state["dirty_at_end"] is None

    def test_captures_commit_on_clean_tree(
        self, tmp_path: Path,
    ) -> None:
        """A clean git repo yields ``commit_at_end`` and ``dirty_at_end=False``."""
        repo = tmp_path / "clean-repo"
        repo.mkdir()
        head = _init_git_repo(repo)
        state = capture_code_state(repo)
        assert state["commit_at_end"] == head
        assert state["dirty_at_end"] is False
        # commit_at_start is always None from this function — that
        # capture would need a SessionStart-hook side-channel.
        assert state["commit_at_start"] is None

    def test_detects_dirty_tree_via_untracked(
        self, tmp_path: Path,
    ) -> None:
        """An untracked file counts as dirty per ``git status --porcelain``."""
        repo = tmp_path / "dirty-repo"
        repo.mkdir()
        _init_git_repo(repo)
        (repo / "scratch.txt").write_text("untracked content\n")
        state = capture_code_state(repo)
        assert state["commit_at_end"] is not None
        assert state["dirty_at_end"] is True


# -------------------------------------------------------------------------
# Provenance audit Gaps 2 + 3 — session.meta.json provenance fields
# -------------------------------------------------------------------------


class TestSessionMetaProvenance:
    """Provenance fields landed on ``session.meta.json`` (Gaps 2 + 3)."""

    def test_code_state_present_with_default_capture(
        self, sample_session_jsonl: Path,
    ) -> None:
        """``code_state`` is present and has the three expected keys."""
        stats = extract_session_stats(sample_session_jsonl)
        meta = create_session_metadata(
            session_id="abc12345",
            session_path=sample_session_jsonl,
            stats=stats,
        )
        assert "code_state" in meta
        assert set(meta["code_state"].keys()) == {
            "commit_at_start", "commit_at_end", "dirty_at_end",
        }

    def test_code_state_populated_when_project_root_is_git_repo(
        self, sample_session_jsonl: Path, tmp_path: Path,
    ) -> None:
        """When *project_root* is a git repo, code_state captures HEAD."""
        repo = tmp_path / "session-repo"
        repo.mkdir()
        head = _init_git_repo(repo)
        stats = extract_session_stats(sample_session_jsonl)
        meta = create_session_metadata(
            session_id="abc12345",
            session_path=sample_session_jsonl,
            stats=stats,
            project_root=repo,
        )
        assert meta["code_state"]["commit_at_end"] == head
        assert meta["code_state"]["dirty_at_end"] is False

    def test_code_state_override(
        self, sample_session_jsonl: Path,
    ) -> None:
        """An explicit ``code_state`` dict overrides capture."""
        stats = extract_session_stats(sample_session_jsonl)
        explicit = {
            "commit_at_start": "deadbeef" * 5,
            "commit_at_end": "feedface" * 5,
            "dirty_at_end": True,
        }
        meta = create_session_metadata(
            session_id="abc12345",
            session_path=sample_session_jsonl,
            stats=stats,
            code_state=explicit,
        )
        assert meta["code_state"] == explicit

    def test_licence_defaults_to_none(
        self, sample_session_jsonl: Path,
    ) -> None:
        """RO-Crate ``licence`` defaults to ``None`` (user opts in later)."""
        stats = extract_session_stats(sample_session_jsonl)
        meta = create_session_metadata(
            session_id="abc12345",
            session_path=sample_session_jsonl,
            stats=stats,
        )
        assert meta["licence"] is DEFAULT_LICENCE
        assert meta["licence"] is None

    def test_licence_explicit_override(
        self, sample_session_jsonl: Path,
    ) -> None:
        """An explicit ``licence`` string lands on the record."""
        stats = extract_session_stats(sample_session_jsonl)
        meta = create_session_metadata(
            session_id="abc12345",
            session_path=sample_session_jsonl,
            stats=stats,
            licence="CC-BY-4.0",
        )
        assert meta["licence"] == "CC-BY-4.0"

    def test_extractor_model_id_defaults_to_config_constant(
        self, sample_session_jsonl: Path,
    ) -> None:
        """``extractor_model_id`` defaults to the toolkit-wide constant."""
        stats = extract_session_stats(sample_session_jsonl)
        meta = create_session_metadata(
            session_id="abc12345",
            session_path=sample_session_jsonl,
            stats=stats,
        )
        assert meta["extractor_model_id"] == EXTRACTOR_MODEL_ID

    def test_extractor_model_id_explicit_override(
        self, sample_session_jsonl: Path,
    ) -> None:
        """An explicit ``extractor_model_id`` overrides the default."""
        stats = extract_session_stats(sample_session_jsonl)
        meta = create_session_metadata(
            session_id="abc12345",
            session_path=sample_session_jsonl,
            stats=stats,
            extractor_model_id="claude-opus-4-7",
        )
        assert meta["extractor_model_id"] == "claude-opus-4-7"
