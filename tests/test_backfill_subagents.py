"""Tests for scripts/backfill-subagents.py (one-shot back-archival)."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType

import pytest

from cc_session_toolkit import archive as archive_mod


# -------------------------------------------------------------------------
# Dynamic import — the script filename contains a hyphen, so it cannot be
# imported via normal ``import`` syntax.  Loaded once per test session.
# -------------------------------------------------------------------------

_BACKFILL_PATH = (
    Path(__file__).parent.parent / "scripts" / "backfill-subagents.py"
)


def _load_backfill_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "backfill_subagents_under_test", _BACKFILL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def backfill() -> ModuleType:
    return _load_backfill_module()


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

def _write_parent_jsonl(path: Path, session_id: str) -> None:
    """Write a minimal parent JSONL with two Agent tool_use blocks."""
    t0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=timezone.utc)
    lines: list[str] = []
    for i, tool_use_id in enumerate(["toolu_AAA", "toolu_BBB"]):
        lines.append(json.dumps({
            "parentUuid": None,
            "isSidechain": False,
            "sessionId": session_id,
            "type": "user",
            "uuid": f"parent-user-{i}",
            "timestamp": (t0 + timedelta(seconds=i * 10)).isoformat(),
            "message": {"role": "user", "content": f"spawn {i}"},
        }))
        lines.append(json.dumps({
            "parentUuid": f"parent-user-{i}",
            "isSidechain": False,
            "sessionId": session_id,
            "type": "assistant",
            "uuid": f"parent-asst-{i}",
            "timestamp": (t0 + timedelta(seconds=i * 10 + 1)).isoformat(),
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-7",
                "content": [{
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": "Agent",
                    "input": {
                        "description": f"work {i}",
                        "subagent_type": "Explore",
                        "prompt": f"prompt-{i}",
                    },
                }],
                "usage": {
                    "input_tokens": 100, "output_tokens": 20,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
        }))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_subagent_jsonl(
    path: Path,
    agent_id: str,
    session_id: str,
    first_prompt: str,
) -> None:
    """Write a minimal sub-agent transcript."""
    t0 = datetime(2026, 4, 22, 12, 0, 5, tzinfo=timezone.utc)
    record = {
        "parentUuid": "parent-user-0",
        "isSidechain": True,
        "agentId": agent_id,
        "sessionId": session_id,
        "type": "user",
        "uuid": f"msg-{agent_id}-0",
        "timestamp": t0.isoformat(),
        "message": {"role": "user", "content": first_prompt},
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


@pytest.fixture()
def synthetic_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backfill: ModuleType,
) -> dict[str, Path]:
    """
    Stand up a synthetic ``~/.claude/projects/`` and ``~/cc-archives/``
    layout so backfill can run end-to-end without touching the real home.

    Patches ``CLAUDE_PROJECTS_DIR`` in both the archive module and the
    backfill module.  Both modules imported the constant via
    ``from ... import CLAUDE_PROJECTS_DIR`` at load time, so their local
    bindings must be overridden separately — patching the source module
    after the fact has no effect on already-imported names.
    """
    # Fake ~/.claude/projects/<proj>/ with a parent JSONL + subagents/
    fake_cc_projects = tmp_path / "claude-projects"
    project_dir_name = "-home-shawn-synthetic-project"
    proj_cc_dir = fake_cc_projects / project_dir_name
    proj_cc_dir.mkdir(parents=True)
    monkeypatch.setattr(
        archive_mod, "CLAUDE_PROJECTS_DIR", fake_cc_projects,
    )
    monkeypatch.setattr(
        backfill, "CLAUDE_PROJECTS_DIR", fake_cc_projects,
    )

    session_id = "syn12345-1234-5678-9abc-def012345678"
    parent = proj_cc_dir / f"{session_id}.jsonl"
    _write_parent_jsonl(parent, session_id)

    sub_dir = proj_cc_dir / session_id / "subagents"
    sub_dir.mkdir(parents=True)
    _write_subagent_jsonl(
        sub_dir / "agent-aaaaaaaa.jsonl",
        "aaaaaaaa", session_id, "prompt-0",
    )
    _write_subagent_jsonl(
        sub_dir / "agent-bbbbbbbb.jsonl",
        "bbbbbbbb", session_id, "prompt-1",
    )

    # Fake ~/cc-archives/<proj>/<slug>/session.meta.json
    archive_root = tmp_path / "cc-archives"
    archive_dir = archive_root / "synthetic-project" / "2026-04-22T12-00_syn"
    archive_dir.mkdir(parents=True)
    meta = {
        "schema_version": "1.1",
        "session": {"id": session_id, "started_at": "2026-04-22T12:00:00Z"},
        "project": {
            "name": "synthetic-project",
            "directory": "/home/shawn/synthetic-project",
        },
        "statistics": {"turns": 2, "tool_calls": {"total": 2}},
        "archive": {"jsonl_path": "session.jsonl.gz"},
    }
    (archive_dir / "session.meta.json").write_text(
        json.dumps(meta, indent=2),
    )

    return {
        "archive_root": archive_root,
        "archive_dir": archive_dir,
        "parent_jsonl": parent,
        "project_cc_slug": project_dir_name,
        "session_id": session_id,
    }


# -------------------------------------------------------------------------
# _resolve_parent_jsonl
# -------------------------------------------------------------------------

class TestResolveParentJsonl:
    """Tests for :func:`_resolve_parent_jsonl`."""

    def test_resolves_when_file_exists(
        self,
        backfill: ModuleType,
        synthetic_archive: dict[str, Path],
    ) -> None:
        meta = json.loads(
            (synthetic_archive["archive_dir"] / "session.meta.json")
            .read_text()
        )
        resolved = backfill._resolve_parent_jsonl(meta)
        assert resolved == synthetic_archive["parent_jsonl"]

    def test_returns_none_when_session_id_missing(
        self, backfill: ModuleType,
    ) -> None:
        meta = {"project": {"directory": "/tmp/x"}}
        assert backfill._resolve_parent_jsonl(meta) is None

    def test_returns_none_when_project_directory_missing(
        self, backfill: ModuleType,
    ) -> None:
        meta = {"session": {"id": "some-id"}}
        assert backfill._resolve_parent_jsonl(meta) is None

    def test_returns_none_when_file_missing(
        self, backfill: ModuleType, tmp_path: Path,
    ) -> None:
        meta = {
            "session": {"id": "does-not-exist"},
            "project": {"directory": str(tmp_path / "nowhere")},
        }
        assert backfill._resolve_parent_jsonl(meta) is None


# -------------------------------------------------------------------------
# backfill_archive
# -------------------------------------------------------------------------

class TestBackfillArchive:
    """Tests for :func:`backfill_archive`."""

    def test_ok_status_when_subagents_archived(
        self,
        backfill: ModuleType,
        synthetic_archive: dict[str, Path],
    ) -> None:
        report = backfill.backfill_archive(
            synthetic_archive["archive_dir"],
            dry_run=False,
            use_gzip=True,
        )
        assert report["status"] == "ok"
        assert report["count"] == 2

    def test_patches_meta_to_v1_2(
        self,
        backfill: ModuleType,
        synthetic_archive: dict[str, Path],
    ) -> None:
        backfill.backfill_archive(
            synthetic_archive["archive_dir"],
            dry_run=False, use_gzip=True,
        )
        meta = json.loads(
            (synthetic_archive["archive_dir"] / "session.meta.json")
            .read_text()
        )
        assert meta["schema_version"] == "1.2"
        assert len(meta["subagents"]) == 2
        assert meta["statistics"]["subagents_summary"]["count"] == 2

    def test_writes_subagent_files_to_archive_dir(
        self,
        backfill: ModuleType,
        synthetic_archive: dict[str, Path],
    ) -> None:
        backfill.backfill_archive(
            synthetic_archive["archive_dir"],
            dry_run=False, use_gzip=True,
        )
        sub_dir = synthetic_archive["archive_dir"] / "subagents"
        assert sub_dir.is_dir()
        names = {p.name for p in sub_dir.glob("*.jsonl.gz")}
        assert "agent-aaaaaaaa.jsonl.gz" in names
        assert "agent-bbbbbbbb.jsonl.gz" in names

    def test_dry_run_does_not_write_or_patch(
        self,
        backfill: ModuleType,
        synthetic_archive: dict[str, Path],
    ) -> None:
        meta_path = synthetic_archive["archive_dir"] / "session.meta.json"
        meta_before = meta_path.read_text()
        report = backfill.backfill_archive(
            synthetic_archive["archive_dir"],
            dry_run=True, use_gzip=True,
        )
        assert report["status"] == "dry-run"
        assert report["count"] == 2
        assert meta_path.read_text() == meta_before
        assert not (synthetic_archive["archive_dir"] / "subagents").exists()

    def test_skipped_when_meta_missing(
        self,
        backfill: ModuleType,
        tmp_path: Path,
    ) -> None:
        empty_archive = tmp_path / "empty"
        empty_archive.mkdir()
        report = backfill.backfill_archive(empty_archive)
        assert report["status"] == "skipped"
        assert "no session.meta.json" in report["reason"]

    def test_skipped_when_parent_jsonl_missing(
        self,
        backfill: ModuleType,
        tmp_path: Path,
    ) -> None:
        archive_dir = tmp_path / "orphan"
        archive_dir.mkdir()
        meta = {
            "schema_version": "1.1",
            "session": {"id": "orphan-id"},
            "project": {
                "name": "orphan-project",
                "directory": "/nonexistent/project",
            },
        }
        (archive_dir / "session.meta.json").write_text(json.dumps(meta))
        report = backfill.backfill_archive(archive_dir)
        assert report["status"] == "skipped"
        assert "parent JSONL not found" in report["reason"]

    def test_error_when_meta_is_invalid_json(
        self,
        backfill: ModuleType,
        tmp_path: Path,
    ) -> None:
        archive_dir = tmp_path / "corrupt"
        archive_dir.mkdir()
        (archive_dir / "session.meta.json").write_text(
            "{not valid json", encoding="utf-8",
        )
        report = backfill.backfill_archive(archive_dir)
        assert report["status"] == "error"
        assert "bad json" in report["reason"]

    def test_idempotent_on_rerun(
        self,
        backfill: ModuleType,
        synthetic_archive: dict[str, Path],
    ) -> None:
        backfill.backfill_archive(
            synthetic_archive["archive_dir"],
            dry_run=False, use_gzip=True,
        )
        # Re-run — should succeed with the same count, and each sub-agent
        # should carry the idempotent capture note.
        report = backfill.backfill_archive(
            synthetic_archive["archive_dir"],
            dry_run=False, use_gzip=True,
        )
        assert report["status"] == "ok"
        assert report["count"] == 2
        meta = json.loads(
            (synthetic_archive["archive_dir"] / "session.meta.json")
            .read_text()
        )
        for sub in meta["subagents"]:
            assert any(
                "idempotent" in note for note in sub["capture_notes"]
            )
