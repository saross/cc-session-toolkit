"""
Tests for writing the session record before subagent summaries (2026-10-08).

On 2026-08-21 a zbook hook was killed after the first of 18 subagent
summaries, about two minutes in, leaving the transcript archived with no
``session.meta.json``: the session counted as unarchived until repaired by
hand. The record is now written as soon as the parent metadata exists, the
caller is told (the hook catalogues it), and the summaries are added after.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cc_session_toolkit import archive as archive_mod
from cc_session_toolkit.archive import archive_session
from tests.test_subagent_archive import current_layout_session  # noqa: F401 — fixture

PARENT = {
    "title": "Audit and plan", "purpose": "p", "tags": [],
    "three_ps": {"prompt_summary": "a", "process_summary": "b", "provenance_summary": "c"},
    "phases": [], "decisions": [], "key_exchanges": [],
}


@pytest.fixture()
def parent_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """The parent model call succeeds instantly."""
    monkeypatch.setattr(archive_mod, "generate_auto_metadata", lambda *_a, **_k: dict(PARENT))


def _archive(session: dict[str, Path], root: Path, **kwargs: Any) -> dict | None:
    return archive_session(
        session["parent"], None, stats_only=True, archive_root=root,
        project_name_override="proj", auto_metadata=True,
        capture_type="session_end", session_id_override=session["session_id"], **kwargs,
    )


def test_killed_during_summaries_leaves_a_record(
    current_layout_session: dict[str, Path],  # noqa: F811
    parent_metadata: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    def _killed(**_kwargs: Any) -> list:
        raise KeyboardInterrupt  # what a hook timeout's SIGTERM amounts to here

    monkeypatch.setattr(archive_mod, "generate_subagent_summaries", _killed)
    seen: list[dict] = []
    root = tmp_path / "cc-archives"
    with pytest.raises(KeyboardInterrupt):
        _archive(current_layout_session, root, on_record_written=seen.append)

    assert len(seen) == 1
    record_dir = Path(seen[0]["_archive_directory"])
    meta = json.loads((record_dir / "session.meta.json").read_text())
    assert meta["auto_generated"]["title"] == PARENT["title"]
    assert meta["subagent_summaries"] == []
    assert len(meta["subagents"]) == 2


def test_completed_run_adds_the_summaries(
    current_layout_session: dict[str, Path],  # noqa: F811
    parent_metadata: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(archive_mod, "generate_subagent_summaries", lambda **_k: [
        {"agent_id": "a1a1a1a1a1", "narrative": "n1"},
        {"agent_id": "b2b2b2b2b2", "narrative": "n2"},
    ])
    seen: list[dict] = []
    result = _archive(current_layout_session, tmp_path / "cc-archives",
                      on_record_written=seen.append)

    assert len(seen) == 1 and seen[0]["subagent_summaries"] == []
    meta = json.loads(
        (Path(result["_archive_directory"]) / "session.meta.json").read_text()
    )
    assert [s["agent_id"] for s in meta["subagent_summaries"]] == ["a1a1a1a1a1", "b2b2b2b2b2"]


def test_no_subagents_to_summarise_writes_once(
    parent_metadata: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    sid = "dddddddd-0000-0000-0000-000000000004"
    live = tmp_path / f"{sid}.jsonl"
    lines = []
    for i in range(6):
        lines.append(json.dumps({"timestamp": f"2026-03-15T10:{i:02d}:00+00:00",
                                 "sessionId": sid, "message": {"role": "user",
                                 "content": "x" * 200}}))
        lines.append(json.dumps({"timestamp": f"2026-03-15T10:{i:02d}:30+00:00",
                                 "sessionId": sid, "message": {"role": "assistant",
                                 "content": [{"type": "text", "text": "y" * 200}]}}))
    live.write_text("\n".join(lines) + "\n", encoding="utf-8")
    seen: list[dict] = []
    result = archive_session(live, None, stats_only=True, archive_root=tmp_path / "a",
                             project_name_override="proj", auto_metadata=True,
                             capture_type="session_end", session_id_override=sid,
                             on_record_written=seen.append)
    assert result is not None and seen == []


def test_hook_catalogues_before_the_summaries(
    current_layout_session: dict[str, Path],  # noqa: F811
    parent_metadata: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The CLI's callback puts the session in the catalogue before summaries."""
    from cc_session_toolkit.catalogue import update_catalogue

    root = tmp_path / "cc-archives"
    catalogue = root / "CATALOG.json"

    def _killed(**_kwargs: Any) -> list:
        assert catalogue.is_file(), "catalogued before the summaries start"
        raise KeyboardInterrupt

    monkeypatch.setattr(archive_mod, "generate_subagent_summaries", _killed)
    with pytest.raises(KeyboardInterrupt):
        _archive(current_layout_session, root, on_record_written=lambda meta: update_catalogue(
            [meta], catalogue, root, "proj"))
    ids = {s["id"] for s in json.loads(catalogue.read_text())["sessions"]}
    assert current_layout_session["session_id"] in ids
