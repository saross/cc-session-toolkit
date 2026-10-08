"""
Tests for superseding an existing archive when its live transcript grows.

Background (diagnosed 2026-10-08): the hook dedup in ``cli.py`` let the
first archive of a session win. A session that compacted was archived at
its first ``PreCompact`` and every later capture, including
``SessionEnd``, was skipped — so the archived transcript and its
metadata stopped at the first compaction. A resumed session stopped at
its first end in the same way. Measured on amd-tower: 10 of 10
``pre_compact`` archives shorter than their live transcripts (27% of
lines missing), and 43 of 371 ``session_end`` archives.

These tests pin the replacement policy:

* re-archive only when the live transcript has grown, or the archived
  metadata is a placeholder;
* reuse the existing archive directory (its name came from an earlier
  title and project, and must not fork);
* regenerate model metadata at ``SessionEnd`` when growth is material,
  or whenever the metadata is a placeholder; otherwise carry it forward,
  so repeated compactions do not each pay for a Gemini call;
* never downgrade good metadata to a placeholder when regeneration
  fails — keep the prior block, marked stale by ``extractor_source_bytes``;
* record what was replaced, so replication can propagate it.
"""

from __future__ import annotations

import gzip
import io
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from cc_session_toolkit import archive as archive_mod
from cc_session_toolkit.archive import (
    archive_session,
    archived_transcript_bytes,
    is_placeholder_metadata,
    plan_supersede,
)
from cc_session_toolkit.catalogue import update_catalogue

SID = "cccccccc-1111-2222-3333-444444444444"
PROJECT = "proj"

FAKE_AUTO: dict[str, Any] = {
    "title": "Fake model title",
    "purpose": "A purpose written by the fake model",
    "tags": ["alpha", "beta"],
    "three_ps": {
        "prompt_summary": "p",
        "process_summary": "q",
        "provenance_summary": "r",
    },
    "phases": [],
    "decisions": [],
    "key_exchanges": [],
}

START = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)


def _turns(first: int, count: int) -> list[str]:
    """Return JSONL lines for ``count`` user/assistant turn pairs."""
    lines: list[str] = []
    for i in range(first, first + count):
        ts = (START + timedelta(minutes=i * 2)).isoformat()
        lines.append(json.dumps({
            "timestamp": ts,
            "sessionId": SID,
            "message": {"role": "user", "content": f"Message {i} " + "x" * 200},
        }))
        lines.append(json.dumps({
            "timestamp": ts,
            "sessionId": SID,
            "message": {
                "role": "assistant",
                "model": "claude-opus-5-5",
                "content": [{"type": "text", "text": f"Reply {i} " + "y" * 200}],
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
        }))
    return lines


def _write(path: Path, first: int, count: int, *, append: bool = False) -> None:
    """Write (or append) ``count`` turn pairs to a live transcript."""
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as fh:
        fh.write("\n".join(_turns(first, count)) + "\n")


@pytest.fixture()
def fake_generator(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the Gemini call with a counter that returns ``FAKE_AUTO``.

    Tests flip ``state["result"]`` to ``None`` to simulate a Flex 503
    give-up. Keeps every test free of network access and spend.
    """
    state: dict[str, Any] = {"calls": 0, "result": dict(FAKE_AUTO)}

    def _fake(session_path: Path, stats: dict[str, Any]) -> dict | None:
        state["calls"] += 1
        res = state["result"]
        return json.loads(json.dumps(res)) if res is not None else None

    monkeypatch.setattr(archive_mod, "generate_auto_metadata", _fake)
    return state


def _initial_archive(
    tmp_path: Path, live: Path, *, capture_type: str = "pre_compact",
) -> tuple[Path, Path, Path]:
    """Archive ``live`` once, register it, and return (root, catalogue, dir)."""
    archive_root = tmp_path / "cc-archives"
    archive_root.mkdir(exist_ok=True)
    catalogue = archive_root / "CATALOG.json"
    result = archive_session(
        live,
        None,
        stats_only=True,
        archive_root=archive_root,
        project_name_override=PROJECT,
        auto_metadata=True,
        capture_type=capture_type,
        session_id_override=SID,
    )
    assert result is not None
    dest = Path(result["_archive_directory"])
    update_catalogue([result], catalogue, archive_root, PROJECT)
    return archive_root, catalogue, dest


def _meta(dest: Path) -> dict[str, Any]:
    return json.loads((dest / "session.meta.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_archived_transcript_bytes_counts_decompressed_length(
        self, tmp_path: Path, fake_generator: dict[str, Any],
    ) -> None:
        live = tmp_path / f"{SID}.jsonl"
        _write(live, 0, 10)
        _, _, dest = _initial_archive(tmp_path, live)
        assert archived_transcript_bytes(dest) == live.stat().st_size

    def test_recorded_uncompressed_bytes_match_archive(
        self, tmp_path: Path, fake_generator: dict[str, Any],
    ) -> None:
        """The meta must describe the bytes actually archived."""
        live = tmp_path / f"{SID}.jsonl"
        _write(live, 0, 10)
        _, _, dest = _initial_archive(tmp_path, live)
        meta = _meta(dest)
        assert meta["archive"]["jsonl_bytes_uncompressed"] == (
            archived_transcript_bytes(dest)
        )
        raw = gzip.decompress((dest / "session.jsonl.gz").read_bytes())
        import hashlib
        assert meta["archive"]["jsonl_sha256_uncompressed"] == (
            hashlib.sha256(raw).hexdigest()
        )

    def test_placeholder_detection(self) -> None:
        assert is_placeholder_metadata(
            {"auto_generated": {"purpose": "Auto-metadata unavailable"}}
        )
        assert not is_placeholder_metadata(
            {"auto_generated": {"purpose": "Real purpose"}}
        )

    def test_fresh_metadata_records_source_bytes(
        self, tmp_path: Path, fake_generator: dict[str, Any],
    ) -> None:
        live = tmp_path / f"{SID}.jsonl"
        _write(live, 0, 10)
        _, _, dest = _initial_archive(tmp_path, live)
        assert _meta(dest)["extractor_source_bytes"] == live.stat().st_size
        from cc_session_toolkit.config import AUTO_METADATA_THINKING_LEVEL
        assert _meta(dest)["extractor_thinking_level"] == AUTO_METADATA_THINKING_LEVEL

    def test_placeholder_records_no_source_bytes(
        self, tmp_path: Path, fake_generator: dict[str, Any],
    ) -> None:
        fake_generator["result"] = None
        live = tmp_path / f"{SID}.jsonl"
        _write(live, 0, 10)
        _, _, dest = _initial_archive(tmp_path, live)
        meta = _meta(dest)
        assert is_placeholder_metadata(meta)
        assert meta["extractor_source_bytes"] is None

    def test_placeholder_names_no_extractor_model(
        self, tmp_path: Path, fake_generator: dict[str, Any],
    ) -> None:
        """No model wrote a placeholder, so none is named.

        The metadata writer used to fall back to ``EXTRACTOR_MODEL_ID``,
        so a Flex give-up was attributed to the model that failed, and a
        model-off repair run on 2026-10-08 labelled 24 placeholders
        ``gemini-3.8-flash``.
        """
        fake_generator["result"] = None
        live = tmp_path / f"{SID}.jsonl"
        _write(live, 0, 10)
        _, _, dest = _initial_archive(tmp_path, live)
        assert _meta(dest)["extractor_model_id"] is None


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


class TestPlanSupersede:
    def test_skips_when_unchanged(
        self, tmp_path: Path, fake_generator: dict[str, Any],
    ) -> None:
        live = tmp_path / f"{SID}.jsonl"
        _write(live, 0, 10)
        root, cat, _ = _initial_archive(tmp_path, live)
        assert plan_supersede(
            SID, live, cat, root, regenerate_on_growth=True
        ) is None

    def test_pre_compact_growth_refreshes_without_regenerating(
        self, tmp_path: Path, fake_generator: dict[str, Any],
    ) -> None:
        live = tmp_path / f"{SID}.jsonl"
        _write(live, 0, 10)
        root, cat, dest = _initial_archive(tmp_path, live)
        _write(live, 10, 10, append=True)
        plan = plan_supersede(SID, live, cat, root, regenerate_on_growth=False)
        assert plan is not None
        assert plan.dest_dir == dest
        assert plan.regenerate is False

    def test_session_end_material_growth_regenerates(
        self, tmp_path: Path, fake_generator: dict[str, Any],
    ) -> None:
        live = tmp_path / f"{SID}.jsonl"
        _write(live, 0, 10)
        root, cat, _ = _initial_archive(tmp_path, live)
        _write(live, 10, 10, append=True)  # ~100% growth
        plan = plan_supersede(SID, live, cat, root, regenerate_on_growth=True)
        assert plan is not None and plan.regenerate is True

    def test_session_end_small_growth_carries_forward(
        self, tmp_path: Path, fake_generator: dict[str, Any],
    ) -> None:
        live = tmp_path / f"{SID}.jsonl"
        _write(live, 0, 50)
        root, cat, _ = _initial_archive(tmp_path, live)
        _write(live, 50, 1, append=True)  # ~2% growth
        plan = plan_supersede(SID, live, cat, root, regenerate_on_growth=True)
        assert plan is not None and plan.regenerate is False

    def test_placeholder_retries_without_growth(
        self, tmp_path: Path, fake_generator: dict[str, Any],
    ) -> None:
        fake_generator["result"] = None
        live = tmp_path / f"{SID}.jsonl"
        _write(live, 0, 10)
        root, cat, _ = _initial_archive(tmp_path, live)
        plan = plan_supersede(SID, live, cat, root, regenerate_on_growth=False)
        assert plan is not None and plan.regenerate is True

    def test_none_when_catalogue_has_no_directory(self, tmp_path: Path) -> None:
        live = tmp_path / f"{SID}.jsonl"
        _write(live, 0, 10)
        root = tmp_path / "cc-archives"
        root.mkdir()
        cat = root / "CATALOG.json"
        cat.write_text(json.dumps({"sessions": [{"id": SID}]}))
        assert plan_supersede(SID, live, cat, root, regenerate_on_growth=True) is None

    def test_never_shrinks_an_archive(
        self, tmp_path: Path, fake_generator: dict[str, Any],
    ) -> None:
        """A live transcript SHORTER than the archive is never written back."""
        live = tmp_path / f"{SID}.jsonl"
        _write(live, 0, 20)
        root, cat, _ = _initial_archive(tmp_path, live)
        _write(live, 0, 5)  # truncated live copy
        assert plan_supersede(SID, live, cat, root, regenerate_on_growth=True) is None


# ---------------------------------------------------------------------------
# Superseding archive_session
# ---------------------------------------------------------------------------


class TestArchiveSessionSupersede:
    def _supersede(self, live: Path, cat: Path, root: Path, *, regen_on_growth: bool):
        plan = plan_supersede(SID, live, cat, root, regenerate_on_growth=regen_on_growth)
        assert plan is not None
        return plan, archive_session(
            live,
            None,
            stats_only=True,
            archive_root=root,
            project_name_override=PROJECT,
            auto_metadata=True,
            capture_type="pre_compact" if not regen_on_growth else "session_end",
            session_id_override=SID,
            existing_dest_dir=plan.dest_dir,
            prior_metadata=plan.prior_metadata,
            regenerate_metadata=plan.regenerate,
        )

    def test_reuses_directory_and_carries_metadata_forward(
        self, tmp_path: Path, fake_generator: dict[str, Any],
    ) -> None:
        live = tmp_path / f"{SID}.jsonl"
        _write(live, 0, 10)
        root, cat, dest = _initial_archive(tmp_path, live)
        before = _meta(dest)
        calls_before = fake_generator["calls"]
        _write(live, 10, 10, append=True)

        _, result = self._supersede(live, cat, root, regen_on_growth=False)

        assert result is not None
        assert Path(result["_archive_directory"]) == dest
        assert [p.name for p in (root / PROJECT).iterdir()] == [dest.name]
        assert fake_generator["calls"] == calls_before  # no model call
        after = _meta(dest)
        assert after["auto_generated"] == before["auto_generated"]
        assert after["extractor_source_bytes"] == before["extractor_source_bytes"]
        assert archived_transcript_bytes(dest) == live.stat().st_size

    def test_records_what_was_superseded(
        self, tmp_path: Path, fake_generator: dict[str, Any],
    ) -> None:
        live = tmp_path / f"{SID}.jsonl"
        _write(live, 0, 10)
        root, cat, dest = _initial_archive(tmp_path, live)
        old = _meta(dest)["archive"]
        _write(live, 10, 10, append=True)

        self._supersede(live, cat, root, regen_on_growth=False)

        sup = _meta(dest)["archive"]["supersedes"]
        assert sup["previous_bytes_uncompressed"] == old["jsonl_bytes_uncompressed"]
        assert sup["previous_sha256"] == old["jsonl_sha256"]
        assert sup["previous_capture_type"] == "pre_compact"
        assert "superseded_at" in sup

    def test_regenerates_when_planned(
        self, tmp_path: Path, fake_generator: dict[str, Any],
    ) -> None:
        live = tmp_path / f"{SID}.jsonl"
        _write(live, 0, 10)
        root, cat, dest = _initial_archive(tmp_path, live)
        calls_before = fake_generator["calls"]
        fake_generator["result"] = dict(FAKE_AUTO, title="Second title")
        _write(live, 10, 10, append=True)

        self._supersede(live, cat, root, regen_on_growth=True)

        after = _meta(dest)
        assert fake_generator["calls"] == calls_before + 1
        assert after["auto_generated"]["title"] == "Second title"
        assert after["extractor_source_bytes"] == live.stat().st_size

    def test_failed_regeneration_keeps_prior_metadata(
        self, tmp_path: Path, fake_generator: dict[str, Any],
    ) -> None:
        live = tmp_path / f"{SID}.jsonl"
        _write(live, 0, 10)
        root, cat, dest = _initial_archive(tmp_path, live)
        before = _meta(dest)
        fake_generator["result"] = None  # simulate Flex 503 give-up
        _write(live, 10, 10, append=True)

        self._supersede(live, cat, root, regen_on_growth=True)

        after = _meta(dest)
        assert not is_placeholder_metadata(after)
        assert after["auto_generated"] == before["auto_generated"]
        # Stale marker: metadata covers fewer bytes than the archive holds.
        assert after["extractor_source_bytes"] < archived_transcript_bytes(dest)

    def test_model_off_never_downgrades_real_metadata(
        self, tmp_path: Path, fake_generator: dict[str, Any],
    ) -> None:
        """A repair run with auto-metadata off keeps real metadata.

        Material growth plans a regeneration, but with no model to call
        the prior block must be carried forward (marked stale by
        ``extractor_source_bytes``), never replaced by a placeholder.
        """
        live = tmp_path / f"{SID}.jsonl"
        _write(live, 0, 10)
        root, cat, dest = _initial_archive(tmp_path, live)
        before = _meta(dest)
        calls_before = fake_generator["calls"]
        _write(live, 10, 10, append=True)
        plan = plan_supersede(SID, live, cat, root, regenerate_on_growth=True)
        assert plan is not None and plan.regenerate is True

        archive_session(
            live, None, stats_only=True, archive_root=root,
            project_name_override=PROJECT, auto_metadata=False,
            capture_type=None, session_id_override=SID,
            existing_dest_dir=plan.dest_dir, prior_metadata=plan.prior_metadata,
            regenerate_metadata=plan.regenerate,
        )

        after = _meta(dest)
        assert fake_generator["calls"] == calls_before
        assert not is_placeholder_metadata(after)
        assert after["auto_generated"] == before["auto_generated"]
        assert after["extractor_model_id"] == before["extractor_model_id"]
        assert archived_transcript_bytes(dest) == live.stat().st_size
        assert after["extractor_source_bytes"] < archived_transcript_bytes(dest)

    def test_carry_forward_keeps_an_absent_label_absent(
        self, tmp_path: Path, fake_generator: dict[str, Any],
    ) -> None:
        """Carried-forward metadata keeps its own label, even none.

        Records written before labels existed carry no extractor_model_id.
        Carrying such a block forward must not attribute it to today's
        model: on 2026-10-08 a model-off refresh did exactly that.
        """
        live = tmp_path / f"{SID}.jsonl"
        _write(live, 0, 10)
        root, cat, dest = _initial_archive(tmp_path, live)
        meta = _meta(dest)
        meta.pop("extractor_model_id")
        (dest / "session.meta.json").write_text(json.dumps(meta), encoding="utf-8")
        _write(live, 10, 10, append=True)

        self._supersede(live, cat, root, regen_on_growth=False)

        assert _meta(dest)["extractor_model_id"] is None

    def test_stale_metadata_is_regenerated_later(
        self, tmp_path: Path, fake_generator: dict[str, Any],
    ) -> None:
        """Metadata carried forward over a refreshed transcript is not final.

        Once the transcript is complete the session no longer looks grown,
        so without a staleness rule metadata that describes only the first
        part would never be regenerated: after a repair run with the model
        off, or a ``PreCompact`` refresh followed by a ``SessionEnd`` with
        nothing new.
        """
        live = tmp_path / f"{SID}.jsonl"
        _write(live, 0, 10)
        root, cat, dest = _initial_archive(tmp_path, live)
        _write(live, 10, 10, append=True)
        self._supersede(live, cat, root, regen_on_growth=False)  # carry forward
        assert archived_transcript_bytes(dest) == live.stat().st_size

        # PreCompact still never pays for a call...
        assert plan_supersede(
            SID, live, cat, root, regenerate_on_growth=False
        ) is None
        # ...but SessionEnd or a repair run regenerates the stale block.
        plan = plan_supersede(SID, live, cat, root, regenerate_on_growth=True)
        assert plan is not None and plan.regenerate is True
        assert "stale" in plan.reason

    def test_legacy_record_without_source_bytes_is_not_stale(
        self, tmp_path: Path, fake_generator: dict[str, Any],
    ) -> None:
        """Records written before 2026-10-08 lack the field: not stale."""
        live = tmp_path / f"{SID}.jsonl"
        _write(live, 0, 10)
        root, cat, dest = _initial_archive(tmp_path, live)
        meta = _meta(dest)
        meta.pop("extractor_source_bytes", None)
        (dest / "session.meta.json").write_text(json.dumps(meta), encoding="utf-8")
        assert plan_supersede(
            SID, live, cat, root, regenerate_on_growth=True
        ) is None

    def test_failed_write_leaves_previous_archive_intact(
        self,
        tmp_path: Path,
        fake_generator: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A corrupt round-trip must not destroy the archive it replaces."""
        live = tmp_path / f"{SID}.jsonl"
        _write(live, 0, 10)
        root, cat, dest = _initial_archive(tmp_path, live)
        old_gz = (dest / "session.jsonl.gz").read_bytes()
        _write(live, 10, 10, append=True)

        real_open = gzip.open

        def _corrupting_open(path, mode="rb", *a, **k):
            fh = real_open(path, mode, *a, **k)
            if "r" in mode:
                data = fh.read()
                fh.close()
                return io.BytesIO(data[:-10])  # round-trip mismatch
            return fh

        monkeypatch.setattr(archive_mod.gzip, "open", _corrupting_open)
        with pytest.raises(RuntimeError, match="round-trip"):
            self._supersede(live, cat, root, regen_on_growth=False)
        monkeypatch.undo()
        assert (dest / "session.jsonl.gz").read_bytes() == old_gz


# ---------------------------------------------------------------------------
# Hook integration
# ---------------------------------------------------------------------------


class TestHookSupersede:
    def _run_hook(
        self,
        monkeypatch: pytest.MonkeyPatch,
        root: Path,
        live: Path,
        cwd: Path,
        *,
        pre_compact: bool,
    ) -> None:
        from cc_session_toolkit.cli import main

        args = ["cc-session", "archive", "--from-hook", "--gzip",
                "--auto-metadata", "--archive-root", str(root)]
        if pre_compact:
            args.append("--pre-compact")
        monkeypatch.setattr("sys.argv", args)
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({
            "session_id": SID,
            "transcript_path": str(live),
            "cwd": str(cwd),
        })))
        try:
            main()
        except SystemExit as exc:  # pragma: no cover - surfaced by assert
            assert not exc.code

    def test_session_end_after_pre_compact_supersedes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
        fake_generator: dict[str, Any],
    ) -> None:
        root = tmp_path / "cc-archives"
        root.mkdir()
        cwd = tmp_path / PROJECT
        cwd.mkdir()
        live = tmp_path / f"{SID}.jsonl"
        _write(live, 0, 10)

        self._run_hook(monkeypatch, root, live, cwd, pre_compact=True)
        _write(live, 10, 10, append=True)
        self._run_hook(monkeypatch, root, live, cwd, pre_compact=False)

        out = capsys.readouterr().out
        assert "Superseding archived session" in out
        catalogue = json.loads((root / "CATALOG.json").read_text())
        assert [s["id"] for s in catalogue["sessions"]].count(SID) == 1
        dirs = [p for p in (root / PROJECT).iterdir() if p.is_dir()]
        assert len(dirs) == 1
        assert archived_transcript_bytes(dirs[0]) == live.stat().st_size
        assert fake_generator["calls"] == 2  # PreCompact + SessionEnd regen

    def test_unchanged_session_is_still_skipped(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
        fake_generator: dict[str, Any],
    ) -> None:
        root = tmp_path / "cc-archives"
        root.mkdir()
        cwd = tmp_path / PROJECT
        cwd.mkdir()
        live = tmp_path / f"{SID}.jsonl"
        _write(live, 0, 10)

        self._run_hook(monkeypatch, root, live, cwd, pre_compact=False)
        self._run_hook(monkeypatch, root, live, cwd, pre_compact=False)

        assert "Skipping already-archived session" in capsys.readouterr().out
        assert fake_generator["calls"] == 1


# ---------------------------------------------------------------------------
# Repair sweep: archive --archive-root ROOT --refresh-grown
# ---------------------------------------------------------------------------


class TestRefreshGrownSweep:
    def _run(self, monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
        from cc_session_toolkit.cli import main

        monkeypatch.setattr("sys.argv", ["cc-session"] + argv)
        try:
            main()
        except SystemExit as exc:  # pragma: no cover - surfaced by assert
            assert not exc.code

    def _setup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[Path, Path, Path]:
        monkeypatch.setenv("HOME", str(tmp_path))
        live_dir = tmp_path / ".claude" / "projects" / "-home-user-proj"
        live_dir.mkdir(parents=True)
        live = live_dir / f"{SID}.jsonl"
        _write(live, 0, 10)
        root, _, dest = _initial_archive(tmp_path, live)
        _write(live, 10, 10, append=True)
        old = live.stat().st_mtime - 48 * 3600
        os.utime(live, (old, old))
        return root, live, dest

    def test_dry_run_writes_nothing_and_calls_no_model(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
        fake_generator: dict[str, Any],
    ) -> None:
        root, _live, dest = self._setup(tmp_path, monkeypatch)
        before_gz = (dest / "session.jsonl.gz").read_bytes()
        calls = fake_generator["calls"]

        self._run(monkeypatch, ["archive", "--archive-root", str(root),
                                "--refresh-grown", "--auto-metadata",
                                "--dry-run"])

        out = capsys.readouterr().out
        assert "Refresh plan: 1 archive(s) to update, 1 with a model call" in out
        assert (dest / "session.jsonl.gz").read_bytes() == before_gz
        assert fake_generator["calls"] == calls

    def test_sweep_updates_in_place(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fake_generator: dict[str, Any],
    ) -> None:
        root, live, dest = self._setup(tmp_path, monkeypatch)

        self._run(monkeypatch, ["archive", "--archive-root", str(root),
                                "--refresh-grown", "--auto-metadata"])

        assert archived_transcript_bytes(dest) == live.stat().st_size
        assert _meta(dest)["archive"]["supersedes"]["supersede_count"] == 1
        assert [p.name for p in (root / PROJECT).iterdir()] == [dest.name]


    def test_active_sessions_are_left_to_their_hooks(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
        fake_generator: dict[str, Any],
    ) -> None:
        root, live, _ = self._setup(tmp_path, monkeypatch)
        os.utime(live, None)  # touched now: still active

        self._run(monkeypatch, ["archive", "--archive-root", str(root),
                                "--refresh-grown", "--dry-run"])

        out = capsys.readouterr().out
        assert "Refresh plan: 0 archive(s)" in out
        assert "1 active within 24h" in out

class TestBookkeepingTail:
    """Claude Code appends records such as ``bridge-session`` after
    SessionEnd fires. Growth made only of such records is not content and
    must not trigger a re-archive (found 2026-10-08: many archives were
    exactly 146 bytes short for this reason)."""

    def test_bookkeeping_only_growth_is_skipped(
        self, tmp_path: Path, fake_generator: dict[str, Any],
    ) -> None:
        live = tmp_path / f"{SID}.jsonl"
        _write(live, 0, 10)
        root, cat, _ = _initial_archive(tmp_path, live)
        with open(live, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "type": "bridge-session", "sessionId": SID,
                "bridgeSessionId": "cse_x", "lastSequenceNum": 0,
            }) + "\n")
        assert plan_supersede(SID, live, cat, root, regenerate_on_growth=True) is None

    def test_content_growth_after_bookkeeping_is_planned(
        self, tmp_path: Path, fake_generator: dict[str, Any],
    ) -> None:
        live = tmp_path / f"{SID}.jsonl"
        _write(live, 0, 10)
        root, cat, _ = _initial_archive(tmp_path, live)
        with open(live, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"type": "bridge-session"}) + "\n")
            fh.write(json.dumps({
                "type": "user", "timestamp": START.isoformat(),
                "message": {"role": "user", "content": "resumed"},
            }) + "\n")
        assert plan_supersede(SID, live, cat, root, regenerate_on_growth=True) is not None

