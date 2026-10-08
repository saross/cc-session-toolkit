"""
Tests for catalogue integrity: complete scanning and safe writes.

Background (diagnosed 2026-10-08). Both machines' catalogues were missing
archived sessions (47 on amd-tower, 60 on zbook). Three causes:

* the rebuild scanned only ``<project>/<entry>``, so ``_legacy/<project>/
  <entry>`` and other nested entries were never catalogued;
* hooks rewrote ``CATALOG.json`` with a bare ``write_text`` and no lock, so
  two sessions ending together could lose one entry, and a hook killed
  mid-write could truncate the file;
* a catalogue that failed to parse was replaced by an EMPTY one, keeping
  only the session being added.

(The fourth cause, replication of the catalogue between machines, is fixed
in personal-assistant's ``daily-sync.sh``.)

The hooks find an archived session through the catalogue, so a missing
entry makes the next hook archive the session again into a new directory
instead of refreshing the existing one.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from cc_session_toolkit import catalogue as catalogue_mod
from cc_session_toolkit.archive import find_archive_directory
from cc_session_toolkit.catalogue import (
    rebuild_catalogue,
    update_catalogue,
    update_catalogue_entry,
    write_catalogue,
)


def _meta(sid: str, *, project: str | None = None, recorded: int | None = None,
          tags: list[str] | None = None, started: str = "2026-03-02T09:00:00Z") -> dict:
    """Return a minimal session.meta.json body."""
    body: dict = {
        "session": {"id": sid, "started_at": started, "duration_minutes": 10},
        "auto_generated": {"title": f"Session {sid}", "purpose": "p",
                           "tags": tags if tags is not None else ["t"]},
        "archive": {},
    }
    if project is not None:
        body["project"] = {"name": project}
    if recorded is not None:
        body["archive"]["jsonl_bytes_uncompressed"] = recorded
    return body


def _entry(root: Path, rel: str, meta: dict) -> Path:
    """Write ``<root>/<rel>/session.meta.json`` and return the directory."""
    directory = root / rel
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "session.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return directory


class TestScanning:
    def test_nested_entries_are_catalogued(self, tmp_path: Path) -> None:
        """``_legacy`` and other nested entries were invisible before."""
        _entry(tmp_path, "proj/2026-03-02_a", _meta("aaaa"))
        _entry(tmp_path, "_legacy/trap/trap-extraction/2025-12-01_b",
               _meta("bbbb", project="trap-extraction"))
        _entry(tmp_path, "map-reader-llm/vlm/2026-03-03_c", _meta("cccc"))

        cat = rebuild_catalogue(tmp_path)
        by_id = {s["id"]: s for s in cat["sessions"]}

        assert set(by_id) == {"aaaa", "bbbb", "cccc"}
        assert by_id["bbbb"]["directory"] == "_legacy/trap/trap-extraction/2025-12-01_b"
        # Nested: the meta's project name, else the top-level directory.
        assert by_id["bbbb"]["project"] == "trap-extraction"
        assert by_id["cccc"]["project"] == "map-reader-llm"
        # Depth two keeps its old behaviour: the top-level directory.
        assert by_id["aaaa"]["project"] == "proj"

    def test_subagents_and_queries_are_skipped(self, tmp_path: Path) -> None:
        _entry(tmp_path, "proj/2026-03-02_a", _meta("aaaa"))
        _entry(tmp_path, "proj/2026-03-02_a/subagents/agent-x", _meta("sub1"))
        _entry(tmp_path, "queries/q1", _meta("qqqq"))

        ids = {s["id"] for s in rebuild_catalogue(tmp_path)["sessions"]}
        assert ids == {"aaaa"}


class TestDuplicateDirectories:
    def test_most_complete_copy_comes_first(self, tmp_path: Path) -> None:
        """Lookups take the first entry, so the order must be deliberate."""
        _entry(tmp_path, "vlm/2026-03-02_short", _meta("dddd", recorded=100))
        _entry(tmp_path, "map-reader-llm/vlm/2026-03-02_long",
               _meta("dddd", recorded=900))
        _entry(tmp_path, "zz/2026-03-02_mid", _meta("dddd", recorded=500))

        cat = rebuild_catalogue(tmp_path)
        dirs = [s["directory"] for s in cat["sessions"] if s["id"] == "dddd"]
        assert dirs[0] == "map-reader-llm/vlm/2026-03-02_long"

        catalogue_file = tmp_path / "CATALOG.json"
        write_catalogue(catalogue_file, cat)
        found = find_archive_directory("dddd", catalogue_file, tmp_path)
        assert found == tmp_path / "map-reader-llm/vlm/2026-03-02_long"

    def test_equal_copies_prefer_the_shallower(self, tmp_path: Path) -> None:
        _entry(tmp_path, "map-reader-llm/vlm/2026-03-02_x", _meta("eeee", recorded=10))
        _entry(tmp_path, "vlm/2026-03-02_x", _meta("eeee", recorded=10))

        cat = rebuild_catalogue(tmp_path)
        first = next(s for s in cat["sessions"] if s["id"] == "eeee")
        assert first["directory"] == "vlm/2026-03-02_x"

    def test_rollups_and_tags_count_each_session_once(self, tmp_path: Path) -> None:
        _entry(tmp_path, "vlm/2026-03-02_x", _meta("ffff", tags=["mounds"]))
        _entry(tmp_path, "map-reader-llm/vlm/2026-03-02_x",
               _meta("ffff", project="vlm", tags=["mounds"]))

        cat = rebuild_catalogue(tmp_path)
        assert len([s for s in cat["sessions"] if s["id"] == "ffff"]) == 2
        assert cat["tag_index"]["mounds"] == ["ffff"]
        assert cat["projects"]["vlm"]["session_count"] == 1
        assert cat["unique_sessions"] == 1


class TestSafeWrites:
    def test_concurrent_updates_both_survive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two hooks adding sessions at once must not lose either entry.

        The first writer is held inside its critical section; without the
        lock the second reads the old catalogue and overwrites the first.
        """
        catalogue_file = tmp_path / "CATALOG.json"
        write_catalogue(catalogue_file, {"schema_version": "1.3", "sessions": []})
        real_write = catalogue_mod._write_json_atomic
        first_inside = threading.Event()
        release_first = threading.Event()

        def slow_write(path: Path, data: dict) -> None:
            if not first_inside.is_set():
                first_inside.set()
                release_first.wait(timeout=5)
            real_write(path, data)

        monkeypatch.setattr(catalogue_mod, "_write_json_atomic", slow_write)

        def add(sid: str) -> None:
            update_catalogue([_meta(sid)], catalogue_file, tmp_path, "proj")

        first = threading.Thread(target=add, args=("one1",))
        first.start()
        assert first_inside.wait(timeout=5)
        second = threading.Thread(target=add, args=("two2",))
        second.start()
        time.sleep(0.2)  # the second must now be blocked on the lock
        release_first.set()
        first.join(timeout=5)
        second.join(timeout=5)

        ids = {s["id"] for s in json.loads(catalogue_file.read_text())["sessions"]}
        assert ids == {"one1", "two2"}

    def test_failed_write_leaves_previous_catalogue_intact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        catalogue_file = tmp_path / "CATALOG.json"
        update_catalogue([_meta("keep")], catalogue_file, tmp_path, "proj")
        before = catalogue_file.read_text()

        def boom(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(catalogue_mod.json, "dump", boom)
        with pytest.raises(OSError):
            update_catalogue([_meta("lost")], catalogue_file, tmp_path, "proj")

        assert catalogue_file.read_text() == before
        assert not list(tmp_path.glob("CATALOG.json.*.tmp"))

    def test_entry_update_is_atomic_too(self, tmp_path: Path) -> None:
        catalogue_file = tmp_path / "CATALOG.json"
        update_catalogue([_meta("abcd")], catalogue_file, tmp_path, "proj")
        meta = _meta("abcd")
        meta["auto_generated"]["title"] = "Renamed"
        update_catalogue_entry("abcd", meta, catalogue_file)
        data = json.loads(catalogue_file.read_text())
        assert data["sessions"][0]["title"] == "Renamed"
        assert not list(tmp_path.glob("CATALOG.json.*.tmp"))

    def test_lock_is_shared_with_bulk_archive(self, tmp_path: Path) -> None:
        """personal-assistant's bulk-archive.py locks ``CATALOG.json.lock``."""
        catalogue_file = tmp_path / "CATALOG.json"
        write_catalogue(catalogue_file, {"sessions": []})
        assert (tmp_path / "CATALOG.json.lock").exists()

    def test_unreadable_catalogue_is_rebuilt_not_emptied(self, tmp_path: Path) -> None:
        """A corrupt catalogue used to be replaced by one holding one session."""
        _entry(tmp_path, "proj/2026-03-02_a", _meta("old1"))
        _entry(tmp_path, "proj/2026-03-02_b", _meta("old2"))
        catalogue_file = tmp_path / "CATALOG.json"
        catalogue_file.write_text('{"sessions": [', encoding="utf-8")  # truncated

        update_catalogue([_meta("new1")], catalogue_file, tmp_path, "proj")

        ids = {s["id"] for s in json.loads(catalogue_file.read_text())["sessions"]}
        assert ids == {"old1", "old2", "new1"}
