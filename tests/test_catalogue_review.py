"""
Tests for the catalogue fixes from Astra's review of PA #172 (2026-10-08).

* P1: a rebuild holds ONE lock across baseline read, scan, check and
  write, so an incremental update made meanwhile is never overwritten.
* P1: the lock fails closed; a strict scan refuses unreadable metadata;
  an unreadable baseline stops the rebuild; entries may leave only when
  their directory no longer holds metadata, and never en masse.
* P2: the newest-first sort keeps each session's copies in preference
  order, so lookups find the most complete copy even when dates differ.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from cc_session_toolkit import catalogue as catalogue_mod
from cc_session_toolkit.archive import find_archive_directory
from cc_session_toolkit.catalogue import (
    CatalogueLockError,
    CatalogueRebuildRefused,
    CatalogueScanError,
    catalogue_lock,
    check_rebuild_keeps_entries,
    rebuild_and_write_catalogue,
    rebuild_catalogue,
    update_catalogue,
)


def _entry(root: Path, rel: str, sid: str, *, started: str | None = "2026-03-02T09:00:00Z",
           recorded: int | None = None) -> Path:
    """Write a minimal ``<root>/<rel>/session.meta.json``; return the dir."""
    directory = root / rel
    directory.mkdir(parents=True, exist_ok=True)
    session: dict = {"id": sid, "duration_minutes": 10}
    if started is not None:
        session["started_at"] = started
    meta = {"session": session,
            "auto_generated": {"title": f"T {sid}", "purpose": "p", "tags": []},
            "archive": {} if recorded is None else {"jsonl_bytes_uncompressed": recorded}}
    (directory / "session.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return directory


class TestLock:
    def test_fails_closed_when_flock_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _refuse(*_a: object) -> None:
            raise OSError(38, "Function not implemented")

        monkeypatch.setattr(catalogue_mod.fcntl, "flock", _refuse)
        ran = False
        with pytest.raises(CatalogueLockError), catalogue_lock(tmp_path / "CATALOG.json"):
            ran = True
        assert ran is False

    def test_bounded_wait(self, tmp_path: Path) -> None:
        import fcntl

        cat = tmp_path / "CATALOG.json"
        with open(cat.with_name("CATALOG.json.lock"), "a") as holder:
            fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
            with pytest.raises(CatalogueLockError, match="still held"), \
                    catalogue_lock(cat, timeout=0.2):
                pytest.fail("body must not run")


def test_rebuild_and_hook_update_cannot_lose_an_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Paused-scan versus incremental update, made deterministic with events.

    The rebuild scans (seeing only A), then pauses while still holding the
    lock. A hook archives B and calls update_catalogue, which must wait for
    the lock. When the rebuild publishes, the hook's update lands on top of
    it, so B survives. Before the fix the scan ran unlocked and the rebuild
    overwrote the hook's entry.
    """
    _entry(tmp_path, "proj/a", "A")
    cat = tmp_path / "CATALOG.json"
    rebuild_and_write_catalogue(tmp_path, cat)

    scanned, resume = threading.Event(), threading.Event()
    real_scan = catalogue_mod._scan_sessions

    def _paused_scan(*args: object, **kwargs: object) -> list:
        result = real_scan(*args, **kwargs)
        scanned.set()
        assert resume.wait(5)
        return result

    monkeypatch.setattr(catalogue_mod, "_scan_sessions", _paused_scan)
    rebuild = threading.Thread(target=rebuild_and_write_catalogue, args=(tmp_path, cat))
    rebuild.start()
    assert scanned.wait(5)

    b_dir = _entry(tmp_path, "proj/b", "B")
    # The shape archive_session returns: the metadata plus its directory.
    b_result = {**json.loads((b_dir / "session.meta.json").read_text()),
                "_archive_directory": str(b_dir)}
    hook = threading.Thread(target=update_catalogue, args=([b_result], cat, tmp_path, "proj"))
    hook.start()
    hook.join(0.5)
    assert hook.is_alive(), "the hook must wait for the rebuild's lock"

    resume.set()
    rebuild.join(5)
    hook.join(5)
    ids = {s["id"] for s in json.loads(cat.read_text())["sessions"]}
    assert ids == {"A", "B"}


class TestStrictRebuild:
    def test_unreadable_metadata_refuses_and_keeps_the_file(self, tmp_path: Path) -> None:
        for i in range(12):
            _entry(tmp_path, f"proj/s{i}", f"S{i}")
        cat = tmp_path / "CATALOG.json"
        rebuild_and_write_catalogue(tmp_path, cat)
        before = cat.read_text()
        # Ten of twelve become unreadable: a count-ratio guard would pass this.
        for i in range(10):
            (tmp_path / f"proj/s{i}" / "session.meta.json").write_text("{truncated")
        with pytest.raises(CatalogueScanError):
            rebuild_and_write_catalogue(tmp_path, cat)
        assert cat.read_text() == before

    def test_unreadable_baseline_refuses_and_keeps_the_file(self, tmp_path: Path) -> None:
        _entry(tmp_path, "proj/a", "A")
        cat = tmp_path / "CATALOG.json"
        cat.write_text("{not json")
        with pytest.raises(CatalogueRebuildRefused, match="unreadable"):
            rebuild_and_write_catalogue(tmp_path, cat)
        assert cat.read_text() == "{not json"

    def test_first_creation_needs_no_baseline(self, tmp_path: Path) -> None:
        _entry(tmp_path, "proj/a", "A")
        assert len(rebuild_and_write_catalogue(tmp_path)["sessions"]) == 1
        assert (tmp_path / "CATALOG.json").is_file()

    def test_lenient_rebuild_still_skips_unreadable(self, tmp_path: Path) -> None:
        _entry(tmp_path, "proj/a", "A")
        (tmp_path / "proj/bad").mkdir(parents=True)
        (tmp_path / "proj/bad/session.meta.json").write_text("{")
        assert len(rebuild_catalogue(tmp_path)["sessions"]) == 1


class TestIdentityCheck:
    def test_dropping_an_entry_that_still_exists_is_refused(self, tmp_path: Path) -> None:
        _entry(tmp_path, "proj/a", "A")
        baseline = {"sessions": [{"id": "A", "directory": "proj/a"}]}
        with pytest.raises(CatalogueRebuildRefused, match="still holds metadata"):
            check_rebuild_keeps_entries(baseline, {"sessions": []}, tmp_path)

    def test_a_moved_directory_may_leave(self, tmp_path: Path) -> None:
        _entry(tmp_path, "proj/kept", "K")
        baseline = {"sessions": [{"id": "K", "directory": "proj/kept"},
                                 {"id": "K", "directory": "proj/moved-away"}]}
        rebuilt = {"sessions": [{"id": "K", "directory": "proj/kept"}]}
        check_rebuild_keeps_entries(baseline, rebuilt, tmp_path)

    def test_mass_removal_is_refused(self, tmp_path: Path) -> None:
        baseline = {"sessions": [{"id": str(i), "directory": f"p/{i}"} for i in range(10)]}
        rebuilt = {"sessions": [{"id": str(i), "directory": f"p/{i}"} for i in range(4)]}
        for i in range(4):
            _entry(tmp_path, f"p/{i}", str(i))
        with pytest.raises(CatalogueRebuildRefused, match="stale or empty mount"):
            check_rebuild_keeps_entries(baseline, rebuilt, tmp_path)
        check_rebuild_keeps_entries(baseline, rebuilt, tmp_path, max_removed_fraction=0.7)


@pytest.mark.parametrize("short_started", ["2026-05-01T00:00:00Z", None])
def test_preferred_copy_stays_first_despite_dates(
    tmp_path: Path, short_started: str | None,
) -> None:
    """A shorter copy with a later (or missing) date must not overtake."""
    _entry(tmp_path, "proj/2026-03-01_full", "S", started="2026-03-01T00:00:00Z", recorded=9000)
    _entry(tmp_path, "proj/2026-03-01_short", "S", started=short_started, recorded=100)
    _entry(tmp_path, "proj/2026-04-01_other", "O", started="2026-04-01T00:00:00Z")
    cat = tmp_path / "CATALOG.json"
    rebuild_and_write_catalogue(tmp_path, cat)
    order = [s["directory"] for s in json.loads(cat.read_text())["sessions"] if s["id"] == "S"]
    assert order[0] == "proj/2026-03-01_full"
    assert find_archive_directory("S", cat, tmp_path) == tmp_path / "proj/2026-03-01_full"
