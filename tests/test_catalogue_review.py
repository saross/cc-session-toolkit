"""
Tests for the catalogue fixes from Astra's review of PA #172 (2026-10-08).

* P1: a rebuild holds ONE lock across baseline read, scan, check and
  write, so an incremental update made meanwhile is never overwritten.
* P1: the lock fails closed; a strict scan refuses unreadable metadata;
  an unreadable baseline stops the rebuild.
* Round three: the hooks' incremental update no longer rebuilds an
  unreadable catalogue leniently; it refuses and keeps the file.
* Round two (F2): absence is not evidence of removal. An entry may leave
  only when an authorised cleanup names it (``allow_removed``) or it is a
  redundant copy of a session still catalogued from a copy at least as
  long; a minority of directories vanishing (a stale mount) is refused.
* P2: the newest-first sort keeps each session's copies in preference
  order, so lookups find the most complete copy even when dates differ.
* Single owner (2026-10-09): a store's ``CATALOG.owner`` names the one host
  that may write its catalogue; every writer refuses on any other host.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from cc_session_toolkit import catalogue as catalogue_mod
from cc_session_toolkit.archive import find_archive_directory
from cc_session_toolkit.catalogue import (
    CATALOGUE_OWNER_FILE,
    CatalogueLockError,
    CatalogueNotOwner,
    CatalogueRebuildRefused,
    CatalogueScanError,
    catalogue_lock,
    check_rebuild_keeps_entries,
    rebuild_and_write_catalogue,
    rebuild_catalogue,
    update_catalogue,
    update_catalogue_entry,
    write_catalogue,
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
        # Two of twelve become unreadable, leaving ten readable: above the
        # old guard's half-the-baseline threshold, so a count-ratio check
        # would have published the partial scan. (Ten of twelve would not
        # test that: two readable is below the threshold.)
        for i in range(2):
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
    """Round two (F2): an entry leaves the index only with proof."""

    def test_a_disappearing_minority_is_refused(self, tmp_path: Path) -> None:
        # 100 sessions catalogued; ten directories then vanish from both the
        # scan and stat, as on a partially stale mount. The scan sees no
        # read error, and 90 of 100 is above any half-the-baseline ratio.
        for i in range(100):
            _entry(tmp_path, f"proj/s{i:03d}", f"S{i}", recorded=1000)
        cat = tmp_path / "CATALOG.json"
        rebuild_and_write_catalogue(tmp_path, cat)
        before = cat.read_text()
        for i in range(10):
            directory = tmp_path / f"proj/s{i:03d}"
            (directory / "session.meta.json").unlink()
            directory.rmdir()
        with pytest.raises(CatalogueRebuildRefused, match="10 entries would be dropped"):
            rebuild_and_write_catalogue(tmp_path, cat)
        assert cat.read_text() == before

    def test_an_entry_the_scan_did_not_return_is_refused(self) -> None:
        baseline = {"sessions": [{"id": "A", "directory": "proj/a", "transcript_bytes": 5}]}
        with pytest.raises(CatalogueRebuildRefused, match="not evidence"):
            check_rebuild_keeps_entries(baseline, {"sessions": []})

    def test_an_authorised_removal_is_named_explicitly(self, tmp_path: Path) -> None:
        for rel, sid in (("proj/a", "A"), ("proj/b", "B"), ("proj/c", "C")):
            _entry(tmp_path, rel, sid, recorded=10)
        cat = tmp_path / "CATALOG.json"
        rebuild_and_write_catalogue(tmp_path, cat)
        (tmp_path / "proj/b/session.meta.json").unlink()
        (tmp_path / "proj/b").rmdir()
        # Naming a different directory does not excuse this one.
        with pytest.raises(CatalogueRebuildRefused):
            rebuild_and_write_catalogue(tmp_path, cat, allow_removed=["proj/c"])
        rebuilt = rebuild_and_write_catalogue(tmp_path, cat, allow_removed=["proj/b"])
        assert {s["id"] for s in rebuilt["sessions"]} == {"A", "C"}
        assert {s["id"] for s in json.loads(cat.read_text())["sessions"]} == {"A", "C"}

    def test_a_redundant_shorter_copy_may_leave(self) -> None:
        baseline = {"sessions": [
            {"id": "K", "directory": "proj/long", "transcript_bytes": 200},
            {"id": "K", "directory": "proj/short", "transcript_bytes": 100},
        ]}
        rebuilt = {"sessions": [{"id": "K", "directory": "proj/long", "transcript_bytes": 200}]}
        assert check_rebuild_keeps_entries(baseline, rebuilt) == ["proj/short"]

    def test_the_longer_copy_may_not_leave(self) -> None:
        # The surviving copy holds less than the one that vanished, so the
        # session would stay findable but lose conversation.
        baseline = {"sessions": [
            {"id": "K", "directory": "proj/long", "transcript_bytes": 200},
            {"id": "K", "directory": "proj/short", "transcript_bytes": 100},
        ]}
        rebuilt = {"sessions": [{"id": "K", "directory": "proj/short", "transcript_bytes": 100}]}
        with pytest.raises(CatalogueRebuildRefused, match="proj/long"):
            check_rebuild_keeps_entries(baseline, rebuilt)

    def test_a_copy_without_a_recorded_length_is_not_provably_redundant(self) -> None:
        # Catalogues written before ``transcript_bytes`` existed.
        baseline = {"sessions": [{"id": "K", "directory": "proj/kept"},
                                 {"id": "K", "directory": "proj/moved-away"}]}
        rebuilt = {"sessions": [{"id": "K", "directory": "proj/kept", "transcript_bytes": 9}]}
        with pytest.raises(CatalogueRebuildRefused, match="moved-away"):
            check_rebuild_keeps_entries(baseline, rebuilt)
        assert check_rebuild_keeps_entries(
            baseline, rebuilt, allow_removed=["proj/moved-away"],
        ) == ["proj/moved-away"]

    def test_entries_without_a_directory_match_by_id(self) -> None:
        baseline = {"sessions": [{"id": "A"}, {"id": "B"}]}
        rebuilt = {"sessions": [{"id": "A", "directory": "proj/a", "transcript_bytes": 1}]}
        with pytest.raises(CatalogueRebuildRefused, match="session B"):
            check_rebuild_keeps_entries(baseline, rebuilt)


class TestIncrementalUpdateKeepsAnUnreadableCatalogue:
    """Round three: no unattended lenient recovery on the hook path.

    The old fallback rebuilt a corrupt catalogue with a LENIENT scan, which
    skips unreadable metadata, and published the partial result as valid
    JSON; the next scheduled rebuild then trusted it as its baseline.
    """

    CORRUPT = '{"sessions": ['

    def _store(self, root: Path, *, bad_sibling: bool) -> Path:
        _entry(root, "proj/2026-03-02_a", "A", recorded=10)
        if bad_sibling:
            (root / "proj/2026-03-03_bad").mkdir(parents=True)
            (root / "proj/2026-03-03_bad/session.meta.json").write_text("{trunc")
        cat = root / "CATALOG.json"
        cat.write_text(self.CORRUPT)
        return cat

    @pytest.mark.parametrize("bad_sibling", [True, False])
    def test_update_refuses_and_keeps_the_bytes(
        self, tmp_path: Path, bad_sibling: bool,
    ) -> None:
        cat = self._store(tmp_path, bad_sibling=bad_sibling)
        new = json.loads((_entry(tmp_path, "proj/2026-03-04_n", "N", recorded=5)
                          / "session.meta.json").read_text())
        new["_archive_directory"] = str(tmp_path / "proj/2026-03-04_n")
        with pytest.raises(CatalogueRebuildRefused, match="unreadable"):
            update_catalogue([new], cat, tmp_path, "proj")
        assert cat.read_text() == self.CORRUPT
        assert not list(tmp_path.glob("CATALOG.json.*.tmp"))

    def test_a_wrong_shape_is_refused_too(self, tmp_path: Path) -> None:
        cat = tmp_path / "CATALOG.json"
        cat.write_text('{"sessions": "not-a-list"}')
        with pytest.raises(CatalogueRebuildRefused, match="no sessions list"):
            update_catalogue([], cat, tmp_path, "proj")
        assert cat.read_text() == '{"sessions": "not-a-list"}'

    def test_entry_update_refuses_and_keeps_the_bytes(self, tmp_path: Path) -> None:
        cat = self._store(tmp_path, bad_sibling=True)
        with pytest.raises(CatalogueRebuildRefused):
            update_catalogue_entry("A", {"auto_generated": {"title": "t"}}, cat)
        assert cat.read_text() == self.CORRUPT

    def test_a_missing_catalogue_is_still_created(self, tmp_path: Path) -> None:
        cat = tmp_path / "CATALOG.json"
        meta = json.loads((_entry(tmp_path, "proj/2026-03-04_n", "N", recorded=5)
                           / "session.meta.json").read_text())
        update_catalogue([meta], cat, tmp_path, "proj")
        assert [s["id"] for s in json.loads(cat.read_text())["sessions"]] == ["N"]


class TestCatalogueOwner:
    """``CATALOG.owner``: only the named host writes the store's catalogue."""

    @pytest.fixture()
    def store(self, tmp_path: Path) -> Path:
        _entry(tmp_path, "proj/2026-03-02_a", "A", recorded=10)
        rebuild_and_write_catalogue(tmp_path)
        return tmp_path

    @staticmethod
    def _own(root: Path, text: str) -> None:
        (root / CATALOGUE_OWNER_FILE).write_text(text, encoding="utf-8")

    def test_another_hosts_store_is_refused_everywhere(
        self, store: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(catalogue_mod.socket, "gethostname", lambda: "zbook-ubuntu")
        self._own(store, "# the R2 push owner\nAMD-tower-ubuntu\n")
        cat = store / "CATALOG.json"
        before = cat.read_bytes()
        new = json.loads((_entry(store, "proj/2026-03-04_n", "N", recorded=5)
                          / "session.meta.json").read_text())
        new["_archive_directory"] = str(store / "proj/2026-03-04_n")
        writers = [
            lambda: rebuild_and_write_catalogue(store),
            lambda: update_catalogue([new], cat, store, "proj"),
            lambda: update_catalogue_entry("A", {"auto_generated": {"title": "t"}}, cat),
            lambda: write_catalogue(cat, {"sessions": []}),
        ]
        for write in writers:
            with pytest.raises(CatalogueNotOwner, match="owned by AMD-tower-ubuntu"):
                write()
        assert cat.read_bytes() == before
        assert not list(store.glob("CATALOG.json.*.tmp"))

    def test_refused_before_the_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Another host's store is not touched at all, not even its lock file."""
        monkeypatch.setattr(catalogue_mod.socket, "gethostname", lambda: "zbook-ubuntu")
        _entry(tmp_path, "proj/a", "A")
        self._own(tmp_path, "AMD-tower-ubuntu\n")
        with pytest.raises(CatalogueNotOwner):
            rebuild_and_write_catalogue(tmp_path)
        assert not (tmp_path / "CATALOG.json").exists()
        assert not (tmp_path / "CATALOG.json.lock").exists()

    def test_the_owner_writes_and_case_does_not_matter(
        self, store: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(catalogue_mod.socket, "gethostname", lambda: "amd-tower-ubuntu")
        self._own(store, "AMD-tower-ubuntu\n")
        _entry(store, "proj/2026-03-05_b", "B", recorded=7)
        rebuilt = rebuild_and_write_catalogue(store)
        assert {s["id"] for s in rebuilt["sessions"]} == {"A", "B"}

    def test_a_refusal_is_a_rebuild_refusal(self) -> None:
        """Existing callers that catch the refusal handle this one too."""
        assert issubclass(CatalogueNotOwner, CatalogueRebuildRefused)

    @pytest.mark.parametrize("text", ["", "# only a comment\n", "host-a\nhost-b\n"])
    def test_an_owner_file_must_name_one_host(
        self, store: Path, monkeypatch: pytest.MonkeyPatch, text: str,
    ) -> None:
        monkeypatch.setattr(catalogue_mod.socket, "gethostname", lambda: "host-a")
        self._own(store, text)
        with pytest.raises(CatalogueNotOwner, match="exactly one host"):
            rebuild_and_write_catalogue(store)

    def test_an_unreadable_owner_file_refuses(
        self, store: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(catalogue_mod.socket, "gethostname", lambda: "host-a")
        (store / CATALOGUE_OWNER_FILE).write_bytes(b"\xff\xfe not utf-8")
        with pytest.raises(CatalogueNotOwner, match="cannot read"):
            rebuild_and_write_catalogue(store)

    def test_a_store_without_an_owner_file_is_unchanged(self, store: Path) -> None:
        _entry(store, "proj/2026-03-05_b", "B", recorded=7)
        assert len(rebuild_and_write_catalogue(store)["sessions"]) == 2


class TestRecordedLength:
    """``transcript_bytes`` is what the identity check proves redundancy with."""

    def test_rebuild_records_each_copys_length(self, tmp_path: Path) -> None:
        _entry(tmp_path, "proj/a", "A", recorded=321)
        _entry(tmp_path, "proj/b", "B")
        lengths = {s["id"]: s["transcript_bytes"]
                   for s in rebuild_catalogue(tmp_path)["sessions"]}
        assert lengths == {"A": 321, "B": 0}

    def test_hook_append_records_the_length(self, tmp_path: Path) -> None:
        cat = tmp_path / "CATALOG.json"
        meta = json.loads(
            (_entry(tmp_path, "proj/a", "A", recorded=77) / "session.meta.json").read_text())
        meta["_archive_directory"] = str(tmp_path / "proj/a")
        update_catalogue([meta], cat, tmp_path, "proj")
        assert json.loads(cat.read_text())["sessions"][0]["transcript_bytes"] == 77

    def test_a_supersede_updates_the_copy_it_wrote(self, tmp_path: Path) -> None:
        cat = tmp_path / "CATALOG.json"
        cat.write_text(json.dumps({"sessions": [
            {"id": "K", "directory": "proj/first", "transcript_bytes": 500},
            {"id": "K", "directory": "proj/second", "transcript_bytes": 100},
        ]}))
        meta = {"auto_generated": {"title": "t"}, "archive": {"jsonl_bytes_uncompressed": 300},
                "_archive_directory": str(tmp_path / "proj/second")}
        update_catalogue_entry("K", meta, cat)
        lengths = {s["directory"]: s["transcript_bytes"]
                   for s in json.loads(cat.read_text())["sessions"]}
        assert lengths == {"proj/first": 500, "proj/second": 300}

    def test_an_edit_without_a_length_keeps_the_known_one(self, tmp_path: Path) -> None:
        cat = tmp_path / "CATALOG.json"
        cat.write_text(json.dumps({"sessions": [
            {"id": "K", "directory": "proj/a", "transcript_bytes": 500}]}))
        update_catalogue_entry("K", {"auto_generated": {"title": "new"}}, cat)
        entry = json.loads(cat.read_text())["sessions"][0]
        assert (entry["title"], entry["transcript_bytes"]) == ("new", 500)


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
