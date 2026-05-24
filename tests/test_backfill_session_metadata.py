"""Tests for scripts/backfill-session-metadata.py (v1.3 upgrade path)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

# -------------------------------------------------------------------------
# Dynamic import — script filename contains a hyphen.
# -------------------------------------------------------------------------

_BACKFILL_PATH = (
    Path(__file__).parent.parent / "scripts" / "backfill-session-metadata.py"
)


def _load_backfill_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "backfill_session_metadata_under_test", _BACKFILL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def backfill() -> ModuleType:
    return _load_backfill_module()


def _write_meta(
    archive_root: Path,
    rel: str,
    *,
    schema_version: str | None,
    purpose: str,
) -> Path:
    """Helper: write a minimal session.meta.json for finder tests."""
    archive_dir = archive_root / rel
    archive_dir.mkdir(parents=True, exist_ok=True)
    meta_path = archive_dir / "session.meta.json"
    payload: dict = {
        "auto_generated": {
            "title": "Test",
            "purpose": purpose,
            "tags": [],
        },
    }
    if schema_version is not None:
        payload["schema_version"] = schema_version
    meta_path.write_text(json.dumps(payload), encoding="utf-8")
    return meta_path


# -------------------------------------------------------------------------
# find_sessions_needing_v13_upgrade
# -------------------------------------------------------------------------


class TestFindSessionsNeedingV13Upgrade:
    """Partition the archive into upgrade-needed / already-current / empty."""

    def test_picks_up_v12_populated_session(
        self,
        backfill: ModuleType,
        tmp_path: Path,
    ) -> None:
        """A populated v1.2 session is in scope for the upgrade."""
        _write_meta(
            tmp_path, "proj/2026-01-01_a",
            schema_version="1.2", purpose="Build the parser",
        )
        results = backfill.find_sessions_needing_v13_upgrade(tmp_path)
        assert len(results) == 1
        assert results[0].name == "session.meta.json"

    def test_skips_v13_session(
        self,
        backfill: ModuleType,
        tmp_path: Path,
    ) -> None:
        """A session already on v1.3 is NOT in scope."""
        _write_meta(
            tmp_path, "proj/2026-01-01_a",
            schema_version="1.3", purpose="Build the parser",
        )
        assert backfill.find_sessions_needing_v13_upgrade(tmp_path) == []

    def test_skips_empty_metadata_session(
        self,
        backfill: ModuleType,
        tmp_path: Path,
    ) -> None:
        """The 'Auto-metadata unavailable' marker is the default-backfill
        population, NOT the upgrade population."""
        _write_meta(
            tmp_path, "proj/2026-01-01_a",
            schema_version="1.2", purpose="Auto-metadata unavailable",
        )
        assert backfill.find_sessions_needing_v13_upgrade(tmp_path) == []

    def test_skips_session_with_missing_purpose(
        self,
        backfill: ModuleType,
        tmp_path: Path,
    ) -> None:
        """A session with no purpose at all is treated as empty, not upgrade."""
        _write_meta(
            tmp_path, "proj/2026-01-01_a",
            schema_version="1.2", purpose="",
        )
        assert backfill.find_sessions_needing_v13_upgrade(tmp_path) == []

    def test_picks_up_session_with_no_schema_version(
        self,
        backfill: ModuleType,
        tmp_path: Path,
    ) -> None:
        """A pre-Phase-1 session with no schema_version key but populated
        metadata IS in scope — schema_version omitted means 'unknown old'."""
        _write_meta(
            tmp_path, "proj/2026-01-01_a",
            schema_version=None, purpose="Some legacy purpose",
        )
        results = backfill.find_sessions_needing_v13_upgrade(tmp_path)
        assert len(results) == 1

    def test_mixed_archive_partitions_correctly(
        self,
        backfill: ModuleType,
        tmp_path: Path,
    ) -> None:
        """In a mixed archive, only pre-v1.3 populated sessions are returned."""
        # Three populations side-by-side:
        # (a) v1.2 populated — upgrade target
        # (b) v1.3 populated — skip
        # (c) v1.2 empty marker — skip (default backfill instead)
        _write_meta(
            tmp_path, "proj/2026-01-01_a",
            schema_version="1.2", purpose="Real purpose A",
        )
        _write_meta(
            tmp_path, "proj/2026-01-02_b",
            schema_version="1.3", purpose="Real purpose B",
        )
        _write_meta(
            tmp_path, "proj/2026-01-03_c",
            schema_version="1.2", purpose="Auto-metadata unavailable",
        )
        results = backfill.find_sessions_needing_v13_upgrade(tmp_path)
        assert len(results) == 1
        assert "2026-01-01_a" in str(results[0])


# -------------------------------------------------------------------------
# backup_pre_v13_meta
# -------------------------------------------------------------------------


class TestBackupPreV13Meta:
    """Preserve the original meta side-by-side before rebuild."""

    def test_creates_v2_backup_alongside(
        self,
        backfill: ModuleType,
        tmp_path: Path,
    ) -> None:
        """Backup lands at session.meta.v2-backup.json with original content."""
        meta_path = _write_meta(
            tmp_path, "proj/2026-01-01_a",
            schema_version="1.2", purpose="Original purpose",
        )
        original_content = meta_path.read_text()
        backup_path = backfill.backup_pre_v13_meta(meta_path)
        assert backup_path is not None
        assert backup_path.name == "session.meta.v2-backup.json"
        assert backup_path.exists()
        assert backup_path.read_text() == original_content

    def test_does_not_overwrite_existing_backup(
        self,
        backfill: ModuleType,
        tmp_path: Path,
    ) -> None:
        """If a backup already exists, the function leaves it untouched.

        Prevents losing the true original on a second upgrade attempt — if
        a previous run already backed up a v1.2, a second backup would
        overwrite that with a v1.3 (or even the partially-upgraded state).
        """
        meta_path = _write_meta(
            tmp_path, "proj/2026-01-01_a",
            schema_version="1.2", purpose="Original purpose",
        )
        # Pre-existing backup from a prior run.
        prior_backup_content = '{"schema_version": "1.2", "marker": "prior"}'
        prior_backup = meta_path.with_name("session.meta.v2-backup.json")
        prior_backup.write_text(prior_backup_content)
        # Now mutate the canonical meta to simulate a v1.3 state.
        meta_path.write_text('{"schema_version": "1.3"}')
        # Attempt backup again.
        returned = backfill.backup_pre_v13_meta(meta_path)
        assert returned == prior_backup
        # Prior backup untouched.
        assert prior_backup.read_text() == prior_backup_content
