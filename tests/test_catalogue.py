"""Tests for catalogue management."""

from __future__ import annotations

import json
from pathlib import Path

from cc_session_toolkit.catalogue import (
    generate_catalogue_markdown,
    rebuild_catalogue,
    update_catalogue,
    update_catalogue_entry,
)


def _make_session_meta(
    session_id: str = "abc12345",
    title: str = "Test Session",
    started_at: str = "2026-01-15T10:30:00Z",
) -> dict:
    """Create a minimal session metadata dictionary for testing."""
    return {
        "session": {
            "id": session_id,
            "started_at": started_at,
            "ended_at": "2026-01-15T11:30:00Z",
            "duration_minutes": 60,
        },
        "auto_generated": {
            "title": title,
            "purpose": "Test purpose",
            "tags": ["test"],
        },
    }


class TestUpdateCatalogue:
    """Tests for :func:`update_catalogue`."""

    def test_creates_catalogue(self, tmp_path: Path) -> None:
        catalogue_file = tmp_path / "CATALOG.json"
        archive_dir = tmp_path / "archive" / "cc-sessions"
        archive_dir.mkdir(parents=True)

        update_catalogue(
            new_sessions=[_make_session_meta()],
            catalogue_file=catalogue_file,
            archive_dir=archive_dir,
            project_name="test-project",
        )

        assert catalogue_file.exists()
        data = json.loads(catalogue_file.read_text())
        assert data["total_sessions"] == 1
        assert data["sessions"][0]["id"] == "abc12345"

    def test_appends_without_duplicates(self, tmp_path: Path) -> None:
        catalogue_file = tmp_path / "CATALOG.json"
        archive_dir = tmp_path / "archive" / "cc-sessions"
        archive_dir.mkdir(parents=True)

        # Add once
        update_catalogue(
            new_sessions=[_make_session_meta()],
            catalogue_file=catalogue_file,
            archive_dir=archive_dir,
            project_name="test-project",
        )
        # Add same again
        update_catalogue(
            new_sessions=[_make_session_meta()],
            catalogue_file=catalogue_file,
            archive_dir=archive_dir,
            project_name="test-project",
        )

        data = json.loads(catalogue_file.read_text())
        assert data["total_sessions"] == 1


class TestUpdateCatalogueEntry:
    """Tests for :func:`update_catalogue_entry`."""

    def test_updates_entry(self, tmp_path: Path) -> None:
        catalogue_file = tmp_path / "CATALOG.json"
        catalogue_file.write_text(json.dumps({
            "schema_version": "1.1",
            "sessions": [
                {
                    "id": "abc12345",
                    "title": "Old Title",
                    "purpose": "",
                    "tags": [],
                }
            ],
        }))

        update_catalogue_entry(
            session_id="abc12345",
            meta={
                "auto_generated": {
                    "title": "New Title",
                    "purpose": "New purpose",
                    "tags": ["updated"],
                }
            },
            catalogue_file=catalogue_file,
        )

        data = json.loads(catalogue_file.read_text())
        assert data["sessions"][0]["title"] == "New Title"
        assert data["sessions"][0]["tags"] == ["updated"]


class TestRebuildCatalogue:
    """Tests for :func:`rebuild_catalogue`."""

    def test_empty_archive(self, tmp_path: Path) -> None:
        archive_dir = tmp_path / "archive" / "cc-sessions"
        archive_dir.mkdir(parents=True)

        result = rebuild_catalogue(archive_dir)
        assert result["total_sessions"] == 0
        assert result["sessions"] == []

    def test_scans_sessions(self, tmp_path: Path) -> None:
        archive_dir = tmp_path / "archive" / "cc-sessions"
        session_dir = archive_dir / "my-project" / "2026-01-15T10-30_abc"
        session_dir.mkdir(parents=True)

        # Create a session.meta.json
        meta = {
            "session": {
                "id": "abc12345",
                "started_at": "2026-01-15T10:30:00Z",
                "duration_minutes": 60,
            },
            "auto_generated": {
                "title": "Test Session",
                "purpose": "Testing",
                "tags": ["test", "example"],
            },
            "statistics": {
                "turns": 10,
                "tool_calls": {"total": 5},
                "thinking_blocks": 3,
            },
            "model": {"model_id": "claude-sonnet-4-5-20250929"},
            "relationships": {},
            "artifacts": {"created": [], "modified": [], "referenced": []},
            "thinking_blocks": {"total_tokens": 1000},
        }
        (session_dir / "session.meta.json").write_text(json.dumps(meta))

        result = rebuild_catalogue(archive_dir)
        assert result["total_sessions"] == 1
        assert "test" in result["tag_index"]
        assert len(result["tag_index"]["test"]) == 1

    def test_builds_project_rollups(self, tmp_path: Path) -> None:
        archive_dir = tmp_path / "archive" / "cc-sessions"

        for i in range(3):
            session_dir = archive_dir / "proj" / f"session-{i}"
            session_dir.mkdir(parents=True)
            meta = {
                "session": {
                    "id": f"id-{i}",
                    "started_at": f"2026-01-1{i}T10:00:00Z",
                    "duration_minutes": 30,
                },
                "auto_generated": {
                    "title": f"Session {i}",
                    "purpose": "",
                    "tags": [],
                },
                "statistics": {"turns": 5, "tool_calls": {"total": 2},
                               "thinking_blocks": 1},
                "model": {"model_id": "test"},
                "relationships": {},
                "artifacts": {},
                "thinking_blocks": {},
            }
            (session_dir / "session.meta.json").write_text(
                json.dumps(meta)
            )

        result = rebuild_catalogue(archive_dir)
        assert result["projects"]["proj"]["session_count"] == 3
        assert result["projects"]["proj"]["total_duration_minutes"] == 90


class TestGenerateCatalogueMarkdown:
    """Tests for :func:`generate_catalogue_markdown`."""

    def test_generates_markdown(self) -> None:
        catalogue = {
            "total_sessions": 1,
            "projects": {
                "test": {
                    "name": "test",
                    "session_count": 1,
                    "total_duration_minutes": 60,
                }
            },
            "sessions": [
                {
                    "id": "abc",
                    "project": "test",
                    "directory": "test/session-1",
                    "title": "Test Session",
                    "purpose": "Testing the system",
                    "tags": ["test"],
                    "started_at": "2026-01-15T10:30:00Z",
                    "duration_minutes": 60,
                    "model": "claude-sonnet",
                    "statistics": {
                        "turns": 10,
                        "tool_calls": 5,
                        "thinking_blocks": 3,
                    },
                    "artifact_counts": {
                        "created": 2,
                        "modified": 1,
                        "referenced": 3,
                    },
                    "thinking_blocks_tokens": 1000,
                }
            ],
            "tag_index": {"test": ["abc"]},
        }

        md = generate_catalogue_markdown(catalogue)
        assert "# CC Session Catalogue" in md
        assert "Test Session" in md
        assert "test" in md
        assert "1h" in md
