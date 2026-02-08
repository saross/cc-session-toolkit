"""Tests for naming utilities."""

from __future__ import annotations

from pathlib import Path

from cc_session_toolkit.naming import get_archive_directory, slugify


class TestSlugify:
    """Tests for :func:`slugify`."""

    def test_basic_conversion(self) -> None:
        assert slugify("VLM Pipeline Development") == "vlm-pipeline-development"

    def test_special_characters(self) -> None:
        assert slugify("Hello, World! (Test)") == "hello-world-test"

    def test_max_length(self) -> None:
        result = slugify("a very long title that exceeds the limit", max_length=20)
        assert len(result) <= 20
        # Should truncate at word boundary
        assert not result.endswith("-")

    def test_empty_string(self) -> None:
        assert slugify("") == ""

    def test_collapses_hyphens(self) -> None:
        assert slugify("hello---world") == "hello-world"

    def test_strips_leading_trailing_hyphens(self) -> None:
        assert slugify("--hello--") == "hello"


class TestGetArchiveDirectory:
    """Tests for :func:`get_archive_directory`."""

    def test_with_title(self, tmp_path: Path) -> None:
        archive_dir = tmp_path / "archive" / "cc-sessions"
        archive_dir.mkdir(parents=True)

        result = get_archive_directory(
            session_id="abc12345-1234-5678-9abc-def012345678",
            stats={"started_at": "2026-01-15T10:30:00Z"},
            archive_dir=archive_dir,
            project_name="my-project",
            title="Pipeline Development Session",
        )

        assert "my-project" in str(result)
        assert "2026-01-15T10-30" in result.name
        assert "pipeline-development-session" in result.name

    def test_without_title(self, tmp_path: Path) -> None:
        archive_dir = tmp_path / "archive" / "cc-sessions"
        archive_dir.mkdir(parents=True)

        result = get_archive_directory(
            session_id="abc12345-1234-5678-9abc-def012345678",
            stats={"started_at": "2026-01-15T10:30:00Z"},
            archive_dir=archive_dir,
            project_name="my-project",
        )

        assert "abc12345" in result.name

    def test_agent_session(self, tmp_path: Path) -> None:
        archive_dir = tmp_path / "archive" / "cc-sessions"
        archive_dir.mkdir(parents=True)

        result = get_archive_directory(
            session_id="agent-a37c175",
            stats={"started_at": "2026-01-15T10:30:00Z"},
            archive_dir=archive_dir,
            project_name="my-project",
            title="Agent Sub-task",
        )

        assert "agent" in result.name

    def test_uniqueness_suffix(self, tmp_path: Path) -> None:
        """Should append short_id when directory already exists."""
        archive_dir = tmp_path / "archive" / "cc-sessions"
        project_dir = archive_dir / "my-project"
        project_dir.mkdir(parents=True)

        # Pre-create a conflicting directory
        (project_dir / "2026-01-15T10-30_test-title").mkdir()

        result = get_archive_directory(
            session_id="abc12345-1234-5678-9abc-def012345678",
            stats={"started_at": "2026-01-15T10:30:00Z"},
            archive_dir=archive_dir,
            project_name="my-project",
            title="Test Title",
        )

        # Should have the short_id suffix for uniqueness
        assert "abc12345" in result.name

    def test_no_timestamp(self, tmp_path: Path) -> None:
        archive_dir = tmp_path / "archive" / "cc-sessions"
        archive_dir.mkdir(parents=True)

        result = get_archive_directory(
            session_id="abc12345-1234-5678-9abc-def012345678",
            stats={"started_at": None},
            archive_dir=archive_dir,
            project_name="my-project",
        )

        assert result.name == "abc12345"
