"""Tests for project root detection and path utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from cc_session_toolkit.project import (
    find_project_root,
    get_archive_dir,
    get_catalogue_file,
    get_cc_project_path,
    get_defaults_file,
    get_project_name,
)


class TestFindProjectRoot:
    """Tests for :func:`find_project_root`."""

    def test_finds_git_marker(self, tmp_project: Path) -> None:
        """Should find root when starting from the project directory."""
        assert find_project_root(start=tmp_project) == tmp_project

    def test_finds_root_from_subdirectory(self, tmp_project: Path) -> None:
        """Should search upward and find root from a nested directory."""
        subdir = tmp_project / "src" / "deep"
        subdir.mkdir(parents=True)
        assert find_project_root(start=subdir) == tmp_project

    def test_raises_when_no_marker(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError when no marker is found."""
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="No project root found"):
            find_project_root(start=empty)

    def test_finds_pyproject_toml_marker(self, tmp_path: Path) -> None:
        """Should detect pyproject.toml as a valid project marker."""
        project = tmp_path / "pkg"
        project.mkdir()
        (project / "pyproject.toml").write_text("[project]\nname='test'\n")
        assert find_project_root(start=project) == project

    def test_uses_cwd_when_no_start(
        self, tmp_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should default to CWD when start is None."""
        monkeypatch.chdir(tmp_project)
        assert find_project_root() == tmp_project


class TestGetProjectName:
    """Tests for :func:`get_project_name`."""

    def test_reads_claude_md(self, tmp_project: Path) -> None:
        """Should extract name from CLAUDE.md '# Project:' line."""
        assert get_project_name(project_root=tmp_project) == "test-project"

    def test_falls_back_to_dir_name(
        self, tmp_project_no_claude: Path
    ) -> None:
        """Should use directory name when CLAUDE.md is absent."""
        assert (
            get_project_name(project_root=tmp_project_no_claude)
            == "bare-project"
        )


class TestGetCcProjectPath:
    """Tests for :func:`get_cc_project_path`."""

    def test_encodes_slashes(self, tmp_project: Path) -> None:
        """Should replace all '/' with '-' in the resolved path."""
        result = get_cc_project_path(project_root=tmp_project)
        assert "/" not in result
        assert result.startswith("-")


class TestPathHelpers:
    """Tests for convenience path helpers."""

    def test_get_archive_dir(self, tmp_project: Path) -> None:
        """Should return archive/cc-sessions/ under project root."""
        result = get_archive_dir(project_root=tmp_project)
        assert result == tmp_project / "archive" / "cc-sessions"

    def test_get_defaults_file(self, tmp_project: Path) -> None:
        """Should return archive-defaults.yaml inside archive dir."""
        result = get_defaults_file(project_root=tmp_project)
        assert result.name == "archive-defaults.yaml"

    def test_get_catalogue_file(self, tmp_project: Path) -> None:
        """Should return CATALOG.json inside archive dir."""
        result = get_catalogue_file(project_root=tmp_project)
        assert result.name == "CATALOG.json"
