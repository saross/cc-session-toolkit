"""Tests for project initialisation (init command)."""

from __future__ import annotations

from pathlib import Path

from cc_session_toolkit.init import initialise_project


class TestInitialiseProject:
    """Tests for :func:`initialise_project`."""

    def test_creates_archive_structure(self, tmp_project: Path) -> None:
        """Should create archive directory with queries and defaults."""
        initialise_project(
            project_root=tmp_project,
            project_name="test-project",
        )

        queries = tmp_project / "archive" / "cc-sessions" / "queries"
        assert queries.exists()
        assert (queries / "summarise-session.md").exists()
        assert (queries / "extract-decisions.md").exists()

    def test_creates_defaults_yaml(self, tmp_project: Path) -> None:
        """Should render archive-defaults.yaml with project name."""
        initialise_project(
            project_root=tmp_project,
            project_name="my-project",
        )

        defaults = (
            tmp_project / "archive" / "cc-sessions"
            / "archive-defaults.yaml"
        )
        assert defaults.exists()
        content = defaults.read_text()
        assert "my-project" in content
        assert "{{project_name}}" not in content

    def test_creates_reflection_stubs(self, tmp_project: Path) -> None:
        """Should create reflection document stubs with frontmatter."""
        initialise_project(
            project_root=tmp_project,
            project_name="test-project",
        )

        reflections = tmp_project / "docs" / "notes" / "reflections"
        assert reflections.exists()
        assert (reflections / "session-reflection.md").exists()
        assert (reflections / "llm-observations.md").exists()
        assert (reflections / "abductive-reasoning.md").exists()
        assert (reflections / "session-log.md").exists()

        # working-notes.md is the research-notes layer (owned by /observe)
        # and must be a SIBLING of reflections/, never inside it.
        assert (reflections.parent / "working-notes.md").exists()
        assert not (reflections / "working-notes.md").exists()

        # Check frontmatter is present
        content = (reflections / "session-reflection.md").read_text()
        assert "priority: 1" in content

    def test_creates_skill_md(self, tmp_project: Path) -> None:
        """Should render SKILL.md with project name."""
        initialise_project(
            project_root=tmp_project,
            project_name="my-project",
        )

        skill = (
            tmp_project / ".claude" / "skills" / "reflect" / "SKILL.md"
        )
        assert skill.exists()
        content = skill.read_text()
        assert "my-project" in content
        assert "{{project_name}}" not in content

    def test_skips_existing_files(self, tmp_project: Path) -> None:
        """Should not overwrite existing files."""
        # Pre-create a reflection file with custom content
        reflections = tmp_project / "docs" / "notes" / "reflections"
        reflections.mkdir(parents=True)
        custom = reflections / "session-reflection.md"
        custom.write_text("# My custom content\n")

        initialise_project(
            project_root=tmp_project,
            project_name="test-project",
        )

        # Custom content should be preserved
        assert custom.read_text() == "# My custom content\n"

    def test_update_regenerates_skill(self, tmp_project: Path) -> None:
        """--update should regenerate SKILL.md but not content."""
        # First init
        initialise_project(
            project_root=tmp_project,
            project_name="test-project",
        )

        # Modify SKILL.md
        skill = (
            tmp_project / ".claude" / "skills" / "reflect" / "SKILL.md"
        )
        skill.write_text("# Old content\n")

        # Pre-create a reflection file
        reflections = tmp_project / "docs" / "notes" / "reflections"
        custom = reflections / "session-reflection.md"
        original_content = custom.read_text()

        # Update
        initialise_project(
            project_root=tmp_project,
            project_name="test-project",
            update=True,
        )

        # SKILL.md should be regenerated
        assert skill.read_text() != "# Old content\n"
        assert "test-project" in skill.read_text()

        # Reflection content should be preserved
        assert custom.read_text() == original_content

    def test_custom_reflections_dir(self, tmp_project: Path) -> None:
        """Should respect custom reflections directory."""
        initialise_project(
            project_root=tmp_project,
            project_name="test-project",
            reflections_dir="notes/reflect",
        )

        assert (
            tmp_project / "notes" / "reflect"
            / "session-reflection.md"
        ).exists()

        # working-notes.md should land beside the custom reflections dir
        # (its parent), not inside it.
        assert (tmp_project / "notes" / "working-notes.md").exists()
        assert not (
            tmp_project / "notes" / "reflect" / "working-notes.md"
        ).exists()

        # SKILL.md should reference the custom path
        skill = (
            tmp_project / ".claude" / "skills" / "reflect" / "SKILL.md"
        )
        assert "notes/reflect" in skill.read_text()
