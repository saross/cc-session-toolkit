"""Tests for hook-based automated archiving (Phase 1)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from cc_session_toolkit.archive import (
    _is_meta_message,
    generate_auto_metadata,
    is_already_archived,
    is_trivial_session,
)
from cc_session_toolkit.config import (
    DEFAULT_ARCHIVE_ROOT,
    DEFAULT_MIN_DURATION_MINUTES,
    DEFAULT_MIN_TURNS,
)
from cc_session_toolkit.project import (
    detect_project_name_from_cwd,
    get_global_archive_dir,
    get_global_catalogue_file,
)


# -------------------------------------------------------------------------
# Trivial session filter
# -------------------------------------------------------------------------

class TestIsTrivialSession:
    """Tests for :func:`is_trivial_session`."""

    def test_below_turn_threshold(self) -> None:
        """Sessions with fewer turns than the minimum are trivial."""
        stats = {"turns": 2, "duration_minutes": 10}
        assert is_trivial_session(stats) is True

    def test_below_duration_threshold(self) -> None:
        """Sessions shorter than the minimum duration are trivial."""
        stats = {"turns": 10, "duration_minutes": 0}
        assert is_trivial_session(stats) is True

    def test_meets_both_thresholds(self) -> None:
        """Sessions meeting both thresholds are NOT trivial."""
        stats = {"turns": 10, "duration_minutes": 5}
        assert is_trivial_session(stats) is False

    def test_exactly_at_threshold(self) -> None:
        """Sessions exactly at the threshold are NOT trivial."""
        stats = {
            "turns": DEFAULT_MIN_TURNS,
            "duration_minutes": DEFAULT_MIN_DURATION_MINUTES,
        }
        assert is_trivial_session(stats) is False

    def test_custom_thresholds(self) -> None:
        """Custom thresholds override defaults."""
        stats = {"turns": 3, "duration_minutes": 5}
        # Below default (5 turns) but above custom (2 turns)
        assert is_trivial_session(stats, min_turns=2) is False

    def test_missing_turns_key(self) -> None:
        """Missing 'turns' key defaults to 0 (trivial)."""
        stats = {"duration_minutes": 10}
        assert is_trivial_session(stats) is True

    def test_missing_duration_key(self) -> None:
        """Missing 'duration_minutes' key defaults to 0 (trivial)."""
        stats = {"turns": 10}
        assert is_trivial_session(stats) is True


# -------------------------------------------------------------------------
# Deduplication check
# -------------------------------------------------------------------------

class TestIsAlreadyArchived:
    """Tests for :func:`is_already_archived`."""

    def test_not_archived(self, tmp_path: Path) -> None:
        """A session not in the catalogue is NOT archived."""
        catalogue = tmp_path / "CATALOG.json"
        catalogue.write_text(json.dumps({
            "schema_version": "1.1",
            "sessions": [
                {"id": "existing-session-id"},
            ],
        }))
        assert is_already_archived("new-session-id", catalogue) is False

    def test_already_archived(self, tmp_path: Path) -> None:
        """A session in the catalogue IS archived."""
        catalogue = tmp_path / "CATALOG.json"
        catalogue.write_text(json.dumps({
            "schema_version": "1.1",
            "sessions": [
                {"id": "existing-session-id"},
            ],
        }))
        assert is_already_archived("existing-session-id", catalogue) is True

    def test_no_catalogue_file(self, tmp_path: Path) -> None:
        """If no catalogue exists, nothing is archived."""
        catalogue = tmp_path / "nonexistent" / "CATALOG.json"
        assert is_already_archived("any-id", catalogue) is False

    def test_empty_catalogue(self, tmp_path: Path) -> None:
        """An empty catalogue has no archived sessions."""
        catalogue = tmp_path / "CATALOG.json"
        catalogue.write_text(json.dumps({
            "schema_version": "1.1",
            "sessions": [],
        }))
        assert is_already_archived("any-id", catalogue) is False


# -------------------------------------------------------------------------
# Auto-metadata generation
# -------------------------------------------------------------------------

class TestGenerateAutoMetadata:
    """Tests for :func:`generate_auto_metadata`."""

    def test_returns_none_without_anthropic(
        self, sample_session_jsonl: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Falls back gracefully when anthropic is not installed."""
        # Simulate ImportError for anthropic
        import builtins
        real_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "anthropic":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        stats = {"turns": 10, "duration_minutes": 30, "tool_calls": {
            "total": 5, "by_type": {"Read": 3, "Write": 2},
        }}
        result = generate_auto_metadata(sample_session_jsonl, stats)
        assert result is None

    def test_returns_none_for_empty_session(
        self, tmp_path: Path
    ) -> None:
        """Returns None when the session has no user messages."""
        empty_session = tmp_path / "empty.jsonl"
        empty_session.write_text("")
        stats = {"turns": 0, "duration_minutes": 0, "tool_calls": {
            "total": 0, "by_type": {},
        }}
        result = generate_auto_metadata(empty_session, stats)
        assert result is None


# -------------------------------------------------------------------------
# Meta-message filter
# -------------------------------------------------------------------------

class TestIsMetaMessage:
    """Unit tests for :func:`_is_meta_message`."""

    @pytest.mark.parametrize("text", [
        "/recap",
        "/done task-123",
        "/review",
        "/standup",
        "yes",
        "ok",
        "looks good",
        "lgtm",
        "commit",
        "push",
        "done",
        "please",
        "sure!",
        "go ahead.",
        "commit and push this please",
        "please commit this",
        "Could you push this to main",
        "  /reflect  ",
        "",
        "   ",
    ])
    def test_meta_messages_detected(self, text: str) -> None:
        assert _is_meta_message(text) is True

    @pytest.mark.parametrize("text", [
        "Implement the parser for CSV files",
        "Add error handling to the validation module",
        "yes, but also add the validation logic for edge cases",
        "The config loader needs to handle missing keys gracefully",
        "Update the README with installation instructions",
        "Fix the bug where timestamps are off by one hour",
        "Can you commit the archive module changes separately",
    ])
    def test_substantive_messages_not_filtered(self, text: str) -> None:
        assert _is_meta_message(text) is False


# -------------------------------------------------------------------------
# Session JSONL builder (for meta-filter and artefact tests)
# -------------------------------------------------------------------------

def _build_session_jsonl(
    tmp_path: Path,
    user_messages: list[str],
    *,
    write_files: list[str] | None = None,
    edit_files: list[str] | None = None,
) -> Path:
    """
    Build a session JSONL file with controlled user messages and
    optional Write/Edit tool_use blocks in assistant messages.

    Args:
        tmp_path: pytest tmp_path for file creation.
        user_messages: List of user message strings.
        write_files: File paths for Write tool_use blocks.
        edit_files: File paths for Edit tool_use blocks.

    Returns:
        Path to the created JSONL file.
    """
    now = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
    entries: list[dict[str, Any]] = []

    # Add Write/Edit tool calls as early assistant messages
    tool_uses: list[dict[str, Any]] = []
    for fp in (write_files or []):
        tool_uses.append({
            "type": "tool_use",
            "id": f"tool_{len(tool_uses)}",
            "name": "Write",
            "input": {"file_path": fp, "content": "..."},
        })
    for fp in (edit_files or []):
        tool_uses.append({
            "type": "tool_use",
            "id": f"tool_{len(tool_uses)}",
            "name": "Edit",
            "input": {"file_path": fp, "old_string": "a", "new_string": "b"},
        })

    if tool_uses:
        entries.append({
            "timestamp": now.isoformat(),
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-5-20250929",
                "content": tool_uses,
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
        })

    # Add user messages with interleaved assistant replies
    from datetime import timedelta
    for i, msg in enumerate(user_messages):
        ts = (now + timedelta(minutes=i * 2 + 1)).isoformat()
        entries.append({
            "timestamp": ts,
            "message": {"role": "user", "content": msg},
        })
        ts_reply = (now + timedelta(minutes=i * 2 + 2)).isoformat()
        entries.append({
            "timestamp": ts_reply,
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-5-20250929",
                "content": [{"type": "text", "text": f"Reply {i + 1}"}],
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
        })

    session = tmp_path / "test-session.jsonl"
    session.write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n"
    )
    return session


# -------------------------------------------------------------------------
# Auto-metadata sampling integration tests
# -------------------------------------------------------------------------

class TestAutoMetadataSampling:
    """Tests for meta-message filtering and artefact collection in
    :func:`generate_auto_metadata`."""

    _STATS: dict[str, Any] = {
        "turns": 10, "duration_minutes": 30,
        "tool_calls": {"total": 5, "by_type": {"Read": 5}},
    }

    @staticmethod
    def _mock_haiku_response(title: str = "Test Title") -> Any:
        """Build a mock Anthropic messages.create response."""
        from unittest.mock import MagicMock

        response = MagicMock()
        content_block = MagicMock()
        content_block.text = json.dumps({
            "title": title,
            "purpose": "Test purpose",
            "tags": ["test"],
        })
        response.content = [content_block]
        return response

    def test_filters_slash_commands(self, tmp_path: Path) -> None:
        """Slash commands are excluded from sampled messages."""
        session = _build_session_jsonl(tmp_path, [
            "Implement the parser",
            "Add error handling",
            "/recap",
            "/done",
        ])
        with patch("anthropic.Anthropic") as MockClient:
            mock_client = MockClient.return_value
            mock_client.api_key = "test-key"
            mock_client.messages.create.return_value = (
                self._mock_haiku_response("Parser Implementation")
            )
            result = generate_auto_metadata(session, self._STATS)
            assert result is not None
            assert result["title"] == "Parser Implementation"
            # Verify the prompt sent to Haiku excludes /recap and /done
            call_args = mock_client.messages.create.call_args
            prompt_text = call_args.kwargs["messages"][0]["content"]
            assert "/recap" not in prompt_text
            assert "/done" not in prompt_text
            assert "Implement the parser" in prompt_text

    def test_fallback_when_all_meta(self, tmp_path: Path) -> None:
        """All-meta sessions fall back to unfiltered messages."""
        session = _build_session_jsonl(tmp_path, [
            "/recap",
            "yes",
            "ok",
        ])
        with patch("anthropic.Anthropic") as MockClient:
            mock_client = MockClient.return_value
            mock_client.api_key = "test-key"
            mock_client.messages.create.return_value = (
                self._mock_haiku_response("Fallback Session")
            )
            result = generate_auto_metadata(session, self._STATS)
            # Should succeed — messages exist after fallback
            assert result is not None

    def test_collects_write_edit_file_paths(
        self, tmp_path: Path
    ) -> None:
        """Write/Edit file paths are collected during JSONL parsing."""
        session = _build_session_jsonl(
            tmp_path,
            ["Refactor the config module"],
            write_files=["/home/user/project/config.py"],
            edit_files=["/home/user/project/cli.py"],
        )
        with patch("anthropic.Anthropic") as MockClient:
            mock_client = MockClient.return_value
            mock_client.api_key = "test-key"
            mock_client.messages.create.return_value = (
                self._mock_haiku_response("Config Refactor")
            )
            result = generate_auto_metadata(session, self._STATS)
            assert result is not None
            # Verify file paths appear in the prompt
            call_args = mock_client.messages.create.call_args
            prompt_text = call_args.kwargs["messages"][0]["content"]
            assert "config.py" in prompt_text
            assert "cli.py" in prompt_text

    def test_empty_after_filtering(self, tmp_path: Path) -> None:
        """Sessions with only meta messages still parse without error."""
        session = _build_session_jsonl(tmp_path, [
            "yes", "ok", "sure", "/done",
        ])
        with patch("anthropic.Anthropic") as MockClient:
            mock_client = MockClient.return_value
            mock_client.api_key = "test-key"
            mock_client.messages.create.return_value = (
                self._mock_haiku_response("Short Session")
            )
            result = generate_auto_metadata(session, self._STATS)
            # All messages are short/meta, but fallback includes them
            assert result is not None


# -------------------------------------------------------------------------
# Project detection from CWD
# -------------------------------------------------------------------------

class TestDetectProjectNameFromCwd:
    """Tests for :func:`detect_project_name_from_cwd`."""

    def test_detects_from_project_root(
        self, tmp_project: Path
    ) -> None:
        """Finds project name when CWD is a project root."""
        name = detect_project_name_from_cwd(tmp_project)
        assert name == "test-project"

    def test_detects_from_subdirectory(
        self, tmp_project: Path
    ) -> None:
        """Finds project name from a subdirectory of a project."""
        sub = tmp_project / "src" / "deep"
        sub.mkdir(parents=True)
        name = detect_project_name_from_cwd(sub)
        assert name == "test-project"

    def test_falls_back_to_directory_name(
        self, tmp_path: Path
    ) -> None:
        """Falls back to CWD name when no project markers exist."""
        bare_dir = tmp_path / "my-standalone-dir"
        bare_dir.mkdir()
        name = detect_project_name_from_cwd(bare_dir)
        assert name == "my-standalone-dir"


# -------------------------------------------------------------------------
# Global archive paths
# -------------------------------------------------------------------------

class TestGlobalArchivePaths:
    """Tests for global archive directory and catalogue functions."""

    def test_default_archive_dir(self) -> None:
        """Default archive root is ~/cc-archives/."""
        result = get_global_archive_dir()
        assert result == DEFAULT_ARCHIVE_ROOT
        assert result.name == "cc-archives"

    def test_custom_archive_dir(self, tmp_path: Path) -> None:
        """Custom archive root overrides default."""
        custom = tmp_path / "my-archives"
        result = get_global_archive_dir(custom)
        assert result == custom

    def test_default_catalogue_file(self) -> None:
        """Default catalogue is at ~/cc-archives/CATALOG.json."""
        result = get_global_catalogue_file()
        assert result == DEFAULT_ARCHIVE_ROOT / "CATALOG.json"

    def test_custom_catalogue_file(self, tmp_path: Path) -> None:
        """Custom archive root places CATALOG.json within it."""
        custom = tmp_path / "my-archives"
        result = get_global_catalogue_file(custom)
        assert result == custom / "CATALOG.json"


# -------------------------------------------------------------------------
# Config defaults
# -------------------------------------------------------------------------

class TestConfigDefaults:
    """Tests for Phase 1 configuration constants."""

    def test_min_turns_is_five(self) -> None:
        assert DEFAULT_MIN_TURNS == 5

    def test_min_duration_is_one(self) -> None:
        assert DEFAULT_MIN_DURATION_MINUTES == 1

    def test_archive_root_is_under_home(self) -> None:
        assert DEFAULT_ARCHIVE_ROOT.parent == Path.home()


# -------------------------------------------------------------------------
# Archive session with global root
# -------------------------------------------------------------------------

class TestArchiveSessionGlobalRoot:
    """Tests for :func:`archive_session` with ``archive_root`` param."""

    # noinspection DuplicatedCode
    def _make_session_with_turns(
        self, tmp_path: Path, *, turns: int = 10
    ) -> Path:
        """Create a session JSONL file with enough turns to pass filter."""
        from datetime import timedelta

        session_file = tmp_path / "abc12345-1234-5678-9abc-def012345678.jsonl"
        now = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
        entries: list[dict[str, Any]] = []

        for i in range(turns):
            # User message
            ts = (now + timedelta(minutes=i * 2)).isoformat()
            entries.append({
                "timestamp": ts,
                "message": {
                    "role": "user",
                    "content": f"User message {i + 1}",
                },
            })
            # Assistant message
            ts_reply = (now + timedelta(minutes=i * 2 + 1)).isoformat()
            entries.append({
                "timestamp": ts_reply,
                "message": {
                    "role": "assistant",
                    "model": "claude-sonnet-4-5-20250929",
                    "content": [
                        {"type": "text", "text": f"Reply {i + 1}"},
                    ],
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                    },
                },
            })

        lines = [json.dumps(entry) for entry in entries]
        session_file.write_text("\n".join(lines) + "\n")
        return session_file

    def test_archives_to_global_root(self, tmp_path: Path) -> None:
        """archive_session with archive_root writes to that root."""
        from cc_session_toolkit.archive import archive_session

        session = self._make_session_with_turns(tmp_path)
        archive_root = tmp_path / "cc-archives"

        result = archive_session(
            session,
            archive_root=archive_root,
            project_name_override="test-project",
            stats_only=True,
            use_gzip=False,
        )

        assert result is not None
        # Check metadata has the right project name
        assert result["project"]["name"] == "test-project"
        # Check archive was written under archive_root
        project_dir = archive_root / "test-project"
        assert project_dir.exists()
        session_dirs = list(project_dir.iterdir())
        assert len(session_dirs) == 1
        meta_file = session_dirs[0] / "session.meta.json"
        assert meta_file.exists()

    def test_capture_type_in_metadata(self, tmp_path: Path) -> None:
        """capture_type is recorded in the archive section."""
        from cc_session_toolkit.archive import archive_session

        session = self._make_session_with_turns(tmp_path)
        archive_root = tmp_path / "cc-archives"

        result = archive_session(
            session,
            archive_root=archive_root,
            project_name_override="test-project",
            stats_only=True,
            capture_type="pre_compact",
        )

        assert result is not None
        assert result["archive"]["capture_type"] == "pre_compact"

    def test_session_id_override(self, tmp_path: Path) -> None:
        """session_id_override takes priority over filename."""
        from cc_session_toolkit.archive import archive_session

        session = self._make_session_with_turns(tmp_path)
        archive_root = tmp_path / "cc-archives"
        custom_id = "custom-override-id"

        result = archive_session(
            session,
            archive_root=archive_root,
            project_name_override="test-project",
            stats_only=True,
            session_id_override=custom_id,
        )

        assert result is not None
        assert result["session"]["id"] == custom_id

    def test_gzip_compression(self, tmp_path: Path) -> None:
        """Gzip compression creates .jsonl.gz in archive."""
        from cc_session_toolkit.archive import archive_session

        session = self._make_session_with_turns(tmp_path)
        archive_root = tmp_path / "cc-archives"

        result = archive_session(
            session,
            archive_root=archive_root,
            project_name_override="test-project",
            stats_only=True,
            use_gzip=True,
        )

        assert result is not None
        project_dir = archive_root / "test-project"
        session_dirs = list(project_dir.iterdir())
        gz_files = list(session_dirs[0].glob("*.jsonl.gz"))
        assert len(gz_files) == 1
        assert result["archive"]["jsonl_path"] == "session.jsonl.gz"

    def test_archive_directory_stored_in_metadata(
        self, tmp_path: Path
    ) -> None:
        """Bug #2b: _archive_directory is stored in returned metadata."""
        from cc_session_toolkit.archive import archive_session

        session = self._make_session_with_turns(tmp_path)
        archive_root = tmp_path / "cc-archives"

        result = archive_session(
            session,
            archive_root=archive_root,
            project_name_override="test-project",
            stats_only=True,
        )

        assert result is not None
        assert "_archive_directory" in result
        actual_dir = Path(result["_archive_directory"])
        assert actual_dir.exists()
        assert actual_dir.is_dir()

    def test_catalogue_uses_actual_directory(
        self, tmp_path: Path
    ) -> None:
        """Bug #2b: catalogue 'directory' field matches real path."""
        from cc_session_toolkit.archive import archive_session
        from cc_session_toolkit.catalogue import update_catalogue

        session = self._make_session_with_turns(tmp_path)
        archive_root = tmp_path / "cc-archives"

        result = archive_session(
            session,
            archive_root=archive_root,
            project_name_override="test-project",
            stats_only=True,
        )

        assert result is not None
        # Save before update_catalogue pops the transient key
        actual_dir = Path(result["_archive_directory"])

        catalogue_file = archive_root / "CATALOG.json"
        update_catalogue(
            [result], catalogue_file, archive_root, "test-project"
        )

        catalogue = json.loads(catalogue_file.read_text())
        cat_dir = catalogue["sessions"][0]["directory"]
        expected_rel = str(actual_dir.relative_to(archive_root))
        assert cat_dir == expected_rel

    def test_no_interactive_prompt_in_hook_mode(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """Bug #1: stats_only=True suppresses interactive prompt even
        when auto_metadata=True and Haiku is unavailable."""
        from cc_session_toolkit.archive import archive_session

        session = self._make_session_with_turns(tmp_path)
        archive_root = tmp_path / "cc-archives"

        result = archive_session(
            session,
            archive_root=archive_root,
            project_name_override="test-project",
            stats_only=True,
            auto_metadata=True,  # will fail (no anthropic in test)
        )

        assert result is not None
        captured = capsys.readouterr()
        # Must NOT contain the interactive metadata prompt
        assert "METADATA GENERATION" not in captured.out


# -------------------------------------------------------------------------
# CLI from-hook integration
# -------------------------------------------------------------------------

class TestCliFromHook:
    """Integration tests for ``cc-session archive --from-hook``."""

    def _run_cli(
        self,
        args: list[str],
        stdin_data: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> int:
        """Run the CLI with monkeypatched argv and stdin."""
        import io

        from cc_session_toolkit.cli import main

        monkeypatch.setattr(
            "sys.argv", ["cc-session"] + args
        )
        monkeypatch.setattr(
            "sys.stdin", io.StringIO(stdin_data)
        )
        try:
            main()
            return 0
        except SystemExit as exc:
            return exc.code if exc.code else 0

    def test_rejects_empty_stdin(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """Empty stdin should produce an error."""
        code = self._run_cli(
            ["archive", "--from-hook"], "", monkeypatch
        )
        assert code != 0
        captured = capsys.readouterr()
        assert "no input on stdin" in captured.out

    def test_rejects_invalid_json(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid JSON on stdin should produce an error."""
        code = self._run_cli(
            ["archive", "--from-hook"], "not json", monkeypatch
        )
        assert code != 0

    def test_rejects_missing_session_id(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """Missing session_id in hook input should produce an error."""
        code = self._run_cli(
            ["archive", "--from-hook"],
            json.dumps({"cwd": "/tmp"}),
            monkeypatch,
        )
        assert code != 0
        captured = capsys.readouterr()
        assert "session_id" in captured.out

    def test_skips_trivial_session(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Sessions with too few turns are skipped."""
        # Create a trivial session (1 turn only)
        now = datetime.now(tz=timezone.utc)
        entries = [
            {
                "timestamp": now.isoformat(),
                "message": {
                    "role": "user",
                    "content": "Hello",
                },
            },
            {
                "timestamp": now.isoformat(),
                "message": {
                    "role": "assistant",
                    "model": "claude-sonnet-4-5-20250929",
                    "content": [{"type": "text", "text": "Hi"}],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                    },
                },
            },
        ]
        session = tmp_path / "trivial.jsonl"
        session.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n"
        )

        hook_input = json.dumps({
            "session_id": "trivial-session-id",
            "transcript_path": str(session),
            "cwd": str(tmp_path),
        })

        code = self._run_cli(
            ["archive", "--from-hook"], hook_input, monkeypatch
        )
        assert code == 0
        captured = capsys.readouterr()
        assert "Skipping trivial" in captured.out

    def test_skips_duplicate_session(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Sessions already in the catalogue are skipped."""
        from datetime import timedelta

        archive_root = tmp_path / "cc-archives"
        archive_root.mkdir()

        # Pre-populate catalogue with the session ID
        catalogue = archive_root / "CATALOG.json"
        catalogue.write_text(json.dumps({
            "schema_version": "1.1",
            "sessions": [{"id": "existing-id"}],
        }))

        # Create a session with enough turns to pass trivial filter
        now = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
        entries: list[dict[str, Any]] = []
        for i in range(6):
            ts = (now + timedelta(minutes=i * 2)).isoformat()
            entries.append({
                "timestamp": ts,
                "message": {
                    "role": "user",
                    "content": f"Message {i + 1}",
                },
            })
            ts_reply = (now + timedelta(minutes=i * 2 + 1)).isoformat()
            entries.append({
                "timestamp": ts_reply,
                "message": {
                    "role": "assistant",
                    "model": "claude-sonnet-4-5-20250929",
                    "content": [
                        {"type": "text", "text": f"Reply {i + 1}"},
                    ],
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                    },
                },
            })

        session = tmp_path / "session.jsonl"
        session.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n"
        )

        hook_input = json.dumps({
            "session_id": "existing-id",
            "transcript_path": str(session),
            "cwd": str(tmp_path),
        })

        code = self._run_cli(
            ["archive", "--from-hook", "--archive-root", str(archive_root)],
            hook_input,
            monkeypatch,
        )
        assert code == 0
        captured = capsys.readouterr()
        assert "already-archived" in captured.out
