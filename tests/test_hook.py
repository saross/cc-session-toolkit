"""Tests for hook-based automated archiving (Phase 1)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from cc_session_toolkit.archive import (
    _build_auto_metadata_user_message,
    _ensure_gemini_api_key,
    _load_auto_metadata_prompt,
    _parse_metadata_response_json,
    generate_auto_metadata,
    is_already_archived,
    is_trivial_session,
)
from cc_session_toolkit.config import (
    AUTO_METADATA_MAX_OUTPUT_TOKENS,
    DEFAULT_ARCHIVE_ROOT,
    DEFAULT_MIN_DURATION_MINUTES,
    DEFAULT_MIN_TURNS,
    EXTRACTOR_MODEL_ID,
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
    """Tests for :func:`generate_auto_metadata` (Gemini Flex path)."""

    _STATS: dict[str, Any] = {
        "turns": 10, "duration_minutes": 30,
        "tool_calls": {"total": 5, "by_type": {"Read": 5}},
        "session_id": "test-session-abc",
        "project_name": "test-project",
        "started_at": "2026-05-18T10:00:00+00:00",
    }

    def test_returns_none_without_google_genai(
        self, sample_session_jsonl: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Falls back gracefully when google-genai is not installed."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "google" or name.startswith("google."):
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        result = generate_auto_metadata(sample_session_jsonl, self._STATS)
        assert result is None

    def test_returns_none_without_api_key(
        self,
        sample_session_jsonl: Path,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Returns None when no GEMINI_API_KEY / GOOGLE_API_KEY is available."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        # Point the .env-sniffing fallback at an empty file so it
        # doesn't accidentally pick up the real PA .env.
        empty_home = tmp_path / "fake-home"
        (empty_home / "personal-assistant").mkdir(parents=True)
        (empty_home / "personal-assistant" / ".env").write_text("")
        monkeypatch.setattr(Path, "home", lambda: empty_home)
        result = generate_auto_metadata(sample_session_jsonl, self._STATS)
        assert result is None

    def test_returns_none_for_empty_session(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Returns None when the distilled transcript is empty.

        Skip when ``google-genai`` is not installed: without the import
        the function would early-return ``None`` from the ImportError
        branch and the test would pass for the wrong reason — never
        reaching the empty-transcript code path this test is meant to
        exercise.
        """
        pytest.importorskip("google.genai")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        empty_session = tmp_path / "empty.jsonl"
        empty_session.write_text("")
        result = generate_auto_metadata(empty_session, self._STATS)
        assert result is None


# -------------------------------------------------------------------------
# Helper-level unit tests for the Gemini path
# -------------------------------------------------------------------------

class TestAutoMetadataHelpers:
    """Tests for the Gemini-path helper functions."""

    def test_load_prompt_from_package_data(self) -> None:
        """The bundled prompt loads via importlib.resources."""
        prompt = _load_auto_metadata_prompt()
        assert "Session Metadata Extraction Prompt" in prompt
        # Spot-check known sections from prompt-gemini-v2.md.
        assert "Specifics requirement" in prompt
        assert "three_ps" in prompt or "three Ps" in prompt.lower()

    def test_load_prompt_override_via_env(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``CC_AUTO_METADATA_PROMPT_PATH`` overrides the bundled prompt."""
        custom = tmp_path / "custom-prompt.md"
        custom.write_text("# custom-prompt-sentinel\nbody\n")
        monkeypatch.setenv("CC_AUTO_METADATA_PROMPT_PATH", str(custom))
        prompt = _load_auto_metadata_prompt()
        assert "custom-prompt-sentinel" in prompt

    def test_build_user_message_wraps_transcript_in_tags(self) -> None:
        """The user message wraps the transcript in ``<transcript>`` tags."""
        msg = _build_auto_metadata_user_message(
            session_id="sid",
            project="proj",
            started_at="2026-05-18T10:00:00",
            content_tokens=1234,
            transcript_text="--- User ---\nHello",
        )
        assert "<transcript>" in msg
        assert "</transcript>" in msg
        # Output reminder lives AFTER the closing tag so recency
        # favours the contract.
        assert msg.index("</transcript>") < msg.index("Output reminder")
        # Square-bracket chat markers must NOT appear — they confuse the
        # extractor into continuing the conversation.
        assert "[user]" not in msg.lower()
        assert "[assistant]" not in msg.lower()

    def test_build_user_message_includes_header_fields(self) -> None:
        """Session header carries the identifying fields."""
        msg = _build_auto_metadata_user_message(
            session_id="abc-123",
            project="my-proj",
            started_at="2026-05-18T10:00:00",
            content_tokens=42,
            transcript_text="--- User ---\nx",
        )
        assert "abc-123" in msg
        assert "my-proj" in msg
        assert "2026-05-18" in msg
        assert "42" in msg

    def test_parse_response_handles_bare_json(self) -> None:
        """Bare JSON parses cleanly."""
        result = _parse_metadata_response_json('{"title": "t", "tags": []}')
        assert result == {"title": "t", "tags": []}

    def test_parse_response_strips_code_fences(self) -> None:
        """A fenced ```json block is stripped and parsed."""
        raw = '```json\n{"title": "fenced"}\n```'
        result = _parse_metadata_response_json(raw)
        assert result == {"title": "fenced"}

    def test_parse_response_raises_on_malformed(self) -> None:
        """Malformed JSON raises ValueError so callers can record raw text."""
        with pytest.raises(ValueError):
            _parse_metadata_response_json("not json at all")

    def test_ensure_gemini_api_key_prefers_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Existing ``GEMINI_API_KEY`` takes precedence over the .env file."""
        monkeypatch.setenv("GEMINI_API_KEY", "env-value")
        assert _ensure_gemini_api_key() == "env-value"

    def test_ensure_gemini_api_key_falls_back_to_google(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``GOOGLE_API_KEY`` is accepted when GEMINI_API_KEY is unset."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "google-value")
        assert _ensure_gemini_api_key() == "google-value"


# -------------------------------------------------------------------------
# Session-JSONL builder + Gemini integration tests
# -------------------------------------------------------------------------

def _build_session_jsonl(
    tmp_path: Path,
    user_messages: list[str],
) -> Path:
    """Build a minimal session JSONL fixture with the given user messages.

    The Gemini path reads the full distilled transcript, so we no longer
    need elaborate Write/Edit tool_use scaffolding here — tests just need
    enough user-role text to produce a non-empty distillation.
    """
    from datetime import timedelta

    now = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
    entries: list[dict[str, Any]] = []
    for i, msg in enumerate(user_messages):
        ts_user = (now + timedelta(minutes=i * 2)).isoformat()
        entries.append({
            "type": "user",
            "timestamp": ts_user,
            "message": {"role": "user", "content": msg},
        })
        ts_reply = (now + timedelta(minutes=i * 2 + 1)).isoformat()
        entries.append({
            "type": "assistant",
            "timestamp": ts_reply,
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-5-20250929",
                "content": [{"type": "text", "text": f"Reply {i + 1}"}],
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
        })

    session = tmp_path / "test-session.jsonl"
    session.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return session


class TestAutoMetadataGeminiIntegration:
    """Integration tests for :func:`generate_auto_metadata` with a mocked Gemini client."""

    _STATS: dict[str, Any] = {
        "turns": 10,
        "duration_minutes": 30,
        "tool_calls": {"total": 5, "by_type": {"Read": 5}},
        "session_id": "integration-test-001",
        "project_name": "test-project",
        "started_at": "2026-05-18T10:00:00+00:00",
    }

    @staticmethod
    def _mock_gemini_response(payload: dict[str, Any]) -> Any:
        """Build a minimal mock of ``client.models.generate_content``'s return."""
        from unittest.mock import MagicMock

        response = MagicMock()
        response.text = json.dumps(payload)
        return response

    def test_happy_path_returns_parsed_metadata(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A clean response yields a populated auto_meta dict."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        session = _build_session_jsonl(
            tmp_path,
            [
                "Implement the parser for CSV files",
                "Add error handling to the validation module",
            ],
        )
        mock_payload = {
            "title": "CSV Parser Implementation",
            "purpose": "Build a robust CSV parsing layer",
            "tags": ["csv", "parser", "implementation"],
            "three_ps": {
                "prompt_summary": "User asked for a CSV parser",
                "process_summary": "Iterative implementation with tests",
                "provenance_summary": "Part of the data-pipeline rewrite",
            },
        }
        with patch("google.genai.Client") as MockClient:
            mock_client = MockClient.return_value
            mock_client.models.generate_content.return_value = (
                self._mock_gemini_response(mock_payload)
            )
            result = generate_auto_metadata(session, self._STATS)

        assert result is not None
        assert result["title"] == "CSV Parser Implementation"
        assert result["purpose"] == "Build a robust CSV parsing layer"
        assert result["tags"] == ["csv", "parser", "implementation"]
        assert result["three_ps"]["prompt_summary"] == (
            "User asked for a CSV parser"
        )

    def test_call_shape_passes_required_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify the Gemini call uses thinking_budget=0, Flex tier, and the production model."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        session = _build_session_jsonl(tmp_path, ["A substantive message"])
        with patch("google.genai.Client") as MockClient:
            mock_client = MockClient.return_value
            mock_client.models.generate_content.return_value = (
                self._mock_gemini_response({"title": "x", "tags": []})
            )
            generate_auto_metadata(session, self._STATS)
            call_args = mock_client.models.generate_content.call_args

        assert call_args.kwargs["model"] == EXTRACTOR_MODEL_ID
        cfg = call_args.kwargs["config"]
        assert cfg["service_tier"] == "flex"
        assert cfg["thinking_config"] == {"thinking_budget": 0}
        assert cfg["max_output_tokens"] == AUTO_METADATA_MAX_OUTPUT_TOKENS
        # System prompt carries the role + contracts; user message carries
        # the delimited transcript.
        assert "Session Metadata Extraction" in cfg["system_instruction"]
        user_content = call_args.kwargs["contents"]
        assert "<transcript>" in user_content
        assert "</transcript>" in user_content

    def test_returns_none_on_persistent_503(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Exhausted 503 retries collapse to None gracefully."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        # Don't actually sleep during retries.
        monkeypatch.setattr(
            "cc_session_toolkit.archive.AUTO_METADATA_FLEX_RETRY_WAITS_SECONDS",
            (0, 0, 0),
        )
        session = _build_session_jsonl(tmp_path, ["Substantive message"])

        with patch("google.genai.Client") as MockClient:
            mock_client = MockClient.return_value
            mock_client.models.generate_content.side_effect = RuntimeError(
                "503 Service Unavailable: preempted"
            )
            result = generate_auto_metadata(session, self._STATS)
            # Initial attempt plus three retries (one per wait in the
            # patched ``(0, 0, 0)`` tuple) — total 4 calls before
            # collapsing to None. Without this assertion a bug that
            # broke out of the retry loop after a single attempt would
            # still produce a None result and the test would still pass.
            assert mock_client.models.generate_content.call_count == 4
        assert result is None

    def test_503_retry_recovers(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A 503 followed by success returns the parsed metadata."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr(
            "cc_session_toolkit.archive.AUTO_METADATA_FLEX_RETRY_WAITS_SECONDS",
            (0, 0, 0),
        )
        session = _build_session_jsonl(tmp_path, ["Substantive message"])

        success_response = self._mock_gemini_response(
            {"title": "Recovered", "tags": []}
        )
        with patch("google.genai.Client") as MockClient:
            mock_client = MockClient.return_value
            mock_client.models.generate_content.side_effect = [
                RuntimeError("503 Service Unavailable"),
                success_response,
            ]
            result = generate_auto_metadata(session, self._STATS)
            # First attempt raises 503, second attempt succeeds — so
            # exactly two calls. Without this assertion a bug that
            # short-circuited the retry loop would produce a misleading
            # green result.
            assert mock_client.models.generate_content.call_count == 2
        assert result is not None
        assert result["title"] == "Recovered"

    def test_non_503_error_propagates_to_none(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Non-503 errors don't retry; auto-metadata gives up cleanly."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        session = _build_session_jsonl(tmp_path, ["Substantive message"])

        with patch("google.genai.Client") as MockClient:
            mock_client = MockClient.return_value
            mock_client.models.generate_content.side_effect = RuntimeError(
                "400 invalid argument"
            )
            result = generate_auto_metadata(session, self._STATS)
        assert result is None

    def test_unparseable_json_collapses_to_none(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Response that doesn't parse as JSON yields None, not an exception."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        session = _build_session_jsonl(tmp_path, ["Substantive message"])

        bad_response = type("MockResp", (), {"text": "definitely not json"})()
        with patch("google.genai.Client") as MockClient:
            mock_client = MockClient.return_value
            mock_client.models.generate_content.return_value = bad_response
            result = generate_auto_metadata(session, self._STATS)
        assert result is None


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
        when auto_metadata=True and the metadata extractor is unavailable."""
        from cc_session_toolkit.archive import archive_session

        session = self._make_session_with_turns(tmp_path)
        archive_root = tmp_path / "cc-archives"

        result = archive_session(
            session,
            archive_root=archive_root,
            project_name_override="test-project",
            stats_only=True,
            auto_metadata=True,  # will fail (no google-genai key in test)
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
        # Create a trivial session (1 turn only). Use the fixed
        # timestamp convention shared by sibling tests in this file
        # (``datetime(2026, 3, 15, ...)``) rather than ``datetime.now``
        # so the fixture is deterministic across runs.
        now = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
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


class TestCliArchiveGlobal:
    """
    Tests for ``cc-session archive --archive-root ...`` (global mode).

    Global mode is the cross-project bulk-backfill counterpart to
    ``--from-hook``: it scans every ``~/.claude/projects/<project>/``
    for live JSONLs and writes archives to the supplied ``archive_root``
    with the project name derived per-session from each transcript's
    ``cwd`` field.
    """

    def _run_cli_no_stdin(
        self,
        args: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> int:
        """Run the CLI with monkeypatched argv (no stdin needed)."""
        from cc_session_toolkit.cli import main

        monkeypatch.setattr("sys.argv", ["cc-session"] + args)
        try:
            main()
            return 0
        except SystemExit as exc:
            return exc.code if exc.code else 0

    def _make_session(
        self,
        live_dir: Path,
        sid: str,
        cwd: str,
        turns: int = 6,
    ) -> Path:
        """
        Create a fake live JSONL with *turns* user/assistant pairs.

        The first entry carries the ``cwd`` field that
        ``_extract_cwd_from_jsonl`` reads. Each pair adds 1 min between
        entries so the session passes the trivial-session filter.
        """
        from datetime import timedelta

        live_dir.mkdir(parents=True, exist_ok=True)
        start = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
        entries: list[dict[str, Any]] = []
        for i in range(turns):
            ts_user = (start + timedelta(minutes=i * 2)).isoformat()
            entries.append({
                "timestamp": ts_user,
                "cwd": cwd,
                "message": {
                    "role": "user",
                    "content": f"User msg {i + 1}",
                },
            })
            ts_asst = (start + timedelta(minutes=i * 2 + 1)).isoformat()
            entries.append({
                "timestamp": ts_asst,
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
        jsonl = live_dir / f"{sid}.jsonl"
        jsonl.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        return jsonl

    def test_archive_root_dispatches_to_global_mode(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        ``--archive-root`` routes to ``_cmd_archive_global``, not the
        per-project path. Verified by spying on the dispatcher.
        """
        # Set HOME so Path.home() under test points at tmp_path
        monkeypatch.setenv("HOME", str(tmp_path))

        # Create a single session under the mock CC live store
        live_dir = tmp_path / ".claude" / "projects" / "-home-shawn-foo"
        self._make_session(live_dir, "sid-001", str(tmp_path / "foo"))

        called = {"count": 0}

        def fake_global(_args: Any) -> None:
            called["count"] += 1

        monkeypatch.setattr(
            "cc_session_toolkit.cli._cmd_archive_global", fake_global
        )

        archive_root = tmp_path / "cc-archives"
        code = self._run_cli_no_stdin(
            ["archive", "--all", "--archive-root", str(archive_root)],
            monkeypatch,
        )
        assert code == 0
        assert called["count"] == 1

    def test_global_mode_scans_all_projects(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Global mode discovers sessions across multiple project dirs."""
        monkeypatch.setenv("HOME", str(tmp_path))

        # Two projects, two sessions each. Test ids deliberately have
        # distinct 8-char prefixes — the archive uses the prefix as
        # part of the session-dir name, so collisions there would mask
        # a real multi-project bug. (Not real UUIDs; just non-colliding
        # filenames for `get_session_id` via `Path.stem`.)
        for proj_dir, cwd_str, sid in [
            ("-home-shawn-foo", str(tmp_path / "foo"),
             "aaaaaaaa-foo1-0000-0000-000000000001"),
            ("-home-shawn-foo", str(tmp_path / "foo"),
             "bbbbbbbb-foo2-0000-0000-000000000002"),
            ("-home-shawn-bar", str(tmp_path / "bar"),
             "cccccccc-bar1-0000-0000-000000000003"),
            ("-home-shawn-bar", str(tmp_path / "bar"),
             "dddddddd-bar2-0000-0000-000000000004"),
        ]:
            live_dir = tmp_path / ".claude" / "projects" / proj_dir
            self._make_session(live_dir, sid, cwd_str)

        archive_root = tmp_path / "cc-archives"

        code = self._run_cli_no_stdin(
            ["archive", "--all", "--gzip",
             "--archive-root", str(archive_root)],
            monkeypatch,
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "Archiving 4 session(s)" in out
        # Each session should land under archive_root/<project_name>/
        # detect_project_name_from_cwd falls back to the cwd's
        # directory basename when no project markers are present, so
        # we expect "foo" and "bar" subtrees.
        assert (archive_root / "foo").is_dir()
        assert (archive_root / "bar").is_dir()
        # Two session-dirs per project
        assert len(list((archive_root / "foo").iterdir())) == 2
        assert len(list((archive_root / "bar").iterdir())) == 2

    def test_global_mode_passes_auto_metadata_to_archive_session(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Regression: ``--auto-metadata`` must flow through to
        ``archive_session`` in global mode (it was silently dropped
        before the 2026-05-22 cli.py patch).
        """
        monkeypatch.setenv("HOME", str(tmp_path))
        live_dir = tmp_path / ".claude" / "projects" / "-home-shawn-foo"
        self._make_session(live_dir, "sid-001", str(tmp_path / "foo"))

        captured_kwargs: dict[str, Any] = {}

        def fake_archive_session(*_args: Any, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)
            return None

        monkeypatch.setattr(
            "cc_session_toolkit.cli.archive_session", fake_archive_session
        )

        archive_root = tmp_path / "cc-archives"
        code = self._run_cli_no_stdin(
            ["archive", "--all", "--auto-metadata",
             "--archive-root", str(archive_root)],
            monkeypatch,
        )
        assert code == 0
        assert captured_kwargs.get("auto_metadata") is True
        assert captured_kwargs.get("archive_root") == archive_root

    def test_global_mode_applies_trivial_session_filter(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """
        Regression: trivial sessions (< min-turns) must be filtered in
        global mode, mirroring ``_cmd_archive_from_hook``. Without
        this, ``--all --auto-metadata`` fires a Gemini Flex call on
        every empty session in the live store.
        """
        monkeypatch.setenv("HOME", str(tmp_path))
        live_dir = tmp_path / ".claude" / "projects" / "-home-shawn-foo"

        # One trivial session (2 turns) + one non-trivial (6 turns)
        self._make_session(
            live_dir,
            "aaaaaaaa-trivial-0000-0000-000000000001",
            str(tmp_path / "foo"),
            turns=2,
        )
        self._make_session(
            live_dir,
            "bbbbbbbb-real-0000-0000-000000000002",
            str(tmp_path / "foo"),
            turns=6,
        )

        archive_root = tmp_path / "cc-archives"
        code = self._run_cli_no_stdin(
            ["archive", "--all", "--gzip",
             "--archive-root", str(archive_root)],
            monkeypatch,
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "Skipping trivial" in out
        assert "Archiving 1 session(s)" in out
        # Only the non-trivial session should land in the archive
        foo_dir = archive_root / "foo"
        assert foo_dir.is_dir()
        archived_dirs = list(foo_dir.iterdir())
        assert len(archived_dirs) == 1

    def test_global_mode_dry_run_does_not_update_catalogue(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Regression: ``--dry-run`` must NOT call ``update_catalogue``.
        A dry-run that mutates the catalogue would defeat the purpose
        of dry-running (state pollution from a "preview" command).
        """
        monkeypatch.setenv("HOME", str(tmp_path))
        live_dir = tmp_path / ".claude" / "projects" / "-home-shawn-foo"
        self._make_session(
            live_dir,
            "aaaaaaaa-foo1-0000-0000-000000000001",
            str(tmp_path / "foo"),
        )

        update_calls: list[Any] = []

        def fake_update(*args: Any, **kwargs: Any) -> None:
            update_calls.append((args, kwargs))

        monkeypatch.setattr(
            "cc_session_toolkit.cli.update_catalogue", fake_update
        )

        archive_root = tmp_path / "cc-archives"
        code = self._run_cli_no_stdin(
            ["archive", "--all", "--gzip", "--dry-run",
             "--archive-root", str(archive_root)],
            monkeypatch,
        )
        assert code == 0
        assert len(update_calls) == 0, (
            f"update_catalogue called {len(update_calls)} times "
            "during --dry-run; expected 0"
        )

    def test_global_mode_dedup_skips_already_archived(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """
        Sessions whose id is in the global catalogue should be skipped
        unless ``--force`` is set. Critical for safe re-runs of a
        production sweep — a second invocation must not double-archive
        and double-spend on Gemini Flex.
        """
        monkeypatch.setenv("HOME", str(tmp_path))
        live_dir = tmp_path / ".claude" / "projects" / "-home-shawn-foo"
        sid_known = "aaaaaaaa-known-0000-0000-000000000001"
        sid_new = "bbbbbbbb-newxx-0000-0000-000000000002"
        self._make_session(live_dir, sid_known, str(tmp_path / "foo"))
        self._make_session(live_dir, sid_new, str(tmp_path / "foo"))

        # Pre-populate the catalogue so the known id is already there
        archive_root = tmp_path / "cc-archives"
        archive_root.mkdir()
        (archive_root / "CATALOG.json").write_text(json.dumps({
            "schema_version": "1.1",
            "sessions": [{"id": sid_known}],
        }))

        code = self._run_cli_no_stdin(
            ["archive", "--all", "--gzip",
             "--archive-root", str(archive_root)],
            monkeypatch,
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "Archiving 1 session(s)" in out
        foo_dir = archive_root / "foo"
        assert foo_dir.is_dir()
        # Only the new (not pre-archived) session should land
        archived = list(foo_dir.iterdir())
        assert len(archived) == 1

    def test_global_mode_requires_all_or_session_id(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """
        Global mode without --all or --session-id is almost always a
        user mistake (the previous "lexicographically-last globally"
        default was surprising). The CLI should error rather than
        silently archive one session.
        """
        monkeypatch.setenv("HOME", str(tmp_path))
        live_dir = tmp_path / ".claude" / "projects" / "-home-shawn-foo"
        self._make_session(
            live_dir,
            "aaaaaaaa-foo1-0000-0000-000000000001",
            str(tmp_path / "foo"),
        )

        archive_root = tmp_path / "cc-archives"
        code = self._run_cli_no_stdin(
            ["archive", "--gzip", "--archive-root", str(archive_root)],
            monkeypatch,
        )
        assert code == 2
        err = capsys.readouterr().out
        assert "requires --all or --session-id" in err

    def test_global_mode_force_rearchives_existing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """
        With ``--force`` set, already-archived sessions are
        re-processed instead of skipped. Important for the
        re-run-after-prompt-iteration workflow.
        """
        monkeypatch.setenv("HOME", str(tmp_path))
        live_dir = tmp_path / ".claude" / "projects" / "-home-shawn-foo"
        sid_known = "aaaaaaaa-known-0000-0000-000000000001"
        self._make_session(live_dir, sid_known, str(tmp_path / "foo"))

        archive_root = tmp_path / "cc-archives"
        archive_root.mkdir()
        # Pre-populate catalogue so the session is "already archived"
        (archive_root / "CATALOG.json").write_text(json.dumps({
            "schema_version": "1.1",
            "sessions": [{"id": sid_known}],
        }))

        # Without --force: skipped
        code = self._run_cli_no_stdin(
            ["archive", "--all", "--gzip",
             "--archive-root", str(archive_root)],
            monkeypatch,
        )
        assert code == 0
        out_no_force = capsys.readouterr().out
        assert "No sessions to archive" in out_no_force

        # With --force: archived
        code = self._run_cli_no_stdin(
            ["archive", "--all", "--force", "--gzip",
             "--archive-root", str(archive_root)],
            monkeypatch,
        )
        assert code == 0
        out_force = capsys.readouterr().out
        assert "Archiving 1 session(s)" in out_force

    def test_global_mode_session_id_targets_one_session(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """
        ``--session-id X`` in global mode targets a single session by
        its UUID, regardless of which project_dir it lives under.
        Tests both the match and the not-found error path.
        """
        monkeypatch.setenv("HOME", str(tmp_path))
        live_dir_foo = (
            tmp_path / ".claude" / "projects" / "-home-shawn-foo"
        )
        live_dir_bar = (
            tmp_path / ".claude" / "projects" / "-home-shawn-bar"
        )
        target_sid = "aaaaaaaa-target-0000-0000-000000000001"
        other_sid = "bbbbbbbb-other-0000-0000-000000000002"
        self._make_session(live_dir_foo, target_sid, str(tmp_path / "foo"))
        self._make_session(live_dir_bar, other_sid, str(tmp_path / "bar"))

        archive_root = tmp_path / "cc-archives"

        # Match: only the target session is archived
        code = self._run_cli_no_stdin(
            ["archive", "--session-id", target_sid, "--gzip",
             "--archive-root", str(archive_root)],
            monkeypatch,
        )
        assert code == 0
        out_match = capsys.readouterr().out
        assert "Archiving 1 session(s)" in out_match
        assert (archive_root / "foo").is_dir()
        assert not (archive_root / "bar").exists()

        # Not found: error returned
        code = self._run_cli_no_stdin(
            ["archive", "--session-id", "no-such-sid", "--gzip",
             "--archive-root", str(archive_root)],
            monkeypatch,
        )
        assert code == 0
        out_miss = capsys.readouterr().out
        assert "Session not found: no-such-sid" in out_miss

    def test_global_mode_update_catalogue_called_once_per_project(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Regression: ``update_catalogue`` takes a single ``project_name``
        argument, so global mode groups results by project and calls
        it once per project. Verify the call count matches the number
        of distinct projects, not the total session count.

        A bug that called it once per session would n²-bloat writes on
        large sweeps. A bug that called it once total would silently
        overwrite all but the last project's catalogue contribution.
        """
        monkeypatch.setenv("HOME", str(tmp_path))
        # 4 sessions across 2 projects
        for proj_dir, cwd_str, sid in [
            ("-home-shawn-foo", str(tmp_path / "foo"),
             "aaaaaaaa-foo1-0000-0000-000000000001"),
            ("-home-shawn-foo", str(tmp_path / "foo"),
             "bbbbbbbb-foo2-0000-0000-000000000002"),
            ("-home-shawn-bar", str(tmp_path / "bar"),
             "cccccccc-bar1-0000-0000-000000000003"),
            ("-home-shawn-bar", str(tmp_path / "bar"),
             "dddddddd-bar2-0000-0000-000000000004"),
        ]:
            live_dir = tmp_path / ".claude" / "projects" / proj_dir
            self._make_session(live_dir, sid, cwd_str)

        update_calls: list[tuple[str, int]] = []

        def fake_update(
            results: list[Any],
            _cat: Any,
            _root: Any,
            project_name: str,
        ) -> None:
            update_calls.append((project_name, len(results)))

        monkeypatch.setattr(
            "cc_session_toolkit.cli.update_catalogue", fake_update
        )

        archive_root = tmp_path / "cc-archives"
        code = self._run_cli_no_stdin(
            ["archive", "--all", "--gzip",
             "--archive-root", str(archive_root)],
            monkeypatch,
        )
        assert code == 0
        # 2 distinct projects → exactly 2 catalogue calls
        assert len(update_calls) == 2
        project_names = sorted(name for name, _ in update_calls)
        assert project_names == ["bar", "foo"]
        # Each call should batch both sessions for that project
        counts = {name: n for name, n in update_calls}
        assert counts == {"foo": 2, "bar": 2}

    def test_global_mode_archive_root_with_tilde_is_expanded(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        ``--archive-root ~/cc-archives`` must be tilde-expanded before
        use. ``Path("~/cc-archives")`` doesn't expand `~` on its own.
        """
        # Use HOME=tmp_path so ~ expands predictably under test
        monkeypatch.setenv("HOME", str(tmp_path))
        live_dir = tmp_path / ".claude" / "projects" / "-home-shawn-foo"
        self._make_session(
            live_dir,
            "aaaaaaaa-foo1-0000-0000-000000000001",
            str(tmp_path / "foo"),
        )

        # Pass a tilde-prefixed path; expansion should resolve it
        # to tmp_path/cc-archives
        code = self._run_cli_no_stdin(
            ["archive", "--all", "--gzip",
             "--archive-root", "~/cc-archives"],
            monkeypatch,
        )
        assert code == 0
        # The archive should land under HOME-expanded path, not a
        # literal "~" directory
        assert (tmp_path / "cc-archives").is_dir()
        assert not (Path.cwd() / "~").exists()  # no literal-~ dir

    def test_global_mode_excludes_flat_agent_jsonls(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """
        ``agent-*.jsonl`` files at the top of a project dir are
        subagent transcripts; they are archived under their parent
        session's ``subagents/`` subdir during the main archive pass.
        Global mode must exclude them from discovery to avoid
        misrouting and double-counting. Matches the exclusion in
        ``archive.get_session_files``.
        """
        monkeypatch.setenv("HOME", str(tmp_path))
        live_dir = tmp_path / ".claude" / "projects" / "-home-shawn-foo"
        # One regular session
        self._make_session(
            live_dir,
            "aaaaaaaa-main-0000-0000-000000000001",
            str(tmp_path / "foo"),
        )
        # One agent-*.jsonl that should be ignored
        self._make_session(live_dir, "agent-deadbeef", str(tmp_path / "foo"))

        archive_root = tmp_path / "cc-archives"
        code = self._run_cli_no_stdin(
            ["archive", "--all", "--gzip",
             "--archive-root", str(archive_root)],
            monkeypatch,
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "Archiving 1 session(s)" in out
        foo_dir = archive_root / "foo"
        assert foo_dir.is_dir()
        assert len(list(foo_dir.iterdir())) == 1

    def test_per_project_mode_passes_auto_metadata_to_archive_session(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Regression: ``--auto-metadata`` must flow through in
        per-project mode too (it was silently dropped before the
        2026-05-22 cli.py patch). Same bug as global mode but in a
        different code path.
        """
        # Set HOME and create a fake project with .git so find_project_root works
        monkeypatch.setenv("HOME", str(tmp_path))
        proj = tmp_path / "myproject"
        proj.mkdir()
        (proj / ".git").mkdir()
        # CC live store sits under tmp_path/.claude/projects/<encoded>/
        live_dir = tmp_path / ".claude" / "projects" / "-tmp-myproject"
        self._make_session(live_dir, "sid-001", str(proj))

        # Per-project mode reads sessions via get_session_files(project_root).
        # Mock that to return our session directly, sidestepping the
        # CC live-store discovery path (which uses get_cc_project_path).
        sess_file = live_dir / "sid-001.jsonl"

        captured_kwargs: dict[str, Any] = {}

        def fake_archive_session(*_args: Any, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)
            return None

        monkeypatch.setattr(
            "cc_session_toolkit.cli.archive_session", fake_archive_session
        )
        monkeypatch.setattr(
            "cc_session_toolkit.cli.get_session_files",
            lambda _root: [sess_file],
        )
        monkeypatch.setattr(
            "cc_session_toolkit.cli.get_archived_session_ids",
            lambda _f: set(),
        )
        monkeypatch.setattr(
            "cc_session_toolkit.cli.find_project_root",
            lambda *_a, **_k: proj,
        )

        code = self._run_cli_no_stdin(
            ["archive", "--all", "--auto-metadata"],
            monkeypatch,
        )
        assert code == 0
        assert captured_kwargs.get("auto_metadata") is True


class TestCliCatalogueGlobal:
    """
    Tests for ``cc-session catalogue --rebuild --archive-root ...``
    (global mode). Symmetric with ``cc-session archive --archive-root ...``.
    """

    def _run_cli(
        self,
        args: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> int:
        """Run the CLI with monkeypatched argv (no stdin needed)."""
        from cc_session_toolkit.cli import main

        monkeypatch.setattr("sys.argv", ["cc-session"] + args)
        try:
            main()
            return 0
        except SystemExit as exc:
            return exc.code if exc.code else 0

    def test_archive_root_bypasses_find_project_root(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """
        ``catalogue --rebuild --archive-root X`` operates on X directly,
        without requiring the caller to be inside a project. Verified by
        running from a non-project cwd.
        """
        # Build a minimal archive structure at archive_root
        archive_root = tmp_path / "cc-archives"
        proj_session_dir = (
            archive_root / "demo-project"
            / "2026-03-15T10-00_aaaaaaaa-1234"
        )
        proj_session_dir.mkdir(parents=True)
        meta = {
            "schema_version": "1.2",
            "session": {
                "id": "aaaaaaaa-1234-0000-0000-000000000001",
                "started_at": "2026-03-15T10:00:00Z",
                "ended_at": "2026-03-15T10:10:00Z",
                "duration_minutes": 10,
            },
            "auto_generated": {
                "title": "Test session",
                "purpose": "demo",
                "tags": ["test"],
            },
            "project": {"name": "demo-project"},
            "statistics": {"turns": 6},
            "archive": {"created_at": "2026-03-15T10:11:00Z"},
        }
        (proj_session_dir / "session.meta.json").write_text(
            json.dumps(meta)
        )

        # Run from a non-project cwd (tmp_path itself has no .git etc.)
        monkeypatch.chdir(tmp_path)

        code = self._run_cli(
            ["catalogue", "--rebuild",
             "--archive-root", str(archive_root)],
            monkeypatch,
        )
        assert code == 0

        # Global catalogue lands at archive_root/CATALOG.json
        catalogue_file = archive_root / "CATALOG.json"
        assert catalogue_file.is_file()
        data = json.loads(catalogue_file.read_text())
        # The session we placed should now appear in the catalogue
        session_ids = {s.get("id") for s in data.get("sessions", [])}
        assert "aaaaaaaa-1234-0000-0000-000000000001" in session_ids

    def test_catalogue_errors_on_missing_archive_root(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """
        Typo'd ``--archive-root`` should error explicitly rather than
        silently writing an empty CATALOG.json to a path that didn't
        exist. (Pre-fix behaviour was the silent-empty-catalogue
        misfire flagged by the audit.)
        """
        monkeypatch.chdir(tmp_path)
        bogus = tmp_path / "no-such-archive-root"

        code = self._run_cli(
            ["catalogue", "--rebuild", "--archive-root", str(bogus)],
            monkeypatch,
        )
        assert code == 2
        err = capsys.readouterr().out
        assert "not found or not a directory" in err
        # And the catalogue file must NOT have been created
        assert not (bogus / "CATALOG.json").exists()


class TestExtractCwdFromJsonl:
    """Robustness of :func:`_extract_cwd_from_jsonl`."""

    def test_reads_cwd_from_first_line(self, tmp_path: Path) -> None:
        from cc_session_toolkit.cli import _extract_cwd_from_jsonl

        jsonl = tmp_path / "s.jsonl"
        jsonl.write_text(
            json.dumps({"cwd": "/some/where", "msg": "hi"}) + "\n"
        )
        assert _extract_cwd_from_jsonl(jsonl) == Path("/some/where")

    def test_falls_back_to_home_when_cwd_missing(
        self, tmp_path: Path
    ) -> None:
        from cc_session_toolkit.cli import _extract_cwd_from_jsonl

        jsonl = tmp_path / "s.jsonl"
        jsonl.write_text(json.dumps({"msg": "no cwd"}) + "\n")
        assert _extract_cwd_from_jsonl(jsonl) == Path.home()

    def test_falls_back_to_home_on_unparseable_jsonl(
        self, tmp_path: Path
    ) -> None:
        from cc_session_toolkit.cli import _extract_cwd_from_jsonl

        jsonl = tmp_path / "bad.jsonl"
        jsonl.write_text("not json at all\n")
        assert _extract_cwd_from_jsonl(jsonl) == Path.home()

    def test_falls_back_to_home_on_missing_file(
        self, tmp_path: Path
    ) -> None:
        from cc_session_toolkit.cli import _extract_cwd_from_jsonl

        assert _extract_cwd_from_jsonl(
            tmp_path / "no-such-file.jsonl"
        ) == Path.home()

    def test_scans_past_summary_record_to_find_cwd(
        self, tmp_path: Path
    ) -> None:
        """
        Regression: real CC live JSONLs commonly open with a
        ``type: summary`` or ``type: permission-mode`` record that
        lacks a ``cwd`` field; the actual session record with ``cwd``
        appears later. Empirically ~90% of live JSONLs have ``cwd`` on
        line 2 or later. The helper must scan past leading metadata
        records, not stop at line 1.
        """
        from cc_session_toolkit.cli import _extract_cwd_from_jsonl

        jsonl = tmp_path / "s.jsonl"
        jsonl.write_text(
            json.dumps(
                {"leafUuid": "abc", "summary": "x", "type": "summary"}
            ) + "\n"
            + json.dumps(
                {"permissionMode": "default",
                 "sessionId": "s1", "type": "permission-mode"}
            ) + "\n"
            + json.dumps(
                {"type": "user",
                 "cwd": "/home/shawn/personal-assistant",
                 "isSidechain": False, "sessionId": "s1"}
            ) + "\n"
        )
        assert _extract_cwd_from_jsonl(jsonl) == Path(
            "/home/shawn/personal-assistant"
        )

    def test_scans_past_multiple_metadata_records(
        self, tmp_path: Path
    ) -> None:
        """The scan should look at up to 20 records before giving up."""
        from cc_session_toolkit.cli import _extract_cwd_from_jsonl

        jsonl = tmp_path / "s.jsonl"
        lines = []
        # 15 metadata records without cwd, then the real one
        for _ in range(15):
            lines.append(json.dumps({"type": "progress"}))
        lines.append(json.dumps({"type": "user", "cwd": "/somewhere"}))
        jsonl.write_text("\n".join(lines) + "\n")

        assert _extract_cwd_from_jsonl(jsonl) == Path("/somewhere")

    def test_treats_non_string_cwd_as_missing(
        self, tmp_path: Path
    ) -> None:
        """
        Regression: a malformed record with a non-string ``cwd``
        (None / int / list / dict) must not raise TypeError and abort
        the batch. The helper treats such values as if the field were
        absent, so the scan continues to subsequent lines.
        """
        from cc_session_toolkit.cli import _extract_cwd_from_jsonl

        # cwd=None then a valid record
        jsonl1 = tmp_path / "s1.jsonl"
        jsonl1.write_text(
            json.dumps({"cwd": None, "type": "x"}) + "\n"
            + json.dumps({"cwd": "/real/path", "type": "y"}) + "\n"
        )
        assert _extract_cwd_from_jsonl(jsonl1) == Path("/real/path")

        # cwd=int — also a valid record after it
        jsonl2 = tmp_path / "s2.jsonl"
        jsonl2.write_text(
            json.dumps({"cwd": 42, "type": "x"}) + "\n"
            + json.dumps({"cwd": "/real/path-2", "type": "y"}) + "\n"
        )
        assert _extract_cwd_from_jsonl(jsonl2) == Path("/real/path-2")

        # cwd=list — falls back when no valid record follows
        jsonl3 = tmp_path / "s3.jsonl"
        jsonl3.write_text(
            json.dumps({"cwd": ["a", "b"], "type": "x"}) + "\n"
        )
        assert _extract_cwd_from_jsonl(jsonl3) == Path.home()

        # cwd="" (empty string) — also treated as missing
        jsonl4 = tmp_path / "s4.jsonl"
        jsonl4.write_text(
            json.dumps({"cwd": "", "type": "x"}) + "\n"
            + json.dumps({"cwd": "/real/path-4", "type": "y"}) + "\n"
        )
        assert _extract_cwd_from_jsonl(jsonl4) == Path("/real/path-4")

    def test_skips_unparseable_lines_then_finds_cwd(
        self, tmp_path: Path
    ) -> None:
        """
        A bad line in the middle should not abort the scan — keep
        going to find the next parseable record with a valid cwd.
        """
        from cc_session_toolkit.cli import _extract_cwd_from_jsonl

        jsonl = tmp_path / "s.jsonl"
        jsonl.write_text(
            json.dumps({"type": "summary"}) + "\n"
            + "this is not json at all\n"
            + json.dumps({"cwd": "/found"}) + "\n"
        )
        assert _extract_cwd_from_jsonl(jsonl) == Path("/found")
