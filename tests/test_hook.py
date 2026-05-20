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
