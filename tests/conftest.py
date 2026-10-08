"""
Shared test fixtures for cc-session-toolkit.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_project(tmp_path: Path) -> Path:
    """
    Create a minimal project directory with markers that
    :func:`find_project_root` will recognise.

    Returns the project root path.
    """
    project = tmp_path / "test-project"
    project.mkdir()
    (project / ".git").mkdir()
    (project / "CLAUDE.md").write_text("# Project: test-project\n")
    return project


@pytest.fixture()
def tmp_project_no_claude(tmp_path: Path) -> Path:
    """
    Project directory with only a ``.git`` marker (no CLAUDE.md).
    """
    project = tmp_path / "bare-project"
    project.mkdir()
    (project / ".git").mkdir()
    return project


@pytest.fixture()
def sample_session_jsonl(tmp_path: Path) -> Path:
    """
    Create a minimal CC session JSONL file with realistic entries.

    Returns the path to the file.
    """
    session_file = tmp_path / "abc12345-1234-5678-9abc-def012345678.jsonl"
    now = datetime.now(tz=timezone.utc)
    entries = [
        # User message
        {
            "type": "user",
            "userType": "external",
            "timestamp": now.isoformat(),
            "message": {
                "role": "user",
                "content": "Hello, please help me with this project.",
            },
        },
        # Assistant message with thinking + tool_use
        {
            "type": "assistant",
            "timestamp": now.isoformat(),
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-5-20250929",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Let me think about this for a moment. "
                        * 10,
                    },
                    {
                        "type": "tool_use",
                        "id": "tool_01",
                        "name": "Read",
                        "input": {"file_path": "/tmp/test/README.md"},
                    },
                ],
                "usage": {
                    "input_tokens": 500,
                    "output_tokens": 200,
                    "cache_read_input_tokens": 100,
                    "cache_creation_input_tokens": 50,
                },
            },
        },
        # Tool result
        {
            "type": "user",
            "timestamp": now.isoformat(),
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_01",
                        "content": "# README\nThis is a test project.\n",
                    }
                ],
            },
        },
        # Assistant message with Write tool
        {
            "type": "assistant",
            "timestamp": now.isoformat(),
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-5-20250929",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_02",
                        "name": "Write",
                        "input": {
                            "file_path": "/tmp/test/output.py",
                            "content": "print('hello')\n",
                        },
                    },
                ],
                "usage": {
                    "input_tokens": 300,
                    "output_tokens": 100,
                    "cache_read_input_tokens": 50,
                    "cache_creation_input_tokens": 0,
                },
            },
        },
    ]

    lines = [json.dumps(entry) for entry in entries]
    session_file.write_text("\n".join(lines) + "\n")
    return session_file


def pytest_configure(config: pytest.Config) -> None:
    """Register the marker that opts a test into the OpenAI extractor."""
    config.addinivalue_line(
        "markers",
        "openai_provider: run with the OpenAI extractor primary (network stubbed)",
    )


@pytest.fixture(autouse=True)
def _isolate_extractor(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep every test off the network and out of the real metadata log.

    * The auto-metadata log goes to a temporary directory. Tests used to
      append to the real ``data/logs/auto-metadata.log`` (54 fake lines
      found there on 2026-10-08).
    * The extractor is pinned to Gemini with no fallback, the path the
      tests written before 2026-10-08 exercise. Tests marked
      ``openai_provider`` keep the production provider order instead.
    * OpenAI requests and key lookups are stubbed for every test, so none
      can reach the network or read a real key; OpenAI tests replace
      ``_openai_post`` with a fake response.
    """
    from cc_session_toolkit import archive, config

    monkeypatch.setenv("CC_SESSION_LOG_DIR", str(tmp_path_factory.mktemp("logs")))

    def _blocked(*_args: object, **_kwargs: object) -> dict:
        raise AssertionError("network blocked in tests: stub archive._openai_post")

    monkeypatch.setattr(archive, "_openai_post", _blocked)
    monkeypatch.setattr(archive, "_ensure_openai_api_key", lambda: "test-openai-key")
    if request.node.get_closest_marker("openai_provider") is None:
        monkeypatch.setattr(archive, "EXTRACTOR_PROVIDER", "gemini")
        monkeypatch.setattr(archive, "EXTRACTOR_FALLBACK_PROVIDER", None)
        monkeypatch.setattr(config, "EXTRACTOR_PROVIDER", "gemini")
        # The primary's defaults follow the pin, as they would in production.
        for module in (archive, config):
            monkeypatch.setattr(module, "EXTRACTOR_MODEL_ID", config.GEMINI_EXTRACTOR_MODEL_ID)
            monkeypatch.setattr(
                module, "EXTRACTOR_THINKING_LEVEL", config.AUTO_METADATA_THINKING_LEVEL
            )
    else:
        monkeypatch.setattr(archive, "EXTRACTOR_PROVIDER", "openai")
        monkeypatch.setattr(archive, "EXTRACTOR_FALLBACK_PROVIDER", "gemini")
        monkeypatch.setattr(config, "EXTRACTOR_PROVIDER", "openai")
