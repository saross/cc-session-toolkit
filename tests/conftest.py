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
