"""Tests for session statistics extraction."""

from __future__ import annotations

from pathlib import Path

from cc_session_toolkit.extraction import (
    detect_relationship_hints,
    estimate_cost,
    extract_artifacts,
    extract_session_stats,
    extract_thinking_block_tokens,
    extract_tool_output_bytes,
)


class TestExtractSessionStats:
    """Tests for :func:`extract_session_stats`."""

    def test_counts_turns(self, sample_session_jsonl: Path) -> None:
        stats = extract_session_stats(sample_session_jsonl)
        assert stats["turns"] >= 1
        assert stats["human_messages"] >= 1
        assert stats["assistant_messages"] >= 1

    def test_counts_thinking_blocks(
        self, sample_session_jsonl: Path
    ) -> None:
        stats = extract_session_stats(sample_session_jsonl)
        assert stats["thinking_blocks"] == 1

    def test_counts_tool_calls(self, sample_session_jsonl: Path) -> None:
        stats = extract_session_stats(sample_session_jsonl)
        assert stats["tool_calls"]["total"] == 2
        assert "Read" in stats["tool_calls"]["by_type"]
        assert "Write" in stats["tool_calls"]["by_type"]

    def test_extracts_model(self, sample_session_jsonl: Path) -> None:
        stats = extract_session_stats(sample_session_jsonl)
        assert stats["model"] == "claude-sonnet-4-5-20250929"

    def test_extracts_tokens(self, sample_session_jsonl: Path) -> None:
        stats = extract_session_stats(sample_session_jsonl)
        assert stats["tokens"]["input"] == 800
        assert stats["tokens"]["output"] == 300
        assert stats["tokens"]["cache_read"] == 150

    def test_computes_timestamps(self, sample_session_jsonl: Path) -> None:
        stats = extract_session_stats(sample_session_jsonl)
        assert stats["started_at"] is not None
        assert stats["ended_at"] is not None
        assert isinstance(stats["duration_minutes"], int)


class TestEstimateCost:
    """Tests for :func:`estimate_cost`."""

    def test_zero_tokens(self) -> None:
        stats = {"tokens": {"input": 0, "output": 0, "cache_read": 0}}
        assert estimate_cost(stats) == 0.0

    def test_calculates_cost(self) -> None:
        stats = {
            "tokens": {
                "input": 1_000_000,
                "output": 1_000_000,
                "cache_read": 1_000_000,
            }
        }
        expected = 3.0 + 15.0 + 0.3
        assert estimate_cost(stats) == expected


class TestExtractThinkingBlockTokens:
    """Tests for :func:`extract_thinking_block_tokens`."""

    def test_counts_blocks(self, sample_session_jsonl: Path) -> None:
        result = extract_thinking_block_tokens(sample_session_jsonl)
        assert result["count"] == 1
        assert result["total_tokens"] > 0
        assert result["token_count_method"] == "estimated"


class TestExtractToolOutputBytes:
    """Tests for :func:`extract_tool_output_bytes`."""

    def test_counts_bytes(self, sample_session_jsonl: Path) -> None:
        result = extract_tool_output_bytes(sample_session_jsonl)
        assert result["total_bytes"] > 0
        assert "Read" in result["by_type"]
        assert result["largest_single_output_bytes"] > 0


class TestExtractArtifacts:
    """Tests for :func:`extract_artifacts`."""

    def test_categorises_operations(
        self, sample_session_jsonl: Path, tmp_path: Path
    ) -> None:
        # The sample session has a Write to /tmp/test/output.py
        # and a Read of /tmp/test/README.md
        # Use /tmp/test as project root so paths are "in project"
        project_root = Path("/tmp/test")
        project_root.mkdir(parents=True, exist_ok=True)

        result = extract_artifacts(sample_session_jsonl, project_root)
        assert isinstance(result["created"], list)
        assert isinstance(result["modified"], list)
        assert isinstance(result["referenced"], list)

        # Write → created, Read → referenced
        created_paths = [a["path"] for a in result["created"]]
        referenced_paths = [a["path"] for a in result["referenced"]]
        assert "output.py" in created_paths
        assert "README.md" in referenced_paths


class TestDetectRelationshipHints:
    """Tests for :func:`detect_relationship_hints`."""

    def test_returns_structure(self, sample_session_jsonl: Path) -> None:
        result = detect_relationship_hints(sample_session_jsonl)
        assert "continues_hint" in result
        assert "references_hints" in result
        assert "detection_notes" in result
        assert isinstance(result["references_hints"], list)
