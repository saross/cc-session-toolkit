"""
Session statistics extraction from Claude Code JSONL transcript files.

All functions that need to resolve project-relative paths accept an
explicit ``project_root`` parameter rather than using ``__file__``.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from cc_session_toolkit.config import get_file_type


# -------------------------------------------------------------------------
# Session statistics
# -------------------------------------------------------------------------

def extract_session_stats(session_path: Path) -> dict[str, Any]:
    """
    Extract statistics from a session JSONL file.

    Returns a dictionary containing:

    - timestamps (``started_at``, ``ended_at``, ``duration_minutes``)
    - message counts (``human_messages``, ``assistant_messages``, ``turns``)
    - ``thinking_blocks`` count
    - ``tool_calls`` with total and by-type breakdown
    - ``tokens`` with input, output, and cache_read counts
    - ``model`` identifier

    Args:
        session_path: Path to the session JSONL file.

    Returns:
        Statistics dictionary.
    """
    stats: dict[str, Any] = {
        "turns": 0,
        "human_messages": 0,
        "assistant_messages": 0,
        "thinking_blocks": 0,
        "tool_calls": {"total": 0, "by_type": {}},
        "tokens": {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0},
        "model": None,
        "timestamps": [],
    }

    with open(session_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Collect timestamp
            ts = entry.get("timestamp")
            if ts:
                stats["timestamps"].append(ts)

            # Skip non-message entries
            if entry.get("type") == "file-history-snapshot":
                continue

            # Process message content
            message = entry.get("message", {})
            role = message.get("role", entry.get("type", ""))

            if role == "user":
                # Distinguish real human messages from tool_result messages
                user_content = message.get("content", "")
                is_tool_result = False
                if isinstance(user_content, list):
                    is_tool_result = any(
                        isinstance(b, dict) and b.get("type") == "tool_result"
                        for b in user_content
                    )
                if not is_tool_result:
                    stats["human_messages"] += 1
                    stats["turns"] += 1

            elif role == "assistant":
                stats["assistant_messages"] += 1
                content = message.get("content", [])

                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        block_type = block.get("type")

                        if block_type == "thinking":
                            stats["thinking_blocks"] += 1
                        elif block_type == "tool_use":
                            tool_name = block.get("name", "unknown")
                            stats["tool_calls"]["total"] += 1
                            by_type = stats["tool_calls"]["by_type"]
                            by_type[tool_name] = by_type.get(tool_name, 0) + 1

                # Extract model from response metadata
                if not stats["model"]:
                    model = message.get("model")
                    if model:
                        stats["model"] = model

            # Token usage (lives inside message object in real CC data)
            usage = (
                message.get("usage", {})
                if isinstance(message, dict)
                else {}
            )
            if usage:
                stats["tokens"]["input"] += usage.get("input_tokens", 0)
                stats["tokens"]["output"] += usage.get("output_tokens", 0)
                stats["tokens"]["cache_read"] += usage.get(
                    "cache_read_input_tokens", 0
                )
                stats["tokens"]["cache_creation"] += usage.get(
                    "cache_creation_input_tokens", 0
                )

    # Derive timestamps and duration
    if stats["timestamps"]:
        stats["started_at"] = min(stats["timestamps"])
        stats["ended_at"] = max(stats["timestamps"])
        try:
            start = datetime.fromisoformat(
                stats["started_at"].replace("Z", "+00:00")
            )
            end = datetime.fromisoformat(
                stats["ended_at"].replace("Z", "+00:00")
            )
            stats["duration_minutes"] = int(
                (end - start).total_seconds() / 60
            )
        except (ValueError, TypeError):
            stats["duration_minutes"] = 0
    else:
        stats["started_at"] = None
        stats["ended_at"] = None
        stats["duration_minutes"] = 0

    del stats["timestamps"]
    return stats


# -------------------------------------------------------------------------
# Cost estimation
# -------------------------------------------------------------------------

def estimate_cost(stats: dict[str, Any]) -> float:
    """
    Estimate API cost based on token usage.

    Uses approximate pricing for Claude Sonnet (may vary by model).

    Args:
        stats: Statistics dictionary from :func:`extract_session_stats`.

    Returns:
        Estimated cost in USD, rounded to two decimal places.
    """
    # Pricing per 1M tokens (USD) by model family
    model = stats.get("model") or ""
    if "opus" in model:
        input_price = 15.0
        output_price = 75.0
        cache_read_price = 1.50
        cache_creation_price = 18.75  # 1.25x input
    else:
        # Sonnet pricing (default)
        input_price = 3.0
        output_price = 15.0
        cache_read_price = 0.30
        cache_creation_price = 3.75  # 1.25x input

    tokens = stats.get("tokens", {})
    cost = (
        (tokens.get("input", 0) / 1_000_000) * input_price
        + (tokens.get("output", 0) / 1_000_000) * output_price
        + (tokens.get("cache_read", 0) / 1_000_000) * cache_read_price
        + (tokens.get("cache_creation", 0) / 1_000_000) * cache_creation_price
    )
    return round(cost, 2)


# -------------------------------------------------------------------------
# Thinking-block token extraction
# -------------------------------------------------------------------------

def extract_thinking_block_tokens(
    session_path: Path,
) -> dict[str, Any]:
    """
    Extract thinking-block statistics including token estimates.

    The JSONL may not include per-block token counts, so tokens are
    estimated at roughly 4 characters per token.

    Args:
        session_path: Path to the session JSONL file.

    Returns:
        Dictionary with ``count``, ``total_tokens``, and
        ``token_count_method``.
    """
    thinking_count = 0
    estimated_tokens = 0

    with open(session_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            content = entry.get("message", {}).get("content", [])
            if isinstance(content, list):
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "thinking"
                    ):
                        thinking_count += 1
                        thinking_text = block.get("thinking", "")
                        estimated_tokens += len(thinking_text) // 4

    return {
        "count": thinking_count,
        "total_tokens": estimated_tokens,
        "token_count_method": "estimated",
    }


# -------------------------------------------------------------------------
# Tool output byte tracking
# -------------------------------------------------------------------------

def extract_tool_output_bytes(session_path: Path) -> dict[str, Any]:
    """
    Extract tool output byte statistics for storage analysis.

    Parses ``tool_result`` blocks and sums byte counts by tool type.

    Args:
        session_path: Path to the session JSONL file.

    Returns:
        Dictionary with ``total_bytes``, ``by_type``, and
        ``largest_single_output_bytes``.
    """
    by_type: dict[str, dict[str, int]] = {}
    total_bytes = 0
    largest_output = 0
    tool_use_names: dict[str, str] = {}

    with open(session_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            content = entry.get("message", {}).get("content", [])
            if not isinstance(content, list):
                continue

            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")

                # Map tool_use IDs to tool names
                if block_type == "tool_use":
                    tool_id = block.get("id")
                    tool_name = block.get("name", "unknown")
                    if tool_id:
                        tool_use_names[tool_id] = tool_name

                # Count tool_result bytes
                elif block_type == "tool_result":
                    tool_id = block.get("tool_use_id")
                    tool_name = tool_use_names.get(tool_id, "unknown")

                    result_content = block.get("content", "")
                    if isinstance(result_content, str):
                        output_bytes = len(result_content.encode("utf-8"))
                    elif isinstance(result_content, list):
                        output_bytes = sum(
                            len(json.dumps(item).encode("utf-8"))
                            for item in result_content
                        )
                    else:
                        output_bytes = len(
                            str(result_content).encode("utf-8")
                        )

                    total_bytes += output_bytes
                    largest_output = max(largest_output, output_bytes)

                    if tool_name not in by_type:
                        by_type[tool_name] = {"count": 0, "bytes": 0}
                    by_type[tool_name]["count"] += 1
                    by_type[tool_name]["bytes"] += output_bytes

    return {
        "total_bytes": total_bytes,
        "by_type": by_type,
        "largest_single_output_bytes": largest_output,
    }


# -------------------------------------------------------------------------
# Artifact tracking
# -------------------------------------------------------------------------

def extract_artifacts(
    session_path: Path,
    project_root: Path,
) -> dict[str, list[dict[str, str]]]:
    """
    Extract artifacts (files created, modified, referenced) from a session.

    Parses tool calls to identify file operations:

    - ``Write`` → created
    - ``Edit`` → modified
    - ``Read`` → referenced

    Deduplication priority:

    - created + modified → created only (it was new)
    - read + modified → modified only (read was context)
    - read only → referenced

    Files outside *project_root* are excluded.

    Args:
        session_path: Path to the session JSONL file.
        project_root: Project root for filtering and relative paths.

    Returns:
        Dictionary with ``created``, ``modified``, ``referenced`` lists.
    """
    project_dir = str(project_root.resolve())

    written_files: set[str] = set()
    edited_files: set[str] = set()
    read_files: set[str] = set()

    with open(session_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            content = entry.get("message", {}).get("content", [])
            if not isinstance(content, list):
                continue

            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_use":
                    continue

                tool_name = block.get("name", "")
                tool_input = block.get("input", {})
                file_path = tool_input.get("file_path")

                if not file_path:
                    continue

                if tool_name == "Write":
                    written_files.add(file_path)
                elif tool_name == "Edit":
                    edited_files.add(file_path)
                elif tool_name == "Read":
                    read_files.add(file_path)

    # Helpers
    def is_in_project(path: str) -> bool:
        try:
            resolved = Path(path).resolve()
            return resolved.is_relative_to(project_root.resolve())
        except (OSError, ValueError):
            return False

    def make_relative(path: str) -> str:
        try:
            return str(Path(path).resolve().relative_to(project_dir))
        except (OSError, ValueError):
            return path

    # Deduplication
    created = written_files
    modified = edited_files - written_files
    referenced = read_files - edited_files - written_files

    result: dict[str, list[dict[str, str]]] = {
        "created": [],
        "modified": [],
        "referenced": [],
    }

    for path in sorted(created):
        if is_in_project(path):
            result["created"].append({
                "path": make_relative(path),
                "type": get_file_type(path),
                "description": "",
            })

    for path in sorted(modified):
        if is_in_project(path):
            result["modified"].append({
                "path": make_relative(path),
                "type": get_file_type(path),
                "description": "",
            })

    for path in sorted(referenced):
        if is_in_project(path):
            result["referenced"].append({
                "path": make_relative(path),
                "type": get_file_type(path),
                "description": "",
            })

    return result


# -------------------------------------------------------------------------
# Relationship detection
# -------------------------------------------------------------------------

def detect_relationship_hints(session_path: Path) -> dict[str, Any]:
    """
    Detect potential session relationships from conversation context.

    Looks for mentions of previous sessions (continuation language,
    session UUIDs). These are *hints* for user confirmation, not
    auto-applied relationships.

    Args:
        session_path: Path to the session JSONL file.

    Returns:
        Dictionary with ``continues_hint``, ``references_hints``, and
        ``detection_notes``.
    """
    hints: dict[str, Any] = {
        "continues_hint": None,
        "references_hints": [],
        "detection_notes": [],
    }

    uuid_pattern = re.compile(
        r"\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b",
        re.IGNORECASE,
    )
    continuation_patterns = [
        r"continu(?:e|ing|ed)\s+from",
        r"pick(?:ing)?\s+up\s+(?:from\s+)?where",
        r"previous\s+session",
        r"last\s+session",
        r"earlier\s+(?:session|conversation)",
    ]

    with open(session_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            message = entry.get("message", {})
            if message.get("role") != "user":
                continue

            text = message.get("content", "")
            if isinstance(text, list):
                text = " ".join(
                    b.get("text", "")
                    for b in text
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            if not isinstance(text, str):
                continue

            # Session UUIDs
            for match in uuid_pattern.finditer(text):
                uuid_str = match.group(1)
                if uuid_str not in hints["references_hints"]:
                    hints["references_hints"].append(uuid_str)
                    hints["detection_notes"].append(
                        f"Found session ID reference: {uuid_str}"
                    )

            # Continuation language
            for pattern in continuation_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    if not hints["continues_hint"]:
                        hints["continues_hint"] = "detected"
                        hints["detection_notes"].append(
                            f"Found continuation language: '{pattern}'"
                        )

    return hints
