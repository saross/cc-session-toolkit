"""
Archive operations — copy, compress, and create metadata for CC sessions.

Merges features from both diverged versions:

- map-reader-llm v1.1: title-based naming, catalogue project rollups
- llm-reproducibility v1.2: find_archive_by_id, update_catalogue_entry
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from cc_session_toolkit.config import (
    DEFAULT_MIN_DURATION_MINUTES,
    DEFAULT_MIN_TURNS,
    DEFAULT_THINKING_EXCLUDED_USES,
    DEFAULT_THINKING_NATURE_NOTE,
    DEFAULT_THINKING_SHARING,
    DEFAULT_THINKING_USE_CONSTRAINTS,
    SCHEMA_VERSION,
    load_defaults,
)
from cc_session_toolkit.extraction import (
    detect_relationship_hints,
    estimate_cost,
    extract_artifacts,
    extract_session_stats,
    extract_thinking_block_tokens,
    extract_tool_output_bytes,
)
from cc_session_toolkit.naming import get_archive_directory
from cc_session_toolkit.project import (
    CLAUDE_PROJECTS_DIR,
    get_archive_dir,
    get_cc_project_path,
    get_defaults_file,
    get_project_name,
)


# -------------------------------------------------------------------------
# Session discovery
# -------------------------------------------------------------------------

def get_session_files(project_root: Path) -> list[Path]:
    """
    Find all session JSONL files for a project in ``~/.claude/projects/``.

    Args:
        project_root: Project root directory.

    Returns:
        List of session file paths, sorted by modification time.
    """
    cc_project_path = get_cc_project_path(project_root)
    project_dir = CLAUDE_PROJECTS_DIR / cc_project_path

    if not project_dir.exists():
        return []

    session_files = list(project_dir.glob("*.jsonl"))
    return sorted(session_files, key=lambda p: p.stat().st_mtime)


def get_archived_session_ids(
    catalogue_file: Path,
) -> set[str]:
    """
    Get session IDs that have already been archived.

    Args:
        catalogue_file: Path to ``CATALOG.json``.

    Returns:
        Set of archived session ID strings.
    """
    if not catalogue_file.exists():
        return set()

    try:
        catalogue = json.loads(catalogue_file.read_text(encoding="utf-8"))
        return {s["id"] for s in catalogue.get("sessions", [])}
    except (json.JSONDecodeError, KeyError):
        return set()


def get_session_id(session_path: Path) -> str:
    """
    Extract session ID from a JSONL filename.

    Main sessions use a UUID stem; agent sessions use ``agent-<short_id>``.

    Args:
        session_path: Path to the JSONL file.

    Returns:
        Session identifier string.
    """
    return session_path.stem


# -------------------------------------------------------------------------
# Hook-mode helpers (trivial filter, dedup, auto-metadata)
# -------------------------------------------------------------------------

def is_trivial_session(
    stats: dict[str, Any],
    min_turns: int = DEFAULT_MIN_TURNS,
    min_duration: int = DEFAULT_MIN_DURATION_MINUTES,
) -> bool:
    """
    Check whether a session is too short to be worth archiving.

    Used by hook-based archiving to skip trivial sessions (e.g.
    accidental opens, quick ``/help`` queries).

    Args:
        stats: Session statistics from :func:`extract_session_stats`.
        min_turns: Minimum number of human turns.
        min_duration: Minimum duration in minutes.

    Returns:
        *True* if the session is trivial and should be skipped.
    """
    if stats.get("turns", 0) < min_turns:
        return True
    if stats.get("duration_minutes", 0) < min_duration:
        return True
    return False


def is_already_archived(
    session_id: str,
    catalogue_file: Path,
) -> bool:
    """
    Check whether a session has already been archived.

    Used for deduplication when both PreCompact and SessionEnd hooks
    fire for the same session — the first one to archive wins.

    Args:
        session_id: Session identifier to check.
        catalogue_file: Path to ``CATALOG.json`` (global or per-project).

    Returns:
        *True* if the session is already in the catalogue.
    """
    return session_id in get_archived_session_ids(catalogue_file)


# Short confirmations and housekeeping phrases (case-insensitive).
# Only checked when the message is under 40 characters.
_META_SHORT_PATTERNS: frozenset[str] = frozenset({
    "yes", "no", "ok", "okay", "sure", "nope",
    "go ahead", "looks good", "lgtm",
    "thanks", "thank you", "perfect", "great", "nice",
    "do it", "commit", "push", "done",
    "agreed", "exactly", "correct", "right", "good",
    "yep", "yeah", "please", "proceed", "continue", "approved",
})

# Substrings that mark a message as meta regardless of length.
_META_SUBSTRINGS: tuple[str, ...] = (
    "commit and push",
    "commit this",
    "push this",
    "session summary",
    "please commit",
    "go ahead and commit",
)


def _is_meta_message(text: str) -> bool:
    """
    Check whether a user message is meta/housekeeping rather than
    substantive work direction.

    Meta messages include slash commands, short confirmations, and
    commit/push instructions.  These are filtered before sampling
    to ensure the "last N" messages reflect actual work, not session
    wrap-up.

    Args:
        text: The user message text.

    Returns:
        *True* if the message should be excluded from sampling.
    """
    stripped = text.strip()
    if not stripped:
        return True

    # Slash commands (/recap, /done, /reflect, etc.)
    if stripped.startswith("/"):
        return True

    lower = stripped.lower().rstrip(".!?,")

    # Short confirmations / housekeeping
    if len(stripped) < 40 and lower in _META_SHORT_PATTERNS:
        return True

    # Substring matches (any length)
    lower_full = stripped.lower()
    return any(sub in lower_full for sub in _META_SUBSTRINGS)


def _ensure_anthropic_api_key() -> None:
    """
    Ensure ``ANTHROPIC_API_KEY`` is available in the environment.

    When called from Claude Code hooks, the environment variable may
    not survive the shell export chain.  This function provides a
    fallback: if the key is missing, it reads ``~/personal-assistant/.env``
    directly and injects the key into ``os.environ``.
    """
    import os

    if os.environ.get("ANTHROPIC_API_KEY"):
        return

    env_path = Path.home() / "personal-assistant" / ".env"
    if not env_path.is_file():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key == "ANTHROPIC_API_KEY" and value:
            os.environ["ANTHROPIC_API_KEY"] = value
            return


def _log_metadata_event(
    message: str,
    *,
    level: str = "INFO",
) -> None:
    """Append a timestamped line to the auto-metadata log."""
    import os

    log_dir = Path(
        os.environ.get(
            "CC_SESSION_LOG_DIR",
            Path.home() / "personal-assistant" / "data" / "logs",
        )
    )
    log_file = log_dir / "auto-metadata.log"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as fh:
            fh.write(f"{timestamp} [{level}] {message}\n")
    except OSError:
        pass  # logging must never break the archive


def generate_auto_metadata(
    session_path: Path,
    stats: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Generate title, purpose, and tags via Haiku Application Programming
    Interface (API) call.

    Extracts the first few user messages from the session transcript
    and sends them (with session statistics) to Haiku for automatic
    metadata generation.  Falls back to *None* if the ``anthropic``
    package is not installed or the API call fails.

    Budget: ~$0.001 per session (Haiku pricing).

    Args:
        session_path: Path to the session JSONL file.
        stats: Session statistics from :func:`extract_session_stats`.

    Returns:
        Dictionary with ``title``, ``purpose``, and ``tags`` keys,
        or *None* on failure.
    """
    try:
        import anthropic  # noqa: WPS433 — optional dependency
    except ImportError:
        _log_metadata_event(
            "anthropic package not installed — skipping",
            level="WARNING",
        )
        print(
            "  Warning: anthropic package not installed, "
            "skipping auto-metadata"
        )
        return None

    _ensure_anthropic_api_key()

    # Extract representative user messages and file paths from the
    # transcript in a single pass.  Collects:
    # - All user messages (for sampling)
    # - File paths from Write/Edit tool calls (for artefact context)
    all_user_messages: list[str] = []
    files_modified: set[str] = set()

    with open(session_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            message = entry.get("message", {})
            role = message.get("role")

            # Collect file paths from Write/Edit tool calls
            if role == "assistant":
                a_content = message.get("content", [])
                if isinstance(a_content, list):
                    for block in a_content:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "tool_use"
                            and block.get("name") in {"Write", "Edit"}
                        ):
                            fp = block.get("input", {}).get("file_path")
                            if fp:
                                files_modified.add(fp)
                continue

            if role != "user":
                continue

            content = message.get("content", "")
            if isinstance(content, list):
                # Skip tool results
                if any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in content
                ):
                    continue
                content = " ".join(
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )

            if content:
                all_user_messages.append(content[:500])

    if not all_user_messages:
        return None

    # Filter out meta/housekeeping messages (slash commands, short
    # confirmations, commit instructions) so the "last 2" reflect
    # substantive work rather than session wrap-up.
    substantive = [
        msg for msg in all_user_messages
        if not _is_meta_message(msg)
    ]
    # Fall back to unfiltered if filtering removed everything
    sample_source = substantive or all_user_messages

    # Sample: first 2 (intent) + last 2 (outcome), deduplicated
    # for short sessions where they overlap
    first = sample_source[:2]
    last = sample_source[-2:]
    seen: set[str] = set()
    sampled: list[str] = []
    for msg in first + last:
        if msg not in seen:
            seen.add(msg)
            sampled.append(msg)

    messages_text = "\n---\n".join(sampled)
    tool_summary = ", ".join(
        f"{k}: {v}"
        for k, v in stats.get("tool_calls", {}).get("by_type", {}).items()
    )

    # Build artefact context from collected file paths
    artefact_basenames = sorted(
        {Path(fp).name for fp in files_modified}
    )
    files_line = (
        f"Files modified: {', '.join(artefact_basenames)}\n"
        if artefact_basenames else ""
    )

    # Label the sample so Haiku understands the temporal spread
    n_total = len(sample_source)
    if len(sampled) == n_total:
        sample_label = f"All {n_total} user messages"
    else:
        sample_label = (
            f"First and last substantive user messages "
            f"(from {n_total} total, meta-messages filtered)"
        )

    prompt = (
        f"Based on the following Claude Code session information, "
        f"generate:\n"
        f"1. A concise title (5-10 words) reflecting the session's "
        f"main accomplishment\n"
        f"2. A one-sentence purpose statement that captures *why*, "
        f"not just *what* (include motivation if evident)\n"
        f"3. 2-5 lowercase hyphenated tags\n"
        f"4. Three Ps metadata:\n"
        f"   - prompt_summary: What was asked and why (1 sentence)\n"
        f"   - process_summary: How the tool was used and why this "
        f"approach (1 sentence)\n"
        f"   - provenance_summary: Where this session fits in the "
        f"broader project (1 sentence)\n\n"
        f"Session stats: {stats.get('duration_minutes', 0)} min, "
        f"{stats.get('turns', 0)} turns, tools: {tool_summary}\n"
        f"{files_line}\n"
        f"{sample_label}:\n"
        f"{messages_text}\n\n"
        f"Respond with ONLY a JSON object, no markdown:\n"
        f'{{"title": "...", "purpose": "...", "tags": ["..."], '
        f'"three_ps": {{"prompt_summary": "...", '
        f'"process_summary": "...", "provenance_summary": "..."}}}}'
    )

    try:
        client = anthropic.Anthropic()
        if not client.api_key:
            _log_metadata_event(
                f"No API key available for {session_path.name}",
                level="ERROR",
            )
            print("  Warning: no ANTHROPIC_API_KEY — skipping auto-metadata")
            return None

        _log_metadata_event(
            f"Calling Haiku for {session_path.name} "
            f"({len(sampled)} messages sampled)"
        )

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text.strip()

        # Extract JSON from potential markdown code blocks.
        # Use regex to handle varied fencing (```json, ```, etc.)
        code_block = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?\s*```",
            response_text,
            re.DOTALL,
        )
        json_str = (
            code_block.group(1).strip() if code_block
            else response_text
        )

        result = json.loads(json_str)
        title = result.get("title", "Untitled Session")
        _log_metadata_event(
            f"Success for {session_path.name}: {title!r}"
        )
        auto_meta: dict[str, Any] = {
            "title": title,
            "purpose": result.get("purpose", ""),
            "tags": result.get("tags", []),
        }
        # Include three_ps if Haiku returned them
        three_ps = result.get("three_ps")
        if isinstance(three_ps, dict):
            auto_meta["three_ps"] = {
                "prompt_summary": three_ps.get(
                    "prompt_summary", ""
                ),
                "process_summary": three_ps.get(
                    "process_summary", ""
                ),
                "provenance_summary": three_ps.get(
                    "provenance_summary", ""
                ),
            }
        return auto_meta
    except Exception as exc:  # noqa: BLE001 — graceful degradation
        _log_metadata_event(
            f"Failed for {session_path.name}: "
            f"{type(exc).__name__}: {exc}",
            level="ERROR",
        )
        print(f"  Warning: auto-metadata generation failed: {exc}")
        return None


# -------------------------------------------------------------------------
# Metadata prompt generation
# -------------------------------------------------------------------------

def generate_metadata_prompt(
    session_id: str, stats: dict[str, Any]
) -> str:
    """
    Generate a prompt for CC to create session metadata.

    The prompt is printed to stdout so CC can respond with the metadata
    JSON during an interactive archiving session.

    Args:
        session_id: Session identifier.
        stats: Extracted session statistics.

    Returns:
        Prompt string.
    """
    tool_summary = ", ".join(
        f"{k}: {v}" for k, v in stats["tool_calls"]["by_type"].items()
    )
    return (
        f"I need you to generate metadata for archiving the CC session "
        f"that just completed.\n\n"
        f"**Session ID**: {session_id}\n"
        f"**Duration**: {stats['duration_minutes']} minutes\n"
        f"**Turns**: {stats['turns']} "
        f"({stats['human_messages']} human, "
        f"{stats['assistant_messages']} assistant)\n"
        f"**Thinking blocks**: {stats['thinking_blocks']}\n"
        f"**Tool calls**: {stats['tool_calls']['total']} "
        f"({tool_summary})\n"
        f"**Tokens**: {stats['tokens']['input']:,} input, "
        f"{stats['tokens']['output']:,} output\n\n"
        f"Based on our conversation, please provide the following in "
        f"JSON format:\n\n"
        f'```json\n'
        f'{{\n'
        f'  "title": "Brief descriptive title (5-10 words)",\n'
        f'  "purpose": "What the user was trying to accomplish '
        f'(1-2 sentences)",\n'
        f'  "tags": ["tag1", "tag2", "tag3"],\n'
        f'  "three_ps": {{\n'
        f'    "prompt_summary": "What was asked (Prompt) - '
        f'1-2 sentences",\n'
        f'    "process_summary": "How the tool was used (Process) - '
        f'1-2 sentences",\n'
        f'    "provenance_summary": "Role in research workflow '
        f'(Provenance) - 1 sentence"\n'
        f"  }}\n"
        f"}}\n"
        f"```\n\n"
        f"Please respond with ONLY the JSON block, no other text."
    )


# -------------------------------------------------------------------------
# Relationship resolution
# -------------------------------------------------------------------------


def _find_previous_session(
    archive_base: Path,
    project_name: str,
    current_session_id: str,
) -> str | None:
    """
    Find the most recent archived session in the same project.

    Scans the project's archive subdirectory for session directories
    (sorted lexically by timestamp-prefixed names) and returns the
    session ID of the most recent one that is not the current session.

    Args:
        archive_base: Root of the archive tree (e.g. ``~/cc-archives``).
        project_name: Project subdirectory name.
        current_session_id: ID of the session being archived (to exclude).

    Returns:
        Session ID of the previous session, or *None* if none found.
    """
    project_dir = archive_base / project_name
    if not project_dir.is_dir():
        return None

    # Session directories are timestamp-prefixed, so lexical sort works
    candidates = sorted(
        (d for d in project_dir.iterdir() if d.is_dir()),
        reverse=True,
    )

    for candidate in candidates:
        meta_file = candidate / "session.meta.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            prev_id = meta.get("session", {}).get("id", "")
            if prev_id and prev_id != current_session_id:
                return prev_id
        except (json.JSONDecodeError, OSError):
            continue

    return None


# -------------------------------------------------------------------------
# Metadata creation
# -------------------------------------------------------------------------

def create_session_metadata(
    session_id: str,
    session_path: Path,
    stats: dict[str, Any],
    project_root: Path | None = None,
    *,
    auto_generated: dict[str, Any] | None = None,
    thinking_block_stats: dict[str, Any] | None = None,
    tool_output_stats: dict[str, Any] | None = None,
    artifacts: dict[str, list] | None = None,
    relationship_hints: dict[str, Any] | None = None,
    compression_info: dict[str, Any] | None = None,
    defaults: dict[str, Any] | None = None,
    project_name_override: str | None = None,
    capture_type: str | None = None,
) -> dict[str, Any]:
    """
    Create the complete ``session.meta.json`` structure (v1.1 schema).

    Args:
        session_id: Session identifier.
        session_path: Path to the source JSONL file.
        stats: Extracted session statistics.
        project_root: Project root directory.  May be *None* in
            hook mode if no project root was detected.
        auto_generated: CC-generated metadata (title, purpose, tags,
            three_ps).
        thinking_block_stats: Thinking-block token statistics.
        tool_output_stats: Tool output byte statistics.
        artifacts: Files created/modified/referenced.
        relationship_hints: Detected relationship hints.
        compression_info: Compression metadata (if using gzip).
        defaults: Loaded defaults from ``archive-defaults.yaml``.
        project_name_override: Explicit project name (overrides
            auto-detection from *project_root*).
        capture_type: How the session was captured — ``"session_end"``,
            ``"pre_compact"``, or *None* for manual archiving.

    Returns:
        Complete metadata dictionary.
    """
    defaults = defaults or {}
    thinking_defaults = defaults.get("thinking_blocks", {})
    relationship_defaults = defaults.get("relationships", {})

    # Compute file hash
    sha256_hash = hashlib.sha256()
    with open(session_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha256_hash.update(chunk)

    # Resolve project name — explicit override takes priority
    if project_name_override:
        project_name = project_name_override
    elif project_root:
        project_name = get_project_name(project_root)
    else:
        project_name = "unknown"

    # Thinking blocks section
    thinking_stats = thinking_block_stats or {
        "count": 0,
        "total_tokens": 0,
    }
    thinking_blocks = {
        "included": True,
        "count": thinking_stats.get(
            "count", stats.get("thinking_blocks", 0)
        ),
        "total_tokens": thinking_stats.get("total_tokens", 0),
        "token_count_method": thinking_stats.get(
            "token_count_method", "estimated"
        ),
        "sharing_preference": thinking_defaults.get(
            "sharing_preference", DEFAULT_THINKING_SHARING
        ),
        "use_constraints": thinking_defaults.get(
            "use_constraints", DEFAULT_THINKING_USE_CONSTRAINTS
        ),
        "excluded_uses": thinking_defaults.get(
            "excluded_uses", DEFAULT_THINKING_EXCLUDED_USES
        ),
        "nature_note": DEFAULT_THINKING_NATURE_NOTE,
    }

    # Relationships section
    default_is_part_of = relationship_defaults.get(
        "default_isPartOf", [project_name]
    )
    relationship_hints_info = relationship_hints or {}

    # Auto-populate "continues" when continuation was detected and
    # a previous session ID was resolved.
    continues_id = relationship_hints_info.get(
        "continues_session_id"
    )

    relationships = {
        "continues": continues_id,
        "continuedBy": None,
        "isPartOf": default_is_part_of,
        "isParallelTo": [],
        "supersedes": None,
        "references": [],
        "branchesFrom": None,
    }

    # Statistics section
    tool_outputs = tool_output_stats or {
        "total_bytes": 0,
        "by_type": {},
        "largest_single_output_bytes": 0,
    }
    statistics = {
        "turns": stats["turns"],
        "human_messages": stats["human_messages"],
        "assistant_messages": stats["assistant_messages"],
        "thinking_blocks": stats["thinking_blocks"],
        "tool_calls": stats["tool_calls"],
        "tokens": stats["tokens"],
        "estimated_cost_usd": estimate_cost(stats),
        "tool_outputs": tool_outputs,
    }

    # Archive section
    archive = {
        "jsonl_path": "session.jsonl",
        "jsonl_sha256": sha256_hash.hexdigest(),
        "jsonl_bytes": session_path.stat().st_size,
        "archived_at": datetime.now().isoformat(),
    }

    if compression_info:
        archive["jsonl_path"] = compression_info.get(
            "path", "session.jsonl.gz"
        )
        archive["jsonl_compression"] = compression_info.get(
            "compression", "gzip"
        )
        archive["jsonl_bytes_compressed"] = compression_info.get(
            "compressed_bytes", 0
        )
        archive["jsonl_bytes_uncompressed"] = compression_info.get(
            "uncompressed_bytes", session_path.stat().st_size
        )
        archive["jsonl_sha256_uncompressed"] = sha256_hash.hexdigest()
        if "compressed_sha256" in compression_info:
            archive["jsonl_sha256"] = compression_info["compressed_sha256"]

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "session": {
            "id": session_id,
            "started_at": stats["started_at"],
            "ended_at": stats["ended_at"],
            "duration_minutes": stats["duration_minutes"],
        },
        "project": {
            "name": project_name,
            "directory": str(project_root or ""),
        },
        "model": {
            "provider": "anthropic",
            "model_id": stats.get("model", "unknown"),
            "access_method": "claude-code-cli",
        },
        "thinking_blocks": thinking_blocks,
        "relationships": relationships,
        "artifacts": artifacts
        or {"created": [], "modified": [], "referenced": []},
        "statistics": statistics,
        "auto_generated": auto_generated
        or {
            "title": "Untitled Session",
            "purpose": "No description provided",
            "tags": [],
        },
        "three_ps": (auto_generated or {}).get(
            "three_ps",
            {
                "prompt_summary": "",
                "process_summary": "",
                "provenance_summary": "",
            },
        ),
        "archive": archive,
    }

    if capture_type:
        metadata["archive"]["capture_type"] = capture_type

    if relationship_hints_info.get("detection_notes"):
        metadata["_relationship_hints"] = relationship_hints_info

    return metadata


# -------------------------------------------------------------------------
# Archive a session
# -------------------------------------------------------------------------

def archive_session(
    session_path: Path,
    project_root: Path | None = None,
    *,
    dry_run: bool = False,
    stats_only: bool = False,
    use_gzip: bool = False,
    title: str | None = None,
    archive_root: Path | None = None,
    project_name_override: str | None = None,
    auto_metadata: bool = False,
    capture_type: str | None = None,
    session_id_override: str | None = None,
) -> dict[str, Any] | None:
    """
    Archive a single session with v1.1 schema.

    Supports two modes:

    - **Project mode** (default): Archives to the project's own
      ``archive/cc-sessions/`` directory.  Requires *project_root*.
    - **Global mode** (``archive_root`` set): Archives to a
      centralised directory (e.g. ``~/cc-archives/``).  Used by
      hook-based automated archiving.

    Args:
        session_path: Path to source JSONL file.
        project_root: Project root directory.  May be *None* in
            global mode if no project root was detected.
        dry_run: Preview without writing files.
        stats_only: Skip CC metadata generation (use placeholders).
        use_gzip: Compress the JSONL file.
        title: Optional session title for human-readable directory name.
        archive_root: Global archive root (overrides per-project
            archive directory).
        project_name_override: Explicit project name (overrides
            auto-detection from *project_root*).
        auto_metadata: Call Haiku API for automatic title/purpose/tags.
        capture_type: How the session was captured — ``"session_end"``,
            ``"pre_compact"``, or *None* for manual archiving.
        session_id_override: Explicit session ID (overrides
            filename-based detection).

    Returns:
        Session metadata dictionary, or *None* if skipped.
    """
    session_id = session_id_override or get_session_id(session_path)
    stats = extract_session_stats(session_path)

    # Resolve project name and archive base directory
    if project_name_override:
        project_name = project_name_override
    elif project_root:
        project_name = get_project_name(project_root)
    else:
        project_name = "unknown"

    if archive_root:
        archive_base = archive_root
    elif project_root:
        archive_base = get_archive_dir(project_root)
    else:
        from cc_session_toolkit.config import DEFAULT_ARCHIVE_ROOT
        archive_base = DEFAULT_ARCHIVE_ROOT

    # Load project defaults (if available)
    if project_root:
        defaults_file = get_defaults_file(project_root)
        defaults = load_defaults(defaults_file)
    else:
        defaults = {}

    # Generate auto-metadata early so Haiku title can inform the
    # directory name (human-readable slug instead of short session ID).
    auto_generated = None
    effective_title = title

    if auto_metadata:
        print("  Generating auto-metadata via Haiku...")
        auto_generated = generate_auto_metadata(session_path, stats)
        if auto_generated:
            if title:
                # Explicit title overrides Haiku title
                auto_generated["title"] = title
            elif auto_generated.get("title"):
                # Use Haiku title for directory naming
                effective_title = auto_generated["title"]

    if not auto_generated and not stats_only:
        # Interactive mode: print prompt for CC to fill in
        print("\n" + "=" * 60)
        print("METADATA GENERATION")
        print("=" * 60)
        print(generate_metadata_prompt(session_id, stats))
        print("=" * 60)
        print(
            "\nPlease provide the JSON metadata above, "
            "or press Enter to skip."
        )

    if not auto_generated:
        auto_generated = {
            "title": title or "Untitled Session",
            "purpose": (
                "Auto-metadata unavailable"
                if auto_metadata
                else "Metadata generation requires interactive CC session"
            ),
            "tags": [],
            "three_ps": {
                "prompt_summary": "",
                "process_summary": "",
                "provenance_summary": "",
            },
        }

    dest_dir = get_archive_directory(
        session_id=session_id,
        stats=stats,
        archive_dir=archive_base,
        project_name=project_name,
        title=effective_title,
    )

    print(f"\nSession: {session_id}")
    print(f"  Source: {session_path}")
    print(f"  Archive: {dest_dir}")
    print(
        f"  Duration: {stats['duration_minutes']} min, "
        f"{stats['turns']} turns"
    )
    print(f"  Model: {stats.get('model', 'unknown')}")

    if dry_run:
        print("  [DRY RUN] Would archive this session")
        return None

    # Extract v1.1 statistics
    print("  Extracting v1.1 metadata...")
    thinking_block_stats = extract_thinking_block_tokens(session_path)
    tool_output_stats = extract_tool_output_bytes(session_path)
    if project_root:
        artifacts = extract_artifacts(session_path, project_root)
    else:
        artifacts = {"created": [], "modified": [], "referenced": []}
    relationship_hints = detect_relationship_hints(session_path)

    # Resolve "continues" relationship: if continuation language was
    # detected, look up the most recent session in the same project.
    prev_session_id = _find_previous_session(
        archive_base, project_name, session_id,
    )
    if (
        relationship_hints.get("continues_hint")
        and prev_session_id
    ):
        relationship_hints["continues_session_id"] = prev_session_id
        relationship_hints["detection_notes"].append(
            f"Auto-linked: continues {prev_session_id}"
        )

    print(
        f"    Thinking blocks: {thinking_block_stats['count']} "
        f"(~{thinking_block_stats['total_tokens']:,} tokens estimated)"
    )
    print(
        f"    Tool outputs: {tool_output_stats['total_bytes']:,} bytes total"
    )
    print(
        f"    Artifacts: {len(artifacts['created'])} created, "
        f"{len(artifacts['modified'])} modified, "
        f"{len(artifacts['referenced'])} referenced"
    )

    if relationship_hints.get("detection_notes"):
        print("    Relationship hints detected:")
        for note in relationship_hints["detection_notes"][:3]:
            print(f"      - {note}")

    # Create archive directory
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Copy JSONL file
    compression_info = None
    if use_gzip:
        dest_jsonl = dest_dir / "session.jsonl.gz"
        uncompressed_size = session_path.stat().st_size

        with open(session_path, "rb") as f_in:
            with gzip.open(dest_jsonl, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        compressed_hash = hashlib.sha256()
        with open(dest_jsonl, "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                compressed_hash.update(chunk)

        compression_info = {
            "path": "session.jsonl.gz",
            "compression": "gzip",
            "compressed_bytes": dest_jsonl.stat().st_size,
            "uncompressed_bytes": uncompressed_size,
            "compressed_sha256": compressed_hash.hexdigest(),
        }
        ratio = (
            compression_info["compressed_bytes"] / uncompressed_size
            if uncompressed_size > 0
            else 0.0
        )
        print(
            f"  Compressed to: {dest_jsonl} "
            f"({compression_info['compressed_bytes']:,} bytes, "
            f"{ratio * 100:.1f}% of original)"
        )
    else:
        dest_jsonl = dest_dir / "session.jsonl"
        shutil.copy2(session_path, dest_jsonl)
        print(f"  Copied to: {dest_jsonl}")

    metadata = create_session_metadata(
        session_id=session_id,
        session_path=session_path,
        stats=stats,
        project_root=project_root,
        auto_generated=auto_generated,
        thinking_block_stats=thinking_block_stats,
        tool_output_stats=tool_output_stats,
        artifacts=artifacts,
        relationship_hints=relationship_hints,
        compression_info=compression_info,
        defaults=defaults,
        project_name_override=project_name_override,
        capture_type=capture_type,
    )

    metadata_path = dest_dir / "session.meta.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"  Metadata: {metadata_path}")

    # Set transient key AFTER writing to disk so it does not pollute
    # session.meta.json.  Consumed by update_catalogue() via pop().
    metadata["_archive_directory"] = str(dest_dir)

    return metadata


# -------------------------------------------------------------------------
# Archive lookup (from v1.2)
# -------------------------------------------------------------------------

def find_archive_by_id(
    session_id: str,
    archive_dir: Path,
    project_name: str,
) -> Path | None:
    """
    Find an archive directory by session ID (full or partial match).

    Args:
        session_id: Full or partial session ID to search for.
        archive_dir: Base archive directory (``archive/cc-sessions/``).
        project_name: Project name subdirectory.

    Returns:
        Path to the matching archive directory, or *None*.
    """
    project_dir = archive_dir / project_name
    if not project_dir.exists():
        return None

    matches = sorted(
        d for d in project_dir.iterdir()
        if d.is_dir() and session_id in d.name
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Prefer exact match on the session ID portion
        for m in matches:
            if session_id == m.name.split("_", 1)[-1]:
                return m
        # Fall back to first sorted match
        return matches[0]

    return None
