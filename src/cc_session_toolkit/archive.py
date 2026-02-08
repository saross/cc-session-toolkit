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
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from cc_session_toolkit.config import (
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
# Metadata creation
# -------------------------------------------------------------------------

def create_session_metadata(
    session_id: str,
    session_path: Path,
    stats: dict[str, Any],
    project_root: Path,
    *,
    auto_generated: dict[str, Any] | None = None,
    thinking_block_stats: dict[str, Any] | None = None,
    tool_output_stats: dict[str, Any] | None = None,
    artifacts: dict[str, list] | None = None,
    relationship_hints: dict[str, Any] | None = None,
    compression_info: dict[str, Any] | None = None,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create the complete ``session.meta.json`` structure (v1.1 schema).

    Args:
        session_id: Session identifier.
        session_path: Path to the source JSONL file.
        stats: Extracted session statistics.
        project_root: Project root directory.
        auto_generated: CC-generated metadata (title, purpose, tags,
            three_ps).
        thinking_block_stats: Thinking-block token statistics.
        tool_output_stats: Tool output byte statistics.
        artifacts: Files created/modified/referenced.
        relationship_hints: Detected relationship hints.
        compression_info: Compression metadata (if using gzip).
        defaults: Loaded defaults from ``archive-defaults.yaml``.

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

    project_name = get_project_name(project_root)

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
    relationships = {
        "continues": None,
        "continuedBy": None,
        "isPartOf": default_is_part_of,
        "isParallelTo": [],
        "supersedes": None,
        "references": [],
        "branchesFrom": None,
    }

    relationship_hints_info = relationship_hints or {}

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
            "directory": str(project_root),
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

    if relationship_hints_info.get("detection_notes"):
        metadata["_relationship_hints"] = relationship_hints_info

    return metadata


# -------------------------------------------------------------------------
# Archive a session
# -------------------------------------------------------------------------

def archive_session(
    session_path: Path,
    project_root: Path,
    *,
    dry_run: bool = False,
    stats_only: bool = False,
    use_gzip: bool = False,
    title: str | None = None,
) -> dict[str, Any] | None:
    """
    Archive a single session with v1.1 schema.

    Args:
        session_path: Path to source JSONL file.
        project_root: Project root directory.
        dry_run: Preview without writing files.
        stats_only: Skip CC metadata generation (use placeholders).
        use_gzip: Compress the JSONL file.
        title: Optional session title for human-readable directory name.

    Returns:
        Session metadata dictionary, or *None* if skipped.
    """
    session_id = get_session_id(session_path)
    stats = extract_session_stats(session_path)
    archive_base = get_archive_dir(project_root)
    project_name = get_project_name(project_root)
    defaults_file = get_defaults_file(project_root)
    defaults = load_defaults(defaults_file)

    dest_dir = get_archive_directory(
        session_id=session_id,
        stats=stats,
        archive_dir=archive_base,
        project_name=project_name,
        title=title,
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
    artifacts = extract_artifacts(session_path, project_root)
    relationship_hints = detect_relationship_hints(session_path)

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

    # Generate metadata
    auto_generated = None
    if not stats_only:
        print("\n" + "=" * 60)
        print("METADATA GENERATION")
        print("=" * 60)
        print(generate_metadata_prompt(session_id, stats))
        print("=" * 60)
        print(
            "\nPlease provide the JSON metadata above, "
            "or press Enter to skip."
        )
        auto_generated = {
            "title": title or "Untitled Session",
            "purpose": "Metadata generation requires interactive CC session",
            "tags": [],
            "three_ps": {
                "prompt_summary": "",
                "process_summary": "",
                "provenance_summary": "",
            },
        }

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
    )

    metadata_path = dest_dir / "session.meta.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"  Metadata: {metadata_path}")

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
