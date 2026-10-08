"""
Catalogue management — CRUD operations, rebuild, and markdown generation.

Merges catalogue update logic from both archive scripts with the full
rebuild, tag index, and relationship graph from
``generate_session_catalog.py``.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

try:  # POSIX only; elsewhere writes stay atomic but unlocked.
    import fcntl
except ImportError:  # pragma: no cover - not exercised on Linux
    fcntl = None  # type: ignore[assignment]

from cc_session_toolkit.config import SCHEMA_VERSION
from cc_session_toolkit.naming import get_archive_directory


# -------------------------------------------------------------------------
# Safe writes (2026-10-08)
# -------------------------------------------------------------------------
#
# Hooks used to rewrite CATALOG.json with a bare ``write_text`` and no lock.
# Two sessions ending together could each read the old catalogue and write
# it back with only their own entry (a lost update), and a hook killed
# mid-write could leave truncated JSON. Every read-modify-write now holds
# an exclusive lock, and every write goes through a temporary file and an
# atomic rename.


def _lock_path(catalogue_file: Path) -> Path:
    """Return the catalogue's lock file.

    A sibling file, not the catalogue itself, so the rename that replaces
    the catalogue cannot pull the lock out from under a waiting writer. The
    name matches personal-assistant's ``bulk-archive.py write_catalogue``,
    so the two writers exclude each other.
    """
    return catalogue_file.with_name(catalogue_file.name + ".lock")


@contextlib.contextmanager
def catalogue_lock(catalogue_file: Path) -> Iterator[None]:
    """Hold an exclusive lock on *catalogue_file* for a read-modify-write.

    Not re-entrant: never nest two of these on the same catalogue in one
    process, because a second ``flock`` on a new descriptor would wait for
    the first. On a filesystem that refuses ``flock`` the body still runs,
    unlocked, rather than failing the archive.
    """
    catalogue_file.parent.mkdir(parents=True, exist_ok=True)
    with open(_lock_path(catalogue_file), "a", encoding="utf-8") as handle:
        locked = False
        if fcntl is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                locked = True
            except OSError:
                locked = False
        try:
            yield
        finally:
            if locked:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write *data* to *path* via a temporary file, fsync and rename.

    Keeps the existing file's permission bits (``mkstemp`` creates 0600),
    and removes the temporary file if anything fails, so a reader sees
    either the old catalogue or the new one, never part of one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        mode = path.stat().st_mode & 0o777
    else:
        umask = os.umask(0)
        os.umask(umask)
        mode = 0o666 & ~umask
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def write_catalogue(catalogue_file: Path, catalogue: dict[str, Any]) -> None:
    """Replace *catalogue_file* with *catalogue*, locked and atomically.

    The writer for full rebuilds. Incremental updates take the same lock
    around their whole read-modify-write instead.
    """
    with catalogue_lock(catalogue_file):
        _write_json_atomic(catalogue_file, catalogue)


# -------------------------------------------------------------------------
# Incremental catalogue updates
# -------------------------------------------------------------------------

def update_catalogue(
    new_sessions: list[dict[str, Any]],
    catalogue_file: Path,
    archive_dir: Path,
    project_name: str,
) -> None:
    """
    Append newly archived sessions to ``CATALOG.json``.

    Args:
        new_sessions: List of session metadata dictionaries.
        catalogue_file: Path to ``CATALOG.json``.
        archive_dir: Base archive directory.
        project_name: Project name.
    """
    with catalogue_lock(catalogue_file):
        _update_catalogue_locked(
            new_sessions, catalogue_file, archive_dir, project_name
        )
    print(f"\nCatalogue updated: {catalogue_file}")


def _update_catalogue_locked(
    new_sessions: list[dict[str, Any]],
    catalogue_file: Path,
    archive_dir: Path,
    project_name: str,
) -> None:
    """Body of :func:`update_catalogue`; the caller holds the lock."""
    if catalogue_file.exists():
        try:
            catalogue = json.loads(
                catalogue_file.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            # Unreadable (e.g. truncated by a writer killed before writes
            # became atomic). Rebuild from disk: the old fallback started
            # an EMPTY catalogue, so every other session vanished from it.
            print(
                f"Warning: {catalogue_file} is unreadable; "
                "rebuilding it from the archive directories"
            )
            catalogue = rebuild_catalogue(archive_dir)
    else:
        catalogue = {"schema_version": SCHEMA_VERSION, "sessions": []}

    existing_ids = {s["id"] for s in catalogue["sessions"]}

    for session in new_sessions:
        sid = session["session"]["id"]
        if sid in existing_ids:
            continue

        # Use actual archive directory if recorded by archive_session()
        # (avoids path reconstruction mismatch).  The transient
        # ``_archive_directory`` key is consumed here and not persisted.
        actual_dir = session.pop("_archive_directory", None)
        if actual_dir:
            actual_path = Path(actual_dir)
            try:
                rel_dir = str(actual_path.relative_to(archive_dir))
            except ValueError:
                rel_dir = str(actual_path)
        else:
            # Fallback: reconstruct path (legacy callers without the key)
            title = session.get("auto_generated", {}).get("title")
            if title == "Untitled Session":
                title = None
            directory = get_archive_directory(
                session_id=sid,
                stats={"started_at": session["session"]["started_at"]},
                archive_dir=archive_dir,
                project_name=project_name,
                title=title,
            )
            try:
                rel_dir = str(directory.relative_to(archive_dir))
            except ValueError:
                rel_dir = str(directory)

        auto = session.get("auto_generated", {})
        sess = session.get("session", {})
        subagents_summary = (
            session.get("statistics", {}).get("subagents_summary", {}) or {}
        )
        catalogue["sessions"].append({
            "id": sid,
            "title": auto.get("title", "Untitled"),
            "directory": rel_dir,
            "started_at": sess.get("started_at"),
            "duration_minutes": sess.get("duration_minutes", 0),
            "tags": auto.get("tags", []),
            "purpose": auto.get("purpose", ""),
            "subagent_count": subagents_summary.get("count", 0),
            "subagent_cost_usd": subagents_summary.get(
                "estimated_cost_usd", 0.0
            ),
        })

    catalogue["generated_at"] = datetime.now().isoformat()
    catalogue["project"] = project_name
    catalogue["total_sessions"] = len(catalogue["sessions"])

    # Sort by date (None values last)
    catalogue["sessions"].sort(
        key=lambda s: s.get("started_at") or "", reverse=True
    )

    _write_json_atomic(catalogue_file, catalogue)


def update_catalogue_entry(
    session_id: str,
    meta: dict[str, Any],
    catalogue_file: Path,
) -> None:
    """
    Update a single session's entry in the catalogue (from v1.2).

    Args:
        session_id: Full or partial session ID.
        meta: Updated metadata dictionary.
        catalogue_file: Path to ``CATALOG.json``.
    """
    with catalogue_lock(catalogue_file):
        if not catalogue_file.exists():
            return
        catalogue = json.loads(catalogue_file.read_text(encoding="utf-8"))
        _apply_entry_update(catalogue, session_id, meta)
        _write_json_atomic(catalogue_file, catalogue)


def _apply_entry_update(
    catalogue: dict[str, Any], session_id: str, meta: dict[str, Any],
) -> None:
    """Refresh one session's title, purpose and tags in *catalogue*."""
    for session in catalogue.get("sessions", []):
        entry_id = session.get("id", "")
        if entry_id and entry_id == session_id:
            session["title"] = (
                meta.get("auto_generated", {}).get("title", "Untitled")
            )
            session["purpose"] = (
                meta.get("auto_generated", {}).get("purpose", "")
            )
            session["tags"] = (
                meta.get("auto_generated", {}).get("tags", [])
            )
            break

    catalogue["generated_at"] = datetime.now().isoformat()


# -------------------------------------------------------------------------
# Full catalogue rebuild (from generate_session_catalog.py)
# -------------------------------------------------------------------------

def _scan_sessions(archive_dir: Path) -> list[dict[str, Any]]:
    """
    Scan all archived sessions and collect metadata.

    Finds every directory holding a ``session.meta.json`` at any depth,
    skipping ``subagents/`` and the top-level ``queries/``. Until
    2026-10-08 only ``<project>/<entry>`` was scanned, so ``_legacy/
    <project>/<entry>`` (47 sessions on amd-tower) and other nested entries
    were never catalogued, and the hooks could not find them.

    The project of a ``<project>/<entry>`` directory is its top-level
    directory, as before. A nested entry takes the project its metadata
    names, else its top-level directory.

    Args:
        archive_dir: Base archive directory (``archive/cc-sessions/``).

    Returns:
        List of ``{metadata, path, project}`` dictionaries.
    """
    sessions: list[dict[str, Any]] = []

    if not archive_dir.exists():
        return sessions

    for meta_file in sorted(archive_dir.rglob("session.meta.json")):
        rel = meta_file.parent.relative_to(archive_dir)
        if not rel.parts or "subagents" in rel.parts or rel.parts[0] == "queries":
            continue
        try:
            metadata = json.loads(meta_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Warning: Could not read {meta_file}: {exc}")
            continue
        if len(rel.parts) <= 2:
            project = rel.parts[0]
        else:
            project = (
                (metadata.get("project") or {}).get("name") or rel.parts[0]
            )
        sessions.append({
            "metadata": metadata,
            "path": meta_file.parent,
            "project": project,
        })

    return sessions


def _recorded_bytes(meta: dict[str, Any]) -> int:
    """The transcript length a record claims, or 0 when it records none."""
    value = (meta.get("archive") or {}).get("jsonl_bytes_uncompressed")
    return value if isinstance(value, int) else 0


def rebuild_catalogue(archive_dir: Path) -> dict[str, Any]:
    """
    Full catalogue rebuild from archived session metadata files.

    Scans all project subdirectories, builds project rollups, tag index,
    and relationship graph.

    Args:
        archive_dir: Base archive directory (``archive/cc-sessions/``).

    Returns:
        Complete catalogue dictionary (schema v1.1).
    """
    sessions = _scan_sessions(archive_dir)

    catalogue: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(),
        "total_sessions": len(sessions),
        "projects": {},
        "sessions": [],
        "tag_index": {},
        "relationship_graph": {},
    }

    # One session can live in several directories (a renamed title, a
    # re-capture, the old nested layout). Lookups such as
    # archive.find_archive_directory take the FIRST entry for an id, so
    # the copies are ordered deliberately: the most complete (most
    # recorded transcript bytes) first, then the shallowest, then by path.
    # The final date sort below is stable, so this order survives it.
    sessions.sort(key=lambda d: (
        -_recorded_bytes(d["metadata"]),
        len(d["path"].relative_to(archive_dir).parts),
        str(d["path"].relative_to(archive_dir)),
    ))

    seen_ids: set[str] = set()
    for session_data in sessions:
        meta = session_data["metadata"]
        project = session_data["project"]
        rel_path = session_data["path"].relative_to(archive_dir)
        session_id = meta.get("session", {}).get("id", "unknown")
        # Rollups, the tag index and the relationship graph count each
        # session once, from its preferred copy; every copy stays listed.
        first_copy = session_id not in seen_ids
        seen_ids.add(session_id)

        # Project rollup
        if first_copy:
            rollup = catalogue["projects"].setdefault(project, {
                "name": project,
                "session_count": 0,
                "total_duration_minutes": 0,
            })
            rollup["session_count"] += 1
            rollup["total_duration_minutes"] += (
                meta.get("session", {}).get("duration_minutes", 0)
            )

        # Extract v1.1 fields
        relationships = meta.get("relationships", {})
        artifacts = meta.get("artifacts", {})
        thinking_blocks = meta.get("thinking_blocks", {})

        artifact_counts = {
            "created": len(artifacts.get("created", [])),
            "modified": len(artifacts.get("modified", [])),
            "referenced": len(artifacts.get("referenced", [])),
        }

        subagents_summary = (
            meta.get("statistics", {}).get("subagents_summary", {}) or {}
        )

        session_entry = {
            "id": session_id,
            "project": project,
            "directory": str(rel_path),
            "title": meta.get("auto_generated", {}).get(
                "title", "Untitled"
            ),
            "purpose": meta.get("auto_generated", {}).get("purpose", ""),
            "tags": meta.get("auto_generated", {}).get("tags", []),
            "started_at": meta.get("session", {}).get("started_at"),
            "duration_minutes": meta.get("session", {}).get(
                "duration_minutes", 0
            ),
            "model": meta.get("model", {}).get("model_id", "unknown"),
            "statistics": {
                "turns": meta.get("statistics", {}).get("turns", 0),
                "tool_calls": meta.get("statistics", {})
                .get("tool_calls", {})
                .get("total", 0),
                "thinking_blocks": meta.get("statistics", {}).get(
                    "thinking_blocks", 0
                ),
            },
            "relationships": {
                "continues": relationships.get("continues"),
                "isPartOf": relationships.get("isPartOf", []),
            },
            "artifact_counts": artifact_counts,
            "thinking_blocks_tokens": thinking_blocks.get(
                "total_tokens", 0
            ),
            "subagent_count": subagents_summary.get("count", 0),
            "subagent_cost_usd": subagents_summary.get(
                "estimated_cost_usd", 0.0
            ),
        }
        catalogue["sessions"].append(session_entry)
        if not first_copy:
            continue

        # Tag index
        for tag in session_entry["tags"]:
            if tag not in catalogue["tag_index"]:
                catalogue["tag_index"][tag] = []
            catalogue["tag_index"][tag].append(session_id)

        # Relationship graph
        continues_target = relationships.get("continues")
        if continues_target:
            if continues_target not in catalogue["relationship_graph"]:
                catalogue["relationship_graph"][continues_target] = {
                    "continuedBy": [],
                }
            catalogue["relationship_graph"][continues_target][
                "continuedBy"
            ].append(session_id)

    # Sort by date, newest first. Stable, so each session's copies keep
    # the preference order set above.
    catalogue["sessions"].sort(
        key=lambda s: s.get("started_at") or "", reverse=True
    )
    catalogue["tag_index"] = dict(sorted(catalogue["tag_index"].items()))
    catalogue["unique_sessions"] = len(seen_ids)

    return catalogue


# -------------------------------------------------------------------------
# Markdown generation (from generate_session_catalog.py)
# -------------------------------------------------------------------------

def _format_duration(minutes: int) -> str:
    """Format duration in minutes to a readable string."""
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h" if mins == 0 else f"{hours}h {mins}m"


def _format_date(iso_date: str | None) -> str:
    """Format ISO date to readable format."""
    if not iso_date:
        return "Unknown"
    try:
        dt = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return iso_date[:16] if iso_date else "Unknown"


def generate_catalogue_markdown(catalogue: dict[str, Any]) -> str:
    """
    Generate human-readable ``CATALOG.md`` content.

    Args:
        catalogue: Catalogue dictionary from :func:`rebuild_catalogue`.

    Returns:
        Markdown string.
    """
    lines = [
        "# CC Session Catalogue",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## Overview",
        "",
        f"- **Total sessions**: {catalogue['total_sessions']}",
        f"- **Projects**: {len(catalogue['projects'])}",
        "",
    ]

    # Project summary
    if catalogue["projects"]:
        lines.extend([
            "### Projects",
            "",
            "| Project | Sessions | Total Duration |",
            "|---------|----------|----------------|",
        ])
        for project, info in sorted(catalogue["projects"].items()):
            duration = _format_duration(info["total_duration_minutes"])
            lines.append(
                f"| {project} | {info['session_count']} | {duration} |"
            )
        lines.append("")

    # Sessions by project
    lines.extend(["## Sessions", ""])

    current_project = None
    for session in catalogue["sessions"]:
        if session["project"] != current_project:
            current_project = session["project"]
            lines.extend([f"### {current_project}", ""])

        date = _format_date(session["started_at"])
        duration = _format_duration(session["duration_minutes"])
        tags = ", ".join(session["tags"]) if session["tags"] else "-"
        stats = session["statistics"]

        artifact_counts = session.get("artifact_counts", {})
        artifacts_str = (
            f"{artifact_counts.get('created', 0)} created, "
            f"{artifact_counts.get('modified', 0)} modified, "
            f"{artifact_counts.get('referenced', 0)} referenced"
        )
        thinking_tokens = session.get("thinking_blocks_tokens", 0)

        lines.extend([
            f"#### {session['title']}",
            "",
            f"- **Date**: {date}",
            f"- **Duration**: {duration}",
            f"- **Directory**: `{session['directory']}`",
            f"- **Model**: {session['model']}",
            f"- **Statistics**: {stats['turns']} turns, "
            f"{stats['tool_calls']} tool calls, "
            f"{stats['thinking_blocks']} thinking blocks "
            f"(~{thinking_tokens:,} tokens)",
            f"- **Artifacts**: {artifacts_str}",
            f"- **Tags**: {tags}",
            "",
        ])

        if session["purpose"]:
            lines.extend([f"> {session['purpose']}", ""])

    # Tag index
    if catalogue["tag_index"]:
        lines.extend(["## Tag Index", ""])
        for tag, session_ids in sorted(catalogue["tag_index"].items()):
            lines.append(f"- **{tag}**: {len(session_ids)} session(s)")
        lines.append("")

    # Footer
    lines.extend([
        "---",
        "",
        "*This catalogue is auto-generated. Do not edit manually.*",
        "*Regenerate with: `cc-session catalogue --rebuild`*",
        "",
    ])

    return "\n".join(lines)
