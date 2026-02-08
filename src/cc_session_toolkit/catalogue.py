"""
Catalogue management — CRUD operations, rebuild, and markdown generation.

Merges catalogue update logic from both archive scripts with the full
rebuild, tag index, and relationship graph from
``generate_session_catalog.py``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from cc_session_toolkit.config import SCHEMA_VERSION
from cc_session_toolkit.naming import get_archive_directory


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
    if catalogue_file.exists():
        try:
            catalogue = json.loads(
                catalogue_file.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            catalogue = {"schema_version": SCHEMA_VERSION, "sessions": []}
    else:
        catalogue = {"schema_version": SCHEMA_VERSION, "sessions": []}

    existing_ids = {s["id"] for s in catalogue["sessions"]}

    for session in new_sessions:
        sid = session["session"]["id"]
        if sid in existing_ids:
            continue

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

        catalogue["sessions"].append({
            "id": sid,
            "title": session["auto_generated"]["title"],
            "directory": rel_dir,
            "started_at": session["session"]["started_at"],
            "duration_minutes": session["session"]["duration_minutes"],
            "tags": session["auto_generated"]["tags"],
            "purpose": session["auto_generated"]["purpose"],
        })

    catalogue["generated_at"] = datetime.now().isoformat()
    catalogue["project"] = project_name
    catalogue["total_sessions"] = len(catalogue["sessions"])

    # Sort by date (None values last)
    catalogue["sessions"].sort(
        key=lambda s: s.get("started_at") or "", reverse=True
    )

    catalogue_file.parent.mkdir(parents=True, exist_ok=True)
    catalogue_file.write_text(
        json.dumps(catalogue, indent=2), encoding="utf-8"
    )
    print(f"\nCatalogue updated: {catalogue_file}")


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
    if not catalogue_file.exists():
        return

    catalogue = json.loads(catalogue_file.read_text(encoding="utf-8"))

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
    catalogue_file.write_text(
        json.dumps(catalogue, indent=2), encoding="utf-8"
    )


# -------------------------------------------------------------------------
# Full catalogue rebuild (from generate_session_catalog.py)
# -------------------------------------------------------------------------

def _scan_sessions(archive_dir: Path) -> list[dict[str, Any]]:
    """
    Scan all archived sessions and collect metadata.

    Args:
        archive_dir: Base archive directory (``archive/cc-sessions/``).

    Returns:
        List of ``{metadata, path, project}`` dictionaries.
    """
    sessions = []

    if not archive_dir.exists():
        return sessions

    for project_dir in archive_dir.iterdir():
        if not project_dir.is_dir():
            continue
        if project_dir.name in ("queries",):
            continue

        for session_dir in project_dir.iterdir():
            if not session_dir.is_dir():
                continue

            meta_file = session_dir / "session.meta.json"
            if not meta_file.exists():
                continue

            try:
                metadata = json.loads(
                    meta_file.read_text(encoding="utf-8")
                )
                sessions.append({
                    "metadata": metadata,
                    "path": session_dir,
                    "project": project_dir.name,
                })
            except (json.JSONDecodeError, OSError) as exc:
                print(f"Warning: Could not read {meta_file}: {exc}")

    return sessions


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

    for session_data in sessions:
        meta = session_data["metadata"]
        project = session_data["project"]
        rel_path = session_data["path"].relative_to(archive_dir)
        session_id = meta.get("session", {}).get("id", "unknown")

        # Project rollup
        if project not in catalogue["projects"]:
            catalogue["projects"][project] = {
                "name": project,
                "session_count": 0,
                "total_duration_minutes": 0,
            }
        catalogue["projects"][project]["session_count"] += 1
        catalogue["projects"][project]["total_duration_minutes"] += (
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
        }
        catalogue["sessions"].append(session_entry)

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

    # Sort
    catalogue["sessions"].sort(
        key=lambda s: s.get("started_at") or "", reverse=True
    )
    catalogue["tag_index"] = dict(sorted(catalogue["tag_index"].items()))

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
