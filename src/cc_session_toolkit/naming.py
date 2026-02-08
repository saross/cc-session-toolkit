"""
Human-readable archive directory naming utilities.

Ported from map-reader-llm v1.1 (slugify + title-based directory names).
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any


def slugify(title: str, max_length: int = 45) -> str:
    """
    Convert a title to a URL-friendly slug for directory naming.

    Args:
        title: Session title (e.g. "VLM Pipeline Development and Codebase").
        max_length: Maximum slug length (default 45 characters).

    Returns:
        Lowercase, hyphenated slug
        (e.g. ``vlm-pipeline-development-and-codebase``).
    """
    # Lowercase + replace non-alphanumeric with hyphens
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower())
    slug = slug.strip("-")
    slug = re.sub(r"-+", "-", slug)

    # Truncate at word boundary if too long
    if len(slug) > max_length:
        slug = slug[:max_length].rsplit("-", 1)[0]

    return slug


def get_archive_directory(
    session_id: str,
    stats: dict[str, Any],
    archive_dir: Path,
    project_name: str,
    title: str | None = None,
) -> Path:
    """
    Determine the archive directory for a session.

    Format::

        archive/cc-sessions/{project}/{timestamp}[_agent]_{slug}/

    - Timestamp: ``YYYY-MM-DDTHH-MM`` (from session start)
    - Agent prefix: added for agent sub-sessions
    - Slug: derived from *title* if provided, otherwise short session ID
    - Short ID suffix added only when needed for uniqueness

    Args:
        session_id: Session UUID or ``agent-xxx`` identifier.
        stats: Session statistics (must include ``started_at``).
        archive_dir: Base archive directory (``archive/cc-sessions/``).
        project_name: Project name for the subdirectory.
        title: Optional session title for a human-readable slug.

    Returns:
        Path to the archive directory.
    """
    project_dir = archive_dir / project_name

    is_agent = session_id.startswith("agent-")

    # Short ID for uniqueness fallback
    if is_agent:
        short_id = session_id.replace("agent-", "")[:7]
    else:
        short_id = session_id[:8]

    # Parse timestamp
    timestamp = None
    started_at = stats.get("started_at")
    if started_at:
        try:
            dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            timestamp = dt.strftime("%Y-%m-%dT%H-%M")
        except (ValueError, TypeError):
            pass

    # Build directory name
    if title and title != "Untitled Session":
        slug = slugify(title)
        if is_agent:
            dir_name = (
                f"{timestamp}_agent_{slug}" if timestamp
                else f"agent_{slug}"
            )
        else:
            dir_name = f"{timestamp}_{slug}" if timestamp else slug

        # Check for uniqueness; add short_id suffix if needed
        candidate = project_dir / dir_name
        if candidate.exists():
            dir_name = f"{dir_name}_{short_id}"
    else:
        # Fallback to short_id-based naming (legacy behaviour)
        if is_agent:
            dir_name = (
                f"{timestamp}_{session_id}" if timestamp else session_id
            )
        else:
            dir_name = f"{timestamp}_{short_id}" if timestamp else short_id

    return project_dir / dir_name
