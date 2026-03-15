"""
PostgreSQL query layer for session search and review workflows.

Requires the ``psycopg2`` package (optional dependency, install via
``pip install cc-session-toolkit[db]``). All functions degrade
gracefully when PostgreSQL is unavailable — callers handle errors.

Database: ``claude_memories`` via peer authentication (unix socket).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Union


# ============================================================================
# Configuration
# ============================================================================

DB_NAME = "claude_memories"


# ============================================================================
# Database connection
# ============================================================================

def get_connection():
    """
    Open a PostgreSQL connection via peer auth.

    Returns a psycopg2 connection object.

    Raises:
        ImportError: If psycopg2 is not installed.
        psycopg2.OperationalError: If the database is unreachable.
    """
    import psycopg2  # noqa: WPS433 — optional dependency
    return psycopg2.connect(dbname=DB_NAME)


# ============================================================================
# Result container
# ============================================================================

@dataclass
class SessionRow:
    """Lightweight container for a session query result."""

    id: str
    project: str
    title: str | None
    started_at: datetime | None
    duration_minutes: int | None
    estimated_cost_usd: float | None
    capture_type: str | None


# ============================================================================
# Search query
# ============================================================================

def search_sessions(
    query: str,
    *,
    project: str | None = None,
    since: date | None = None,
    before: date | None = None,
    limit: int = 20,
) -> list[SessionRow]:
    """
    Full-text search across sessions using PostgreSQL tsvector.

    Builds a parameterised query with optional project and date filters.
    Returns matching sessions ordered by start time (most recent first).

    Args:
        query: Free-text search terms.
        project: Filter to a specific project name.
        since: Only sessions started on or after this date.
        before: Only sessions started before this date.
        limit: Maximum number of results.

    Returns:
        List of SessionRow objects.

    Raises:
        ImportError: If psycopg2 is not installed.
        Exception: If the database is unreachable or query fails.
    """
    clauses = [
        "is_active = TRUE",
        (
            "to_tsvector('english', "
            "COALESCE(title, '') || ' ' || "
            "COALESCE(purpose, '') || ' ' || "
            "COALESCE(prompt_summary, '')) "
            "@@ plainto_tsquery('english', %s)"
        ),
    ]
    params: list[Any] = [query]

    if project is not None:
        clauses.append("project = %s")
        params.append(project)

    if since is not None:
        clauses.append("started_at >= %s")
        params.append(since)

    if before is not None:
        clauses.append("started_at < %s")
        params.append(before)

    params.append(limit)

    sql = (
        "SELECT id, project, title, started_at, duration_minutes, "
        "       estimated_cost_usd, capture_type "
        "FROM sessions "
        f"WHERE {' AND '.join(clauses)} "
        "ORDER BY started_at DESC "
        "LIMIT %s"
    )

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [SessionRow(*row) for row in rows]
    finally:
        conn.close()


# ============================================================================
# Untagged sessions query
# ============================================================================

def fetch_untagged_sessions(
    *,
    count_only: bool = False,
) -> Union[list[SessionRow], int]:
    """
    Fetch sessions where needs_review = TRUE.

    Uses the existing ``untagged_sessions`` view (defined in schema.sql).

    Args:
        count_only: If True, return an integer count instead of rows.

    Returns:
        List of SessionRow objects, or an integer if count_only is True.

    Raises:
        ImportError: If psycopg2 is not installed.
        Exception: If the database is unreachable or query fails.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if count_only:
                cur.execute("SELECT COUNT(*) FROM untagged_sessions")
                result = cur.fetchone()
                return result[0] if result else 0

            cur.execute(
                "SELECT id, project, title, started_at, "
                "       duration_minutes, NULL, capture_type "
                "FROM untagged_sessions"
            )
            rows = cur.fetchall()
        return [SessionRow(*row) for row in rows]
    finally:
        conn.close()


# ============================================================================
# Output formatting
# ============================================================================

def format_session_table(
    rows: list[SessionRow],
    *,
    show_cost: bool = True,
    show_id: bool = False,
) -> str:
    """
    Format session rows as a human-readable text table.

    Args:
        rows: Session results to format.
        show_cost: Include the cost column (default True for search).
        show_id: Include the session ID column (default False; True
            for untagged, where the user needs the ID for
            ``cc-session update``).

    Returns:
        Complete formatted table string.
    """
    if not rows:
        return "No sessions found."

    # Column widths
    id_w = 36
    proj_w = 20
    title_w = 30
    date_w = 10
    dur_w = 6
    cost_w = 10

    # Build header
    parts = []
    if show_id:
        parts.append(f"{'ID':<{id_w}}")
    parts.append(f"{'Project':<{proj_w}}")
    parts.append(f"{'Title':<{title_w}}")
    parts.append(f"{'Date':<{date_w}}")
    parts.append(f"{'Dur.':>{dur_w}}")
    if show_cost:
        parts.append(f"{'Cost':>{cost_w}}")

    header = "  ".join(parts)
    separator = "\u2500" * len(header)
    lines = [header, separator]

    # Build rows
    for row in rows:
        parts = []

        if show_id:
            parts.append(f"{row.id:<{id_w}}")

        proj = _truncate(row.project or "unknown", proj_w)
        parts.append(f"{proj:<{proj_w}}")

        title = _truncate(row.title or "Untitled", title_w)
        parts.append(f"{title:<{title_w}}")

        date_str = (
            row.started_at.strftime("%Y-%m-%d")
            if row.started_at else "N/A"
        )
        parts.append(f"{date_str:<{date_w}}")

        dur_str = f"{row.duration_minutes}m" if row.duration_minutes else "N/A"
        parts.append(f"{dur_str:>{dur_w}}")

        if show_cost:
            cost_str = (
                f"${row.estimated_cost_usd:.2f}"
                if row.estimated_cost_usd is not None else "N/A"
            )
            parts.append(f"{cost_str:>{cost_w}}")

        lines.append("  ".join(parts))

    return "\n".join(lines)


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis if it exceeds max_len."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 2] + ".."
