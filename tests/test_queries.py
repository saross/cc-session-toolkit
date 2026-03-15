"""
Tests for queries.py — session search, untagged listing, and output formatting.

Tests pure functions with mocked database connections. No running
PostgreSQL instance required.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from cc_session_toolkit.queries import (
    SessionRow,
    fetch_untagged_sessions,
    format_session_table,
    search_sessions,
    _truncate,
)


# ============================================================================
# Fixtures
# ============================================================================


def _make_row(
    id: str = "abc12345-6789-0000-aaaa-bbbbccccdddd",
    project: str = "personal-assistant",
    title: str = "Test Session",
    started_at: datetime | None = None,
    duration_minutes: int = 42,
    estimated_cost_usd: float = 0.12,
    capture_type: str = "session_end",
) -> SessionRow:
    """Create a SessionRow with sensible defaults."""
    if started_at is None:
        started_at = datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc)
    return SessionRow(
        id=id,
        project=project,
        title=title,
        started_at=started_at,
        duration_minutes=duration_minutes,
        estimated_cost_usd=estimated_cost_usd,
        capture_type=capture_type,
    )


@pytest.fixture()
def mock_db():
    """
    Mock psycopg2 connection and cursor.

    Returns the mock cursor so tests can configure fetchall/fetchone.
    Patches get_connection to return the mock connection.
    """
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = (0,)
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with patch(
        "cc_session_toolkit.queries.get_connection",
        return_value=mock_conn,
    ):
        yield mock_cursor


# ============================================================================
# search_sessions
# ============================================================================


class TestSearchSessions:
    """Tests for :func:`search_sessions`."""

    def test_basic_search_returns_session_rows(self, mock_db):
        """Full-text search returns SessionRow objects."""
        now = datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc)
        mock_db.fetchall.return_value = [
            ("id-1", "proj-a", "Title A", now, 30, 1.50, "session_end"),
        ]
        results = search_sessions("pipeline")
        assert len(results) == 1
        assert isinstance(results[0], SessionRow)
        assert results[0].project == "proj-a"
        assert results[0].title == "Title A"

    def test_query_parameter_passed(self, mock_db):
        """The search query is passed as a parameter."""
        mock_db.fetchall.return_value = []
        search_sessions("mound detection")
        call_args = mock_db.execute.call_args
        sql, params = call_args[0]
        assert "plainto_tsquery" in sql
        assert params[0] == "mound detection"

    def test_project_filter(self, mock_db):
        """--project adds a WHERE clause."""
        mock_db.fetchall.return_value = []
        search_sessions("test", project="map-reader-llm")
        sql, params = mock_db.execute.call_args[0]
        assert "project = %s" in sql
        assert "map-reader-llm" in params

    def test_since_filter(self, mock_db):
        """--since adds a started_at >= filter."""
        mock_db.fetchall.return_value = []
        since_date = date(2026, 3, 1)
        search_sessions("test", since=since_date)
        sql, params = mock_db.execute.call_args[0]
        assert "started_at >= %s" in sql
        assert since_date in params

    def test_before_filter(self, mock_db):
        """--before adds a started_at < filter."""
        mock_db.fetchall.return_value = []
        before_date = date(2026, 3, 15)
        search_sessions("test", before=before_date)
        sql, params = mock_db.execute.call_args[0]
        assert "started_at < %s" in sql
        assert before_date in params

    def test_combined_filters(self, mock_db):
        """All filters can be combined."""
        mock_db.fetchall.return_value = []
        search_sessions(
            "test",
            project="proj",
            since=date(2026, 3, 1),
            before=date(2026, 3, 15),
            limit=5,
        )
        sql, params = mock_db.execute.call_args[0]
        assert "project = %s" in sql
        assert "started_at >= %s" in sql
        assert "started_at < %s" in sql
        assert params[-1] == 5  # limit is last param

    def test_empty_results(self, mock_db):
        """Returns empty list when no matches found."""
        mock_db.fetchall.return_value = []
        results = search_sessions("nonexistent")
        assert results == []

    def test_limit_parameter(self, mock_db):
        """Limit is passed as the final parameter."""
        mock_db.fetchall.return_value = []
        search_sessions("test", limit=5)
        _, params = mock_db.execute.call_args[0]
        assert params[-1] == 5

    def test_default_limit_is_20(self, mock_db):
        """Default limit should be 20."""
        mock_db.fetchall.return_value = []
        search_sessions("test")
        _, params = mock_db.execute.call_args[0]
        assert params[-1] == 20

    def test_connection_closed_on_success(self):
        """Connection is closed after successful query."""
        with patch(
            "cc_session_toolkit.queries.get_connection",
        ) as mock_get:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = []
            mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
            mock_cursor.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value = mock_cursor
            mock_get.return_value = mock_conn

            search_sessions("test")
            mock_conn.close.assert_called_once()

    def test_connection_closed_on_error(self):
        """Connection is closed even if query raises."""
        with patch(
            "cc_session_toolkit.queries.get_connection",
        ) as mock_get:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchall.side_effect = RuntimeError("boom")
            mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
            mock_cursor.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value = mock_cursor
            mock_get.return_value = mock_conn

            with pytest.raises(RuntimeError):
                search_sessions("test")
            mock_conn.close.assert_called_once()


# ============================================================================
# fetch_untagged_sessions
# ============================================================================


class TestFetchUntaggedSessions:
    """Tests for :func:`fetch_untagged_sessions`."""

    def test_returns_session_rows(self, mock_db):
        """Untagged query returns SessionRow objects."""
        now = datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc)
        mock_db.fetchall.return_value = [
            ("id-1", "proj-a", "Title A", now, 30, None, "session_end"),
        ]
        results = fetch_untagged_sessions()
        assert len(results) == 1
        assert isinstance(results[0], SessionRow)
        assert results[0].estimated_cost_usd is None

    def test_count_only_returns_integer(self, mock_db):
        """count_only=True returns an integer."""
        mock_db.fetchone.return_value = (7,)
        result = fetch_untagged_sessions(count_only=True)
        assert result == 7
        assert isinstance(result, int)

    def test_count_only_zero(self, mock_db):
        """count_only with no results returns 0."""
        mock_db.fetchone.return_value = (0,)
        result = fetch_untagged_sessions(count_only=True)
        assert result == 0

    def test_empty_results(self, mock_db):
        """Returns empty list when no untagged sessions."""
        mock_db.fetchall.return_value = []
        results = fetch_untagged_sessions()
        assert results == []


# ============================================================================
# format_session_table
# ============================================================================


class TestFormatSessionTable:
    """Tests for :func:`format_session_table`."""

    def test_basic_formatting(self):
        """Formats a list of SessionRow objects as a table."""
        rows = [_make_row()]
        output = format_session_table(rows)
        assert "personal-assistant" in output
        assert "Test Session" in output
        assert "2026-03-15" in output
        assert "42m" in output
        assert "$0.12" in output

    def test_header_present(self):
        """Output includes a header row."""
        rows = [_make_row()]
        output = format_session_table(rows)
        assert "Project" in output
        assert "Title" in output
        assert "Date" in output

    def test_separator_present(self):
        """Output includes a Unicode separator line."""
        rows = [_make_row()]
        output = format_session_table(rows)
        assert "\u2500" in output

    def test_truncates_long_titles(self):
        """Titles exceeding 30 chars are truncated with ellipsis."""
        rows = [_make_row(title="A" * 40)]
        output = format_session_table(rows)
        assert "A" * 40 not in output
        assert "A" * 28 + ".." in output

    def test_truncates_long_project_names(self):
        """Project names exceeding 20 chars are truncated."""
        rows = [_make_row(project="very-long-project-name-here")]
        output = format_session_table(rows)
        assert "very-long-project-name-here" not in output
        assert ".." in output

    def test_handles_none_title(self):
        """None title displays as 'Untitled'."""
        rows = [_make_row(title=None)]
        output = format_session_table(rows)
        assert "Untitled" in output

    def test_handles_none_started_at(self):
        """None started_at displays as 'N/A'."""
        row = _make_row()
        row.started_at = None
        output = format_session_table([row])
        assert "N/A" in output

    def test_handles_zero_duration(self):
        """duration_minutes=0 displays as '0m', not 'N/A'."""
        rows = [_make_row(duration_minutes=0)]
        output = format_session_table(rows)
        assert "0m" in output
        assert "N/A" not in output or "N/A" not in output.split("\n")[-1]

    def test_handles_none_duration(self):
        """None duration displays as 'N/A'."""
        row = _make_row()
        row.duration_minutes = None
        output = format_session_table([row])
        assert "N/A" in output

    def test_handles_none_cost(self):
        """None cost displays as 'N/A'."""
        rows = [_make_row(estimated_cost_usd=None)]
        output = format_session_table(rows)
        assert "N/A" in output

    def test_empty_list(self):
        """Empty list returns 'No sessions found.'."""
        output = format_session_table([])
        assert output == "No sessions found."

    def test_cost_hidden_when_show_cost_false(self):
        """show_cost=False omits the cost column."""
        rows = [_make_row(estimated_cost_usd=99.99)]
        output = format_session_table(rows, show_cost=False)
        assert "$99.99" not in output
        assert "Cost" not in output

    def test_id_shown_when_show_id_true(self):
        """show_id=True includes the session ID column."""
        rows = [_make_row(id="abc12345-6789-0000-aaaa-bbbbccccdddd")]
        output = format_session_table(rows, show_id=True)
        assert "abc12345" in output
        assert "ID" in output

    def test_id_hidden_by_default(self):
        """Session ID is not shown by default."""
        rows = [_make_row(id="abc12345-6789-0000-aaaa-bbbbccccdddd")]
        output = format_session_table(rows)
        assert "abc12345" not in output

    def test_multiple_rows(self):
        """Multiple rows are all formatted."""
        rows = [
            _make_row(title="First Session"),
            _make_row(title="Second Session"),
        ]
        output = format_session_table(rows)
        assert "First Session" in output
        assert "Second Session" in output


# ============================================================================
# _truncate helper
# ============================================================================


class TestTruncate:
    """Tests for the _truncate helper."""

    def test_short_text_unchanged(self):
        """Text shorter than max_len is returned as-is."""
        assert _truncate("hello", 10) == "hello"

    def test_exact_length_unchanged(self):
        """Text exactly at max_len is returned as-is."""
        assert _truncate("12345", 5) == "12345"

    def test_long_text_truncated(self):
        """Text longer than max_len gets ellipsis."""
        result = _truncate("hello world", 8)
        assert len(result) == 8
        assert result.endswith("..")

    def test_truncation_preserves_length(self):
        """Truncated text is exactly max_len characters."""
        result = _truncate("a" * 50, 20)
        assert len(result) == 20
