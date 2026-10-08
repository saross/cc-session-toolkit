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
import time
from collections.abc import Collection, Iterator
from datetime import datetime
from pathlib import Path, PurePosixPath
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


class CatalogueLockError(RuntimeError):
    """The catalogue lock could not be taken (refused, or timed out)."""


class CatalogueScanError(RuntimeError):
    """A strict rebuild met unreadable metadata, so its scan is incomplete."""


class CatalogueRebuildRefused(RuntimeError):
    """A rebuild would drop entries it cannot account for; nothing written."""


#: Version of the transactional rebuild API, for callers outside this
#: package (personal-assistant's scripts) to check before relying on it.
#: 2 (2026-10-08, review round two): ``allow_removed`` replaces
#: ``max_removed_fraction``, and entries carry ``transcript_bytes``.
REBUILD_API_VERSION = 2


def _acquire_lock(handle: Any, timeout: float | None) -> None:
    """Take an exclusive ``flock`` on *handle*, waiting at most *timeout* s."""
    if timeout is None:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            raise CatalogueLockError(f"cannot lock the catalogue: {exc}") from exc
        return
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise CatalogueLockError(
                    f"catalogue lock still held after {timeout:g} s"
                ) from None
            time.sleep(0.1)
        except OSError as exc:
            raise CatalogueLockError(f"cannot lock the catalogue: {exc}") from exc


@contextlib.contextmanager
def catalogue_lock(
    catalogue_file: Path, *, timeout: float | None = None,
) -> Iterator[None]:
    """Hold an exclusive lock on *catalogue_file* for a read-modify-write.

    Fails closed (review 2026-10-08): if the lock cannot be taken, raises
    :class:`CatalogueLockError` and the body never runs. The first version
    ran the body unlocked, which reopened the lost-update race the lock
    exists to close. A hook that fails here has already written its
    metadata, so the next rebuild still catalogues it. *timeout* bounds the
    wait (the daily sync's rebuild must not hang on a stalled mount).

    Not re-entrant: never nest two of these on the same catalogue in one
    process, because a second ``flock`` on a new descriptor would wait for
    the first. Over SSHFS the lock excludes writers on this machine only.
    """
    catalogue_file.parent.mkdir(parents=True, exist_ok=True)
    with open(_lock_path(catalogue_file), "a", encoding="utf-8") as handle:
        if fcntl is not None:
            _acquire_lock(handle, timeout)
        try:
            yield
        finally:
            if fcntl is not None:
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

    Locks only the publication, so a scan done before calling it can miss
    an entry a hook adds meanwhile. For rebuilds use
    :func:`rebuild_and_write_catalogue`, which holds one lock across the
    whole transaction.
    """
    with catalogue_lock(catalogue_file):
        _write_json_atomic(catalogue_file, catalogue)


#: How to recover from an unreadable catalogue, named in every refusal.
#: Deliberately a manual step: with no baseline there is nothing to check a
#: rebuild's identities against, so an operator decides to start afresh.
_RECOVERY_HINT = (
    "to recover, move it aside and rebuild (cc-session catalogue --rebuild), "
    "which then publishes the store as a strict scan finds it"
)


def _read_baseline(catalogue_file: Path) -> dict[str, Any] | None:
    """Return the existing catalogue, *None* if there is none.

    An existing file that cannot be read or parsed raises
    :class:`CatalogueRebuildRefused` and is left untouched: an uncertain
    baseline must stop a writer rather than disable its checks. Both the
    full rebuild and the hooks' incremental updates read through here.
    """
    if not catalogue_file.exists():
        return None
    try:
        data = json.loads(catalogue_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogueRebuildRefused(
            f"existing catalogue unreadable ({exc}); kept as it is; {_RECOVERY_HINT}"
        ) from exc
    if not isinstance(data, dict) or not isinstance(data.get("sessions"), list):
        raise CatalogueRebuildRefused(
            f"existing catalogue has no sessions list; kept as it is; {_RECOVERY_HINT}"
        )
    return data


def check_rebuild_keeps_entries(
    baseline: dict[str, Any] | None,
    rebuilt: dict[str, Any],
    *,
    allow_removed: Collection[str] = (),
) -> list[str]:
    """
    Refuse a rebuild that drops an entry whose session it cannot still find.

    Review 2026-10-08, round two: absence is not evidence of a deliberate
    removal. A partially stale mount can hide directories from both the
    scan and ``stat``; the previous version counted those as removed and
    published the shrunken index whenever they stayed under half of it,
    so whole sessions vanished from the index the hooks look them up in.
    An entry of *baseline* may now leave the catalogue only when:

    * its directory is listed in *allow_removed*: an authorised cleanup
      names exactly the directories it removed; or
    * its session is still catalogued from another directory whose
      RECORDED transcript length (``transcript_bytes``) is at least as
      long: the session stays findable through a copy recorded as no
      shorter. This is a findability and recorded-length check only; it
      does not prove the two copies hold the same content (the separate
      archive-integrity checks compare transcripts). A baseline entry that
      records no length cannot pass it (catalogues written before this
      field existed).

    Entries without a recorded directory (older catalogues) are matched by
    session id, as before. Anything else raises
    :class:`CatalogueRebuildRefused`, whether the directory is missing or
    still holds metadata the scan did not return.

    Returns:
        The directories dropped (named in *allow_removed*, or still
        findable through a copy recorded as no shorter),
        so the caller can report them.
    """
    if baseline is None:
        return []
    allowed = {str(PurePosixPath(path)) for path in allow_removed}
    sessions = rebuilt.get("sessions", [])
    new_keys = {(s.get("id"), s.get("directory")) for s in sessions}
    # The longest recorded transcript among each session's rebuilt copies.
    longest: dict[Any, int] = {}
    for entry in sessions:
        length = entry.get("transcript_bytes")
        if isinstance(length, int):
            longest[entry.get("id")] = max(longest.get(entry.get("id"), 0), length)
    new_ids = {s.get("id") for s in sessions}

    lost: list[str] = []
    dropped: list[str] = []
    for entry in baseline.get("sessions", []):
        session_id, directory = entry.get("id"), entry.get("directory")
        if not directory:
            if session_id not in new_ids:
                lost.append(f"session {session_id} (no directory recorded)")
            continue
        if (session_id, directory) in new_keys:
            continue
        if str(PurePosixPath(directory)) in allowed:
            dropped.append(directory)
            continue
        recorded = entry.get("transcript_bytes")
        if (
            isinstance(recorded, int) and recorded > 0
            and longest.get(session_id, -1) >= recorded
        ):
            dropped.append(directory)
            continue
        lost.append(directory)
    if lost:
        raise CatalogueRebuildRefused(
            f"{len(lost)} entr{'y' if len(lost) == 1 else 'ies'} would be dropped "
            f"although {'its session is' if len(lost) == 1 else 'their sessions are'} "
            f"not catalogued from another copy recorded as no shorter (e.g. "
            f"{lost[0]}); a missing directory is not evidence of a deliberate "
            f"removal. Name an authorised removal in allow_removed"
        )
    return dropped


def rebuild_and_write_catalogue(
    archive_dir: Path,
    catalogue_file: Path | None = None,
    *,
    lock_timeout: float | None = None,
    allow_removed: Collection[str] = (),
) -> dict[str, Any]:
    """
    Rebuild the catalogue from disk and publish it in ONE locked transaction.

    Review 2026-10-08 (P1): a rebuild that scanned before taking the lock
    could miss an entry a hook added meanwhile, then overwrite the hook's
    update. Here the baseline read, the strict scan, the identity check and
    the atomic write all happen under one lock. Raises
    (:class:`CatalogueLockError`, :class:`CatalogueScanError`,
    :class:`CatalogueRebuildRefused`) and leaves the existing file untouched
    on any doubt; an ``OSError`` from the write itself propagates (the
    temporary file is removed, and the rename is the last step, so the old
    file survives any failure before it).

    Unattended callers pass no *allow_removed*: then an entry can leave
    the index only while its session stays catalogued from a copy recorded
    as no shorter (see :func:`check_rebuild_keeps_entries`). The
    directories dropped are printed, so a reconciliation is visible in the
    caller's log.
    """
    catalogue_file = catalogue_file or archive_dir / "CATALOG.json"
    with catalogue_lock(catalogue_file, timeout=lock_timeout):
        baseline = _read_baseline(catalogue_file)
        rebuilt = rebuild_catalogue(archive_dir, strict=True)
        dropped = check_rebuild_keeps_entries(
            baseline, rebuilt, allow_removed=allow_removed,
        )
        _write_json_atomic(catalogue_file, rebuilt)
    if dropped:
        print(
            f"Catalogue rebuild dropped {len(dropped)} entr"
            f"{'y' if len(dropped) == 1 else 'ies'} (named in allow_removed, or "
            f"its session still catalogued from a copy recorded as no shorter): "
            f"{', '.join(dropped[:5])}"
            + (" ..." if len(dropped) > 5 else "")
        )
    return rebuilt


def _sort_newest_first(sessions: list[dict[str, Any]]) -> None:
    """Sort entries newest first by each session's PREFERRED copy's date.

    The list arrives with each session's copies in preference order. A
    plain date sort let a shorter copy with a later date overtake the
    preferred one (review 2026-10-08, P2), and lookups take the first
    match. Copies of one session share a key here, and Python's sort is
    stable (also with ``reverse=True``), so their order survives.
    """
    preferred_date: dict[Any, str] = {}
    for entry in sessions:
        preferred_date.setdefault(entry.get("id"), entry.get("started_at") or "")
    sessions.sort(key=lambda e: preferred_date.get(e.get("id"), ""), reverse=True)


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
    """Body of :func:`update_catalogue`; the caller holds the lock.

    An unreadable catalogue raises :class:`CatalogueRebuildRefused` and is
    left as it is (review of PA #172, round three). This used to fall back
    to a LENIENT rebuild, which skipped any other unreadable metadata and
    published the partial result as valid JSON; the next scheduled rebuild
    then took that partial file as its baseline and could not see what it
    had lost. The session's own metadata is already on disk, so nothing is
    lost by refusing: recovery is the explicit step the error names.
    """
    catalogue = _read_baseline(catalogue_file)
    if catalogue is None:
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
            "transcript_bytes": _recorded_bytes(session),
        })

    catalogue["generated_at"] = datetime.now().isoformat()
    catalogue["project"] = project_name
    catalogue["total_sessions"] = len(catalogue["sessions"])

    # Newest first, keeping each session's copies in their existing order.
    _sort_newest_first(catalogue["sessions"])

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
        # Raises, leaving the file as it is, when it cannot be read.
        catalogue = _read_baseline(catalogue_file)
        if catalogue is None:
            return
        _apply_entry_update(catalogue, session_id, meta)
        _write_json_atomic(catalogue_file, catalogue)


def _entry_written_to(
    matches: list[dict[str, Any]], written_to: str | None,
) -> dict[str, Any] | None:
    """Pick, among one session's entries, the copy in *written_to*.

    *written_to* is an absolute directory; an entry's ``directory`` is
    relative to the archive root, so the match is on trailing path parts.
    Falls back to the first (preferred) copy, or *None* if there is none.
    """
    if not matches:
        return None
    if written_to:
        written = PurePosixPath(Path(written_to).as_posix()).parts
        for candidate in matches:
            parts = PurePosixPath(candidate.get("directory") or "").parts
            if parts and written[-len(parts):] == parts:
                return candidate
    return matches[0]


def _apply_entry_update(
    catalogue: dict[str, Any], session_id: str, meta: dict[str, Any],
) -> None:
    """Refresh one session's title, purpose, tags and length in *catalogue*.

    The entry updated is the copy in the directory the record names
    (``_archive_directory``, set by ``archive_session``) when one matches,
    else the session's first (preferred) copy, as before. Matching the
    directory matters for ``transcript_bytes``: crediting one copy with
    another's length would let a later rebuild drop a longer copy while
    only a shorter one stays catalogued.
    """
    matches = [
        s for s in catalogue.get("sessions", [])
        if s.get("id") and s.get("id") == session_id
    ]
    target = _entry_written_to(matches, meta.get("_archive_directory"))
    if target is not None:
        auto = meta.get("auto_generated", {})
        target["title"] = auto.get("title", "Untitled")
        target["purpose"] = auto.get("purpose", "")
        target["tags"] = auto.get("tags", [])
        # A supersede grows the transcript in place; keep the length the
        # rebuild guard compares current. A caller whose record carries no
        # length (an edit of the summary fields) leaves the known one alone.
        length = _recorded_bytes(meta)
        if length:
            target["transcript_bytes"] = length

    catalogue["generated_at"] = datetime.now().isoformat()


# -------------------------------------------------------------------------
# Full catalogue rebuild (from generate_session_catalog.py)
# -------------------------------------------------------------------------

def _scan_sessions(
    archive_dir: Path, *, strict: bool = False,
) -> list[dict[str, Any]]:
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
        strict: Raise :class:`CatalogueScanError` if any metadata file is
            unreadable, instead of warning and skipping it, so that an
            incomplete scan can never be published as a full rebuild.

    Returns:
        List of ``{metadata, path, project}`` dictionaries.
    """
    sessions: list[dict[str, Any]] = []
    unreadable: list[str] = []

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
            unreadable.append(str(rel))
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

    if strict and unreadable:
        raise CatalogueScanError(
            f"{len(unreadable)} unreadable metadata file(s), e.g. {unreadable[0]}"
        )
    return sessions


def _recorded_bytes(meta: dict[str, Any]) -> int:
    """The transcript length a record claims, or 0 when it records none."""
    value = (meta.get("archive") or {}).get("jsonl_bytes_uncompressed")
    return value if isinstance(value, int) else 0


def rebuild_catalogue(
    archive_dir: Path, *, strict: bool = False,
) -> dict[str, Any]:
    """
    Full catalogue rebuild from archived session metadata files.

    Scans all project subdirectories, builds project rollups, tag index,
    and relationship graph.

    Args:
        archive_dir: Base archive directory (``archive/cc-sessions/``).
        strict: Raise if any metadata file is unreadable (see
            :func:`_scan_sessions`).

    Returns:
        Complete catalogue dictionary (schema v1.1).
    """
    sessions = _scan_sessions(archive_dir, strict=strict)

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
    # The final sort keys each copy on its session's preferred date, so this
    # order survives it.
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
                    # The uncompressed transcript length the record claims (0 when
            # it records none). The next rebuild compares it to decide
            # whether a vanished copy's session stays findable through a
            # copy recorded as no shorter.
            "transcript_bytes": _recorded_bytes(meta),
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

    # Newest first by each session's preferred copy; copies keep their order.
    _sort_newest_first(catalogue["sessions"])
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
