#!/usr/bin/env python3
"""
Back-archive sub-agent transcripts for existing session archives.

Walks ``~/cc-archives/<project>/<datetime>_<slug>/session.meta.json``,
locates the parent JSONL that the archive originated from (via
``project.directory`` + ``session.id`` → ``~/.claude/projects/<slug>/
<session-id>.jsonl``), runs the standard sub-agent discovery and
archival against it, and patches ``session.meta.json`` in place:

- bumps ``schema_version`` to ``"1.2"``
- writes the ``subagents: [...]`` list
- writes the ``statistics.subagents_summary`` rollup

Idempotent — safe to re-run.  Uses the production
``archive_subagent_transcripts`` code path, so any later bug fix in the
main toolkit automatically benefits back-archival too.

Usage:
    venv/bin/python3 scripts/backfill-subagents.py
    venv/bin/python3 scripts/backfill-subagents.py --archive-root /path/to/archives
    venv/bin/python3 scripts/backfill-subagents.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cc_session_toolkit.archive import (
    archive_subagent_transcripts,
    summarise_subagents,
)
from cc_session_toolkit.project import CLAUDE_PROJECTS_DIR


DEFAULT_ARCHIVE_ROOT = Path.home() / "cc-archives"


def _resolve_parent_jsonl(meta: dict) -> Path | None:
    """
    Resolve the path to the parent session's JSONL in ``~/.claude/projects/``
    given a ``session.meta.json`` dictionary.

    Returns *None* if either the project directory or the session id is
    missing, or the file does not exist on disk.
    """
    session_id = (meta.get("session") or {}).get("id")
    project_dir = (meta.get("project") or {}).get("directory")
    if not session_id or not project_dir:
        return None
    cc_slug = project_dir.replace("/", "-")
    candidate = CLAUDE_PROJECTS_DIR / cc_slug / f"{session_id}.jsonl"
    return candidate if candidate.is_file() else None


def backfill_archive(
    archive_dir: Path,
    *,
    dry_run: bool = False,
    use_gzip: bool = True,
) -> dict:
    """
    Back-archive sub-agents for a single archive directory.

    Returns a report dict: ``{"status": ..., "count": ..., "reason": ...}``.
    """
    meta_path = archive_dir / "session.meta.json"
    if not meta_path.is_file():
        return {"status": "skipped", "reason": "no session.meta.json"}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "error", "reason": f"bad json: {exc}"}

    parent_jsonl = _resolve_parent_jsonl(meta)
    if parent_jsonl is None:
        return {
            "status": "skipped",
            "reason": "parent JSONL not found in ~/.claude/projects/",
        }

    session_id = meta["session"]["id"]

    if dry_run:
        # Discover without copying
        from cc_session_toolkit.archive import _find_subagent_jsonls
        found = _find_subagent_jsonls(parent_jsonl, session_id)
        return {"status": "dry-run", "count": len(found)}

    subagents = archive_subagent_transcripts(
        session_path=parent_jsonl,
        session_id=session_id,
        dest_dir=archive_dir,
        use_gzip=use_gzip,
        capture_type="backfill",
    )

    # Patch meta.json in place — bump schema, set subagents, rollup
    meta["schema_version"] = "1.2"
    meta["subagents"] = subagents
    stats = meta.setdefault("statistics", {})
    stats["subagents_summary"] = summarise_subagents(subagents)

    meta_path.write_text(
        json.dumps(meta, indent=2), encoding="utf-8",
    )

    return {"status": "ok", "count": len(subagents)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Back-archive sub-agents for existing session archives.",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=DEFAULT_ARCHIVE_ROOT,
        help="Base archive directory (default: ~/cc-archives/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be archived without writing files.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Limit backfill to a single project subdirectory.",
    )
    parser.add_argument(
        "--no-gzip",
        action="store_true",
        help="Store sub-agent transcripts uncompressed.",
    )
    args = parser.parse_args()

    archive_root: Path = args.archive_root
    if not archive_root.is_dir():
        print(f"Error: archive root not found: {archive_root}")
        return 1

    project_dirs = (
        [archive_root / args.project] if args.project
        else sorted(p for p in archive_root.iterdir() if p.is_dir())
    )

    total = 0
    archived = 0
    skipped = 0
    errored = 0
    total_subagents = 0

    for project_dir in project_dirs:
        if not project_dir.is_dir():
            continue
        for session_dir in sorted(project_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            total += 1
            report = backfill_archive(
                session_dir,
                dry_run=args.dry_run,
                use_gzip=not args.no_gzip,
            )
            status = report["status"]
            if status == "ok":
                archived += 1
                total_subagents += report.get("count", 0)
                if report.get("count", 0):
                    print(
                        f"  [ok] {project_dir.name}/{session_dir.name}: "
                        f"{report['count']} sub-agents"
                    )
            elif status == "dry-run":
                total_subagents += report.get("count", 0)
                if report.get("count", 0):
                    print(
                        f"  [would-archive] "
                        f"{project_dir.name}/{session_dir.name}: "
                        f"{report['count']} sub-agents"
                    )
            elif status == "skipped":
                skipped += 1
            elif status == "error":
                errored += 1
                print(
                    f"  [err] {project_dir.name}/{session_dir.name}: "
                    f"{report['reason']}",
                    file=sys.stderr,
                )

    verb = "Would archive" if args.dry_run else "Archived"
    print()
    print(f"Archives scanned:       {total}")
    print(f"{verb} sub-agents for: {archived}")
    print(f"Skipped (no parent JSONL or meta): {skipped}")
    print(f"Errors:                 {errored}")
    print(f"Total sub-agent transcripts: {total_subagents}")
    return 0 if errored == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
