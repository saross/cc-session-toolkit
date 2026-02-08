"""
Subcommand-based CLI for cc-session-toolkit.

Entry point: ``cc-session`` (registered via pyproject.toml).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from cc_session_toolkit.archive import (
    archive_session,
    find_archive_by_id,
    get_archived_session_ids,
    get_session_files,
    get_session_id,
)
from cc_session_toolkit.catalogue import (
    generate_catalogue_markdown,
    rebuild_catalogue,
    update_catalogue,
    update_catalogue_entry,
)
from cc_session_toolkit.init import initialise_project
from cc_session_toolkit.project import (
    find_project_root,
    get_archive_dir,
    get_catalogue_file,
    get_project_name,
)
from cc_session_toolkit.summarise import summarise_session


# -------------------------------------------------------------------------
# Subcommand handlers
# -------------------------------------------------------------------------

def cmd_init(args: argparse.Namespace) -> None:
    """Handle ``cc-session init``."""
    try:
        root = find_project_root()
    except FileNotFoundError:
        print(
            "Error: no project root found. "
            "Run this command from within a project directory "
            "(must contain .git/, CLAUDE.md, or pyproject.toml)."
        )
        sys.exit(1)

    initialise_project(
        project_root=root,
        project_name=args.project_name,
        update=args.update,
    )


def cmd_archive(args: argparse.Namespace) -> None:
    """Handle ``cc-session archive``."""
    try:
        root = find_project_root()
    except FileNotFoundError:
        print("Error: no project root found.")
        sys.exit(1)

    session_files = get_session_files(root)
    catalogue_file = get_catalogue_file(root)
    archived_ids = get_archived_session_ids(catalogue_file)
    archive_dir = get_archive_dir(root)
    project_name = get_project_name(root)

    if not session_files:
        print("No sessions found.")
        return

    # Determine which sessions to archive
    if args.session_id:
        targets = [
            p for p in session_files
            if get_session_id(p) == args.session_id
        ]
        if not targets:
            print(f"Session not found: {args.session_id}")
            return
    elif args.all:
        if args.force:
            targets = session_files
        else:
            targets = [
                p for p in session_files
                if get_session_id(p) not in archived_ids
            ]
    else:
        targets = [session_files[-1]]

    if not targets:
        print("No sessions to archive (all already archived).")
        print("Use --force to re-archive existing sessions.")
        return

    print(f"Archiving {len(targets)} session(s)...")

    new_sessions: list[dict[str, Any]] = []
    for session_path in targets:
        title = args.title if len(targets) == 1 else None
        result = archive_session(
            session_path,
            root,
            dry_run=args.dry_run,
            stats_only=args.stats_only,
            use_gzip=args.gzip,
            title=title,
        )
        if result:
            new_sessions.append(result)

    if new_sessions and not args.dry_run:
        update_catalogue(
            new_sessions, catalogue_file, archive_dir, project_name
        )

    print("\nDone!")


def cmd_list(args: argparse.Namespace) -> None:
    """Handle ``cc-session list``."""
    try:
        root = find_project_root()
    except FileNotFoundError:
        print("Error: no project root found.")
        sys.exit(1)

    session_files = get_session_files(root)
    catalogue_file = get_catalogue_file(root)
    archived_ids = get_archived_session_ids(catalogue_file)
    project_name = get_project_name(root)

    print(f"Sessions for project: {project_name}")
    print()

    if not session_files:
        print("No sessions found.")
        return

    print(
        f"{'ID':<45} {'Size':>10} {'Modified':>20} {'Archived':>10}"
    )
    print("-" * 90)

    for session_path in session_files:
        sid = get_session_id(session_path)
        size_mb = session_path.stat().st_size / (1024 * 1024)
        modified = datetime.fromtimestamp(
            session_path.stat().st_mtime
        ).strftime("%Y-%m-%d %H:%M")
        archived = "Yes" if sid in archived_ids else "No"

        print(f"{sid:<45} {size_mb:>9.2f}M {modified:>20} {archived:>10}")

    print()
    print(
        f"Total: {len(session_files)} sessions, "
        f"{len(archived_ids)} archived"
    )


def cmd_list_archives(args: argparse.Namespace) -> None:
    """Handle ``cc-session list-archives``."""
    try:
        root = find_project_root()
    except FileNotFoundError:
        print("Error: no project root found.")
        sys.exit(1)

    archive_dir = get_archive_dir(root)
    project_name = get_project_name(root)
    project_dir = archive_dir / project_name

    if not project_dir.exists():
        print("No archives found.")
        return

    print(f"Archived sessions for: {project_name}")
    print(f"Location: {project_dir}")
    print()

    print(f"{'Directory':<50} {'Title':<30} {'Complete':<10}")
    print("-" * 90)

    incomplete_count = 0
    for entry in sorted(project_dir.iterdir()):
        if not entry.is_dir():
            continue

        meta_file = entry / "session.meta.json"
        title = "N/A"
        complete = "No"

        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            title = meta.get("auto_generated", {}).get(
                "title", "Untitled"
            )
            three_ps = meta.get("three_ps", {})
            if (
                title != "Untitled Session"
                and three_ps.get("prompt_summary")
                and three_ps.get("process_summary")
            ):
                complete = "Yes"
            else:
                incomplete_count += 1

        dir_name = entry.name[:48]
        title_display = title[:28] if title else "N/A"
        print(f"{dir_name:<50} {title_display:<30} {complete:<10}")

    print()
    total = len([d for d in project_dir.iterdir() if d.is_dir()])
    print(
        f"Total: {total} archives, {incomplete_count} need metadata"
    )


def cmd_summarise(args: argparse.Namespace) -> None:
    """Handle ``cc-session summarise``."""
    try:
        root = find_project_root()
    except FileNotFoundError:
        print("Error: no project root found.")
        sys.exit(1)

    archive_dir = get_archive_dir(root)
    project_name = get_project_name(root)

    session_dir = find_archive_by_id(
        args.session_id, archive_dir, project_name
    )
    if not session_dir:
        print(f"Archive not found for session: {args.session_id}")
        return

    summarise_session(session_dir)


def cmd_update(args: argparse.Namespace) -> None:
    """Handle ``cc-session update``."""
    try:
        root = find_project_root()
    except FileNotFoundError:
        print("Error: no project root found.")
        sys.exit(1)

    archive_dir = get_archive_dir(root)
    project_name = get_project_name(root)
    catalogue_file = get_catalogue_file(root)

    session_dir = find_archive_by_id(
        args.session_id, archive_dir, project_name
    )
    if not session_dir:
        print(f"Archive not found for session: {args.session_id}")
        return

    meta_file = session_dir / "session.meta.json"
    if not meta_file.exists():
        print(f"Metadata file not found: {meta_file}")
        return

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    print(f"Updating metadata for: {session_dir.name}")
    print(
        f"Current title: "
        f"{meta.get('auto_generated', {}).get('title', 'None')}"
    )

    # Read new metadata
    if args.metadata_file:
        meta_input = Path(args.metadata_file)
        if not meta_input.exists():
            print(f"File not found: {args.metadata_file}")
            return
        try:
            new_meta = json.loads(
                meta_input.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            print(f"Invalid JSON in {args.metadata_file}: {exc}")
            return
    else:
        print("\nEnter metadata JSON (Ctrl+D when done):")
        try:
            new_meta = json.loads(sys.stdin.read())
        except json.JSONDecodeError as exc:
            print(f"Invalid JSON: {exc}")
            return

    # Update fields
    meta.setdefault("auto_generated", {})
    if "title" in new_meta:
        meta["auto_generated"]["title"] = new_meta["title"]
    if "purpose" in new_meta:
        meta["auto_generated"]["purpose"] = new_meta["purpose"]
    if "tags" in new_meta:
        meta["auto_generated"]["tags"] = new_meta["tags"]
    if "three_ps" in new_meta:
        meta["three_ps"] = new_meta["three_ps"]

    if not args.dry_run:
        meta_file.write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print(f"\nUpdated: {meta_file}")
        update_catalogue_entry(
            args.session_id, meta, catalogue_file
        )
        print("Catalogue updated.")
    else:
        print("\n[DRY RUN] Would update metadata to:")
        print(json.dumps(meta["auto_generated"], indent=2))


def cmd_catalogue(args: argparse.Namespace) -> None:
    """Handle ``cc-session catalogue``."""
    try:
        root = find_project_root()
    except FileNotFoundError:
        print("Error: no project root found.")
        sys.exit(1)

    archive_dir = get_archive_dir(root)
    catalogue_file = get_catalogue_file(root)

    if args.rebuild:
        print("Rebuilding catalogue from archived sessions...")
        catalogue = rebuild_catalogue(archive_dir)

        catalogue_file.parent.mkdir(parents=True, exist_ok=True)
        catalogue_file.write_text(
            json.dumps(catalogue, indent=2), encoding="utf-8"
        )
        print(f"Wrote: {catalogue_file}")

        if args.markdown:
            md_file = catalogue_file.with_name("CATALOG.md")
            md_content = generate_catalogue_markdown(catalogue)
            md_file.write_text(md_content, encoding="utf-8")
            print(f"Wrote: {md_file}")

        print()
        print(f"  Sessions: {catalogue['total_sessions']}")
        print(f"  Projects: {len(catalogue['projects'])}")
        print(f"  Tags: {len(catalogue['tag_index'])}")
    else:
        # Just generate markdown from existing catalogue
        if not catalogue_file.exists():
            print(
                "No catalogue found. "
                "Run 'cc-session catalogue --rebuild' first."
            )
            return

        catalogue = json.loads(
            catalogue_file.read_text(encoding="utf-8")
        )
        md_file = catalogue_file.with_name("CATALOG.md")
        md_content = generate_catalogue_markdown(catalogue)
        md_file.write_text(md_content, encoding="utf-8")
        print(f"Wrote: {md_file}")


# -------------------------------------------------------------------------
# Main parser
# -------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for ``cc-session``."""
    parser = argparse.ArgumentParser(
        prog="cc-session",
        description=(
            "Session archiving, catalogue management, and reflection "
            "protocol tooling for Claude Code research sessions."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- init ---------------------------------------------------------
    p_init = subparsers.add_parser(
        "init",
        help="Scaffold project files for cc-session.",
    )
    p_init.add_argument(
        "--project-name",
        help="Explicit project name (default: auto-detect).",
    )
    p_init.add_argument(
        "--update",
        action="store_true",
        help="Regenerate SKILL.md and queries from the current package.",
    )
    p_init.set_defaults(func=cmd_init)

    # -- archive ------------------------------------------------------
    p_archive = subparsers.add_parser(
        "archive",
        help="Archive sessions.",
    )
    p_archive.add_argument(
        "--session-id", "-s",
        help="Archive a specific session by ID.",
    )
    p_archive.add_argument(
        "--all", "-a",
        action="store_true",
        help="Archive all unarchived sessions.",
    )
    p_archive.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-archive sessions even if already archived.",
    )
    p_archive.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview without archiving.",
    )
    p_archive.add_argument(
        "--stats-only",
        action="store_true",
        help="Skip CC metadata generation (use placeholders).",
    )
    p_archive.add_argument(
        "--gzip", "-z",
        action="store_true",
        help="Compress JSONL files with gzip.",
    )
    p_archive.add_argument(
        "--title", "-t",
        help="Session title for human-readable directory name.",
    )
    p_archive.set_defaults(func=cmd_archive)

    # -- list ---------------------------------------------------------
    p_list = subparsers.add_parser(
        "list",
        help="List sessions and their archive status.",
    )
    p_list.set_defaults(func=cmd_list)

    # -- list-archives ------------------------------------------------
    p_list_archives = subparsers.add_parser(
        "list-archives",
        help="Show archived sessions with metadata completion status.",
    )
    p_list_archives.set_defaults(func=cmd_list_archives)

    # -- summarise ----------------------------------------------------
    p_summarise = subparsers.add_parser(
        "summarise",
        help="Analyse an archived session to help generate metadata.",
    )
    p_summarise.add_argument(
        "session_id",
        help="Session ID (full or partial).",
    )
    p_summarise.set_defaults(func=cmd_summarise)

    # -- update -------------------------------------------------------
    p_update = subparsers.add_parser(
        "update",
        help="Update metadata for an existing archive.",
    )
    p_update.add_argument(
        "session_id",
        help="Session ID (full or partial).",
    )
    p_update.add_argument(
        "-m", "--metadata-file",
        help="JSON file containing metadata.",
    )
    p_update.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview without writing.",
    )
    p_update.set_defaults(func=cmd_update)

    # -- catalogue ----------------------------------------------------
    p_catalogue = subparsers.add_parser(
        "catalogue",
        help="Regenerate catalogue from archived session metadata.",
    )
    p_catalogue.add_argument(
        "--rebuild",
        action="store_true",
        help="Full rebuild from session.meta.json files.",
    )
    p_catalogue.add_argument(
        "--markdown",
        action="store_true",
        help="Also generate CATALOG.md (used with --rebuild).",
    )
    p_catalogue.set_defaults(func=cmd_catalogue)

    # -- dispatch -----------------------------------------------------
    args = parser.parse_args()
    args.func(args)


def _get_version() -> str:
    """Return the package version string."""
    try:
        from cc_session_toolkit import __version__
        return __version__
    except ImportError:
        return "unknown"


if __name__ == "__main__":
    main()
