#!/usr/bin/env python3
"""
Backfill auto-metadata for archived sessions missing it.

Finds all session.meta.json files with "Auto-metadata unavailable",
decompresses the session JSONL, runs generate_auto_metadata via Haiku,
and updates the metadata in-place.

Usage:
    python scripts/backfill-session-metadata.py [--dry-run] [--archive-root DIR]

Cost: ~$0.001 per session via Haiku.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import tempfile
from pathlib import Path

# Ensure the src directory is importable when running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cc_session_toolkit.archive import (  # noqa: E402
    _ensure_anthropic_api_key,
    _log_metadata_event,
    generate_auto_metadata,
)
from cc_session_toolkit.config import DEFAULT_ARCHIVE_ROOT  # noqa: E402
from cc_session_toolkit.extraction import (  # noqa: E402
    extract_session_stats,
)


def find_sessions_needing_backfill(
    archive_root: Path,
) -> list[Path]:
    """Return paths to session.meta.json files with missing metadata."""
    results: list[Path] = []
    for meta_path in sorted(archive_root.rglob("session.meta.json")):
        with open(meta_path, encoding="utf-8") as fh:
            data = json.load(fh)
        auto_gen = data.get("auto_generated", {})
        if auto_gen.get("purpose") == "Auto-metadata unavailable":
            results.append(meta_path)
    return results


def decompress_session(archive_dir: Path) -> Path | None:
    """
    Decompress session.jsonl.gz to a temporary file.

    Returns the path to the temporary JSONL file, or None if no
    compressed session file is found.  Caller is responsible for
    cleanup.
    """
    gz_path = archive_dir / "session.jsonl.gz"
    plain_path = archive_dir / "session.jsonl"

    if plain_path.is_file():
        return plain_path

    if not gz_path.is_file():
        return None

    tmp = tempfile.NamedTemporaryFile(
        suffix=".jsonl", delete=False, mode="wb",
    )
    try:
        with gzip.open(gz_path, "rb") as fin:
            shutil.copyfileobj(fin, tmp)
        tmp.close()
        return Path(tmp.name)
    except Exception:
        tmp.close()
        Path(tmp.name).unlink(missing_ok=True)
        raise


def update_metadata(
    meta_path: Path,
    auto_generated: dict,
) -> None:
    """Update auto_generated fields in session.meta.json."""
    with open(meta_path, encoding="utf-8") as fh:
        data = json.load(fh)

    data["auto_generated"] = {
        "title": auto_generated.get("title", "Untitled Session"),
        "purpose": auto_generated.get("purpose", ""),
        "tags": auto_generated.get("tags", []),
        "three_ps": data.get("auto_generated", {}).get("three_ps", {
            "prompt_summary": "",
            "process_summary": "",
            "provenance_summary": "",
        }),
    }

    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def main() -> None:
    """Run the backfill."""
    parser = argparse.ArgumentParser(
        description="Backfill session auto-metadata via Haiku.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List sessions that need backfill without calling the API.",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=DEFAULT_ARCHIVE_ROOT,
        help="Root directory of session archives.",
    )
    args = parser.parse_args()

    # Ensure API key is available before starting.
    _ensure_anthropic_api_key()
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not found in environment or .env")
        sys.exit(1)

    sessions = find_sessions_needing_backfill(args.archive_root)
    if not sessions:
        print("All sessions already have metadata. Nothing to do.")
        return

    print(f"Found {len(sessions)} session(s) needing metadata backfill.")
    if args.dry_run:
        for meta_path in sessions:
            rel = meta_path.parent.relative_to(args.archive_root)
            print(f"  {rel}")
        print(f"\nDry run — no changes made. Cost: ~${len(sessions) * 0.001:.3f}")
        return

    succeeded = 0
    failed = 0

    for i, meta_path in enumerate(sessions, 1):
        archive_dir = meta_path.parent
        rel = archive_dir.relative_to(args.archive_root)
        print(f"[{i}/{len(sessions)}] {rel} ... ", end="", flush=True)

        tmp_path = None
        try:
            tmp_path = decompress_session(archive_dir)
            if tmp_path is None:
                print("SKIP (no session JSONL found)")
                failed += 1
                continue

            stats = extract_session_stats(tmp_path)
            result = generate_auto_metadata(tmp_path, stats)

            if result is None:
                print("FAIL (Haiku returned None)")
                failed += 1
                continue

            update_metadata(meta_path, result)
            title = result.get("title", "?")
            print(f"OK: {title}")
            succeeded += 1

        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}")
            _log_metadata_event(
                f"Backfill error for {rel}: {type(exc).__name__}: {exc}",
                level="ERROR",
            )
            failed += 1

        finally:
            # Clean up temp file (but not if it was the original)
            if (
                tmp_path is not None
                and tmp_path != archive_dir / "session.jsonl"
            ):
                tmp_path.unlink(missing_ok=True)

    print(f"\nDone: {succeeded} succeeded, {failed} failed.")


if __name__ == "__main__":
    main()
