#!/usr/bin/env python3
"""
Backfill auto-metadata for archived sessions missing it.

Finds all session.meta.json files with "Auto-metadata unavailable",
decompresses the session JSONL, runs generate_auto_metadata via Gemini
Flex (production-switched 2026-05-18; see workstream F in
``personal-assistant/planning/continuity.md``), and updates the metadata
in-place.

Usage:
    python scripts/backfill-session-metadata.py [--dry-run] [--archive-root DIR]
                                                [--cost-sample-size N]

Cost: ~$0.08 per session via Gemini 3.5 Flash Flex tier (3× the prior
Gemini 3 Flash Preview ~$0.027 figure; was ~$0.001 under Haiku). The
``--dry-run`` mode samples ``--cost-sample-size`` sessions (default 20),
distils them locally with no API spend, and reports mean / p90 /
worst-case-envelope cost estimates grounded in the actual size
distribution of the sessions in the archive — rather than a flat
per-session average, which can understate cost when the archive
contains long-tail sessions that approach the 850K-token transcript cap.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

# Ensure the src directory is importable when running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cc_session_toolkit.archive import (  # noqa: E402
    _ensure_gemini_api_key,
    _log_metadata_event,
    generate_auto_metadata,
    generate_subagent_summaries,
)
from cc_session_toolkit.config import (  # noqa: E402
    AUTO_METADATA_MAX_OUTPUT_TOKENS,
    DEFAULT_ARCHIVE_ROOT,
    GEMINI_FLEX_INPUT_PRICE_PER_MTOK,
    GEMINI_FLEX_OUTPUT_PRICE_PER_MTOK,
    SCHEMA_VERSION,
)
from cc_session_toolkit.extraction import (  # noqa: E402
    extract_session_stats,
)
from cc_session_toolkit.transcript_text import (  # noqa: E402
    estimate_tokens,
    extract_transcript_text,
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
    subagent_summaries: list[dict[str, str]] | None = None,
) -> None:
    """Update ``auto_generated`` + top-level ``three_ps`` +
    ``subagent_summaries`` fields in ``session.meta.json`` and bump
    ``schema_version`` to v1.3.

    The Gemini path produces ``three_ps`` natively as part of the same
    JSON object, so we overwrite any prior empty-string defaults rather
    than preserving them. ``three_ps`` is also surfaced at the top level
    of ``session.meta.json`` (see ``create_session_metadata``); we
    mirror the new values there.

    v1.3 (2026-05-24): ``auto_generated`` gains optional ``phases``,
    ``decisions``, and ``key_exchanges`` arrays (defaulting to ``[]``
    when the model emits nothing). Top-level ``subagent_summaries[]``
    is populated from the caller-supplied argument (also ``[]`` when
    the session has no subagents or no summaries succeeded).
    """
    with open(meta_path, encoding="utf-8") as fh:
        data = json.load(fh)

    new_three_ps = auto_generated.get("three_ps") or {
        "prompt_summary": "",
        "process_summary": "",
        "provenance_summary": "",
    }

    # ``.get(key) or default`` not ``.get(key, default)`` — Gemini
    # occasionally emits ``"tags": null`` etc.; the default form
    # silently lets None land in the meta file.
    new_auto: dict[str, Any] = {
        "title": auto_generated.get("title") or "Untitled Session",
        "purpose": auto_generated.get("purpose") or "",
        "tags": auto_generated.get("tags") or [],
        "three_ps": new_three_ps,
    }
    # v1.3 optional arrays — always present (possibly empty) so downstream
    # consumers can iterate without isinstance() guards.
    for v3_field in ("phases", "decisions", "key_exchanges"):
        raw = auto_generated.get(v3_field)
        new_auto[v3_field] = raw if isinstance(raw, list) else []

    data["auto_generated"] = new_auto
    # Mirror at top level too — ``create_session_metadata`` keeps a
    # top-level ``three_ps`` block that downstream consumers read.
    data["three_ps"] = new_three_ps
    # Top-level subagent_summaries always present, even when empty.
    data["subagent_summaries"] = (
        subagent_summaries if isinstance(subagent_summaries, list) else []
    )
    # Bump schema_version to reflect the v1.3 fields above. Read from
    # the constant rather than a hardcoded literal so the next schema
    # bump auto-propagates here.
    data["schema_version"] = SCHEMA_VERSION

    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def _per_session_cost(input_tokens: int) -> float:
    """Per-session API cost in USD for one Gemini Flex auto-metadata call.

    Input tokens drive the variable cost; output is bounded by an
    *expected* size (not the max-output-tokens ceiling) because the v3
    schema raised the ceiling to 8192 but typical outputs land at
    ~1000–2000 tokens (parent + arrays). Using the ceiling here would
    inflate the estimate ~4–8× and mislead the user on dry-runs.

    The estimate **does not** cover per-subagent calls (each subagent
    adds 1 Gemini call at ~$0.02–$0.10). Sessions with subagents will
    cost meaningfully more than this function suggests; the dry-run
    output should flag that caveat.
    """
    # Expected output size for the v3 parent call. Calibrated against
    # the 2026-05-24 bake-off: short sessions ~300 output tokens; long
    # multi-thread sessions ~1500-2000 output tokens. 1500 is a
    # reasonable expected upper bound for cost estimation.
    expected_output_tokens = 1500
    return (
        (input_tokens / 1_000_000) * GEMINI_FLEX_INPUT_PRICE_PER_MTOK
        + (expected_output_tokens / 1_000_000)
        * GEMINI_FLEX_OUTPUT_PRICE_PER_MTOK
    )


def _sample_distilled_token_counts(
    meta_paths: list[Path],
    sample_size: int,
) -> list[int]:
    """
    Distil the actual transcripts of a sample of sessions and return
    the per-session input token counts that Gemini Flex would see.

    No API calls — just runs the local extractor. Used to ground the
    dry-run cost estimate in the real distribution of session sizes
    rather than a flat per-session average.

    Sessions whose JSONL cannot be located or decompressed are skipped;
    the returned list may be shorter than ``sample_size``.
    """
    import random

    rng = random.Random(0)  # deterministic sample for reproducible reports
    sample = rng.sample(meta_paths, min(sample_size, len(meta_paths)))
    token_counts: list[int] = []
    for meta_path in sample:
        archive_dir = meta_path.parent
        tmp_path = None
        try:
            tmp_path = decompress_session(archive_dir)
            if tmp_path is None:
                continue
            distilled = extract_transcript_text(tmp_path)
            token_counts.append(estimate_tokens(distilled))
        except Exception:  # noqa: BLE001 — sampling is best-effort
            continue
        finally:
            if (
                tmp_path is not None
                and tmp_path.name.startswith("tmp")
                and tmp_path.exists()
            ):
                tmp_path.unlink(missing_ok=True)
    return token_counts


def _estimate_total_cost(meta_paths: list[Path], sample_size: int) -> str:
    """Build a human-readable cost estimate for the dry-run output.

    Falls back to a flat ``$0.10 × N`` quote if no sample sessions
    could be distilled (calibrated to Gemini 3.5 Flash + v3 schema +
    one subagent typical; the 2026-05-22 head-to-head migration raised
    list price 3×, and v3 produces ~3-5× larger outputs than v2).

    **Caveat**: per-subagent calls are not modelled here. Sessions
    with N subagents cost an extra ~$0.02–$0.10 × N. The refined
    sampled estimate below is still parent-only.
    """
    samples = _sample_distilled_token_counts(meta_paths, sample_size)
    if not samples:
        # Calibrated to Gemini 3.5 Flash + v3 schema; was $0.027/session
        # under Gemini 3 Flash Preview + v2 schema (1024-token output).
        flat_estimate = len(meta_paths) * 0.10
        return (
            f"Est. cost (Gemini Flex, flat $0.10/session): ~${flat_estimate:.2f}\n"
            f"  (could not sample real sessions for a refined estimate;\n"
            f"  per-subagent calls not modelled — add ~$0.05 per subagent)"
        )
    samples_sorted = sorted(samples)
    mean = sum(samples) / len(samples)
    p50 = samples_sorted[len(samples_sorted) // 2]
    p90 = samples_sorted[int(len(samples_sorted) * 0.9)]
    sample_max = max(samples)
    mean_cost = _per_session_cost(int(mean))
    p90_cost = _per_session_cost(p90)
    max_cost = _per_session_cost(sample_max)
    total_estimate = mean_cost * len(meta_paths)
    # Worst-case envelope: assume every session is at the sample's p90.
    worst_envelope = p90_cost * len(meta_paths)
    return (
        f"Est. cost (Gemini Flex, sampled n={len(samples)} sessions):\n"
        f"  per-session input tokens — "
        f"mean: {int(mean):,}  median: {p50:,}  p90: {p90:,}  max: {sample_max:,}\n"
        f"  per-session cost — "
        f"mean: ${mean_cost:.4f}  p90: ${p90_cost:.4f}  max: ${max_cost:.4f}\n"
        f"  total estimate (mean × {len(meta_paths)}): ~${total_estimate:.2f}\n"
        f"  worst-case envelope (p90 × {len(meta_paths)}): ~${worst_envelope:.2f}"
    )


def main() -> None:
    """Run the backfill."""
    parser = argparse.ArgumentParser(
        description="Backfill session auto-metadata via Gemini Flex.",
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
    parser.add_argument(
        "--cost-sample-size",
        type=int,
        default=20,
        help=(
            "Number of sessions to distil locally for a refined dry-run "
            "cost estimate (no API spend). Set to 0 to use a flat "
            "$0.10/session estimate (Gemini 3.5 Flash + v3 schema; "
            "subagent calls not modelled). Default: 20."
        ),
    )
    args = parser.parse_args()

    # Ensure API key is available before starting.
    if not _ensure_gemini_api_key():
        print(
            "Error: neither GEMINI_API_KEY nor GOOGLE_API_KEY found "
            "in environment or ~/personal-assistant/.env"
        )
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
        print("\nDry run — no changes made.")
        if args.cost_sample_size > 0:
            print(
                f"\nSampling {min(args.cost_sample_size, len(sessions))} "
                f"sessions to refine the cost estimate (no API spend) ..."
            )
            print(_estimate_total_cost(sessions, args.cost_sample_size))
        else:
            flat = len(sessions) * 0.10
            print(
                f"Est. cost (Gemini Flex, flat $0.10/session): ~${flat:.2f} "
                f"(subagent calls not modelled — add ~$0.05 per subagent)"
            )
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
                print("FAIL (Gemini returned None)")
                failed += 1
                continue

            # v1.3 (2026-05-24): if the existing session.meta.json
            # already lists archived subagents, generate the matching
            # lightweight per-subagent narrative summaries. Failures
            # don't block the parent's auto_generated update — the
            # backfill still ships, just with empty subagent_summaries.
            with open(meta_path, encoding="utf-8") as fh:
                existing_meta = json.load(fh)
            existing_subagents = existing_meta.get("subagents") or []
            subagent_summaries: list[dict[str, str]] = []
            if existing_subagents:
                try:
                    subagent_summaries = generate_subagent_summaries(
                        dest_dir=archive_dir,
                        subagents=existing_subagents,
                        parent_session_id=existing_meta.get(
                            "session", {}
                        ).get("id", ""),
                    )
                except Exception as sa_exc:
                    print(
                        f"\n    [warn] subagent summaries failed: "
                        f"{type(sa_exc).__name__}: {sa_exc}",
                        end="",
                    )
                    _log_metadata_event(
                        f"Backfill subagent summaries failed for {rel}: "
                        f"{type(sa_exc).__name__}: {sa_exc}",
                        level="WARNING",
                    )

            update_metadata(meta_path, result, subagent_summaries)
            title = result.get("title", "?")
            n_phases = len((result.get("phases") or []))
            n_decisions = len((result.get("decisions") or []))
            n_exchanges = len((result.get("key_exchanges") or []))
            n_sa = len(subagent_summaries)
            print(
                f"OK: {title} "
                f"[phases={n_phases} dec={n_decisions} exch={n_exchanges} "
                f"subagents={n_sa}/{len(existing_subagents)}]"
            )
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
