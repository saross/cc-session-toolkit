#!/usr/bin/env python3
"""
Backfill auto-metadata for archived sessions missing it, or upgrade
existing v1.2-or-earlier metadata to the v1.3 schema.

Default mode: finds all session.meta.json files with
``auto_generated.purpose == "Auto-metadata unavailable"``, decompresses
the session JSONL, runs generate_auto_metadata via Gemini Flex
(production-switched 2026-05-18; see workstream F in
``personal-assistant/planning/continuity.md``), and updates the metadata
in-place to the current schema (v1.3).

--upgrade-to-v13 mode: finds sessions that already have populated
auto_generated metadata but on a pre-v1.3 schema (typically v1.2 —
Three Ps only). Re-summarises them on v1.3, producing the richer
phases / decisions / key_exchanges / subagent_summaries structure. The
original session.meta.json is preserved side-by-side at
``session.meta.v2-backup.json`` before the overwrite. The two finders
partition the archive into disjoint populations: missing-metadata
sessions go through the default path, pre-v1.3 populated sessions go
through the upgrade path, and v1.3 sessions are skipped entirely.

Usage:
    python scripts/backfill-session-metadata.py [--dry-run]
                                                [--archive-root DIR]
                                                [--cost-sample-size N]
                                                [--upgrade-to-v13]

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
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
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


# Threshold below which a session's distilled content is considered
# empty (an abandoned shell, a JSONL with only a header turn, etc.).
# Used by the preflight skip in :func:`main` and by the standalone
# helper that retroactively marks the 11 known-empty sessions from the
# 2026-05-26 archive-wide upgrade. Calibrated against the empty
# theseus-ship and map-reader-llm "empty-abandoned-session" archives
# observed during that run, whose extract_transcript_text outputs
# rounded to ~1 token under the tokeniser.
EMPTY_TRANSCRIPT_TOKEN_THRESHOLD = 50


def find_sessions_needing_backfill(
    archive_root: Path,
) -> list[Path]:
    """Return paths to session.meta.json files with missing metadata."""
    results: list[Path] = []
    for meta_path in sorted(archive_root.rglob("session.meta.json")):
        with open(meta_path, encoding="utf-8") as fh:
            data = json.load(fh)
        # Honour the permanent-skip marker — applied by the preflight
        # when a session's distilled transcript is below the empty
        # threshold (no point spending another API call on the same
        # near-empty content). Audit follow-up 2026-05-28.
        if data.get("auto_metadata_skip_permanent"):
            continue
        auto_gen = data.get("auto_generated", {})
        if auto_gen.get("purpose") == "Auto-metadata unavailable":
            results.append(meta_path)
    return results


def find_sessions_needing_v13_upgrade(
    archive_root: Path,
) -> list[Path]:
    """Return paths to session.meta.json files on a pre-v1.3 schema.

    Targets sessions that already have populated auto_generated metadata
    (i.e., NOT "Auto-metadata unavailable") but were summarised under a
    schema older than v1.3 — typically v1.2 (Three Ps only, no
    phases / decisions / key_exchanges / subagent_summaries).
    Re-summarising these on the v1.3 schema unlocks the richer
    narrative structure for the open-science / transparency artefact
    framing adopted in the 2026-05-24 wire-up.

    Filter is the inverse of :func:`find_sessions_needing_backfill` —
    these two finders partition the "needs API spend" archive into
    distinct populations:
      * find_sessions_needing_backfill: empty (purpose marker)
      * find_sessions_needing_v13_upgrade: populated but pre-v1.3
      * (remainder): already on v1.3 or permanently skipped, nothing to do
    """
    results: list[Path] = []
    for meta_path in sorted(archive_root.rglob("session.meta.json")):
        with open(meta_path, encoding="utf-8") as fh:
            data = json.load(fh)
        # Honour the permanent-skip marker (see find_sessions_needing_backfill).
        if data.get("auto_metadata_skip_permanent"):
            continue
        schema_version = data.get("schema_version")
        auto_gen = data.get("auto_generated", {})
        purpose = auto_gen.get("purpose")
        # Skip if already on the target schema.
        if schema_version == SCHEMA_VERSION:
            continue
        # Skip the empty-metadata population — they belong to the
        # default backfill, not the upgrade path.
        if purpose == "Auto-metadata unavailable" or not purpose:
            continue
        results.append(meta_path)
    return results


def mark_session_permanently_skipped(
    meta_path: Path, reason: str
) -> None:
    """Set the ``auto_metadata_skip_permanent`` marker on a meta file.

    Both finders honour this field — once set, the session is excluded
    from both backfill modes. Applied by the preflight when a session's
    distilled transcript is below the empty threshold, and by the
    standalone helper that retroactively marks the 11 known-empty
    sessions from the 2026-05-26 archive-wide upgrade.

    Writes atomically (tmp + os.replace) and includes a human-readable
    reason for the skip alongside the boolean flag so the marker is
    self-documenting.
    """
    with open(meta_path, encoding="utf-8") as fh:
        data = json.load(fh)
    data["auto_metadata_skip_permanent"] = True
    data["auto_metadata_skip_reason"] = reason
    tmp = meta_path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, meta_path)


def backup_pre_v13_meta(meta_path: Path) -> Path | None:
    """Copy session.meta.json to session.meta.v2-backup.json before upgrade.

    Mirrors the convention used by the 2026-05-24 production-path
    validator: when a v1.2-or-earlier meta is replaced by a v1.3
    rebuild, the original is preserved side-by-side under
    ``session.meta.v2-backup.json`` for comparison.

    If a backup already exists at the destination, the function does
    nothing and returns the existing backup path — the original is
    already preserved from a prior upgrade attempt.
    """
    backup_path = meta_path.with_name("session.meta.v2-backup.json")
    if backup_path.exists():
        return backup_path
    shutil.copy2(meta_path, backup_path)
    return backup_path


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

    # Atomic overwrite: write to a sibling ``.json.tmp`` first, then
    # ``os.replace`` it over the canonical path. The plain
    # ``open(..., "w")`` form truncates ``meta_path`` immediately, so a
    # SIGINT or crash between the truncate and the ``json.dump``
    # completion would leave a zero-length or partial file — fatal given
    # v1.3 mutates three additional fields. ``os.replace`` is atomic on
    # POSIX (and on Windows since 3.3). Audit follow-up 2026-05-24.
    tmp_path = meta_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp_path, meta_path)


def _per_session_cost(input_tokens: int) -> float:
    """Per-session API cost in USD for one Gemini Flex auto-metadata call.

    Input tokens drive the variable cost; output is bounded by an
    *expected* size (not the max-output-tokens ceiling) because the v3
    schema raised the ceiling to 8192 but typical outputs land at
    ~1000–2000 tokens (parent + arrays). Using the ceiling here would
    inflate the estimate ~4–8× and mislead the user on dry-runs.

    The estimate covers the **parent** call only. Subagent calls are
    modelled separately via :func:`_per_subagent_cost` and aggregated by
    :func:`_estimate_total_cost` so the dry-run reports a parent-plus-
    subagent total (audit follow-up 2026-05-24).
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


# Typical per-subagent Gemini Flex call cost in USD. Subagent transcripts
# are typically small (a few thousand distilled tokens) and the v3
# subagent prompt emits short narratives (~60–200 words, ~80–250 output
# tokens). $0.05 is the per-subagent figure already documented in the
# audit follow-ups doc and in ``_per_session_cost`` history; using a
# flat value here keeps the estimate within the ~10% target without
# requiring a second sampling pass over the subagent transcripts.
# Audit follow-up 2026-05-24.
PER_SUBAGENT_COST_USD = 0.05


def _per_subagent_cost() -> float:
    """Estimated USD cost for summarising one subagent.

    Returns the flat ``PER_SUBAGENT_COST_USD`` constant — see its
    docstring for the calibration rationale.
    """
    return PER_SUBAGENT_COST_USD


def _sample_subagent_counts(meta_paths: list[Path]) -> list[int]:
    """
    Return per-session subagent counts for every meta file in the list.

    Reads the ``subagents`` array length from each ``session.meta.json``
    so the dry-run cost estimate can attribute per-subagent calls
    correctly. Meta files that cannot be read or whose ``subagents``
    field is not a list contribute 0 (best-effort; matches the
    summarisation path, which silently emits ``[]`` in those cases).

    No API calls and no transcript decompression — this is a quick scan
    of the meta JSON only, so it runs over the full set rather than a
    sample. Audit follow-up 2026-05-24.
    """
    counts: list[int] = []
    for meta_path in meta_paths:
        try:
            with open(meta_path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:  # noqa: BLE001 — sampling is best-effort
            counts.append(0)
            continue
        subagents = data.get("subagents")
        counts.append(len(subagents) if isinstance(subagents, list) else 0)
    return counts


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

    Reports parent and subagent costs separately so the total reflects
    the full v1.3 call shape (one parent Gemini call plus one Gemini
    call per archived subagent at ~$0.05 each). The 2026-05-22
    head-to-head migration raised list price 3×, and v3 produces ~3–5×
    larger outputs than v2; the parent calibration is captured in
    :func:`_per_session_cost`, the subagent calibration in
    :func:`_per_subagent_cost`. Subagent counts are read directly from
    every ``session.meta.json`` (cheap), so the subagent total is exact
    for the input set rather than sampled. Audit follow-up 2026-05-24.

    Falls back to a flat ``$0.10 × N`` parent quote if no sample
    sessions could be distilled (calibrated to Gemini 3.5 Flash + v3
    schema + one subagent typical); the subagent count is still scanned
    in that branch.
    """
    samples = _sample_distilled_token_counts(meta_paths, sample_size)
    subagent_counts = _sample_subagent_counts(meta_paths)
    total_subagents = sum(subagent_counts)
    subagent_cost = total_subagents * _per_subagent_cost()
    sessions_with_subagents = sum(1 for c in subagent_counts if c > 0)

    if not samples:
        # Calibrated to Gemini 3.5 Flash + v3 schema; was $0.027/session
        # under Gemini 3 Flash Preview + v2 schema (1024-token output).
        flat_parent_estimate = len(meta_paths) * 0.10
        flat_total = flat_parent_estimate + subagent_cost
        return (
            f"Est. cost (Gemini Flex, flat $0.10/session, parent-only): "
            f"~${flat_parent_estimate:.2f}\n"
            f"  (could not sample real sessions for a refined parent estimate)\n"
            f"  subagent calls: {total_subagents} across "
            f"{sessions_with_subagents} session(s) "
            f"× ${PER_SUBAGENT_COST_USD:.2f} ~= ${subagent_cost:.2f}\n"
            f"  total estimate (parent + subagent): "
            f"~${flat_total:.2f}"
        )
    samples_sorted = sorted(samples)
    mean = sum(samples) / len(samples)
    p50 = samples_sorted[len(samples_sorted) // 2]
    p90 = samples_sorted[int(len(samples_sorted) * 0.9)]
    sample_max = max(samples)
    mean_cost = _per_session_cost(int(mean))
    p90_cost = _per_session_cost(p90)
    max_cost = _per_session_cost(sample_max)
    parent_total = mean_cost * len(meta_paths)
    # Worst-case envelope: assume every session is at the sample's p90.
    parent_worst = p90_cost * len(meta_paths)
    grand_total = parent_total + subagent_cost
    grand_worst = parent_worst + subagent_cost
    return (
        f"Est. cost (Gemini Flex, sampled n={len(samples)} sessions):\n"
        f"  per-session input tokens — "
        f"mean: {int(mean):,}  median: {p50:,}  p90: {p90:,}  max: {sample_max:,}\n"
        f"  per-session parent cost — "
        f"mean: ${mean_cost:.4f}  p90: ${p90_cost:.4f}  max: ${max_cost:.4f}\n"
        f"  parent total (mean × {len(meta_paths)}): ~${parent_total:.2f}\n"
        f"  subagent calls: {total_subagents} across "
        f"{sessions_with_subagents} session(s) "
        f"× ${PER_SUBAGENT_COST_USD:.2f} ~= ${subagent_cost:.2f}\n"
        f"  total estimate (parent + subagent): ~${grand_total:.2f}\n"
        f"  worst-case envelope "
        f"(p90 parent + subagent): ~${grand_worst:.2f}"
    )


# ---------------------------------------------------------------------------
# Cost tracking (audit follow-up 2026-05-28).
#
# Mirrors the validate-production-path.py pattern (Agent D commit bbe2a7b in
# pa-data, audit follow-up #12). The production ``_call_gemini_once`` helper
# returns only ``raw_text`` and does not surface ``usage_metadata``, so to
# capture per-call cost without modifying the toolkit we monkey-patch the
# archive module's ``_call_gemini_once`` with a wrapper that:
#
# 1. Performs the same call with the same config.
# 2. Reads ``usage_metadata`` from the response and computes Flex-tier cost.
# 3. Treats a missing ``usage_metadata`` as ``cost = None`` (unknown, but
#    the call WAS billed — distinguish from "no charge").
# 4. Appends a per-call record to ``_CALL_RECORDS``.
#
# ``_call_gemini_with_retry`` (in the same module) resolves the patched name
# on every attempt, so a single replacement covers both the parent path and
# the per-subagent loop. ``_CURRENT_CONTEXT`` is updated before each call
# type so per-call records carry the target session and the phase
# (``parent`` / ``subagent``).
# ---------------------------------------------------------------------------

_CALL_RECORDS: list[dict[str, Any]] = []
_CURRENT_CONTEXT: dict[str, str] = {"target": "?", "phase": "?"}


def _instrumented_call_gemini_once(
    client: Any,
    user_message: str,
    system_prompt: str,
    response_schema: dict[str, Any] | None = None,
) -> str:
    """Wrapper around production ``_call_gemini_once`` that captures cost.

    Mirrors the production helper's behaviour exactly (config keys,
    ``thinking_budget=0``, optional ``response_schema``, RuntimeError on
    ``response.text is None``) while additionally recording per-call
    usage tokens + cost into ``_CALL_RECORDS``.
    """
    from cc_session_toolkit.config import (
        AUTO_METADATA_MAX_OUTPUT_TOKENS,
        EXTRACTOR_MODEL_ID,
    )

    config: dict[str, Any] = {
        "service_tier": "flex",
        "max_output_tokens": AUTO_METADATA_MAX_OUTPUT_TOKENS,
        "system_instruction": system_prompt,
        "thinking_config": {"thinking_budget": 0},
        "response_mime_type": "application/json",
    }
    if response_schema is not None:
        config["response_schema"] = response_schema

    t0 = time.time()
    response = client.models.generate_content(
        model=EXTRACTOR_MODEL_ID,
        contents=user_message,
        config=config,
    )
    wall_seconds = round(time.time() - t0, 2)

    # Capture usage_metadata before touching response.text — if text is
    # None, we still want the cost record (the call was billed regardless).
    um = getattr(response, "usage_metadata", None)
    if um is None:
        in_tok: int | None = None
        out_tok: int | None = None
        cost_usd: float | None = None
        cost_unknown_reason: str | None = "usage_metadata missing on response"
    else:
        in_tok = getattr(um, "prompt_token_count", None)
        out_tok = getattr(um, "candidates_token_count", None)
        if in_tok is None or out_tok is None:
            cost_usd = None
            cost_unknown_reason = (
                "usage_metadata present but token counts missing"
            )
        else:
            cost_in = (
                in_tok / 1_000_000 * GEMINI_FLEX_INPUT_PRICE_PER_MTOK
            )
            cost_out = (
                out_tok / 1_000_000 * GEMINI_FLEX_OUTPUT_PRICE_PER_MTOK
            )
            cost_usd = round(cost_in + cost_out, 6)
            cost_unknown_reason = None

    _CALL_RECORDS.append({
        "target": _CURRENT_CONTEXT["target"],
        "phase": _CURRENT_CONTEXT["phase"],
        "model": EXTRACTOR_MODEL_ID,
        "input_tokens_charged": in_tok,
        "output_tokens": out_tok,
        "cost_usd": cost_usd,
        "cost_unknown_reason": cost_unknown_reason,
        "wall_seconds": wall_seconds,
        "had_response_schema": response_schema is not None,
    })

    raw = response.text
    if raw is None:
        finish_reason = None
        try:
            finish_reason = response.candidates[0].finish_reason
        except (AttributeError, IndexError, TypeError):
            pass
        raise RuntimeError(
            f"Gemini returned response.text=None "
            f"(finish_reason={finish_reason!r}); likely safety-filtered, "
            f"MAX_TOKENS with no parts, or no candidates."
        )
    return raw


def _summarise_cost_records(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the aggregate cost summary written to the cost log.

    Calls with ``cost_usd is None`` are counted but excluded from the
    sum; the ``total_cost_usd`` is then flagged as a lower bound.
    """
    measured_costs = [
        r["cost_usd"] for r in records if r["cost_usd"] is not None
    ]
    unknown_count = sum(1 for r in records if r["cost_usd"] is None)
    total = round(sum(measured_costs), 6) if measured_costs else 0.0
    by_phase: dict[str, dict[str, Any]] = {}
    for r in records:
        p = r["phase"]
        bucket = by_phase.setdefault(
            p, {"calls": 0, "total_cost_usd": 0.0, "unknown_cost_calls": 0}
        )
        bucket["calls"] += 1
        if r["cost_usd"] is None:
            bucket["unknown_cost_calls"] += 1
        else:
            bucket["total_cost_usd"] = round(
                bucket["total_cost_usd"] + r["cost_usd"], 6
            )
    return {
        "total_calls": len(records),
        "total_cost_usd": total,
        "total_cost_usd_is_lower_bound": unknown_count > 0,
        "unknown_cost_calls": unknown_count,
        "by_phase": by_phase,
    }


def _default_cost_log_path() -> Path:
    """Choose a sensible default cost-log path.

    Prefers ``~/personal-assistant/data/logs/`` (matches the existing
    ``auto-metadata.log`` location on Shawn's setup); falls back to
    ``$CWD`` for portability when that tree is absent.
    """
    candidates = [
        Path.home() / "personal-assistant" / "data" / "logs",
        Path.cwd(),
    ]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for d in candidates:
        if d.exists() and d.is_dir():
            return d / f"backfill-cost-log-{ts}.json"
    # Last-ditch: cwd anyway (will fail at write time if cwd is unwritable
    # but that surfaces a clearer error than silently dropping the log).
    return Path.cwd() / f"backfill-cost-log-{ts}.json"


def _write_cost_log(
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    log_path: Path,
    mode: str,
    run_started_at: str,
) -> None:
    """Persist the per-call records + aggregate summary as JSON."""
    payload = {
        "schema": "backfill-cost-log/1",
        "mode": mode,
        "run_started_at": run_started_at,
        "run_finished_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "records": records,
    }
    log_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
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
    parser.add_argument(
        "--upgrade-to-v13",
        action="store_true",
        help=(
            "Re-summarise sessions that already have v1.2-or-earlier "
            "auto_generated metadata, producing v1.3 output (phases, "
            "decisions, key_exchanges, subagent_summaries). The original "
            "session.meta.json is backed up to session.meta.v2-backup.json "
            "side-by-side before the rebuild. Mutually exclusive with the "
            "default missing-metadata backfill."
        ),
    )
    parser.add_argument(
        "--cost-log",
        type=Path,
        default=None,
        help=(
            "Path for the per-call cost-audit JSON log. Defaults to "
            "~/personal-assistant/data/logs/backfill-cost-log-<UTC>.json "
            "(or $CWD if that tree is absent). The log captures each "
            "Gemini call's input/output tokens, Flex-tier cost, wall "
            "time, and phase (parent/subagent); a missing usage_metadata "
            "block produces cost=None with the total flagged as a lower "
            "bound. Audit follow-up 2026-05-28."
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

    if args.upgrade_to_v13:
        sessions = find_sessions_needing_v13_upgrade(args.archive_root)
        mode_label = "v1.3 upgrade"
    else:
        sessions = find_sessions_needing_backfill(args.archive_root)
        mode_label = "metadata backfill"
    if not sessions:
        if args.upgrade_to_v13:
            print("All sessions already on v1.3 schema. Nothing to do.")
        else:
            print("All sessions already have metadata. Nothing to do.")
        return

    print(f"Found {len(sessions)} session(s) needing {mode_label}.")
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
    skipped_empty = 0

    # Monkey-patch the production Gemini call helper so every charged
    # call lands in ``_CALL_RECORDS``. ``_call_gemini_with_retry``
    # resolves the patched name on every attempt, so the single
    # replacement covers both the parent path and the subagent loop.
    # Restored in the ``finally`` block at end of main so any subsequent
    # in-process run starts from a clean accumulator + original symbol.
    from cc_session_toolkit import archive as _archive_module
    original_call_gemini_once = _archive_module._call_gemini_once
    _archive_module._call_gemini_once = _instrumented_call_gemini_once
    run_started_at = datetime.now(timezone.utc).isoformat()

    try:
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

                # Preflight: skip sessions whose distilled transcript is
                # below the empty threshold. These are typically
                # abandoned shells or JSONLs with only a header turn —
                # Gemini reliably returns None on them (the 2026-05-26
                # upgrade run wasted 11 calls on this class). Mark the
                # meta with a permanent-skip flag so future --upgrade-
                # to-v13 / default-backfill runs auto-skip them too.
                distilled = extract_transcript_text(tmp_path)
                content_tokens = estimate_tokens(distilled)
                if content_tokens < EMPTY_TRANSCRIPT_TOKEN_THRESHOLD:
                    print(
                        f"SKIP-EMPTY ({content_tokens} content tokens "
                        f"< {EMPTY_TRANSCRIPT_TOKEN_THRESHOLD} threshold)"
                    )
                    mark_session_permanently_skipped(
                        meta_path,
                        reason=(
                            f"Transcript below empty threshold "
                            f"({content_tokens} < "
                            f"{EMPTY_TRANSCRIPT_TOKEN_THRESHOLD}); marked "
                            f"in backfill preflight on "
                            f"{datetime.now(timezone.utc).date().isoformat()}."
                        ),
                    )
                    skipped_empty += 1
                    continue

                # Tag every Gemini call this iteration emits with the
                # session's relative archive path + the current phase
                # (parent vs subagent). The wrapper reads these on every
                # call when appending to ``_CALL_RECORDS``.
                _CURRENT_CONTEXT["target"] = str(rel)
                _CURRENT_CONTEXT["phase"] = "parent"
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
                    _CURRENT_CONTEXT["phase"] = "subagent"
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

                # Upgrade path: preserve the pre-v1.3 meta side-by-side
                # before overwriting. Matches the 2026-05-24 production-path
                # validator convention. No-op when --upgrade-to-v13 is not
                # set (the empty-metadata backfill has nothing worth
                # preserving).
                if args.upgrade_to_v13:
                    backup_pre_v13_meta(meta_path)
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

        print(
            f"\nDone: {succeeded} succeeded, {failed} failed, "
            f"{skipped_empty} skipped (empty transcript)."
        )
    finally:
        # Always restore the original helper, even on KeyboardInterrupt
        # mid-run, so an interactive re-run starts clean.
        _archive_module._call_gemini_once = original_call_gemini_once
        # Persist the cost log + print a brief summary regardless of
        # how the run exited — partial data is better than none.
        log_path = args.cost_log or _default_cost_log_path()
        summary = _summarise_cost_records(_CALL_RECORDS)
        try:
            _write_cost_log(
                _CALL_RECORDS, summary, log_path, mode_label, run_started_at,
            )
            print(f"\nCost log written: {log_path}")
        except OSError as log_exc:
            print(
                f"\nWARN: could not write cost log to {log_path}: {log_exc}"
            )
        bound_note = (
            " (lower bound — some calls missing usage_metadata)"
            if summary["total_cost_usd_is_lower_bound"] else ""
        )
        parent = summary["by_phase"].get(
            "parent", {"calls": 0, "total_cost_usd": 0.0}
        )
        subagent = summary["by_phase"].get(
            "subagent", {"calls": 0, "total_cost_usd": 0.0}
        )
        print(
            f"Total spend: ${summary['total_cost_usd']:.4f} across "
            f"{summary['total_calls']} calls{bound_note}\n"
            f"  parent:   {parent['calls']:>4} calls  "
            f"${parent['total_cost_usd']:.4f}\n"
            f"  subagent: {subagent['calls']:>4} calls  "
            f"${subagent['total_cost_usd']:.4f}"
        )


if __name__ == "__main__":
    main()
