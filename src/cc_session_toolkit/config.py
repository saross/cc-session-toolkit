"""
Constants, file type mappings, and defaults loading for CC session archiving.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.3"
# 1.3 (2026-05-24): auto_generated gains optional phases[], decisions[],
# and key_exchanges[] arrays; top-level subagent_summaries[] added for
# lightweight per-subagent narrative summaries. All new fields are
# additive — 1.2 consumers continue to work unchanged.

# ---------------------------------------------------------------------------
# Extractor model
# ---------------------------------------------------------------------------
# The model used for automatic metadata generation (title / purpose /
# tags / Three Ps summaries). Surfaced in ``session.meta.json`` under
# ``extractor_model_id`` so that downstream RO-Crate / FAIR consumers
# can attribute the generated fields to a specific model version —
# provenance audit Gap 3, 2026-05-17.
#
# 2026-05-18: switched from ``claude-haiku-4-5-20251001`` (Anthropic
# Batch) to ``gemini-3-flash-preview`` (Google, Flex tier) after the
# bake-off in
# ``personal-assistant/data/experiments/bake-off-metadata-2026-05-18/``
# established Gemini-tuned-v2 as decisively better on quality (17–7 of
# 42 cells against Haiku, 18 ties), reliability (10/10 vs 7/10),
# cost (~½ Haiku), and architectural simplicity (single one-shot
# call vs chunking + stitching).
#
# 2026-05-22: migrated to ``gemini-3.5-flash`` (Google, Flex tier, GA)
# after a 3-session head-to-head comparison (small/medium/large amd-tower
# unarchived sessions). 3.5 Flash showed materially better named-entity
# preservation (commit hashes, tags, CI bounds, people's names), more
# numeric specificity, ~20% faster wall-clock, and zero JSON structural
# defects vs 1-in-3 for 3 Flash Preview (a stray ``three_ps.``-prefixed
# key under ``three_ps``). 3× Flex price accepted (envelope $25 for the
# 107-session Step 2 sweep). 3 Flash Preview retained "Preview" status
# and would have needed migration eventually anyway.
#
# Bumping this constant: change in lockstep with the call shape in
# ``archive.generate_auto_metadata`` and flag in the changelog. PA's
# ``hooks/extraction-hook.py`` carries an independent ``HAIKU_MODEL``
# constant for memory extraction; the two are deliberately
# loose-coupled because they may move on different cadences.
#
# 2026-10-08: migrated to ``gemini-3.8-flash`` at thinking level
# ``medium`` after a 10-session comparison (PA
# ``data/experiments/extractor-comparison-2026-10-08/report.md``). 3.8
# matched or beat 3.5 on schema validity, verbatim quotes, and invented
# specifics (none for either), and covered more side topics in long
# sessions (19 vs 12 of 22), the known weakness. It cost ~52% of 3.5 at
# the introductory 3.8 Flex price, ~103% once that price doubles on
# 2027-01-01. Google's 2026-10-07 notice also deprecates
# ``thinking_budget`` (upcoming models reject it with 400).
#
# 2026-10-08 (later): the Gemini model is now one of two extractor
# providers, see ``EXTRACTOR_PROVIDER`` below. Its constant is renamed
# ``GEMINI_EXTRACTOR_MODEL_ID``; every Gemini call and token count uses it.
GEMINI_EXTRACTOR_MODEL_ID = "gemini-3.8-flash"

# Thinking level for the extractor (2026-10-08). ``medium`` is the
# documented default for 3.8 Flash; it is set explicitly so the record
# states it. ``low`` produced no thinking tokens and weaker side-topic
# coverage; ``high`` exceeded an 8,192-token cap on 2 of 3 sessions.
# ``minimal`` is rejected by 3.8. Sending ``thinking_budget`` together
# with ``thinking_level`` returns 400, so only the level is sent.
AUTO_METADATA_THINKING_LEVEL = "medium"

# ---------------------------------------------------------------------------
# Extractor provider (added 2026-10-08)
# ---------------------------------------------------------------------------
# Which provider writes the auto-metadata: ``"openai"`` (GPT-6 Luna) or
# ``"gemini"`` (the Gemini path above, kept intact). Shawn chose Luna on
# 2026-10-08 after the 10-session comparison (PA
# ``data/experiments/extractor-comparison-2026-10-08/report.md``): topic
# coverage matched 3.8-medium (62 vs 61 of 64; side topics in long sessions
# 20 vs 19 of 22), every one of its 44 quotes was verbatim (3.8: 33 of 35,
# two with added punctuation), and at standard tier it cost about a third
# of 3.8 Flex (US$0.29 vs US$0.81 for the 10 sessions) at a median 14 s
# against 33 s. Standard tier is never preempted, which removes the Flex
# 503s that left 25 sessions with placeholder metadata.
#
# The environment variable ``CC_EXTRACTOR_PROVIDER`` overrides the choice
# without a code change. When the primary provider fails (after its own
# retries), the call falls back to ``EXTRACTOR_FALLBACK_PROVIDER``; set it
# to ``None`` to disable. Each record names the model that actually wrote
# it, so a fallback is visible in ``extractor_model_id``.
EXTRACTOR_PROVIDER: str = os.environ.get("CC_EXTRACTOR_PROVIDER") or "openai"
EXTRACTOR_FALLBACK_PROVIDER: str | None = "gemini"

# OpenAI model and reasoning effort. ``none`` is the effort the comparison
# measured as strongest (``luna6-none``); ``low`` scored the same within
# noise at ~1.3x the latency. Recorded as ``extractor_thinking_level``.
OPENAI_EXTRACTOR_MODEL_ID = "gpt-6-luna"
OPENAI_REASONING_EFFORT = "none"

# The OpenAI key is role-scoped, never a generic ``OPENAI_API_KEY``, so the
# spend lands in the right OpenAI project: role ``PA`` keys belong to the
# ``personal-assistant`` project (Shawn, 2026-10-08). Resolution order:
# ``OPENAI_API_KEY_<ROLE>`` (an override), then
# ``OPENAI_API_KEY_<ROLE>_<SUFFIX>``, with the suffix from
# ``OPENAI_KEY_SUFFIX`` or the hostname below; environment first, then the
# PA ``.env`` file (hooks do not inherit it). Mirrors PA
# ``scripts/_openai_key.py``.
OPENAI_KEY_ROLE = "PA"
OPENAI_KEY_HOST_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("zbook", "ZBOOK"),
    ("amd-tower", "AMDT"),
)

# Request handling. Standard tier (no ``service_tier`` sent). Retries cover
# rate limits and transient server errors; a 429 for exhausted quota and
# any 4xx other than 429 are not retried. The timeout is per attempt.
OPENAI_RETRY_WAITS_SECONDS = (10, 30, 60)
OPENAI_REQUEST_TIMEOUT_SECONDS = 300
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"

# GPT-6 Luna list prices, standard tier (USD per million tokens), verified
# 2026-10-08 (PA ``extractor-comparison-2026-10-08/pricing/
# openai-model-gpt-6-luna-2026-10-08.txt``, line 9). A request whose prompt
# exceeds 272,000 tokens pays 2x input and 1.5x output for the whole
# request. Output prices include reasoning tokens.
OPENAI_INPUT_PRICE_PER_MTOK = 0.10
OPENAI_CACHED_INPUT_PRICE_PER_MTOK = 0.01
OPENAI_OUTPUT_PRICE_PER_MTOK = 0.50
OPENAI_LONG_PROMPT_THRESHOLD_TOKENS = 272_000
OPENAI_LONG_PROMPT_INPUT_MULTIPLIER = 2.0
OPENAI_LONG_PROMPT_OUTPUT_MULTIPLIER = 1.5

# The primary model and its thinking setting: the defaults recorded when a
# caller names no model. A fallback records its own.
EXTRACTOR_MODEL_ID = (
    OPENAI_EXTRACTOR_MODEL_ID if EXTRACTOR_PROVIDER == "openai"
    else GEMINI_EXTRACTOR_MODEL_ID
)
EXTRACTOR_THINKING_LEVEL = (
    OPENAI_REASONING_EFFORT if EXTRACTOR_PROVIDER == "openai"
    else AUTO_METADATA_THINKING_LEVEL
)

# ---------------------------------------------------------------------------
# Auto-metadata extraction tuning
# ---------------------------------------------------------------------------
# Output cap for the JSON object the extractor emits.
# v3 schema (2026-05-24) added phases[], decisions[], key_exchanges[]
# alongside the existing title/purpose/tags/three_ps; long sessions
# can produce ~3-5K-token outputs (process_summary up to ~1000 words
# plus the arrays). Raised from 1024 → 8192 so the v3 ceiling is the
# binding constraint, not the API budget. Short sessions still produce
# small responses; this is an upper bound, not a target. The same cap
# applies to the subagent-narrative call path — subagent outputs are
# much smaller (~60–200 words) so the cap is non-binding there, and a
# separate constant would be dead code per the 2026-05-24 audit.
#
# 2026-10-08: raised 8192 → 16384 because the cap counts thinking tokens.
# At thinking level medium, one comparison session used 6,316 of 8,192
# (77%), and a truncated response yields unparseable JSON, i.e. no
# metadata at all.
AUTO_METADATA_MAX_OUTPUT_TOKENS = 16384

# Wait pattern for Flex preemption (HTTP 503) retries. Per Google's
# Flex documentation, preemption surfaces as HTTP 503 "Service
# Unavailable". The bake-off used (30, 60, 120) and observed zero
# preemptions across 10 sessions, but the schedule stays in place as
# cheap insurance for the production-cadence higher-volume path.
#
# 2026-10-08: lengthened, with one standard-tier attempt after the last
# Flex failure (see ``archive._call_gemini_with_retry``). Production log:
# Flex give-ups rose to 36% of main-session extractions in October; on
# the comparison day 3.8 Flex returned 503 on 32 of 59 calls and 3 of 27
# jobs needed more than 4 attempts. The comparison suggested waits up
# to (30, 60, 120, 240, 300, 300); the schedule stops at 240 s because
# the hooks run async with a 120 s timeout and the longest extraction
# the log shows surviving is ~7 minutes. The standard-tier fallback
# covers the rest.
AUTO_METADATA_FLEX_RETRY_WAITS_SECONDS = (30, 60, 120, 240)

# Re-archive (supersede) policy, added 2026-10-08. When a session is
# archived again after its transcript has grown (a later PreCompact, the
# SessionEnd after a PreCompact, or a resumed session's next end), the
# transcript is always refreshed, because that is local and free. The
# model metadata is regenerated only at SessionEnd, and only when the
# transcript has grown by at least this fraction since the bytes the
# metadata was generated from (``extractor_source_bytes``). Smaller
# growth carries the existing metadata forward: a few trailing hook
# records do not justify a Gemini call. Placeholder metadata is always
# retried regardless of growth.
AUTO_METADATA_REGEN_GROWTH_FRACTION = 0.10

# Gemini list prices (USD per million tokens) — see
# https://ai.google.dev/gemini-api/docs/pricing. Track
# ``EXTRACTOR_MODEL_ID`` and must be re-verified whenever it changes.
# Surfaced for cost estimation in backfill / batch code.
#
# Gemini 3.8 Flash, verified 2026-10-08 (PA experiment
# ``extractor-comparison-2026-10-08/pricing/gemini-pricing-2026-10-08.txt``).
# Output prices include thinking tokens. ⚠ These are INTRODUCTORY prices
# through 2026-12-31; from 2027-01-01 they double (Flex 0.75 / 3.75,
# standard 1.50 / 7.50). Update them then.
GEMINI_FLEX_INPUT_PRICE_PER_MTOK = 0.375
GEMINI_FLEX_OUTPUT_PRICE_PER_MTOK = 1.875
GEMINI_STANDARD_INPUT_PRICE_PER_MTOK = 0.75
GEMINI_STANDARD_OUTPUT_PRICE_PER_MTOK = 3.75

# ---------------------------------------------------------------------------
# Subagent summary fan-out cap
# ---------------------------------------------------------------------------
# Upper bound on the number of subagents summarised in a single archive
# call. Each subagent triggers an independent Gemini Flex call costing
# ~$0.04 actual ($0.05 budgeted), so a runaway orchestrator with 100+
# subagents could quietly spend $4-5 per session. The cap is a circuit
# breaker, not a target — typical sessions have 0–5 subagents and never
# approach it. When the cap is hit, ``generate_subagent_summaries``
# slices the list and emits a WARN log line recording the actual N and
# the cap so the truncation is auditable.
#
# Calibrated 2026-05-28 against the live archive: 17 sessions out of 637
# in the v1.3 upgrade run exceeded the original 20-cap, with the
# heaviest at 68 subagents. Raised to 70 to cover Shawn's full empirical
# distribution (worst-case session ~$3 vs ~$1 at the 20-cap; per-session
# blast radius still small relative to $216 archive-wide envelope).
# Originally 20 (audit follow-up 2026-05-24); bumped to 70 (2026-05-28).
MAX_SUBAGENT_SUMMARIES = 70

# ---------------------------------------------------------------------------
# Sharing licence default
# ---------------------------------------------------------------------------
# Default ``licence`` field for ``session.meta.json`` records (provenance
# audit Gap 3, 2026-05-17). Left as ``None`` so the user explicitly opts
# in to a licence string at the moment a session record becomes
# shareable. RO-Crate spec requires a licence URI on shared records;
# emitting one by default would either (a) lie about a non-existent
# project-wide policy or (b) silently commit Shawn to a default he never
# chose. Both worse than ``None``.
DEFAULT_LICENCE: str | None = None

# ---------------------------------------------------------------------------
# code_state sidecar directory (provenance audit Gap 2 — commit_at_start)
# ---------------------------------------------------------------------------
# The archive runs at session-close, so the git HEAD it can capture
# directly is ``commit_at_end``.  To populate ``commit_at_start``, a
# SessionStart hook in the personal-assistant repo writes a sidecar
# keyed by session id when each session begins:
#
#     ~/personal-assistant/hooks/session-start-code-state.py
#
# ``capture_code_state`` reads this directory at archive time.  The
# default points at the personal-assistant sidecar location; downstream
# users without that hook can leave it untouched (the lookup is
# best-effort, and a missing sidecar simply leaves ``commit_at_start``
# as ``None``).  Environment variable ``CC_CODE_STATE_SIDECAR_DIR``
# overrides the default for non-default deployments.
import os as _os

CODE_STATE_SIDECAR_DIR = Path(
    _os.environ.get(
        "CC_CODE_STATE_SIDECAR_DIR",
        Path.home() / "personal-assistant" / "data" / "code-state",
    )
)

# ---------------------------------------------------------------------------
# Global archive defaults (for hook-based automated archiving)
# ---------------------------------------------------------------------------

DEFAULT_ARCHIVE_ROOT = Path.home() / "cc-archives"

# Trivial session thresholds — sessions below these limits are skipped
DEFAULT_MIN_TURNS = 5
DEFAULT_MIN_DURATION_MINUTES = 1

# ---------------------------------------------------------------------------
# Default thinking-block ethics preferences
# ---------------------------------------------------------------------------

DEFAULT_THINKING_SHARING = "research-only"

DEFAULT_THINKING_USE_CONSTRAINTS = [
    "analysis-for-improvement",
    "research-publication-aggregated",
]

DEFAULT_THINKING_EXCLUDED_USES = [
    "training-data",
    "public-display-individual",
]

DEFAULT_THINKING_NATURE_NOTE = (
    "Work-in-progress reasoning traces, not polished output. "
    "May contain abandoned paths and self-corrections."
)

# ---------------------------------------------------------------------------
# File type mappings for artifact categorisation
# ---------------------------------------------------------------------------

FILE_TYPE_MAPPINGS: dict[str, str] = {
    ".py": "code",
    ".js": "code",
    ".ts": "code",
    ".sh": "code",
    ".r": "code",
    ".R": "code",
    ".sql": "code",
    ".md": "document",
    ".txt": "document",
    ".rst": "document",
    ".json": "data",
    ".csv": "data",
    ".jsonl": "data",
    ".geojson": "data",
    ".yaml": "config",
    ".yml": "config",
    ".toml": "config",
    ".ini": "config",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".svg": "image",
    ".tif": "image",
    ".tiff": "image",
}


def get_file_type(file_path: str | Path) -> str:
    """
    Determine file type from extension.

    Args:
        file_path: Path to the file (or just a filename).

    Returns:
        File type string: ``code``, ``document``, ``data``, ``config``,
        ``image``, or ``other``.
    """
    ext = Path(file_path).suffix.lower()
    return FILE_TYPE_MAPPINGS.get(ext, "other")


def load_defaults(defaults_file: Path) -> dict[str, Any]:
    """
    Load default configuration from an ``archive-defaults.yaml`` file.

    Args:
        defaults_file: Path to the YAML defaults file.

    Returns:
        Dictionary of defaults, or empty dict if the file is missing
        or PyYAML is not installed.
    """
    try:
        import yaml  # noqa: WPS433 — optional dependency
    except ImportError:
        return {}

    if not defaults_file.exists():
        return {}

    try:
        with open(defaults_file, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:  # noqa: BLE001 — graceful degradation
        return {}
