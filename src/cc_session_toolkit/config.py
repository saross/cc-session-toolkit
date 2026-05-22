"""
Constants, file type mappings, and defaults loading for CC session archiving.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.2"

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
EXTRACTOR_MODEL_ID = "gemini-3.5-flash"

# ---------------------------------------------------------------------------
# Auto-metadata extraction tuning
# ---------------------------------------------------------------------------
# Output cap for the JSON object the extractor emits. The target object
# runs ~300 tokens (title, purpose, tags, three_ps); 1024 is a safe
# ceiling that still beats Gemini Flash's per-call limit comfortably.
AUTO_METADATA_MAX_OUTPUT_TOKENS = 1024

# Wait pattern for Flex preemption (HTTP 503) retries. Per Google's
# Flex documentation, preemption surfaces as HTTP 503 "Service
# Unavailable". The bake-off used (30, 60, 120) and observed zero
# preemptions across 10 sessions, but the schedule stays in place as
# cheap insurance for the production-cadence higher-volume path.
AUTO_METADATA_FLEX_RETRY_WAITS_SECONDS = (30, 60, 120)

# Gemini Flex list price (USD per million tokens) — see
# https://ai.google.dev/gemini-api/docs/pricing#flex. Verified
# 2026-05-22 for Gemini 3.5 Flash (3× the 3 Flash Preview Flex rate);
# tracks ``EXTRACTOR_MODEL_ID`` and must be re-verified whenever it
# changes. Surfaced for cost estimation in backfill / batch code.
GEMINI_FLEX_INPUT_PRICE_PER_MTOK = 0.75
GEMINI_FLEX_OUTPUT_PRICE_PER_MTOK = 4.50

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
