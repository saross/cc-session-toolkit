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
# The Anthropic model used for automatic metadata generation
# (title / purpose / tags / Three Ps summaries). Surfaced in
# ``session.meta.json`` under ``provenance.extractor_model_id`` so that
# downstream RO-Crate / FAIR consumers can attribute the generated
# fields to a specific model version — provenance audit Gap 3, 2026-05-17.
#
# Bumping this constant: change in lockstep with any other Haiku
# references in the toolkit and flag in the changelog. PA's
# ``hooks/extraction-hook.py`` carries an independent ``HAIKU_MODEL``
# constant for memory extraction; the two are deliberately
# loose-coupled because they may move on different cadences.
EXTRACTOR_MODEL_ID = "claude-haiku-4-5-20251001"

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
