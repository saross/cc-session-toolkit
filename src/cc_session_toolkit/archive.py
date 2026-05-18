"""
Archive operations — copy, compress, and create metadata for CC sessions.

Merges features from both diverged versions:

- map-reader-llm v1.1: title-based naming, catalogue project rollups
- llm-reproducibility v1.2: find_archive_by_id, update_catalogue_entry
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from cc_session_toolkit.config import (
    AUTO_METADATA_FLEX_RETRY_WAITS_SECONDS,
    AUTO_METADATA_MAX_OUTPUT_TOKENS,
    CODE_STATE_SIDECAR_DIR,
    DEFAULT_LICENCE,
    DEFAULT_MIN_DURATION_MINUTES,
    DEFAULT_MIN_TURNS,
    DEFAULT_THINKING_EXCLUDED_USES,
    DEFAULT_THINKING_NATURE_NOTE,
    DEFAULT_THINKING_SHARING,
    DEFAULT_THINKING_USE_CONSTRAINTS,
    EXTRACTOR_MODEL_ID,
    SCHEMA_VERSION,
    load_defaults,
)
from cc_session_toolkit.extraction import (
    detect_relationship_hints,
    estimate_cost,
    extract_artifacts,
    extract_session_stats,
    extract_thinking_block_tokens,
    extract_tool_output_bytes,
)
from cc_session_toolkit.naming import get_archive_directory
from cc_session_toolkit.project import (
    CLAUDE_PROJECTS_DIR,
    get_archive_dir,
    get_cc_project_path,
    get_defaults_file,
    get_project_name,
)


# -------------------------------------------------------------------------
# Session discovery
# -------------------------------------------------------------------------

def get_session_files(project_root: Path) -> list[Path]:
    """
    Find all parent session JSONL files for a project in ``~/.claude/projects/``.

    Excludes ``agent-*.jsonl`` files, which are sub-agent transcripts from
    the pre-2026-01-12 flat layout (current CC places them under
    ``<session_id>/subagents/``). Sub-agent transcripts are archived
    alongside their parent, not as top-level sessions.

    Args:
        project_root: Project root directory.

    Returns:
        List of session file paths, sorted by modification time.
    """
    cc_project_path = get_cc_project_path(project_root)
    project_dir = CLAUDE_PROJECTS_DIR / cc_project_path

    if not project_dir.exists():
        return []

    session_files = [
        p for p in project_dir.glob("*.jsonl")
        if not p.name.startswith("agent-")
    ]
    return sorted(session_files, key=lambda p: p.stat().st_mtime)


def get_archived_session_ids(
    catalogue_file: Path,
) -> set[str]:
    """
    Get session IDs that have already been archived.

    Args:
        catalogue_file: Path to ``CATALOG.json``.

    Returns:
        Set of archived session ID strings.
    """
    if not catalogue_file.exists():
        return set()

    try:
        catalogue = json.loads(catalogue_file.read_text(encoding="utf-8"))
        return {s["id"] for s in catalogue.get("sessions", [])}
    except (json.JSONDecodeError, KeyError):
        return set()


def get_session_id(session_path: Path) -> str:
    """
    Extract session ID from a JSONL filename.

    Main sessions use a UUID stem; agent sessions use ``agent-<short_id>``.

    Args:
        session_path: Path to the JSONL file.

    Returns:
        Session identifier string.
    """
    return session_path.stem


# -------------------------------------------------------------------------
# Hook-mode helpers (trivial filter, dedup, auto-metadata)
# -------------------------------------------------------------------------

def is_trivial_session(
    stats: dict[str, Any],
    min_turns: int = DEFAULT_MIN_TURNS,
    min_duration: int = DEFAULT_MIN_DURATION_MINUTES,
) -> bool:
    """
    Check whether a session is too short to be worth archiving.

    Used by hook-based archiving to skip trivial sessions (e.g.
    accidental opens, quick ``/help`` queries).

    Args:
        stats: Session statistics from :func:`extract_session_stats`.
        min_turns: Minimum number of human turns.
        min_duration: Minimum duration in minutes.

    Returns:
        *True* if the session is trivial and should be skipped.
    """
    if stats.get("turns", 0) < min_turns:
        return True
    if stats.get("duration_minutes", 0) < min_duration:
        return True
    return False


def is_already_archived(
    session_id: str,
    catalogue_file: Path,
) -> bool:
    """
    Check whether a session has already been archived.

    Used for deduplication when both PreCompact and SessionEnd hooks
    fire for the same session — the first one to archive wins.

    Args:
        session_id: Session identifier to check.
        catalogue_file: Path to ``CATALOG.json`` (global or per-project).

    Returns:
        *True* if the session is already in the catalogue.
    """
    return session_id in get_archived_session_ids(catalogue_file)


def _ensure_gemini_api_key() -> str | None:
    """
    Resolve the Gemini API key from the environment, with a PA
    ``.env`` fallback.

    When called from Claude Code hooks, environment variables exported
    via the launcher may not survive the shell chain. This helper
    mirrors the old ``_ensure_anthropic_api_key`` semantics for the
    Gemini case: check ``GEMINI_API_KEY`` first, then ``GOOGLE_API_KEY``
    (both are accepted by ``google.genai.Client()``), then sniff the
    personal-assistant ``.env`` file for the same names.

    Returns the resolved key (after setting it into ``os.environ`` if it
    came from the .env file), or *None* when no key is available. The
    caller is expected to refuse to make the API call when *None* is
    returned.
    """
    import os

    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value

    env_path = Path.home() / "personal-assistant" / ".env"
    if not env_path.is_file():
        return None

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key in ("GEMINI_API_KEY", "GOOGLE_API_KEY") and value:
            os.environ[key] = value
            return value
    return None


def _log_metadata_event(
    message: str,
    *,
    level: str = "INFO",
) -> None:
    """Append a timestamped line to the auto-metadata log."""
    import os

    log_dir = Path(
        os.environ.get(
            "CC_SESSION_LOG_DIR",
            Path.home() / "personal-assistant" / "data" / "logs",
        )
    )
    log_file = log_dir / "auto-metadata.log"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as fh:
            fh.write(f"{timestamp} [{level}] {message}\n")
    except OSError:
        pass  # logging must never break the archive


def _load_auto_metadata_prompt() -> str:
    """
    Load the production auto-metadata system prompt.

    The prompt is shipped as package data at
    ``src/cc_session_toolkit/prompts/auto_metadata.md`` (see
    ``pyproject.toml`` ``[tool.setuptools.package-data]``). It is the
    production-candidate ``prompt-gemini-v2.md`` from the 2026-05-18
    bake-off; see continuity log workstream F.

    Environment variable ``CC_AUTO_METADATA_PROMPT_PATH`` overrides the
    bundled prompt with an arbitrary filesystem path — useful for
    iterating on the prompt without re-installing the package.
    """
    import os

    override = os.environ.get("CC_AUTO_METADATA_PROMPT_PATH")
    if override:
        return Path(override).read_text(encoding="utf-8")

    # importlib.resources is the canonical way to read package data
    # under modern Python. The ``files`` API arrived in 3.9 and is
    # what setuptools' package_data ships with.
    from importlib.resources import files

    return (
        files("cc_session_toolkit.prompts")
        .joinpath("auto_metadata.md")
        .read_text(encoding="utf-8")
    )


def _build_auto_metadata_user_message(
    *,
    session_id: str,
    project: str,
    started_at: str,
    content_tokens: int,
    transcript_text: str,
) -> str:
    """
    Build the user-message payload for the Gemini auto-metadata call.

    Structural contract (architectural decision 2026-05-18):
    - System instruction carries the role + contracts + JSON output
      spec; this user message carries the *input* (header + transcript)
      plus a post-transcript output reminder.
    - Transcript is wrapped in ``<transcript>`` tags with neutral
      ``--- Role ---`` dividers (not square-bracket chat markers).
      This combination defeats the failure mode where Haiku and Gemini
      would *continue* the conversation instead of summarising it.
    - Output reminder lives *after* the closing ``</transcript>`` tag
      so recency favours the contract rather than the transcript tail.
    """
    header = (
        "## Session metadata header (not authoritative — transcript wins)\n"
        f"- Session ID: {session_id}\n"
        f"- Project: {project}\n"
        f"- Started at: {started_at}\n"
        f"- Distilled content tokens (chars/4): {content_tokens:,}\n"
    )
    postamble = (
        "## Output reminder\n\n"
        "You have now read the complete transcript. Return a single JSON "
        "object with keys ``title``, ``purpose``, ``tags``, and "
        "``three_ps`` (an object with ``prompt_summary``, "
        "``process_summary``, ``provenance_summary``). Field contracts "
        "and anti-satisficing rules are in the system prompt; apply them.\n\n"
        "You are an outside observer summarising the transcript. You are "
        "not a participant. Do not continue the conversation.\n\n"
        "Begin output with ``{`` on the very next character. End with "
        "``}``. No markdown code fence. Nothing before, nothing after."
    )
    return (
        f"{header}\n"
        f"<transcript>\n"
        f"{transcript_text}\n"
        f"</transcript>\n\n"
        f"{postamble}"
    )


def _parse_metadata_response_json(raw_text: str) -> dict[str, Any]:
    """
    Parse a JSON object out of a model response.

    The prompt instructs the model to emit bare JSON (no fences), but
    real-world models intermittently wrap output in ``` json blocks.
    This strips any single leading/trailing fence and then tries
    ``json.loads``. On any failure, raises ``ValueError`` so callers
    can record the raw text for diagnosis.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[-1].startswith("```"):
            text = "\n".join(lines[1:-1])
        else:
            text = "\n".join(lines[1:])
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parse failed: {exc}") from exc


def _call_gemini_once(
    client: Any,
    user_message: str,
    system_prompt: str,
) -> str:
    """
    Single Flex-tier Gemini call. Raises on any error; returns raw text.

    ``thinking_budget=0`` is mandatory (architectural decision
    2026-05-18): Gemini 3 Flash Preview is a reasoning model and without
    this flag thinking tokens consume the output budget before any JSON
    is emitted. ``service_tier="flex"`` selects the discounted tier;
    list price ~½ Haiku Batch.
    """
    response = client.models.generate_content(
        model=EXTRACTOR_MODEL_ID,
        contents=user_message,
        config={
            "service_tier": "flex",
            "max_output_tokens": AUTO_METADATA_MAX_OUTPUT_TOKENS,
            "system_instruction": system_prompt,
            "thinking_config": {"thinking_budget": 0},
        },
    )
    return response.text


def _call_gemini_with_retry(
    client: Any,
    user_message: str,
    system_prompt: str,
) -> str:
    """
    Call Gemini Flex with exponential-backoff retries on HTTP 503.

    Flex preemption surfaces as HTTP 503 "Service Unavailable" (see
    Google's Flex documentation). We retry on 503 specifically; other
    errors propagate. The schedule is configurable via
    ``AUTO_METADATA_FLEX_RETRY_WAITS_SECONDS``.
    """
    import time

    last_exc: Exception | None = None
    waits: tuple[int, ...] = (0,) + tuple(
        AUTO_METADATA_FLEX_RETRY_WAITS_SECONDS
    )
    for attempt, wait_seconds in enumerate(waits):
        if wait_seconds:
            _log_metadata_event(
                f"Gemini Flex preempted; waiting {wait_seconds}s before "
                f"retry (attempt {attempt + 1}/{len(waits)})",
                level="WARNING",
            )
            time.sleep(wait_seconds)
        try:
            return _call_gemini_once(client, user_message, system_prompt)
        except Exception as exc:  # noqa: BLE001 — narrow below
            last_exc = exc
            # Detect 503 from the exception text rather than importing
            # the specific error class — the SDK may not be available
            # at module-load time on every machine.
            text = str(exc).lower()
            is_503 = (
                "503" in text
                or "service unavailable" in text
                or "preempt" in text
            )
            if not is_503:
                raise
            # Else: retry on next loop iteration.
    raise RuntimeError(
        f"Gemini Flex preempted {len(waits)} times; last error: {last_exc}"
    )


def generate_auto_metadata(
    session_path: Path,
    stats: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Generate title / purpose / tags / Three Ps via Gemini Flex.

    Architecture (2026-05-18 bake-off winner):
    - Full distilled transcript via :func:`transcript_text.extract_transcript_text`
      (not first-and-last sampled messages).
    - Production prompt loaded from package data at
      ``cc_session_toolkit/prompts/auto_metadata.md`` (override via
      ``CC_AUTO_METADATA_PROMPT_PATH`` env var).
    - Gemini 3 Flash Preview (Flex tier) via ``google.genai`` with
      ``thinking_budget=0`` and 503-retry backoff.

    Budget: ~$0.027 per session (~½ Haiku Batch list price).

    Falls back to *None* on any of: ``google-genai`` not installed, no
    API key available, transcript extraction failure, persistent 503,
    or response not parseable as JSON. The archive remains usable
    without auto-metadata; the field collapses to the schema-level
    "Untitled Session" default.

    Args:
        session_path: Path to the session JSONL (or JSONL.gz) file.
        stats: Session statistics from :func:`extract_session_stats`.
            Used only for the input-header summary; the model grounds
            every claim in the transcript text.

    Returns:
        Dictionary with ``title``, ``purpose``, ``tags``, and
        ``three_ps`` keys, or *None* on failure.
    """
    try:
        from google import genai  # noqa: WPS433 — optional dependency
    except ImportError:
        _log_metadata_event(
            "google-genai package not installed — skipping auto-metadata",
            level="WARNING",
        )
        print(
            "  Warning: google-genai package not installed, "
            "skipping auto-metadata"
        )
        return None

    api_key = _ensure_gemini_api_key()
    if not api_key:
        _log_metadata_event(
            f"No GEMINI_API_KEY / GOOGLE_API_KEY for {session_path.name}",
            level="ERROR",
        )
        print(
            "  Warning: no GEMINI_API_KEY / GOOGLE_API_KEY — "
            "skipping auto-metadata"
        )
        return None

    # Distil the full transcript. The extractor strips framing,
    # preserves tool calls + results, and is gzip-aware.
    try:
        from cc_session_toolkit.transcript_text import (
            estimate_tokens,
            extract_transcript_text,
        )
        transcript_text = extract_transcript_text(session_path)
    except (FileNotFoundError, OSError) as exc:
        _log_metadata_event(
            f"Transcript extraction failed for {session_path.name}: "
            f"{type(exc).__name__}: {exc}",
            level="ERROR",
        )
        return None

    if not transcript_text:
        _log_metadata_event(
            f"Empty distilled transcript for {session_path.name}; "
            "skipping auto-metadata",
            level="WARNING",
        )
        return None

    content_tokens = estimate_tokens(transcript_text)
    system_prompt = _load_auto_metadata_prompt()
    user_message = _build_auto_metadata_user_message(
        session_id=stats.get("session_id") or session_path.stem,
        project=stats.get("project_name") or "",
        started_at=stats.get("started_at") or "",
        content_tokens=content_tokens,
        transcript_text=transcript_text,
    )

    try:
        client = genai.Client()
        _log_metadata_event(
            f"Calling Gemini Flex for {session_path.name} "
            f"({content_tokens:,} content tokens)"
        )
        raw_text = _call_gemini_with_retry(
            client, user_message, system_prompt
        )
    except Exception as exc:  # noqa: BLE001 — graceful degradation
        _log_metadata_event(
            f"Gemini call failed for {session_path.name}: "
            f"{type(exc).__name__}: {exc}",
            level="ERROR",
        )
        print(f"  Warning: auto-metadata generation failed: {exc}")
        return None

    try:
        result = _parse_metadata_response_json(raw_text)
    except ValueError as exc:
        _log_metadata_event(
            f"Gemini response not parseable as JSON for "
            f"{session_path.name}: {exc}; raw[:300]={raw_text[:300]!r}",
            level="ERROR",
        )
        print(f"  Warning: auto-metadata JSON parse failed: {exc}")
        return None

    title = result.get("title", "Untitled Session")
    _log_metadata_event(
        f"Success for {session_path.name}: {title!r}"
    )
    auto_meta: dict[str, Any] = {
        "title": title,
        "purpose": result.get("purpose", ""),
        "tags": result.get("tags", []),
    }
    three_ps = result.get("three_ps")
    if isinstance(three_ps, dict):
        auto_meta["three_ps"] = {
            "prompt_summary": three_ps.get("prompt_summary", ""),
            "process_summary": three_ps.get("process_summary", ""),
            "provenance_summary": three_ps.get("provenance_summary", ""),
        }
    return auto_meta


# -------------------------------------------------------------------------
# Metadata prompt generation
# -------------------------------------------------------------------------

def generate_metadata_prompt(
    session_id: str, stats: dict[str, Any]
) -> str:
    """
    Generate a prompt for CC to create session metadata.

    The prompt is printed to stdout so CC can respond with the metadata
    JSON during an interactive archiving session.

    Args:
        session_id: Session identifier.
        stats: Extracted session statistics.

    Returns:
        Prompt string.
    """
    tool_summary = ", ".join(
        f"{k}: {v}" for k, v in stats["tool_calls"]["by_type"].items()
    )
    return (
        f"I need you to generate metadata for archiving the CC session "
        f"that just completed.\n\n"
        f"**Session ID**: {session_id}\n"
        f"**Duration**: {stats['duration_minutes']} minutes\n"
        f"**Turns**: {stats['turns']} "
        f"({stats['human_messages']} human, "
        f"{stats['assistant_messages']} assistant)\n"
        f"**Thinking blocks**: {stats['thinking_blocks']}\n"
        f"**Tool calls**: {stats['tool_calls']['total']} "
        f"({tool_summary})\n"
        f"**Tokens**: {stats['tokens']['input']:,} input, "
        f"{stats['tokens']['output']:,} output\n\n"
        f"Based on our conversation, please provide the following in "
        f"JSON format:\n\n"
        f'```json\n'
        f'{{\n'
        f'  "title": "Brief descriptive title (5-10 words)",\n'
        f'  "purpose": "What the user was trying to accomplish '
        f'(1-2 sentences)",\n'
        f'  "tags": ["tag1", "tag2", "tag3"],\n'
        f'  "three_ps": {{\n'
        f'    "prompt_summary": "What was asked (Prompt) - '
        f'1-2 sentences",\n'
        f'    "process_summary": "How the tool was used (Process) - '
        f'1-2 sentences",\n'
        f'    "provenance_summary": "Role in research workflow '
        f'(Provenance) - 1 sentence"\n'
        f"  }}\n"
        f"}}\n"
        f"```\n\n"
        f"Please respond with ONLY the JSON block, no other text."
    )


# -------------------------------------------------------------------------
# Relationship resolution
# -------------------------------------------------------------------------


def _find_previous_session(
    archive_base: Path,
    project_name: str,
    current_session_id: str,
) -> str | None:
    """
    Find the most recent archived session in the same project.

    Scans the project's archive subdirectory for session directories
    (sorted lexically by timestamp-prefixed names) and returns the
    session ID of the most recent one that is not the current session.

    Args:
        archive_base: Root of the archive tree (e.g. ``~/cc-archives``).
        project_name: Project subdirectory name.
        current_session_id: ID of the session being archived (to exclude).

    Returns:
        Session ID of the previous session, or *None* if none found.
    """
    project_dir = archive_base / project_name
    if not project_dir.is_dir():
        return None

    # Session directories are timestamp-prefixed, so lexical sort works
    candidates = sorted(
        (d for d in project_dir.iterdir() if d.is_dir()),
        reverse=True,
    )

    for candidate in candidates:
        meta_file = candidate / "session.meta.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            prev_id = meta.get("session", {}).get("id", "")
            if prev_id and prev_id != current_session_id:
                return prev_id
        except (json.JSONDecodeError, OSError):
            continue

    return None


# -------------------------------------------------------------------------
# Code-state capture (provenance audit Gap 2)
# -------------------------------------------------------------------------

def _run_git(args: list[str], cwd: Path) -> tuple[bool, str]:
    """
    Run a git command in ``cwd`` and return ``(ok, stdout)``.

    ``ok`` is *True* only when the binary executed and exited with code
    0. ``stdout`` is the (possibly empty) stripped stdout. The two-tuple
    shape lets callers disambiguate "command succeeded with empty
    output" (e.g. ``git status --porcelain`` on a clean tree) from
    "command failed" — both produce empty strings under a single-return
    contract.

    Provenance capture is best-effort: any failure mode (missing git
    binary, not a git repo, timeout, non-zero exit) yields
    ``(False, "")`` and is never surfaced as an exception. Archiving
    must not break when git is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False, ""
    if result.returncode != 0:
        return False, ""
    return True, result.stdout.strip()


def _load_code_state_sidecar(
    session_id: str | None,
    sidecar_dir: Path | None = None,
) -> str | None:
    """
    Look up the SessionStart ``commit_at_start`` sidecar.

    The PA hook ``hooks/session-start-code-state.py`` writes a JSON
    sidecar at ``<sidecar_dir>/<session_id>.json`` when each session
    begins.  This helper reads it and returns the captured commit hash,
    or *None* if the sidecar is missing, unreadable, or malformed.

    Returns *None* when *session_id* is falsy or no sidecar exists.
    Never raises — sidecar lookup is best-effort, mirroring the
    semantics of :func:`capture_code_state` itself.
    """
    if not session_id:
        return None
    base = sidecar_dir if sidecar_dir is not None else CODE_STATE_SIDECAR_DIR
    sidecar_path = base / f"{session_id}.json"
    if not sidecar_path.is_file():
        return None
    try:
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    commit = data.get("commit_at_start")
    if isinstance(commit, str) and commit:
        return commit
    return None


def capture_code_state(
    project_root: Path | None,
    session_id: str | None = None,
    *,
    sidecar_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Capture the working tree's git state at archive time.

    Returns a dict with three keys (provenance audit Gap 2, 2026-05-17):

    - ``commit_at_start``: HEAD at session start. Populated from the
      sidecar at ``<sidecar_dir>/<session_id>.json`` if both
      *session_id* is given and the SessionStart hook has written one.
      *None* when no sidecar exists.
    - ``commit_at_end``: HEAD at the moment of archive (i.e. now).
      Resolved via ``git rev-parse HEAD`` in *project_root*. *None* if
      *project_root* is missing, not a git repo, or the command fails.
    - ``dirty_at_end``: *True* when ``git status --porcelain`` returns
      non-empty output (i.e. there are uncommitted changes, including
      untracked files, in the working tree). *None* when the dirty
      check could not be run.

    Best-effort: any git failure produces *None* for the affected
    field rather than raising. Provenance capture must never break
    archiving.

    Args:
        project_root: Project root directory; used for the ``end``
            captures. May be *None*.
        session_id: Session identifier; used to look up the sidecar
            for ``commit_at_start``. May be *None*; in that case
            ``commit_at_start`` is always *None*.
        sidecar_dir: Override for the sidecar directory. Defaults to
            :data:`cc_session_toolkit.config.CODE_STATE_SIDECAR_DIR`.
    """
    state: dict[str, Any] = {
        "commit_at_start": _load_code_state_sidecar(
            session_id, sidecar_dir=sidecar_dir
        ),
        "commit_at_end": None,
        "dirty_at_end": None,
    }
    if project_root is None or not project_root.is_dir():
        return state

    ok, commit = _run_git(["rev-parse", "HEAD"], project_root)
    if not ok or not commit:
        # Not a git repo (or git unavailable) — leave end fields None.
        # The dirty check is meaningless without a repo to dirty.
        # ``commit_at_start`` may still be populated from the sidecar,
        # since that capture happens against the cwd at session start
        # and is independent of whether the project root is a git repo
        # at archive time.
        return state
    state["commit_at_end"] = commit

    ok, porcelain = _run_git(["status", "--porcelain"], project_root)
    if ok:
        # Empty porcelain output means a clean tree; any output at all
        # means uncommitted changes (including untracked files).
        state["dirty_at_end"] = bool(porcelain)
    # else: dirty_at_end stays None — we recorded the commit hash but
    # couldn't ascertain dirtiness; better to surface uncertainty than
    # to assume one way.

    return state


# -------------------------------------------------------------------------
# Metadata creation
# -------------------------------------------------------------------------

def create_session_metadata(
    session_id: str,
    session_path: Path,
    stats: dict[str, Any],
    project_root: Path | None = None,
    *,
    auto_generated: dict[str, Any] | None = None,
    thinking_block_stats: dict[str, Any] | None = None,
    tool_output_stats: dict[str, Any] | None = None,
    artifacts: dict[str, list] | None = None,
    relationship_hints: dict[str, Any] | None = None,
    compression_info: dict[str, Any] | None = None,
    defaults: dict[str, Any] | None = None,
    project_name_override: str | None = None,
    capture_type: str | None = None,
    subagents: list[dict[str, Any]] | None = None,
    licence: str | None = None,
    extractor_model_id: str | None = None,
    code_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create the complete ``session.meta.json`` structure (v1.2 schema).

    Args:
        session_id: Session identifier.
        session_path: Path to the source JSONL file.
        stats: Extracted session statistics.
        project_root: Project root directory.  May be *None* in
            hook mode if no project root was detected.
        auto_generated: CC-generated metadata (title, purpose, tags,
            three_ps).
        thinking_block_stats: Thinking-block token statistics.
        tool_output_stats: Tool output byte statistics.
        artifacts: Files created/modified/referenced.
        relationship_hints: Detected relationship hints.
        compression_info: Compression metadata (if using gzip).
        defaults: Loaded defaults from ``archive-defaults.yaml``.
        project_name_override: Explicit project name (overrides
            auto-detection from *project_root*).
        capture_type: How the session was captured — ``"session_end"``,
            ``"pre_compact"``, or *None* for manual archiving.
        subagents: Sub-agent info dictionaries from
            :func:`archive_subagent_transcripts`.  Populates the v1.2
            ``subagents`` top-level field and the
            ``statistics.subagents_summary`` rollup.
        licence: Optional licence string for sharing the record (RO-Crate
            field, provenance audit Gap 3).  Defaults to
            :data:`cc_session_toolkit.config.DEFAULT_LICENCE` (``None``)
            so the user explicitly opts in to a licence at the moment
            the record becomes shareable.
        extractor_model_id: Model identifier responsible for any
            auto-generated metadata in this record (Three Ps summaries,
            title, purpose, tags).  Defaults to
            :data:`cc_session_toolkit.config.EXTRACTOR_MODEL_ID`.
            Surfaced for RO-Crate attribution (provenance audit Gap 3).
        code_state: Pre-computed code-state dict
            ``{commit_at_start, commit_at_end, dirty_at_end}``.  When
            *None*, :func:`capture_code_state` is invoked against
            *project_root* to populate it.  Pass an explicit dict to
            override capture (e.g. when restoring from a sidecar with
            ``commit_at_start`` already known).  Provenance audit Gap 2.

    Returns:
        Complete metadata dictionary.
    """
    defaults = defaults or {}
    thinking_defaults = defaults.get("thinking_blocks", {})
    relationship_defaults = defaults.get("relationships", {})

    # Compute file hash
    sha256_hash = hashlib.sha256()
    with open(session_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha256_hash.update(chunk)

    # Resolve project name — explicit override takes priority
    if project_name_override:
        project_name = project_name_override
    elif project_root:
        project_name = get_project_name(project_root)
    else:
        project_name = "unknown"

    # Thinking blocks section
    thinking_stats = thinking_block_stats or {
        "count": 0,
        "total_tokens": 0,
    }
    thinking_blocks = {
        "included": True,
        "count": thinking_stats.get(
            "count", stats.get("thinking_blocks", 0)
        ),
        "total_tokens": thinking_stats.get("total_tokens", 0),
        "token_count_method": thinking_stats.get(
            "token_count_method", "estimated"
        ),
        "sharing_preference": thinking_defaults.get(
            "sharing_preference", DEFAULT_THINKING_SHARING
        ),
        "use_constraints": thinking_defaults.get(
            "use_constraints", DEFAULT_THINKING_USE_CONSTRAINTS
        ),
        "excluded_uses": thinking_defaults.get(
            "excluded_uses", DEFAULT_THINKING_EXCLUDED_USES
        ),
        "nature_note": DEFAULT_THINKING_NATURE_NOTE,
    }

    # Relationships section
    default_is_part_of = relationship_defaults.get(
        "default_isPartOf", [project_name]
    )
    relationship_hints_info = relationship_hints or {}

    # Auto-populate "continues" when continuation was detected and
    # a previous session ID was resolved.
    continues_id = relationship_hints_info.get(
        "continues_session_id"
    )

    relationships = {
        "continues": continues_id,
        "continuedBy": None,
        "isPartOf": default_is_part_of,
        "isParallelTo": [],
        "supersedes": None,
        "references": [],
        "branchesFrom": None,
    }

    # Statistics section
    tool_outputs = tool_output_stats or {
        "total_bytes": 0,
        "by_type": {},
        "largest_single_output_bytes": 0,
    }
    subagents_list = subagents or []
    statistics = {
        "turns": stats["turns"],
        "human_messages": stats["human_messages"],
        "assistant_messages": stats["assistant_messages"],
        "thinking_blocks": stats["thinking_blocks"],
        "tool_calls": stats["tool_calls"],
        "tokens": stats["tokens"],
        "estimated_cost_usd": estimate_cost(stats),
        "tool_outputs": tool_outputs,
        "subagents_summary": summarise_subagents(subagents_list),
    }

    # Archive section
    archive = {
        "jsonl_path": "session.jsonl",
        "jsonl_sha256": sha256_hash.hexdigest(),
        "jsonl_bytes": session_path.stat().st_size,
        "archived_at": datetime.now().isoformat(),
    }

    if compression_info:
        archive["jsonl_path"] = compression_info.get(
            "path", "session.jsonl.gz"
        )
        archive["jsonl_compression"] = compression_info.get(
            "compression", "gzip"
        )
        archive["jsonl_bytes_compressed"] = compression_info.get(
            "compressed_bytes", 0
        )
        archive["jsonl_bytes_uncompressed"] = compression_info.get(
            "uncompressed_bytes", session_path.stat().st_size
        )
        archive["jsonl_sha256_uncompressed"] = sha256_hash.hexdigest()
        if "compressed_sha256" in compression_info:
            archive["jsonl_sha256"] = compression_info["compressed_sha256"]

    # Provenance audit Gap 2 (2026-05-17): capture working-tree git
    # state at archive time. Best-effort; missing/unreachable git
    # collapses to None values rather than aborting. ``session_id``
    # is passed so ``capture_code_state`` can look up the SessionStart
    # sidecar for ``commit_at_start``.
    if code_state is None:
        code_state = capture_code_state(project_root, session_id=session_id)

    # Provenance audit Gap 3 (2026-05-17): resolve sharing licence and
    # the model that produced auto-generated metadata. Defaults live in
    # ``cc_session_toolkit.config`` so they can be swept in lockstep.
    resolved_licence = licence if licence is not None else DEFAULT_LICENCE
    resolved_extractor = (
        extractor_model_id if extractor_model_id is not None
        else EXTRACTOR_MODEL_ID
    )

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "session": {
            "id": session_id,
            "started_at": stats["started_at"],
            "ended_at": stats["ended_at"],
            "duration_minutes": stats["duration_minutes"],
        },
        "project": {
            "name": project_name,
            "directory": str(project_root or ""),
        },
        "model": {
            "provider": "anthropic",
            "model_id": stats.get("model", "unknown"),
            "access_method": "claude-code-cli",
        },
        "thinking_blocks": thinking_blocks,
        "relationships": relationships,
        "artifacts": artifacts
        or {"created": [], "modified": [], "referenced": []},
        "statistics": statistics,
        "auto_generated": auto_generated
        or {
            "title": "Untitled Session",
            "purpose": "No description provided",
            "tags": [],
        },
        "three_ps": (auto_generated or {}).get(
            "three_ps",
            {
                "prompt_summary": "",
                "process_summary": "",
                "provenance_summary": "",
            },
        ),
        # Provenance audit Gaps 2 + 3 (2026-05-17). ``code_state``
        # anchors the record to the working tree's git HEAD at archive
        # time; ``licence`` is RO-Crate's sharing field (populated by
        # the user when records become shareable); ``extractor_model_id``
        # attributes the Three Ps / auto_generated fields to a specific
        # model version.
        "code_state": code_state,
        "licence": resolved_licence,
        "extractor_model_id": resolved_extractor,
        "archive": archive,
        "subagents": subagents_list,
    }

    if capture_type:
        metadata["archive"]["capture_type"] = capture_type

    if relationship_hints_info.get("detection_notes"):
        metadata["_relationship_hints"] = relationship_hints_info

    return metadata


# -------------------------------------------------------------------------
# Archive a session
# -------------------------------------------------------------------------

def archive_session(
    session_path: Path,
    project_root: Path | None = None,
    *,
    dry_run: bool = False,
    stats_only: bool = False,
    use_gzip: bool = False,
    title: str | None = None,
    archive_root: Path | None = None,
    project_name_override: str | None = None,
    auto_metadata: bool = False,
    capture_type: str | None = None,
    session_id_override: str | None = None,
) -> dict[str, Any] | None:
    """
    Archive a single session with v1.1 schema.

    Supports two modes:

    - **Project mode** (default): Archives to the project's own
      ``archive/cc-sessions/`` directory.  Requires *project_root*.
    - **Global mode** (``archive_root`` set): Archives to a
      centralised directory (e.g. ``~/cc-archives/``).  Used by
      hook-based automated archiving.

    Args:
        session_path: Path to source JSONL file.
        project_root: Project root directory.  May be *None* in
            global mode if no project root was detected.
        dry_run: Preview without writing files.
        stats_only: Skip CC metadata generation (use placeholders).
        use_gzip: Compress the JSONL file.
        title: Optional session title for human-readable directory name.
        archive_root: Global archive root (overrides per-project
            archive directory).
        project_name_override: Explicit project name (overrides
            auto-detection from *project_root*).
        auto_metadata: Call Haiku API for automatic title/purpose/tags.
        capture_type: How the session was captured — ``"session_end"``,
            ``"pre_compact"``, or *None* for manual archiving.
        session_id_override: Explicit session ID (overrides
            filename-based detection).

    Returns:
        Session metadata dictionary, or *None* if skipped.
    """
    session_id = session_id_override or get_session_id(session_path)
    stats = extract_session_stats(session_path)

    # Resolve project name and archive base directory
    if project_name_override:
        project_name = project_name_override
    elif project_root:
        project_name = get_project_name(project_root)
    else:
        project_name = "unknown"

    if archive_root:
        archive_base = archive_root
    elif project_root:
        archive_base = get_archive_dir(project_root)
    else:
        from cc_session_toolkit.config import DEFAULT_ARCHIVE_ROOT
        archive_base = DEFAULT_ARCHIVE_ROOT

    # Load project defaults (if available)
    if project_root:
        defaults_file = get_defaults_file(project_root)
        defaults = load_defaults(defaults_file)
    else:
        defaults = {}

    # Generate auto-metadata early so Haiku title can inform the
    # directory name (human-readable slug instead of short session ID).
    auto_generated = None
    effective_title = title

    if auto_metadata:
        print("  Generating auto-metadata via Haiku...")
        auto_generated = generate_auto_metadata(session_path, stats)
        if auto_generated:
            if title:
                # Explicit title overrides Haiku title
                auto_generated["title"] = title
            elif auto_generated.get("title"):
                # Use Haiku title for directory naming
                effective_title = auto_generated["title"]

    if not auto_generated and not stats_only:
        # Interactive mode: print prompt for CC to fill in
        print("\n" + "=" * 60)
        print("METADATA GENERATION")
        print("=" * 60)
        print(generate_metadata_prompt(session_id, stats))
        print("=" * 60)
        print(
            "\nPlease provide the JSON metadata above, "
            "or press Enter to skip."
        )

    if not auto_generated:
        auto_generated = {
            "title": title or "Untitled Session",
            "purpose": (
                "Auto-metadata unavailable"
                if auto_metadata
                else "Metadata generation requires interactive CC session"
            ),
            "tags": [],
            "three_ps": {
                "prompt_summary": "",
                "process_summary": "",
                "provenance_summary": "",
            },
        }

    dest_dir = get_archive_directory(
        session_id=session_id,
        stats=stats,
        archive_dir=archive_base,
        project_name=project_name,
        title=effective_title,
    )

    print(f"\nSession: {session_id}")
    print(f"  Source: {session_path}")
    print(f"  Archive: {dest_dir}")
    print(
        f"  Duration: {stats['duration_minutes']} min, "
        f"{stats['turns']} turns"
    )
    print(f"  Model: {stats.get('model', 'unknown')}")

    if dry_run:
        print("  [DRY RUN] Would archive this session")
        return None

    # Extract v1.1 statistics
    print("  Extracting v1.1 metadata...")
    thinking_block_stats = extract_thinking_block_tokens(session_path)
    tool_output_stats = extract_tool_output_bytes(session_path)
    if project_root:
        artifacts = extract_artifacts(session_path, project_root)
    else:
        artifacts = {"created": [], "modified": [], "referenced": []}
    relationship_hints = detect_relationship_hints(session_path)

    # Resolve "continues" relationship: if continuation language was
    # detected, look up the most recent session in the same project.
    prev_session_id = _find_previous_session(
        archive_base, project_name, session_id,
    )
    if (
        relationship_hints.get("continues_hint")
        and prev_session_id
    ):
        relationship_hints["continues_session_id"] = prev_session_id
        relationship_hints["detection_notes"].append(
            f"Auto-linked: continues {prev_session_id}"
        )

    print(
        f"    Thinking blocks: {thinking_block_stats['count']} "
        f"(~{thinking_block_stats['total_tokens']:,} tokens estimated)"
    )
    print(
        f"    Tool outputs: {tool_output_stats['total_bytes']:,} bytes total"
    )
    print(
        f"    Artifacts: {len(artifacts['created'])} created, "
        f"{len(artifacts['modified'])} modified, "
        f"{len(artifacts['referenced'])} referenced"
    )

    if relationship_hints.get("detection_notes"):
        print("    Relationship hints detected:")
        for note in relationship_hints["detection_notes"][:3]:
            print(f"      - {note}")

    # Create archive directory
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Copy JSONL file
    compression_info = None
    if use_gzip:
        dest_jsonl = dest_dir / "session.jsonl.gz"
        uncompressed_size = session_path.stat().st_size

        with open(session_path, "rb") as f_in:
            with gzip.open(dest_jsonl, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        compressed_hash = hashlib.sha256()
        with open(dest_jsonl, "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                compressed_hash.update(chunk)

        compression_info = {
            "path": "session.jsonl.gz",
            "compression": "gzip",
            "compressed_bytes": dest_jsonl.stat().st_size,
            "uncompressed_bytes": uncompressed_size,
            "compressed_sha256": compressed_hash.hexdigest(),
        }
        ratio = (
            compression_info["compressed_bytes"] / uncompressed_size
            if uncompressed_size > 0
            else 0.0
        )
        print(
            f"  Compressed to: {dest_jsonl} "
            f"({compression_info['compressed_bytes']:,} bytes, "
            f"{ratio * 100:.1f}% of original)"
        )
    else:
        dest_jsonl = dest_dir / "session.jsonl"
        shutil.copy2(session_path, dest_jsonl)
        print(f"  Copied to: {dest_jsonl}")

    # Archive any sub-agent transcripts alongside the parent (v1.2).
    # Keeps parent + sub-agents co-located as one atomic archive unit.
    # Exceptions here never propagate — the parent archive is more
    # valuable than any single sub-agent rescue — but we route the
    # failure through _log_metadata_event so it lands in
    # auto-metadata.log with enough context to diagnose, rather than
    # vanishing into hook stdout.
    try:
        subagents = archive_subagent_transcripts(
            session_path=session_path,
            session_id=session_id,
            dest_dir=dest_dir,
            use_gzip=use_gzip,
            capture_type=capture_type,
        )
    except Exception as exc:  # noqa: BLE001 — never fail parent archive
        print(f"  [warn] Sub-agent archival failed: {exc}")
        _log_metadata_event(
            (
                f"Sub-agent archival failed for session_id={session_id} "
                f"source={session_path} dest={dest_dir}: "
                f"{type(exc).__name__}: {exc}"
            ),
            level="WARNING",
        )
        subagents = []
    if subagents:
        print(
            f"  Sub-agents: {len(subagents)} archived "
            f"({sum(1 for s in subagents if s['agent_kind'] == 'user_invoked')} "
            f"user-invoked)"
        )

    metadata = create_session_metadata(
        session_id=session_id,
        session_path=session_path,
        stats=stats,
        project_root=project_root,
        auto_generated=auto_generated,
        thinking_block_stats=thinking_block_stats,
        tool_output_stats=tool_output_stats,
        artifacts=artifacts,
        relationship_hints=relationship_hints,
        compression_info=compression_info,
        defaults=defaults,
        project_name_override=project_name_override,
        capture_type=capture_type,
        subagents=subagents,
    )

    metadata_path = dest_dir / "session.meta.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"  Metadata: {metadata_path}")

    # Set transient key AFTER writing to disk so it does not pollute
    # session.meta.json.  Consumed by update_catalogue() via pop().
    metadata["_archive_directory"] = str(dest_dir)

    return metadata


# -------------------------------------------------------------------------
# Sub-agent transcript discovery and archival
# -------------------------------------------------------------------------

# CC emits internal agents with semantic prefixes (e.g. the session
# compactor and the prompt-suggestion agent).  Everything else is treated
# as a user-invoked Agent/Task tool call for linkage purposes.
_CC_INTERNAL_AGENT_PREFIXES: tuple[str, ...] = (
    "acompact-",
    "aprompt_suggestion-",
)

# Tool names in parent JSONL that spawn sub-agents.  CC currently emits
# ``Agent``; older traces and some documentation use ``Task``.  Accept both.
_SUBAGENT_SPAWN_TOOL_NAMES: frozenset[str] = frozenset({"Agent", "Task"})


def _classify_agent_kind(agent_id: str) -> str:
    """
    Classify a sub-agent by its agent-id prefix.

    Returns one of ``"acompact"``, ``"aprompt_suggestion"``,
    ``"user_invoked"``, or ``"unknown"`` (empty id).
    """
    if not agent_id:
        return "unknown"
    for prefix in _CC_INTERNAL_AGENT_PREFIXES:
        if agent_id.startswith(prefix):
            return prefix.rstrip("-")
    return "user_invoked"


def _read_first_record(path: Path) -> dict[str, Any] | None:
    """Read and parse the first non-blank JSON line of a JSONL file."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    return None
    except OSError:
        return None
    return None


def _extract_user_prompt_from_entry(
    entry: dict[str, Any],
    limit: int = 200,
) -> str | None:
    """
    Return the first ``limit`` characters of a user message's content,
    or *None* if the entry is not a user message with recognisable content.

    Kept separate from :func:`_first_user_prompt_preview` so a single-pass
    walker can reuse the extraction logic without re-opening the file.
    """
    if entry.get("type") != "user":
        return None
    message = entry.get("message", {})
    if not isinstance(message, dict):
        return None
    content = message.get("content", "")
    if isinstance(content, str):
        return content[:limit]
    if isinstance(content, list):
        # Prefer the first text block when content is a list
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return str(block.get("text", ""))[:limit]
        return json.dumps(content)[:limit]
    return ""


def _first_user_prompt_preview(path: Path, limit: int = 200) -> str:
    """Return the first ``limit`` characters of the first user message."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                preview = _extract_user_prompt_from_entry(entry, limit)
                if preview is not None:
                    return preview
    except OSError:
        return ""
    return ""


def _file_sha256(path: Path) -> str:
    """Compute the sha256 of a file as it is on disk (bytes read verbatim)."""
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _existing_uncompressed_sha(
    path: Path,
    is_gzip: bool,
) -> str | None:
    """
    Compute the sha256 of an archived sub-agent file's *uncompressed*
    contents, for idempotency comparison against a new source file.

    Returns *None* if the file cannot be read or decompressed.
    """
    try:
        sha = hashlib.sha256()
        opener = gzip.open if is_gzip else open
        with opener(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()
    except (OSError, gzip.BadGzipFile, EOFError):
        return None


def _duration_seconds(
    started: str | None,
    ended: str | None,
) -> float | None:
    """Return the seconds between two ISO timestamps, or *None* on failure."""
    if not started or not ended:
        return None
    try:
        s = datetime.fromisoformat(started.replace("Z", "+00:00"))
        e = datetime.fromisoformat(ended.replace("Z", "+00:00"))
        return round((e - s).total_seconds(), 1)
    except (ValueError, TypeError):
        return None


def _find_subagent_jsonls(
    session_path: Path,
    session_id: str,
) -> list[Path]:
    """
    Discover sub-agent JSONL files belonging to a parent session.

    Handles both the current Claude Code layout and the legacy flat
    layout that was in use before roughly 2026-01-12.

    Current layout
        ``<session_path.parent>/<session_id>/subagents/agent-<agentId>.jsonl``
        (walked recursively in case future CC versions nest sub-agents).

    Legacy layout
        ``<session_path.parent>/agent-<short>.jsonl`` — flat files alongside
        the parent JSONL.  Filtered by ``sessionId`` in their first
        record to ensure they belong to this parent.

    Returns paths ordered by the timestamp of each file's first record
    (so downstream ordinal matching against parent tool-use order is
    stable).
    """
    discovered: list[tuple[str, Path]] = []

    # Current layout — use the parent JSONL's stem as the nesting dir name
    current_dir = session_path.parent / session_path.stem / "subagents"
    if current_dir.is_dir():
        for p in current_dir.rglob("agent-*.jsonl"):
            first = _read_first_record(p)
            ts = (first or {}).get("timestamp") or ""
            discovered.append((ts, p))

    # Legacy layout — flat siblings of the parent JSONL
    for p in session_path.parent.glob("agent-*.jsonl"):
        first = _read_first_record(p)
        if not first:
            continue
        if first.get("sessionId") != session_id:
            continue
        ts = first.get("timestamp") or ""
        discovered.append((ts, p))

    # De-duplicate (defensive; paths from the two sources should not collide)
    seen: set[Path] = set()
    ordered: list[Path] = []
    for _, p in sorted(discovered, key=lambda x: x[0]):
        if p in seen:
            continue
        seen.add(p)
        ordered.append(p)
    return ordered


def _extract_subagent_info(agent_path: Path) -> dict[str, Any] | None:
    """
    Build a per-sub-agent info dictionary from its JSONL file.

    Single JSON-parsing pass captures first record, line count, and the
    first user-message preview.  A second pass through
    :func:`extract_session_stats` (shared with the parent-session code
    path — not duplicated here) extracts counts, tokens, and tool-call
    breakdowns.  A third binary pass computes the uncompressed sha256.
    Three passes rather than five; the remaining two are either shared
    code or necessarily separate (binary vs text).

    ``duration_seconds`` here is deliberately seconds, not minutes:
    sub-agent runs are commonly under 60 s and minute-resolution would
    round most to zero.  Parent-session metadata uses
    ``duration_minutes`` because sessions span hours.  Readers
    interpreting both at once should key off the unit in the field name.

    Returns *None* if the file cannot be parsed or lacks an agent id.
    The transient key ``_internal_path`` is included for downstream
    helpers and is stripped before persisting.
    """
    first: dict[str, Any] | None = None
    records = 0
    first_prompt_preview = ""
    try:
        fh = open(agent_path, "r", encoding="utf-8")
    except OSError:
        return None
    with fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            records += 1
            if first is None:
                first = entry
            if not first_prompt_preview:
                preview = _extract_user_prompt_from_entry(entry)
                if preview is not None:
                    first_prompt_preview = preview

    if not first:
        return None
    agent_id = str(first.get("agentId") or "")
    if not agent_id:
        return None

    stats = extract_session_stats(agent_path)
    uncompressed_bytes = agent_path.stat().st_size
    uncompressed_sha = _file_sha256(agent_path)

    return {
        "agent_id": agent_id,
        "agent_kind": _classify_agent_kind(agent_id),
        "parent_session_id": first.get("sessionId"),
        "parent_session_message_uuid": first.get("parentUuid"),
        "records": records,
        "user_messages": stats.get("human_messages", 0),
        "assistant_messages": stats.get("assistant_messages", 0),
        "tool_calls": stats.get(
            "tool_calls", {"total": 0, "by_type": {}}
        ),
        "tokens": stats.get(
            "tokens",
            {
                "input": 0,
                "output": 0,
                "cache_read": 0,
                "cache_creation": 0,
            },
        ),
        "estimated_cost_usd": estimate_cost(stats),
        "started_at": stats.get("started_at"),
        "ended_at": stats.get("ended_at"),
        "duration_seconds": _duration_seconds(
            stats.get("started_at"), stats.get("ended_at"),
        ),
        "first_prompt_preview": first_prompt_preview,
        "jsonl_sha256_uncompressed": uncompressed_sha,
        "jsonl_bytes_uncompressed": uncompressed_bytes,
        "source_path": str(agent_path),
        "_internal_path": agent_path,
    }


def _collect_parent_tool_uses(parent_path: Path) -> list[dict[str, Any]]:
    """
    Walk the parent JSONL and return Agent/Task ``tool_use`` blocks in order.

    Each entry carries ``id`` (toolu_…), ``name``, and the input fields
    we care about for linkage (``subagent_type``, ``description``,
    ``prompt``).
    """
    results: list[dict[str, Any]] = []
    try:
        fh = open(parent_path, "r", encoding="utf-8")
    except OSError:
        return results
    with fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = entry.get("message", {})
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_use":
                    continue
                if block.get("name") not in _SUBAGENT_SPAWN_TOOL_NAMES:
                    continue
                tool_input = block.get("input") or {}
                if not isinstance(tool_input, dict):
                    tool_input = {}
                results.append({
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "subagent_type": tool_input.get("subagent_type"),
                    "description": tool_input.get("description"),
                    "prompt": tool_input.get("prompt", ""),
                })
    return results


def _match_subagents_to_parent_tool_uses(
    parent_path: Path,
    subagents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Attach parent tool-use linkage to each user-invoked sub-agent.

    Linkage is by ordinal: the k-th Agent/Task ``tool_use`` in the parent
    JSONL is assumed to correspond to the k-th user-invoked sub-agent
    sorted by start time.  A first-prompt-prefix match upgrades the
    linkage from ``"inferred"`` to ``"confirmed"``.

    Sub-agents classified as ``"acompact"`` / ``"aprompt_suggestion"``
    are CC-internal and receive ``"not_applicable"`` linkage.
    """
    parent_calls = _collect_parent_tool_uses(parent_path)
    user_invoked = [s for s in subagents if s["agent_kind"] == "user_invoked"]

    for idx, sub in enumerate(user_invoked):
        if idx >= len(parent_calls):
            sub["parent_tool_use_id"] = None
            sub["parent_tool_use_index"] = None
            sub["subagent_type"] = None
            sub["parent_tool_use_linkage"] = "unknown"
            continue
        call = parent_calls[idx]
        sub["parent_tool_use_id"] = call["id"]
        sub["parent_tool_use_index"] = idx + 1
        sub["subagent_type"] = call.get("subagent_type")

        child_prompt = sub.get("first_prompt_preview") or ""
        parent_prompt_full = call.get("prompt") or ""
        if (
            child_prompt
            and parent_prompt_full
            and parent_prompt_full.startswith(child_prompt)
        ):
            sub["parent_tool_use_linkage"] = "confirmed"
        else:
            sub["parent_tool_use_linkage"] = "inferred"

    for sub in subagents:
        if sub["agent_kind"] != "user_invoked":
            sub.setdefault("parent_tool_use_id", None)
            sub.setdefault("parent_tool_use_index", None)
            sub.setdefault("subagent_type", None)
            sub.setdefault("parent_tool_use_linkage", "not_applicable")

    return subagents


def _resolve_nested_depth(subagents: list[dict[str, Any]]) -> None:
    """
    Populate ``depth``, ``depth_is_exact``, and ``parent_agent_id`` for
    every sub-agent.

    Looks up each sub-agent's ``parent_session_message_uuid`` against the
    set of message uuids emitted by any *other* sub-agent.  If a match is
    found, the sub-agent is nested under another sub-agent; otherwise
    it is a top-level sub-agent of the parent session (depth 0).

    Depth semantics:

    - ``depth = 0`` means top-level (parent message uuid belongs to the
      parent session's main JSONL).  ``depth_is_exact = True`` — we are
      certain.
    - ``depth = 1`` when a parent-agent match is found.  This is a
      *minimum bound*, not an exact depth — a true depth-2 agent (A
      spawns B spawns C) would still report ``depth = 1`` because the
      single-hop lookup cannot distinguish.  ``depth_is_exact = False``
      flags this.  A future CC version that exposes filesystem-nested
      sub-agents (``subagents/.../subagents/...``) would let us compute
      exact depth via directory structure; until then the min-bound is
      honest.

    Readers should use ``depth == 0`` as "definitely top-level" and
    ``depth >= 1 and not depth_is_exact`` as "nested, true depth unknown".
    Absence of ``depth_is_exact`` in pre-2026-04-23 backfilled archives
    should be treated as True — all such entries had ``depth = 0`` and
    depth 0 is always exact.

    Current CC (as of 2026-04-23) produces only flat sub-agents, so this
    function ordinarily assigns ``depth = 0`` to every entry.  It exists
    for forward-compatibility with future CC versions that may nest.
    """
    # Build a message-uuid -> agent-id index across sub-agents
    agent_uuids: dict[str, str] = {}
    for sub in subagents:
        agent_path = sub.get("_internal_path")
        if not isinstance(agent_path, Path) or not agent_path.is_file():
            continue
        with open(agent_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg_uuid = entry.get("uuid")
                if msg_uuid:
                    agent_uuids[msg_uuid] = sub["agent_id"]

    for sub in subagents:
        parent_msg_uuid = sub.get("parent_session_message_uuid")
        if parent_msg_uuid and parent_msg_uuid in agent_uuids:
            sub["depth"] = 1
            sub["depth_is_exact"] = False
            sub["parent_agent_id"] = agent_uuids[parent_msg_uuid]
        else:
            sub["depth"] = 0
            sub["depth_is_exact"] = True
            sub["parent_agent_id"] = None


def archive_subagent_transcripts(
    session_path: Path,
    session_id: str,
    dest_dir: Path,
    *,
    use_gzip: bool = True,
    capture_type: str | None = None,
) -> list[dict[str, Any]]:
    """
    Discover, archive, and describe all sub-agent transcripts for a session.

    Writes copies into ``dest_dir / "subagents" /`` (creating the
    directory as needed) and returns a list of per-sub-agent dicts ready
    for inclusion in ``session.meta.json``.

    Idempotency: if a target file already exists and decompresses to the
    same uncompressed content as the source, the copy is skipped and an
    ``"idempotent: target unchanged"`` note is recorded.  If the source
    has grown (e.g. mid-session ``PreCompact`` snapshot superseded by
    ``SessionEnd``), the target is overwritten and a note is recorded.

    Args:
        session_path: Path to the parent session JSONL.
        session_id: The parent session identifier (used for legacy-layout
            filtering and directory naming).
        dest_dir: Archive directory for this session (where
            ``session.jsonl.gz`` was written).
        use_gzip: If *True* (default), compress each sub-agent transcript.
        capture_type: Propagates to each sub-agent's ``capture_type``
            field (``"session_end"``, ``"pre_compact"``, or *None* for
            manual/backfill use).

    Returns:
        List of sub-agent info dictionaries, each shaped for the v1.2
        ``subagents`` field of ``session.meta.json``.  Empty list when
        no sub-agents are found.
    """
    discovered = _find_subagent_jsonls(session_path, session_id)
    if not discovered:
        return []

    subagent_infos: list[dict[str, Any]] = []
    for p in discovered:
        try:
            info = _extract_subagent_info(p)
        except OSError:
            continue
        if info is None:
            continue
        subagent_infos.append(info)

    if not subagent_infos:
        return []

    # In-flight heuristic: at PreCompact, if the last record is within
    # 60 seconds of "now" the agent may still be writing; mark as
    # in-flight so a later SessionEnd pass can supersede cleanly.
    now = datetime.now().astimezone()
    for sub in subagent_infos:
        sub["capture_notes"] = []
        sub["capture_type"] = capture_type or "manual"
        sub["capture_status"] = "complete"
        if capture_type == "pre_compact" and sub.get("ended_at"):
            try:
                end_dt = datetime.fromisoformat(
                    sub["ended_at"].replace("Z", "+00:00")
                )
                if (now - end_dt).total_seconds() < 60:
                    sub["capture_status"] = "in_flight"
            except (ValueError, TypeError):
                pass

    # Linkage + nesting resolution
    _match_subagents_to_parent_tool_uses(session_path, subagent_infos)
    _resolve_nested_depth(subagent_infos)

    # Copy each sub-agent JSONL into the archive's subagents/ subdirectory
    sub_dir = dest_dir / "subagents"
    sub_dir.mkdir(parents=True, exist_ok=True)

    for sub in subagent_infos:
        src_path: Path = sub.pop("_internal_path")
        agent_id = sub["agent_id"]
        ext = ".jsonl.gz" if use_gzip else ".jsonl"
        target = sub_dir / f"agent-{agent_id}{ext}"

        if target.exists():
            existing_sha = _existing_uncompressed_sha(target, use_gzip)
            if existing_sha == sub["jsonl_sha256_uncompressed"]:
                sub["archive_path"] = f"subagents/{target.name}"
                sub["archive_compression"] = "gzip" if use_gzip else "none"
                sub["jsonl_bytes_compressed"] = (
                    target.stat().st_size if use_gzip else None
                )
                sub["jsonl_sha256"] = _file_sha256(target)
                sub["capture_notes"].append("idempotent: target unchanged")
                continue
            sub["capture_notes"].append(
                "superseded prior capture; source sha differed"
            )

        if use_gzip:
            with open(src_path, "rb") as f_in, gzip.open(target, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            shutil.copy2(src_path, target)

        sub["archive_path"] = f"subagents/{target.name}"
        sub["archive_compression"] = "gzip" if use_gzip else "none"
        sub["jsonl_bytes_compressed"] = (
            target.stat().st_size if use_gzip else None
        )
        sub["jsonl_sha256"] = _file_sha256(target)

    return subagent_infos


def summarise_subagents(
    subagents: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Produce a rollup block for ``statistics.subagents_summary``.

    Aggregates per-sub-agent records, tokens, estimated cost, and counts
    by ``subagent_type`` / ``capture_status`` / ``agent_kind``.  Empty
    sub-agent lists return a zero-valued rollup so downstream consumers
    can key off a known shape.
    """
    total_input = 0
    total_output = 0
    total_cost = 0.0
    total_records = 0
    by_type: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for sub in subagents:
        tokens = sub.get("tokens") or {}
        total_input += tokens.get("input", 0) or 0
        total_output += tokens.get("output", 0) or 0
        total_cost += float(sub.get("estimated_cost_usd") or 0.0)
        total_records += sub.get("records", 0) or 0
        st = sub.get("subagent_type") or "unspecified"
        by_type[st] = by_type.get(st, 0) + 1
        kind = sub.get("agent_kind", "unknown")
        by_kind[kind] = by_kind.get(kind, 0) + 1
        status = sub.get("capture_status", "complete")
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "count": len(subagents),
        "total_records": total_records,
        "total_tokens": {"input": total_input, "output": total_output},
        "estimated_cost_usd": round(total_cost, 2),
        "by_type": by_type,
        "by_kind": by_kind,
        "capture_status_counts": status_counts,
    }


# -------------------------------------------------------------------------
# Archive lookup (from v1.2)
# -------------------------------------------------------------------------

def find_archive_by_id(
    session_id: str,
    archive_dir: Path,
    project_name: str,
) -> Path | None:
    """
    Find an archive directory by session ID (full or partial match).

    Args:
        session_id: Full or partial session ID to search for.
        archive_dir: Base archive directory (``archive/cc-sessions/``).
        project_name: Project name subdirectory.

    Returns:
        Path to the matching archive directory, or *None*.
    """
    project_dir = archive_dir / project_name
    if not project_dir.exists():
        return None

    matches = sorted(
        d for d in project_dir.iterdir()
        if d.is_dir() and session_id in d.name
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Prefer exact match on the session ID portion
        for m in matches:
            if session_id == m.name.split("_", 1)[-1]:
                return m
        # Fall back to first sorted match
        return matches[0]

    return None
