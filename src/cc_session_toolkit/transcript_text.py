"""
Distil a Claude Code session.jsonl(.gz) transcript into a single clean text
string suitable for sending to a Large Language Model (LLM) for whole-session
summarisation.

Ported from ``personal-assistant/scripts/extract-transcript-text.py`` (the
script-form sibling kept in the PA repo for ad-hoc smoke-testing). The
module-form lives here so the cc-session-toolkit archiver can produce a
faithful, framing-free representation of the whole transcript and feed it
to the auto-metadata extractor (Gemini 3 Flash Preview, Flex tier) without
the sampled-message machinery that preceded it.

What is kept
------------
- User text (string content or ``text``-typed content blocks).
- Assistant text blocks (``type == "text"``).
- ``tool_use`` blocks — tool name plus a compact JSON serialisation of the
  inputs (truncated per block).
- ``tool_result`` text content — the actual results the assistant saw,
  truncated per block to bound runaway log/grep output.

What is stripped
----------------
- Framing scaffolds: ``<system-reminder>`` blocks, the
  ``# claudeMd`` / ``# userEmail`` / ``# currentDate`` injected context, and
  PreToolUse / PostToolUse / Stop hook output wrappers that appear in user
  turns.
- ``thinking`` content blocks (private reasoning — not sent to providers).
- Record types that are pure plumbing:
  ``file-history-snapshot``, ``queue-operation``, ``system``,
  ``custom-title``, ``agent-name``, ``last-prompt``.
- Empty / whitespace-only blocks.

Token counting
--------------
Uses the chars-divided-by-four heuristic. Offline by design; downstream
callers should treat the result as an order-of-magnitude figure.

Output
------
A single string. Turns are separated by a blank line and a distinctive
divider marker (``--- User ---``, ``--- Assistant ---``, ``--- Tool use
(Bash) ---``, ``--- Tool result ---``) so a reader (human or LLM) can
follow the conversation without needing the underlying JSON schema. The
divider style is deliberately distinct from ``[user]`` / ``[assistant]``
chat-turn brackets — when a downstream LLM is asked to *summarise* a
transcript, square-bracket role markers can trigger it to *continue* the
conversation as if it were the assistant.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Record-level types that carry no human-meaningful content.
SKIP_RECORD_TYPES = {
    "file-history-snapshot",
    "queue-operation",
    "system",
    "custom-title",
    "agent-name",
    "last-prompt",
}

# Per-block truncation. Tool results and tool-use inputs can blow up
# enormously (gigabyte log dumps, full file reads). We cap them so a single
# pathological block cannot dominate a session's token budget.
TOOL_RESULT_MAX_CHARS = 4000
TOOL_USE_INPUT_MAX_CHARS = 1500

# Patterns identifying framing material that should be stripped from any
# user-role text block before it is emitted.
SYSTEM_REMINDER_RE = re.compile(
    r"<system-reminder>.*?</system-reminder>", re.DOTALL
)
# The injected context block at session start (claudeMd, userEmail,
# currentDate). It is fenced with a leading "# claudeMd" heading and ends
# before the next non-injected user content.
CLAUDE_MD_BLOCK_RE = re.compile(
    r"#\s*claudeMd\b.*?Today's date is \d{4}-\d{2}-\d{2}\.",
    re.DOTALL,
)
# PreToolUse / PostToolUse / Stop hook wrappers sometimes appear as
# ``<hook-output>`` or ``<tool-use-error>``-style XML in user turns.
HOOK_OUTPUT_RE = re.compile(
    r"<(?:hook-output|tool-use-error|local-command-stdout|"
    r"local-command-stderr|command-message|command-name|command-args)>"
    r".*?</(?:hook-output|tool-use-error|local-command-stdout|"
    r"local-command-stderr|command-message|command-name|command-args)>",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _open_transcript(path: Path):
    """Open a transcript regardless of whether it is gzip-compressed.

    Returns a text-mode file object. Caller is responsible for closing it.
    Some legacy archive entries have a ``.gz`` extension but contain
    plain-text JSONL (an early-archiver bug); we detect that by reading
    the gzip magic bytes and fall back to a plain text open if missing.
    """
    treat_as_gzip = (
        path.suffix == ".gz" or path.name.endswith(".jsonl.gz")
    )
    if treat_as_gzip:
        with open(path, "rb") as probe:
            magic = probe.read(2)
        if magic == b"\x1f\x8b":
            return gzip.open(path, "rt", encoding="utf-8")
        # Mis-labelled .gz that is actually plain text.
    return open(path, "rt", encoding="utf-8")


def _iter_records(path: Path) -> Iterable[dict]:
    """Yield parsed JSON records from a transcript, skipping malformed lines."""
    with _open_transcript(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _strip_framing(text: str) -> str:
    """Remove framing wrappers (system-reminders, hook output, claudeMd)."""
    text = SYSTEM_REMINDER_RE.sub("", text)
    text = CLAUDE_MD_BLOCK_RE.sub("", text)
    text = HOOK_OUTPUT_RE.sub("", text)
    return text.strip()


def _truncate(text: str, limit: int) -> str:
    """Truncate ``text`` to ``limit`` characters with a trailing marker."""
    if len(text) <= limit:
        return text
    return text[:limit] + f" …[truncated; total {len(text):,} chars]"


def _compact_json(value: Any, limit: int) -> str:
    """Serialise a tool-use input dict to compact JSON, truncated to limit."""
    try:
        s = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        s = repr(value)
    return _truncate(s, limit)


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------


def _extract_content_blocks(
    blocks: list[dict], role: str
) -> list[str]:
    """Convert a list of content blocks into role-tagged text fragments."""
    out: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")

        if btype == "text":
            raw = block.get("text", "") or ""
            if role == "user":
                raw = _strip_framing(raw)
            raw = raw.strip()
            if raw:
                out.append(f"--- {role.capitalize()} ---\n{raw}")

        elif btype == "tool_use":
            name = block.get("name", "unknown-tool")
            inputs = block.get("input", {}) or {}
            inputs_text = _compact_json(inputs, TOOL_USE_INPUT_MAX_CHARS)
            out.append(f"--- Tool use ({name}) ---\n{inputs_text}")

        elif btype == "tool_result":
            content = block.get("content")
            text = _stringify_tool_result(content)
            if text:
                out.append(
                    f"--- Tool result ---\n"
                    f"{_truncate(text, TOOL_RESULT_MAX_CHARS)}"
                )

        # ``thinking`` blocks and any other types are intentionally dropped.

    return out


def _stringify_tool_result(content: Any) -> str:
    """Flatten a tool_result ``content`` field to a single string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for sub in content:
            if not isinstance(sub, dict):
                continue
            stype = sub.get("type")
            if stype == "text":
                t = (sub.get("text") or "").strip()
                if t:
                    parts.append(t)
            elif stype == "image":
                # Images are not useful for text summarisation; leave a marker.
                parts.append("[image elided]")
        return "\n".join(parts).strip()
    return str(content).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_transcript_text(path: str | Path) -> str:
    """Return a single distilled-text representation of the transcript.

    See module docstring for the inclusion / exclusion contract.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    fragments: list[str] = []
    for rec in _iter_records(p):
        rtype = rec.get("type")
        if rtype in SKIP_RECORD_TYPES:
            continue
        if rtype not in ("user", "assistant"):
            continue

        msg = rec.get("message")
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or rtype
        content = msg.get("content")

        if isinstance(content, str):
            if role == "user":
                content = _strip_framing(content)
            content = content.strip()
            if content:
                fragments.append(f"--- {role.capitalize()} ---\n{content}")
            continue

        if isinstance(content, list):
            fragments.extend(_extract_content_blocks(content, role))
            continue

        # Anything else (None, dict, etc.) is dropped.

    return "\n\n".join(fragments).strip()


def estimate_tokens(text: str) -> int:
    """Estimate token count using the chars-divided-by-four heuristic."""
    return max(0, len(text) // 4)
