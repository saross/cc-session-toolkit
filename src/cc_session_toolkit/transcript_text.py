"""
Distil a Claude Code session.jsonl(.gz) transcript into a single clean text
string suitable for sending to a Large Language Model (LLM) for whole-session
summarisation.

Ported from ``personal-assistant/scripts/extract-transcript-text.py`` (the
script-form sibling kept in the PA repo for ad-hoc smoke-testing). The
module-form lives here so the cc-session-toolkit archiver can produce a
faithful, framing-free representation of the whole transcript and feed it
to the auto-metadata extractor (Gemini 3.5 Flash, Flex tier) without the
sampled-message machinery that preceded it.

What is kept
------------
- User text (string content or ``text``-typed content blocks).
- Assistant text blocks (``type == "text"``).
- ``tool_use`` blocks — tool name plus a compact JSON serialisation of the
  full inputs. No per-block truncation.
- ``tool_result`` text content — the actual results the assistant saw,
  in full. No per-block truncation.

What is stripped
----------------
- Framing scaffolds: ``<system-reminder>`` blocks, the
  ``# claudeMd`` / ``# userEmail`` / ``# currentDate`` injected context, and
  PreToolUse / PostToolUse / Stop hook output wrappers that appear in user
  turns.
- ``thinking`` content blocks (Claude Code persists only the cryptographic
  ``signature`` field; the ``thinking`` text is empty as of CC v2.1.72 by
  design — see ``docs/open-science/cot-capture-claude-code-investigation-
  2026-05-19.md`` in the personal-assistant repo).
- Record types that are pure plumbing:
  ``file-history-snapshot``, ``queue-operation``, ``system``,
  ``custom-title``, ``agent-name``, ``last-prompt``.
- Empty / whitespace-only blocks.

Session-level emergency cap
---------------------------
A safety bound — ``SESSION_TOKEN_BUDGET = 850_000`` tokens — exists to
ensure the assembled transcript stays comfortably within Gemini 3 Flash
Preview's 1,000,000-token input context. The cap rarely fires (1 in 242
historical sessions on the analysis corpus, 2026-05-19), and when it does
the distillation is **middle-truncated**: the head and tail of the session
are preserved verbatim (the framing and the resolution / handoff), and an
explanatory marker takes the place of the dropped middle fragments. This
exploits the structure of long agentic sessions, which typically have a
repetitive middle (paper-after-paper extractions, file-after-file
refactors) and unique start/end content. See
``data/experiments/transcript-cap-analysis-2026-05-19/findings.md`` in
the personal-assistant repo for the evidence behind this choice.

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
from typing import Any, Callable, Iterable

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

# Per-block truncation is OFF: tool_use inputs and tool_result content
# are passed through in full so downstream summarisers can ground their
# claims in the actual diff / file content / command output. The
# pathological-block worry that motivated per-block caps in the original
# design is handled instead by ``SESSION_TOKEN_BUDGET`` below — an
# absolute session-total ceiling that triggers middle-truncation only on
# rare long-tail outliers. See module docstring for rationale and
# ``data/experiments/transcript-cap-analysis-2026-05-19/findings.md``
# in the personal-assistant repo for the empirical basis.
#
# These sentinels are kept as named constants (rather than removed
# outright) so that callers and tests can still introspect the
# distillation policy and so a future revert to per-block caps is a
# single-line change.
TOOL_RESULT_MAX_CHARS: int | None = None
TOOL_USE_INPUT_MAX_CHARS: int | None = None

# Session-level emergency cap. Triggers middle-truncation when the
# full distilled transcript would exceed this many tokens. Set to 85%
# of Gemini 3.5 Flash's 1,000,000-token input context to leave room
# for the system prompt (~6,400 tokens) plus framing overhead (~1,000
# tokens) and a healthy safety margin. The 2026-05-23 calibration
# finding showed that this 15% absolute margin is consumed by
# chars-per-token undercount on code-heavy sessions —
# ``extract_transcript_text_for_gemini`` adds a real-tokeniser check on
# top so the budget is enforced against actual Gemini tokens, not the
# chars/4 heuristic.
SESSION_TOKEN_BUDGET: int = 850_000
SESSION_CHAR_BUDGET: int = SESSION_TOKEN_BUDGET * 4  # chars-per-token heuristic

# Text inserted in place of dropped fragments when the session-level
# cap fires. The wording is deliberately explicit so a downstream LLM
# treats it as ground-truth metadata about the missing region rather
# than guessing.
MIDDLE_TRUNCATION_MARKER_TEMPLATE = (
    "\n\n--- [SESSION-LEVEL EMERGENCY CAP REACHED] ---\n"
    "{dropped_fragments:,} fragments / ~{dropped_chars:,} chars "
    "(~{dropped_tokens:,} tokens) omitted from the middle of the session "
    "to fit within the {budget_tokens:,}-token transcript budget. The "
    "head and tail of the session are preserved verbatim above and "
    "below this marker.\n"
    "--- [END MARKER] ---\n\n"
)

# Used in the pathological case where natural middle-truncation cannot
# preserve a head/tail structure — typically when a single fragment is
# larger than half the budget (e.g., one ~3.4M-char tool_result), or
# when the largest few fragments force head and tail to overlap. We
# fall back to a hard char-truncation of the leading content with a
# marker spelling out exactly what was lost. Tail content is sacrificed
# rather than the head because the head usually contains the user's
# initial framing (which Gemini needs to summarise the session faithfully).
TAIL_TRUNCATION_MARKER_TEMPLATE = (
    "\n\n--- [SESSION-LEVEL EMERGENCY CAP REACHED — PATHOLOGICAL] ---\n"
    "Could not preserve session head + tail at the {budget_tokens:,}-token "
    "transcript budget (session structure too imbalanced — e.g., one "
    "tool block exceeds half the budget). Output above is the leading "
    "{kept_chars:,} chars (~{kept_tokens:,} tokens) of the session; "
    "~{dropped_chars:,} chars (~{dropped_tokens:,} tokens) follow in the "
    "original transcript but were omitted.\n"
    "--- [END MARKER] ---\n\n"
)

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


def _truncate(text: str, limit: int | None) -> str:
    """Truncate ``text`` to ``limit`` characters with a trailing marker.

    When ``limit`` is ``None``, returns the text untouched. This is the
    default for tool_use / tool_result blocks (see module docstring).
    """
    if limit is None or len(text) <= limit:
        return text
    return text[:limit] + f" …[truncated; total {len(text):,} chars]"


def _compact_json(value: Any, limit: int | None) -> str:
    """Serialise a tool-use input dict to compact JSON, optionally truncated."""
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


def extract_transcript_text(
    path: str | Path,
    budget_tokens: int | None = None,
) -> str:
    """Return a single distilled-text representation of the transcript.

    See module docstring for the inclusion / exclusion contract and the
    session-level emergency-cap policy. ``budget_tokens`` selects the
    session-level cap; when ``None`` (default), the module-level
    ``SESSION_TOKEN_BUDGET`` is used at call time (so test-suite
    monkey-patching of the module constant continues to work). Callers
    that need calibration against a real tokeniser should use
    :func:`extract_transcript_text_for_gemini` instead.
    """
    fragments = _load_fragments(path)
    return _apply_session_budget(fragments, budget_tokens=budget_tokens)


def _load_fragments(path: str | Path) -> list[str]:
    """Walk a transcript and return the list of role-tagged text fragments.

    Carved out of :func:`extract_transcript_text` so callers that want to
    re-apply the session-level budget multiple times (e.g. the
    tokeniser-calibrating helper) can pay the disk read once.
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

    return fragments


def _apply_session_budget(
    fragments: list[str],
    budget_tokens: int | None = None,
    char_budget: int | None = None,
) -> str:
    """
    Join ``fragments`` into the final distilled string, applying the
    session-level emergency cap by middle-truncation if necessary.

    ``budget_tokens`` is the nominal session-level token cap (used in
    marker labels so a downstream LLM sees the actual ceiling). When
    ``None``, the module-level ``SESSION_TOKEN_BUDGET`` is read at call
    time. If ``char_budget`` is ``None``, the module-level
    ``SESSION_CHAR_BUDGET`` is read at call time (so monkey-patching
    either constant continues to work for tests). Callers that have
    measured the true chars-per-token ratio against a real tokeniser may
    pass an explicit ``char_budget`` to override the heuristic.

    The join contract — ``"\\n\\n".join(...)`` — is mirrored here so the
    accumulated char count corresponds exactly to the final output. We
    walk forwards to find the largest prefix that fits in the head
    budget, walk backwards to find the largest suffix that fits in the
    tail budget, and replace the middle fragments with a single marker
    fragment describing the elision. When no truncation is needed the
    function reduces to a single ``"\\n\\n".join(...)`` and ``.strip()``.

    A two-fragment session that already exceeds the budget will not be
    middle-truncated (there is no middle to drop); it is returned in
    full and Gemini's input size becomes the caller's problem. This
    edge case does not occur in practice — a single fragment fitting
    in 425K tokens would already be a pathological session.
    """
    if not fragments:
        return ""

    # Resolve None-defaults with the semantics callers expect:
    #   - Both unset → read module-level globals (so test-suite
    #     monkey-patching of SESSION_TOKEN_BUDGET / SESSION_CHAR_BUDGET
    #     continues to work).
    #   - budget_tokens explicit, char_budget None → derive char_budget
    #     from budget_tokens via the chars-per-token heuristic (×4), so
    #     callers that pass a tighter budget actually get tighter
    #     truncation.
    #   - char_budget explicit (calibrated from a real tokeniser),
    #     budget_tokens None → derive budget_tokens (≈ char_budget // 4)
    #     for the marker labels.
    #   - Both explicit → use both as given.
    if budget_tokens is None and char_budget is None:
        budget_tokens = SESSION_TOKEN_BUDGET
        char_budget = SESSION_CHAR_BUDGET
    elif char_budget is None:
        char_budget = budget_tokens * 4
    elif budget_tokens is None:
        budget_tokens = char_budget // 4

    separator = "\n\n"
    sep_len = len(separator)

    sizes = [len(f) for f in fragments]
    # Total chars if joined verbatim.
    total = sum(sizes) + sep_len * (len(fragments) - 1)
    if total <= char_budget:
        return separator.join(fragments).strip()

    # Need to middle-truncate. Reserve room for the marker.
    marker_overhead = len(MIDDLE_TRUNCATION_MARKER_TEMPLATE.format(
        dropped_fragments=999_999,
        dropped_chars=999_999_999,
        dropped_tokens=999_999_999,
        budget_tokens=budget_tokens,
    ))
    effective_budget = char_budget - marker_overhead
    half_budget = effective_budget // 2

    # Walk forwards: largest prefix [0..head_end) fitting in half_budget.
    running = 0
    head_end = 0
    for i, sz in enumerate(sizes):
        # Cost of adding this fragment: its size plus the separator to
        # the previous one (or to the marker, conservatively).
        added = sz + sep_len
        if running + added > half_budget:
            break
        running += added
        head_end = i + 1

    # Walk backwards: largest suffix [tail_start..end) fitting in half_budget.
    running = 0
    tail_start = len(fragments)
    for i in range(len(fragments) - 1, head_end - 1, -1):
        sz = sizes[i]
        added = sz + sep_len
        if running + added > half_budget:
            break
        running += added
        tail_start = i

    # Pathological cases where natural middle-truncation cannot preserve
    # head + tail structure:
    #   (a) head_end == 0: the very first fragment is itself larger than
    #       half-budget, so no head can fit.
    #   (b) tail_start == len(fragments): the very last fragment is larger
    #       than half-budget, so no tail can fit.
    #   (c) tail_start <= head_end: head and tail walks collided — the
    #       fragments between them are smaller than the slack but the
    #       largest fragments at the ends each consume too much half-budget.
    # In all three, fall back to a hard char-truncation: keep the leading
    # ``effective_budget`` characters of the joined transcript and append
    # the TAIL_TRUNCATION_MARKER_TEMPLATE so Gemini knows it saw a
    # truncated view. Sacrifices the tail (rather than the head) because
    # the user's initial framing is the most important context for an
    # outside-observer summariser.
    if (
        head_end == 0
        or tail_start == len(fragments)
        or tail_start <= head_end
    ):
        # Account for the marker overhead. Use generous placeholder
        # values in the overhead estimate so the actual marker (which
        # carries smaller real numbers) is guaranteed to fit.
        pathological_overhead = len(TAIL_TRUNCATION_MARKER_TEMPLATE.format(
            budget_tokens=budget_tokens,
            kept_chars=999_999_999,
            kept_tokens=999_999_999,
            dropped_chars=999_999_999,
            dropped_tokens=999_999_999,
        ))
        # Clamp to non-negative — in tests with very small budgets the
        # marker text alone can exceed the budget; we still emit the
        # marker (truthful provenance is more important than fitting),
        # but we keep zero content rather than slicing with a negative
        # index (which would silently return content from the tail).
        pathological_keep = max(0, char_budget - pathological_overhead)
        full_text = separator.join(fragments)
        kept = full_text[:pathological_keep]
        dropped_chars = len(full_text) - len(kept)
        marker = TAIL_TRUNCATION_MARKER_TEMPLATE.format(
            budget_tokens=budget_tokens,
            kept_chars=len(kept),
            kept_tokens=len(kept) // 4,
            dropped_chars=dropped_chars,
            dropped_tokens=dropped_chars // 4,
        )
        return (kept + marker).strip()

    dropped = fragments[head_end:tail_start]
    dropped_chars = sum(len(f) for f in dropped) + sep_len * max(
        0, len(dropped) - 1
    )
    marker = MIDDLE_TRUNCATION_MARKER_TEMPLATE.format(
        dropped_fragments=len(dropped),
        dropped_chars=dropped_chars,
        dropped_tokens=dropped_chars // 4,
        budget_tokens=budget_tokens,
    )

    return separator.join(
        list(fragments[:head_end]) + [marker.strip()] + list(fragments[tail_start:])
    ).strip()


def estimate_tokens(text: str) -> int:
    """Estimate token count using the chars-divided-by-four heuristic."""
    return max(0, len(text) // 4)


# Margin applied when re-truncating with an observed chars-per-token ratio.
# Tightens the second-pass char budget by 8% to absorb intra-session
# tokeniser-ratio variation (different content mixes per fragment) and
# keep a small safety buffer below the hard ceiling.
TOKENISER_SECOND_PASS_MARGIN: float = 0.92


def extract_transcript_text_for_gemini(
    path: str | Path,
    budget_tokens: int | None = None,
    count_tokens_fn: Callable[[str], int] | None = None,
) -> str:
    """Distil a transcript with tokeniser-calibrated session-level capping.

    Two-pass strategy:

    1. **First pass** — run :func:`extract_transcript_text` with the
       default chars-per-token heuristic (``budget_tokens * 4``). For the
       vast majority of sessions, this produces text that is already
       under the real-tokeniser budget.

    2. **Verify, then optional second pass** — if ``count_tokens_fn`` is
       provided, call it on the first-pass text. When the real count
       exceeds ``budget_tokens``, re-truncate using the *observed*
       chars-per-token ratio (with a small safety margin) so the second
       pass converges on a result that fits. This handles the code-heavy
       / tool-output-heavy long-tail where the 4-chars-per-token
       heuristic systematically undercounts.

    When ``count_tokens_fn`` is ``None``, the function falls back to the
    first-pass result — preserving offline-test behaviour and providing
    a graceful degrade path when the live tokeniser is unavailable.

    The injected ``count_tokens_fn`` should be a closure that calls the
    Gemini API's ``client.models.count_tokens(model=..., contents=text)``
    and returns ``response.total_tokens``. This call is metered as
    free-tier by Google (no input / output token billing) per the API
    docs, so adding it to every session-end auto-metadata firing is
    cost-neutral.
    """
    if budget_tokens is None:
        budget_tokens = SESSION_TOKEN_BUDGET

    fragments = _load_fragments(path)
    text = _apply_session_budget(fragments, budget_tokens=budget_tokens)

    if not text or count_tokens_fn is None:
        # Empty session or offline / unavailable-tokeniser path: keep the
        # first-pass heuristic result. Empty text would also be rejected
        # by the tokeniser API. Callers using the heuristic-only fallback
        # should treat this as a best-effort output; the heuristic is
        # known to undercount for code-heavy sessions (see workstream F
        # calibration finding, 2026-05-23).
        return text

    # Verify against the real tokeniser. Any failure (network blip,
    # malformed return, type mismatch) falls back to the first-pass
    # output rather than killing the whole metadata-generation path —
    # the heuristic is no worse than the pre-tokeniser-aware code, which
    # is what we had until now.
    try:
        actual_tokens = int(count_tokens_fn(text))
    except (TypeError, ValueError, Exception):  # noqa: BLE001
        return text

    if actual_tokens <= budget_tokens:
        return text

    # First-pass char budget undershot the real tokeniser. Recompute the
    # char budget using the *observed* chars-per-token ratio plus a
    # safety margin (TOKENISER_SECOND_PASS_MARGIN, 8% pull-back), then
    # re-apply the truncation. Operates on the cached fragment list so
    # we do not re-read the transcript from disk.
    observed_chars_per_token = len(text) / max(1, actual_tokens)
    second_pass_char_budget = int(
        budget_tokens * observed_chars_per_token * TOKENISER_SECOND_PASS_MARGIN
    )
    return _apply_session_budget(
        fragments,
        budget_tokens=budget_tokens,
        char_budget=second_pass_char_budget,
    )
