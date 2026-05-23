"""
Tests for :mod:`cc_session_toolkit.transcript_text`.

Covers:
- per-block truncation policy (off by default, on when constants set)
- session-level emergency middle-truncation
- distillation framing-strip behaviour
- gzip vs plain-text transparent open
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from cc_session_toolkit import transcript_text as tt
from cc_session_toolkit.transcript_text import (
    SESSION_CHAR_BUDGET,
    SESSION_TOKEN_BUDGET,
    TOKENISER_SECOND_PASS_MARGIN,
    extract_transcript_text,
    extract_transcript_text_for_gemini,
    estimate_tokens,
    _apply_session_budget,
)


# ---------------------------------------------------------------------------
# Helpers for building synthetic transcripts in tests
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records to a plain-text JSONL file."""
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _user_text(text: str) -> dict:
    return {"type": "user", "message": {"role": "user", "content": text}}


def _assistant_blocks(blocks: list[dict]) -> dict:
    return {
        "type": "assistant",
        "message": {"role": "assistant", "content": blocks},
    }


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def _tool_use(name: str, inputs: dict) -> dict:
    return {"type": "tool_use", "name": name, "input": inputs}


def _tool_result(content: str) -> dict:
    return {"type": "tool_result", "content": content}


def _thinking_block(text: str, signature: str = "sig123") -> dict:
    return {"type": "thinking", "thinking": text, "signature": signature}


# ---------------------------------------------------------------------------
# Per-block truncation: off by default
# ---------------------------------------------------------------------------


class TestPerBlockTruncationOff:
    """Per-block truncation should be off by default (TOOL_*_MAX_CHARS = None)."""

    def test_constants_are_none(self) -> None:
        assert tt.TOOL_RESULT_MAX_CHARS is None
        assert tt.TOOL_USE_INPUT_MAX_CHARS is None

    def test_large_tool_result_passes_through_verbatim(
        self, tmp_path: Path
    ) -> None:
        big_payload = "X" * 50_000
        records = [
            _user_text("read a big file"),
            _assistant_blocks(
                [_tool_use("Read", {"file_path": "/x"})]
            ),
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [_tool_result(big_payload)],
                },
            },
        ]
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, records)
        out = extract_transcript_text(jsonl)
        assert big_payload in out
        # No truncation marker present.
        assert "…[truncated" not in out

    def test_large_tool_use_input_passes_through_verbatim(
        self, tmp_path: Path
    ) -> None:
        big_diff = "Z" * 30_000
        records = [
            _user_text("edit a file"),
            _assistant_blocks(
                [
                    _tool_use(
                        "Edit",
                        {
                            "file_path": "/x.py",
                            "old_string": big_diff,
                            "new_string": "new content",
                        },
                    )
                ]
            ),
        ]
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, records)
        out = extract_transcript_text(jsonl)
        assert big_diff in out
        assert "…[truncated" not in out


# ---------------------------------------------------------------------------
# Per-block truncation: still works when a cap is set (for future revert)
# ---------------------------------------------------------------------------


class TestPerBlockTruncationOnWhenConfigured:
    """If a future operator sets a per-block cap, the marker still appears."""

    def test_tool_result_truncated_when_cap_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tt, "TOOL_RESULT_MAX_CHARS", 100)
        big = "Q" * 5_000
        records = [
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [_tool_result(big)],
                },
            },
        ]
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, records)
        out = extract_transcript_text(jsonl)
        assert "…[truncated; total 5,000 chars]" in out
        # Only the first 100 chars survive.
        assert big[:100] in out
        assert big not in out


# ---------------------------------------------------------------------------
# Session-level middle-truncation
# ---------------------------------------------------------------------------


class TestSessionEmergencyCap:
    """Session-level cap triggers middle-truncation when total exceeds budget."""

    def test_under_budget_no_marker(self) -> None:
        fragments = ["a" * 100, "b" * 100, "c" * 100]
        out = _apply_session_budget(fragments)
        assert "[SESSION-LEVEL EMERGENCY CAP REACHED]" not in out
        # All three fragments present, separator preserved.
        assert "a" * 100 in out
        assert "b" * 100 in out
        assert "c" * 100 in out

    def test_empty_fragments_returns_empty_string(self) -> None:
        assert _apply_session_budget([]) == ""

    def test_over_budget_middle_truncated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Tiny budget for the test so we don't have to build megabytes.
        monkeypatch.setattr(tt, "SESSION_CHAR_BUDGET", 1_000)
        # Use unique markers per fragment so we can check what survived.
        fragments = [
            f"FRAG-{i:03d}-" + "x" * 100 for i in range(20)
        ]  # total ~2,100 chars; 20 fragments
        out = _apply_session_budget(fragments)

        assert "[SESSION-LEVEL EMERGENCY CAP REACHED]" in out
        # Head: at least the first fragment.
        assert "FRAG-000-" in out
        # Tail: at least the last fragment.
        assert "FRAG-019-" in out
        # Middle: at least one of frags 5-14 should be absent.
        absent_in_middle = [
            f"FRAG-{i:03d}-" for i in range(5, 15) if f"FRAG-{i:03d}-" not in out
        ]
        assert absent_in_middle, (
            "expected at least one middle fragment to be dropped, "
            f"but all of frags 5-14 present in output of length {len(out)}"
        )

    def test_marker_describes_dropped_region(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tt, "SESSION_CHAR_BUDGET", 1_000)
        fragments = [f"FRAG-{i:03d}-" + "x" * 100 for i in range(20)]
        out = _apply_session_budget(fragments)
        # The marker carries counts that should be plausible.
        import re

        m = re.search(
            r"(\d+) fragments / ~([\d,]+) chars \(~([\d,]+) tokens\) omitted",
            out,
        )
        assert m is not None, f"marker pattern not found in:\n{out[:500]}"
        dropped_frags = int(m.group(1))
        assert 1 <= dropped_frags <= 18  # head and tail both preserve at least one

    def test_output_after_truncation_within_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point of the cap: the output must fit within budget."""
        monkeypatch.setattr(tt, "SESSION_CHAR_BUDGET", 2_000)
        fragments = [f"F{i:02d}-" + "y" * 200 for i in range(30)]
        out = _apply_session_budget(fragments)
        assert len(out) <= 2_000 + 500, (
            f"output is {len(out)} chars, must be near 2,000 budget "
            f"(allowing slack for marker template)"
        )

    def test_production_budget_constant(self) -> None:
        """Sanity-check the shipped budget."""
        # Gemini 3.5 Flash context = 1,000,000 input tokens; 85% is the
        # documented safety margin (room for system prompt + framing).
        assert SESSION_TOKEN_BUDGET == 850_000
        assert SESSION_CHAR_BUDGET == 850_000 * 4


def _set_budget(monkeypatch: pytest.MonkeyPatch, chars: int) -> None:
    """Helper: monkeypatch both budget constants together.

    The marker template uses ``SESSION_TOKEN_BUDGET`` for documentation
    text (``... within the {budget_tokens:,}-token transcript budget``).
    If only ``SESSION_CHAR_BUDGET`` is monkeypatched, the test's tiny
    char-budget combines with the production 850K-token text to make
    the marker overhead exceed the budget — exposing the implementation
    to a regime it never sees in production. Keep them in lock-step.
    """
    monkeypatch.setattr(tt, "SESSION_CHAR_BUDGET", chars)
    monkeypatch.setattr(tt, "SESSION_TOKEN_BUDGET", chars // 4)


# Marker overhead at production-like ratios is ~400 chars. Test budgets
# below ~600 chars cannot fit a verbose marker AND any session content.
# The implementation correctly emits marker-only in that regime (which
# we test separately). For "normal" pathological tests we use budgets
# comfortably above the marker size.


class TestSessionEmergencyCapPathological:
    """Edge cases where natural middle-truncation cannot preserve head + tail."""

    def test_single_fragment_over_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One huge fragment that exceeds the whole budget.

        Should hard-truncate to the budget and emit the tail-truncation
        marker (NOT silently drop the fragment and return only a
        middle-truncation marker, which was the pre-fix bug behaviour).
        """
        _set_budget(monkeypatch, 5_000)
        big = "Y" * 20_000
        out = _apply_session_budget([big])

        # Must contain the pathological marker, not the middle marker.
        assert "[SESSION-LEVEL EMERGENCY CAP REACHED — PATHOLOGICAL]" in out
        # Must contain SOME of the big fragment's content (i.e., not
        # marker-only).
        assert "Y" * 100 in out
        # Output must fit within the budget plus a small marker-layout slack.
        assert len(out) <= 5_000 + 100
        # Original content size reported in marker.
        assert "20,000" in out or "15," in out  # total or dropped_chars

    def test_first_fragment_exceeds_half_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """head_end stays 0 because the first fragment is too big for half-budget.

        Should fall into the pathological branch and produce the tail-
        truncation marker rather than silently dropping all of fragments[0:N].
        """
        _set_budget(monkeypatch, 5_000)
        # First fragment is 4,000 chars (> half-budget ~2,250 after marker
        # overhead). Total = 4,000 + 3*1,500 = 8,500 > budget. Triggers
        # the pathological head_end==0 branch.
        fragments = ["BIG-" + "z" * 4_000, "m" * 1_500, "m" * 1_500, "m" * 1_500]
        out = _apply_session_budget(fragments)

        assert "[SESSION-LEVEL EMERGENCY CAP REACHED — PATHOLOGICAL]" in out
        # Should keep BIG content at the start.
        assert "BIG-" in out
        # Output within budget.
        assert len(out) <= 5_000 + 100

    def test_last_fragment_exceeds_half_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """tail_start stays at len(fragments) because the last fragment is too big.

        Should fall into the pathological branch.
        """
        _set_budget(monkeypatch, 5_000)
        fragments = ["m" * 1_500, "m" * 1_500, "m" * 1_500, "BIG-" + "w" * 4_000]
        out = _apply_session_budget(fragments)

        assert "[SESSION-LEVEL EMERGENCY CAP REACHED — PATHOLOGICAL]" in out
        # Head content preserved.
        assert "m" in out
        # Output within budget.
        assert len(out) <= 5_000 + 100

    def test_head_tail_overlap_two_oversized_fragments(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two fragments where head + tail walks collide.

        With a 5,000-char budget and two ~3,000-char fragments, both walks
        consume their half-budget and tail_start ends up <= head_end.
        Should fall into the pathological branch rather than silently
        returning the unconstrained join (the pre-fix bug).
        """
        _set_budget(monkeypatch, 5_000)
        fragments = ["HEAD-" + "h" * 3_000, "TAIL-" + "t" * 3_000]
        out = _apply_session_budget(fragments)

        # Must be within budget — the pre-fix bug returned the unconstrained
        # ~6,010-char join.
        assert len(out) <= 5_000 + 100
        # Must use the pathological marker.
        assert "[SESSION-LEVEL EMERGENCY CAP REACHED" in out

    def test_pathological_output_always_within_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Parametric sweep — try several budget sizes against an
        adversarial single-giant-fragment input. Output must always fit
        within budget plus a small slack for the marker layout."""
        for budget in (2_000, 5_000, 10_000, 50_000, 100_000):
            _set_budget(monkeypatch, budget)
            fragments = ["X" * (budget * 10)]
            out = _apply_session_budget(fragments)
            assert len(out) <= budget + 100, (
                f"budget={budget}: out is {len(out)} chars"
            )

    def test_tiny_budget_marker_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """At budgets smaller than the marker text, only the marker is emitted.

        This is the regime where ``pathological_keep`` would have gone
        negative pre-fix; the clamp now produces a marker-only output.
        Production never operates here (850K budget vs ~400-char marker).
        """
        _set_budget(monkeypatch, 200)  # smaller than marker text
        out = _apply_session_budget(["X" * 10_000])
        # Marker is present.
        assert "[SESSION-LEVEL EMERGENCY CAP REACHED" in out
        # No actual content slipped through (would indicate negative slicing).
        assert "X" * 50 not in out


# ---------------------------------------------------------------------------
# Framing-strip + distillation behaviour
# ---------------------------------------------------------------------------


class TestFramingStrip:
    """Framing wrappers in user turns must be stripped before output."""

    def test_system_reminder_block_stripped(self, tmp_path: Path) -> None:
        records = [
            _user_text(
                "real content here\n"
                "<system-reminder>fake content that should be stripped</system-reminder>\n"
                "more real content"
            )
        ]
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, records)
        out = extract_transcript_text(jsonl)
        assert "real content here" in out
        assert "more real content" in out
        assert "fake content" not in out

    def test_thinking_blocks_dropped(self, tmp_path: Path) -> None:
        records = [
            _assistant_blocks(
                [
                    _thinking_block("secret reasoning"),
                    _text_block("public assistant reply"),
                ]
            )
        ]
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, records)
        out = extract_transcript_text(jsonl)
        assert "public assistant reply" in out
        assert "secret reasoning" not in out


# ---------------------------------------------------------------------------
# Gzip-vs-plain transparent open
# ---------------------------------------------------------------------------


class TestGzipTransparency:
    """``.jsonl.gz`` files should be opened transparently, including the
    legacy case where the extension is .gz but the contents are plain text."""

    def test_real_gzip(self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(json.dumps(_user_text("hello from gzip")) + "\n")
        out = extract_transcript_text(path)
        assert "hello from gzip" in out

    def test_misnamed_gz_actually_plaintext(self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl.gz"
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(_user_text("not really gzipped")) + "\n")
        out = extract_transcript_text(path)
        assert "not really gzipped" in out


# ---------------------------------------------------------------------------
# Token estimator
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_chars_div_four(self) -> None:
        assert estimate_tokens("") == 0
        assert estimate_tokens("abcd") == 1
        assert estimate_tokens("x" * 400) == 100


# ---------------------------------------------------------------------------
# Tokeniser-calibrated extraction (extract_transcript_text_for_gemini)
# ---------------------------------------------------------------------------


class TestExtractTranscriptTextForGemini:
    """Tests for the tokeniser-calibrated session-budget path.

    Closes the 2026-05-23 calibration finding: the chars-per-token=4
    heuristic under-counts code-heavy / tool-output-heavy sessions, so
    the helper verifies against the real tokeniser (injected via
    ``count_tokens_fn``) and re-truncates with the observed ratio when
    the first-pass output exceeds the real-tokeniser budget.
    """

    def _write_session(self, tmp_path: Path, fragments: list[str]) -> Path:
        """Build a session with the given user-message fragments."""
        path = tmp_path / "session.jsonl"
        _write_jsonl(path, [_user_text(f) for f in fragments])
        return path

    def test_count_tokens_fn_none_matches_first_pass(
        self, tmp_path: Path
    ) -> None:
        """Without a tokeniser, the helper returns the heuristic-only output."""
        session = self._write_session(
            tmp_path, ["first message", "second message", "third message"]
        )
        with_fn = extract_transcript_text_for_gemini(session, count_tokens_fn=None)
        without_helper = extract_transcript_text(session)
        assert with_fn == without_helper

    def test_under_budget_returns_first_pass_unchanged(
        self, tmp_path: Path
    ) -> None:
        """When count_tokens reports under budget, no second pass fires."""
        session = self._write_session(tmp_path, ["only message"])
        calls: list[str] = []

        def _fake_count(text: str) -> int:
            calls.append(text)
            return 10  # well under any reasonable budget

        text = extract_transcript_text_for_gemini(
            session, budget_tokens=1_000, count_tokens_fn=_fake_count
        )
        assert text == extract_transcript_text(session, budget_tokens=1_000)
        assert len(calls) == 1, (
            "Under-budget path should call count_tokens exactly once"
        )

    def test_over_budget_triggers_calibrated_second_pass(
        self, tmp_path: Path
    ) -> None:
        """When count_tokens reports over budget, the helper re-truncates.

        Build a session whose heuristic-only output exceeds a small custom
        budget under the real tokeniser, and verify the second-pass output
        is shorter than the first-pass output.
        """
        # 5 fragments × 400 chars each → ~2000 chars → 500 chars/token=4
        # heuristic tokens. With budget_tokens=400, heuristic char_budget=1600
        # so the helper will middle-truncate to fit. Then count_tokens
        # reports 600 tokens (over the 400 budget) → second pass.
        fragments = ["x" * 400 for _ in range(5)]
        session = self._write_session(tmp_path, fragments)
        calls: list[str] = []

        def _fake_count(text: str) -> int:
            calls.append(text)
            # Report a count above the budget so a second pass fires;
            # second-pass output is shorter so its real count would be
            # smaller — but the helper trusts the first observed ratio.
            return 600

        first_pass = extract_transcript_text(session, budget_tokens=400)
        second_pass = extract_transcript_text_for_gemini(
            session, budget_tokens=400, count_tokens_fn=_fake_count
        )
        assert len(second_pass) < len(first_pass), (
            "Second pass should produce shorter text than first pass when "
            "the real tokeniser reports over-budget"
        )
        assert len(calls) == 1, (
            "Exactly one tokeniser call expected (first-pass verification); "
            "the second pass uses the observed ratio, not a re-measurement"
        )

    def test_observed_ratio_calibration_formula(
        self, tmp_path: Path
    ) -> None:
        """Second-pass char budget = budget_tokens × observed_chars_per_token × margin.

        Verifies the calibration formula by reproducing it from
        externally-observable inputs and asserting the truncation
        respects the resulting char ceiling.
        """
        fragments = ["abcdefgh" * 100 for _ in range(10)]  # ~8000 chars total
        session = self._write_session(tmp_path, fragments)
        budget_tokens = 500

        first_pass = extract_transcript_text(session, budget_tokens=budget_tokens)
        # Report a count that yields an observed ratio of 2.0 chars/token
        # (i.e., the heuristic over-allocated by 2×). The second pass
        # should then cap chars at budget_tokens × 2.0 × margin.
        reported_count = len(first_pass) // 2

        def _fake_count(_text: str) -> int:
            return reported_count

        result = extract_transcript_text_for_gemini(
            session, budget_tokens=budget_tokens, count_tokens_fn=_fake_count
        )

        observed_cpt = len(first_pass) / reported_count
        expected_max_chars = int(
            budget_tokens * observed_cpt * TOKENISER_SECOND_PASS_MARGIN
        )
        # Result may be slightly shorter (truncation aligns to fragment
        # boundaries) but must not exceed the calibrated char budget.
        assert len(result) <= expected_max_chars, (
            f"Second-pass output ({len(result)} chars) exceeded the "
            f"calibrated char budget ({expected_max_chars}); formula drift"
        )

    def test_empty_session_short_circuits_no_tokeniser_call(
        self, tmp_path: Path
    ) -> None:
        """Empty session must not invoke the tokeniser API."""
        session = tmp_path / "empty.jsonl"
        session.write_text("")
        calls: list[str] = []

        def _fake_count(text: str) -> int:
            calls.append(text)
            return 0

        text = extract_transcript_text_for_gemini(
            session, count_tokens_fn=_fake_count
        )
        assert text == ""
        assert calls == [], (
            "Empty session should bypass the tokeniser entirely "
            "(no wasted API call, and Gemini would reject empty input)"
        )

    def test_count_tokens_exception_graceful_degrade(
        self, tmp_path: Path
    ) -> None:
        """Any tokeniser-side exception falls back to the first-pass output.

        Production resilience: a network blip on count_tokens must not
        kill the whole auto-metadata path. We're no worse off than the
        pre-tokeniser-aware code, which never called count_tokens at all.
        """
        session = self._write_session(tmp_path, ["one", "two", "three"])

        def _broken_count(_text: str) -> int:
            raise RuntimeError("simulated network failure")

        text = extract_transcript_text_for_gemini(
            session, count_tokens_fn=_broken_count
        )
        assert text == extract_transcript_text(session)

    def test_count_tokens_non_int_return_graceful_degrade(
        self, tmp_path: Path
    ) -> None:
        """Non-coercible return value (e.g., a stray None) falls back cleanly."""
        session = self._write_session(tmp_path, ["alpha", "beta"])

        def _bad_count(_text: str) -> int:
            return None  # type: ignore[return-value]

        text = extract_transcript_text_for_gemini(
            session, count_tokens_fn=_bad_count
        )
        assert text == extract_transcript_text(session)

    def test_custom_budget_tokens_narrows_truncation(
        self, tmp_path: Path
    ) -> None:
        """A tighter budget_tokens produces a shorter output than the default."""
        fragments = ["x" * 1000 for _ in range(8)]
        session = self._write_session(tmp_path, fragments)
        big = extract_transcript_text_for_gemini(
            session, budget_tokens=2000, count_tokens_fn=None
        )
        small = extract_transcript_text_for_gemini(
            session, budget_tokens=500, count_tokens_fn=None
        )
        assert len(small) < len(big)
