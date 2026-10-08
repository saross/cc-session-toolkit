"""
Tests for the OpenAI (GPT-6 Luna) extractor and the provider dispatcher.

Added 2026-10-08, when Luna became the primary auto-metadata extractor with
the Gemini path kept intact as a fallback. These tests pin:

* the request shape (model, ``store: false``, reasoning effort, strict
  structured output) and response handling (incomplete, refused, empty);
* the retry policy (rate limits and server errors retried; other 4xx and
  exhausted quota not);
* the role-scoped key resolution, which never falls back to a generic
  ``OPENAI_API_KEY`` that could bill another OpenAI project;
* that each record names the model that actually answered, including when
  the Gemini fallback wrote it;
* the list-price cost, including the long-prompt rates.

No test reaches the network: ``tests/conftest.py`` blocks
``archive._openai_post`` for every test, and these tests replace it with
fakes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cc_session_toolkit import archive
from cc_session_toolkit.config import (
    GEMINI_EXTRACTOR_MODEL_ID,
    OPENAI_EXTRACTOR_MODEL_ID,
    OPENAI_REASONING_EFFORT,
)

# The real key resolver, saved before the autouse fixture stubs it.
REAL_ENSURE_OPENAI_API_KEY = archive._ensure_openai_api_key

PARENT_RESULT: dict[str, Any] = {
    "title": "Plan the survey grid",
    "purpose": "Lay out a field survey grid.",
    "tags": ["survey", "grid"],
    "three_ps": {
        "prompt_summary": "p",
        "process_summary": "q",
        "provenance_summary": "r",
    },
    "phases": [],
    "decisions": [],
    "key_exchanges": [],
}


def _payload(text: str, *, status: str = "completed", **extra: Any) -> dict[str, Any]:
    """A Responses API body carrying *text* as its only output."""
    body: dict[str, Any] = {
        "status": status,
        "model": OPENAI_EXTRACTOR_MODEL_ID,
        "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 200,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
    body.update(extra)
    return body


class FakePost:
    """Stand-in for ``archive._openai_post`` that records each request.

    Responses requests pop the queued *responses*; token-count requests are
    answered with a chars/4 count (or raise *count_error* when given).
    """

    def __init__(self, *responses: Any, count_error: Exception | None = None) -> None:
        self.responses = list(responses)
        self.bodies: list[dict[str, Any]] = []
        self.count_bodies: list[dict[str, Any]] = []
        self.count_error = count_error

    def __call__(
        self,
        body: dict[str, Any],
        api_key: str,
        timeout: float,
        url: str = archive.OPENAI_RESPONSES_URL,
    ) -> dict[str, Any]:
        if url == archive.OPENAI_INPUT_TOKENS_URL:
            self.count_bodies.append(body)
            if self.count_error is not None:
                raise self.count_error
            return {"object": "response.input_tokens", "input_tokens": len(body["input"]) // 4}
        self.bodies.append(body)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture()
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make retry waits instant."""
    import time

    monkeypatch.setattr(time, "sleep", lambda _seconds: None)


# ---------------------------------------------------------------------------
# Request and response
# ---------------------------------------------------------------------------


class TestCallOpenAIOnce:
    def test_request_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakePost(_payload(json.dumps(PARENT_RESULT)))
        monkeypatch.setattr(archive, "_openai_post", fake)

        text = archive._call_openai_once(
            "user", "system", archive.PARENT_METADATA_SCHEMA, api_key="k"
        )

        assert json.loads(text)["title"] == PARENT_RESULT["title"]
        body = fake.bodies[0]
        assert body["model"] == OPENAI_EXTRACTOR_MODEL_ID
        assert body["store"] is False
        assert body["instructions"] == "system"
        assert body["input"] == "user"
        assert body["reasoning"] == {"effort": OPENAI_REASONING_EFFORT}
        assert "service_tier" not in body  # standard tier
        fmt = body["text"]["format"]
        assert fmt["type"] == "json_schema" and fmt["strict"] is True

    def test_strict_schema_closes_every_object(self) -> None:
        strict = archive._openai_strict_schema(archive.PARENT_METADATA_SCHEMA)

        def objects(node: Any):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    yield node
                for value in node.values():
                    yield from objects(value)
            elif isinstance(node, list):
                for value in node:
                    yield from objects(value)

        found = list(objects(strict))
        assert len(found) == 5  # parent, three_ps, phases/decisions/exchanges items
        for node in found:
            assert node["additionalProperties"] is False
            assert sorted(node["required"]) == sorted(node["properties"])
        assert "additionalProperties" not in archive.PARENT_METADATA_SCHEMA  # copy

    def test_incomplete_response_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(archive, "_openai_post", FakePost(_payload(
            '{"title": "cut', status="incomplete",
            incomplete_details={"reason": "max_output_tokens"},
        )))
        with pytest.raises(RuntimeError, match="no usable text"):
            archive._call_openai_once("u", "s", None, api_key="k")

    def test_refusal_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        body = _payload("")
        body["output"][0]["content"] = [{"type": "refusal", "refusal": "no"}]
        monkeypatch.setattr(archive, "_openai_post", FakePost(body))
        with pytest.raises(RuntimeError, match="refused"):
            archive._call_openai_once("u", "s", None, api_key="k")

    def test_observer_receives_usage_and_cost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[dict[str, Any]] = []
        monkeypatch.setattr(archive, "_EXTRACTOR_CALL_OBSERVER", seen.append)
        monkeypatch.setattr(archive, "_openai_post", FakePost(_payload("{}")))

        archive._call_openai_once("u", "s", None, api_key="k")

        assert seen[0]["provider"] == "openai"
        assert seen[0]["input_tokens_charged"] == 1000
        assert seen[0]["cost_usd"] == pytest.approx(archive.openai_cost_usd(1000, 200))


class TestRetry:
    def test_rate_limit_is_retried(
        self, monkeypatch: pytest.MonkeyPatch, no_sleep: None,
    ) -> None:
        fake = FakePost(
            archive.OpenAIRequestError(429, "rate_limit_exceeded"),
            archive.OpenAIRequestError(503, "overloaded"),
            _payload("{}"),
        )
        monkeypatch.setattr(archive, "_openai_post", fake)
        assert archive._call_openai_with_retry("u", "s", None, api_key="k") == "{}"
        assert len(fake.bodies) == 3

    def test_client_error_is_not_retried(
        self, monkeypatch: pytest.MonkeyPatch, no_sleep: None,
    ) -> None:
        fake = FakePost(archive.OpenAIRequestError(400, "invalid schema"))
        monkeypatch.setattr(archive, "_openai_post", fake)
        with pytest.raises(archive.OpenAIRequestError):
            archive._call_openai_with_retry("u", "s", None, api_key="k")
        assert len(fake.bodies) == 1

    def test_exhausted_quota_is_not_retried(
        self, monkeypatch: pytest.MonkeyPatch, no_sleep: None,
    ) -> None:
        fake = FakePost(archive.OpenAIRequestError(429, '{"code": "insufficient_quota"}'))
        monkeypatch.setattr(archive, "_openai_post", fake)
        with pytest.raises(archive.OpenAIRequestError):
            archive._call_openai_with_retry("u", "s", None, api_key="k")
        assert len(fake.bodies) == 1


# ---------------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------------


class TestKeyResolution:
    @pytest.fixture()
    def clean_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
        """An empty home and no OpenAI variables in the environment."""
        import os

        for name in list(os.environ):
            if name.startswith("OPENAI_"):
                monkeypatch.delenv(name)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        (tmp_path / "personal-assistant").mkdir()
        return tmp_path / "personal-assistant" / ".env"

    def test_key_names_follow_the_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_KEY_SUFFIX", raising=False)
        assert archive._openai_key_names("AMD-tower-ubuntu") == [
            "OPENAI_API_KEY_PA", "OPENAI_API_KEY_PA_AMDT",
        ]
        assert archive._openai_key_names("zbook-ubuntu")[-1] == "OPENAI_API_KEY_PA_ZBOOK"
        assert archive._openai_key_names("elsewhere") == ["OPENAI_API_KEY_PA"]
        monkeypatch.setenv("OPENAI_KEY_SUFFIX", "LAPTOP")
        assert archive._openai_key_names("elsewhere")[-1] == "OPENAI_API_KEY_PA_LAPTOP"

    def test_reads_the_role_key_from_env_file(
        self, clean_env: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENAI_KEY_SUFFIX", "AMDT")
        clean_env.write_text(
            "OPENAI_API_KEY=generic-other-project\n"
            'export OPENAI_API_KEY_PA_AMDT="pa-key"\n',
            encoding="utf-8",
        )
        assert REAL_ENSURE_OPENAI_API_KEY() == "pa-key"

    def test_never_uses_a_generic_key(
        self, clean_env: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENAI_KEY_SUFFIX", "AMDT")
        monkeypatch.setenv("OPENAI_API_KEY", "generic-other-project")
        clean_env.write_text("OPENAI_API_KEY=generic-other-project\n", encoding="utf-8")
        assert REAL_ENSURE_OPENAI_API_KEY() is None


# ---------------------------------------------------------------------------
# Dispatcher and provenance
# ---------------------------------------------------------------------------


@pytest.mark.openai_provider
class TestDispatcher:
    def test_primary_answers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(archive, "_openai_post", FakePost(_payload("{}")))
        raw, model, level = archive._call_extractor("u", "s", None, gemini_client=None)
        assert (raw, model, level) == ("{}", OPENAI_EXTRACTOR_MODEL_ID, OPENAI_REASONING_EFFORT)

    def test_falls_back_to_gemini_and_says_so(
        self, monkeypatch: pytest.MonkeyPatch, no_sleep: None,
    ) -> None:
        monkeypatch.setattr(archive, "_openai_post", FakePost(
            archive.OpenAIRequestError(400, "bad request"),
        ))
        monkeypatch.setattr(
            archive, "_call_gemini_with_retry", lambda *a, **k: '{"from": "gemini"}'
        )
        raw, model, _level = archive._call_extractor(
            "u", "s", None, gemini_client=object()
        )
        assert raw == '{"from": "gemini"}'
        assert model == GEMINI_EXTRACTOR_MODEL_ID

    def test_all_providers_failing_raises_with_both_errors(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(archive, "_openai_post", FakePost(
            archive.OpenAIRequestError(400, "bad request"),
        ))
        with pytest.raises(RuntimeError) as info:
            archive._call_extractor("u", "s", None, gemini_client=None)
        assert "openai:" in str(info.value) and "gemini:" in str(info.value)

    def test_generate_auto_metadata_names_the_model(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        # No Gemini key: the heuristic token counter stands in, and Luna answers.
        monkeypatch.setattr(archive, "_ensure_gemini_api_key", lambda: None)
        monkeypatch.setattr(archive, "_openai_post", FakePost(
            _payload(json.dumps(PARENT_RESULT))
        ))
        session = _write_session(tmp_path / "s.jsonl")

        result = archive.generate_auto_metadata(session, {"session_id": "s"})

        assert result is not None
        assert result["title"] == PARENT_RESULT["title"]
        assert result["_extractor"] == {
            "model_id": OPENAI_EXTRACTOR_MODEL_ID,
            "thinking_level": OPENAI_REASONING_EFFORT,
        }


def _write_session(path: Path) -> Path:
    """A minimal two-turn transcript."""
    path.write_text(
        json.dumps({"type": "user", "message": {"role": "user", "content": "x" * 400}})
        + "\n"
        + json.dumps({"type": "assistant", "message": {
            "role": "assistant", "content": [{"type": "text", "text": "y" * 400}],
        }}) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.openai_provider
class TestTokenCounting:
    """2026-10-08: the transcript is counted by the provider that reads it."""

    def test_luna_primary_sends_google_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        from unittest.mock import patch

        pytest.importorskip("google.genai")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        fake = FakePost(_payload(json.dumps(PARENT_RESULT)))
        monkeypatch.setattr(archive, "_openai_post", fake)
        session = _write_session(tmp_path / "s.jsonl")

        with patch("google.genai.Client") as MockClient:
            gemini = MockClient.return_value
            result = archive.generate_auto_metadata(session, {"session_id": "s"})

        assert result is not None and result["title"] == PARENT_RESULT["title"]
        assert fake.count_bodies, "Luna's counter should size the transcript"
        assert all(b["model"] == OPENAI_EXTRACTOR_MODEL_ID for b in fake.count_bodies)
        gemini.models.count_tokens.assert_not_called()
        gemini.models.generate_content.assert_not_called()

    def test_counter_failure_falls_back_to_the_heuristic(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(archive, "_ensure_gemini_api_key", lambda: None)
        monkeypatch.setattr(archive, "_openai_post", FakePost(
            _payload(json.dumps(PARENT_RESULT)),
            count_error=archive.OpenAIRequestError(500, "count failed"),
        ))
        session = _write_session(tmp_path / "s.jsonl")

        result = archive.generate_auto_metadata(session, {"session_id": "s"})

        assert result is not None and result["title"] == PARENT_RESULT["title"]

    def test_luna_budget_applies(self) -> None:
        count_fn, budget = archive._token_counter_for_primary(None)
        assert count_fn is not None
        assert budget == archive.OPENAI_SESSION_TOKEN_BUDGET


def test_gemini_primary_counts_with_gemini() -> None:
    """Gemini primary (the pinned test default) keeps Gemini's counter."""
    from unittest.mock import MagicMock

    client = MagicMock()
    client.models.count_tokens.return_value.total_tokens = 42
    count_fn, budget = archive._token_counter_for_primary(client)
    assert budget is None
    assert count_fn("text") == 42
    assert client.models.count_tokens.call_args.kwargs["model"] == GEMINI_EXTRACTOR_MODEL_ID


def test_archive_session_records_the_fallback_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A record written by the fallback names the fallback's model."""
    from cc_session_toolkit.archive import archive_session

    def _fake(session_path: Path, stats: dict[str, Any]) -> dict[str, Any]:
        return dict(PARENT_RESULT, _extractor={
            "model_id": GEMINI_EXTRACTOR_MODEL_ID, "thinking_level": "medium",
        })

    monkeypatch.setattr(archive, "generate_auto_metadata", _fake)
    live = tmp_path / "cccccccc-0000-0000-0000-000000000001.jsonl"
    lines = []
    for i in range(6):
        lines.append(json.dumps({
            "timestamp": f"2026-03-15T10:{i:02d}:00+00:00", "sessionId": live.stem,
            "message": {"role": "user", "content": f"Message {i} " + "x" * 200},
        }))
        lines.append(json.dumps({
            "timestamp": f"2026-03-15T10:{i:02d}:30+00:00", "sessionId": live.stem,
            "message": {"role": "assistant", "model": "claude-opus-5-5",
                        "content": [{"type": "text", "text": "y" * 200}],
                        "usage": {"input_tokens": 1, "output_tokens": 1}},
        }))
    live.write_text("\n".join(lines) + "\n", encoding="utf-8")
    root = tmp_path / "cc-archives"
    root.mkdir()

    result = archive_session(
        live, None, stats_only=True, archive_root=root,
        project_name_override="proj", auto_metadata=True,
        capture_type="session_end", session_id_override=live.stem,
    )

    meta = json.loads(
        (Path(result["_archive_directory"]) / "session.meta.json").read_text()
    )
    assert meta["extractor_model_id"] == GEMINI_EXTRACTOR_MODEL_ID
    assert meta["extractor_thinking_level"] == "medium"
    assert "_extractor" not in meta["auto_generated"]


def test_long_prompt_cost_uses_the_higher_rates() -> None:
    short = archive.openai_cost_usd(272_000, 1000)
    long = archive.openai_cost_usd(272_001, 1000)
    assert short == pytest.approx((272_000 * 0.10 + 1000 * 0.50) / 1e6)
    assert long == pytest.approx((272_001 * 0.20 + 1000 * 0.75) / 1e6)
