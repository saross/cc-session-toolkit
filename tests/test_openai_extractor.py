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


# ---------------------------------------------------------------------------
# Review follow-ups (2026-10-08)
# ---------------------------------------------------------------------------

# The real request function, saved before the autouse fixture stubs it.
REAL_OPENAI_POST = archive._openai_post


def test_cache_writes_are_billed_at_1_25x() -> None:
    """Luna writes nearly all of an uncached prompt to cache (1.25x input)."""
    cost = archive.openai_cost_usd(100_000, 1000, cache_write_tokens=90_000)
    assert cost == pytest.approx((10_000 * 0.10 + 90_000 * 0.125 + 1000 * 0.50) / 1e6)
    long = archive.openai_cost_usd(300_000, 1000, cached_input_tokens=10_000,
                                   cache_write_tokens=280_000)
    assert long == pytest.approx(
        (10_000 * 0.20 + 280_000 * 0.25 + 10_000 * 0.02 + 1000 * 0.75) / 1e6
    )


def test_observer_prices_cache_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(archive, "_EXTRACTOR_CALL_OBSERVER", seen.append)
    body = _payload("{}")
    body["usage"]["input_tokens_details"]["cache_write_tokens"] = 900
    monkeypatch.setattr(archive, "_openai_post", FakePost(body))
    archive._call_openai_once("u", "s", None, api_key="k")
    assert seen[0]["cache_write_tokens"] == 900
    assert seen[0]["cost_usd"] == pytest.approx(
        archive.openai_cost_usd(1000, 200, 0, 900), abs=1e-6  # rounded to 6 dp
    )


class TestRequestErrors:
    def test_http_error_becomes_openai_request_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import io
        import urllib.error
        import urllib.request

        def _raise(*_a: Any, **_k: Any) -> Any:
            raise urllib.error.HTTPError(
                archive.OPENAI_RESPONSES_URL, 429, "Too Many Requests", {},
                io.BytesIO(b'{"error": {"code": "rate_limit_exceeded"}}'),
            )

        monkeypatch.setattr(urllib.request, "urlopen", _raise)
        with pytest.raises(archive.OpenAIRequestError) as info:
            REAL_OPENAI_POST({"model": "m"}, "sk-test-secret", 5)
        assert info.value.status == 429
        assert "rate_limit_exceeded" in info.value.detail
        assert "sk-test-secret" not in str(info.value)

    def test_invalid_key_characters_never_reach_the_message(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import http.client

        def _no_connect(self: Any) -> None:
            raise AssertionError("must fail before connecting")

        monkeypatch.setattr(http.client.HTTPSConnection, "connect", _no_connect)
        with pytest.raises(RuntimeError) as info:
            REAL_OPENAI_POST({"model": "m"}, "sk-test-secret\n", 5)
        assert "sk-test-secret" not in str(info.value)
        assert info.value.__cause__ is None and info.value.__suppress_context__


class TestRetryLimits:
    def test_persistent_server_error_stops_after_all_attempts(
        self, monkeypatch: pytest.MonkeyPatch, no_sleep: None,
    ) -> None:
        attempts = len(archive.OPENAI_RETRY_WAITS_SECONDS) + 1
        fake = FakePost(*[archive.OpenAIRequestError(503, "busy")] * attempts)
        monkeypatch.setattr(archive, "_openai_post", fake)
        with pytest.raises(RuntimeError, match="after"):
            archive._call_openai_with_retry("u", "s", None, api_key="k")
        assert len(fake.bodies) == attempts

    def test_unreachable_host_fails_fast(
        self, monkeypatch: pytest.MonkeyPatch, no_sleep: None,
    ) -> None:
        import socket
        import urllib.error

        fake = FakePost(urllib.error.URLError(socket.gaierror(-2, "Name or service not known")))
        monkeypatch.setattr(archive, "_openai_post", fake)
        with pytest.raises(urllib.error.URLError):
            archive._call_openai_with_retry("u", "s", None, api_key="k")
        assert len(fake.bodies) == 1

    def test_timeout_is_retried(
        self, monkeypatch: pytest.MonkeyPatch, no_sleep: None,
    ) -> None:
        fake = FakePost(TimeoutError("timed out"), _payload("{}"))
        monkeypatch.setattr(archive, "_openai_post", fake)
        assert archive._call_openai_with_retry("u", "s", None, api_key="k") == "{}"
        assert len(fake.bodies) == 2


class TestKeyOrder:
    @pytest.fixture()
    def env_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
        import os

        for name in list(os.environ):
            if name.startswith("OPENAI_"):
                monkeypatch.delenv(name)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        (tmp_path / "personal-assistant").mkdir()
        path = tmp_path / "personal-assistant" / ".env"
        path.write_text("OPENAI_API_KEY_PA_AMDT=file-key\n", encoding="utf-8")
        return path

    def test_environment_wins_over_the_file(
        self, env_file: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENAI_KEY_SUFFIX", "AMDT")
        monkeypatch.setenv("OPENAI_API_KEY_PA_AMDT", "env-key\r\n")
        assert REAL_ENSURE_OPENAI_API_KEY() == "env-key"

    def test_hostname_selects_the_suffix(
        self, env_file: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import socket

        monkeypatch.setattr(socket, "gethostname", lambda: "AMD-tower-ubuntu")
        assert REAL_ENSURE_OPENAI_API_KEY() == "file-key"

    def test_suffix_is_case_insensitive(
        self, env_file: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENAI_KEY_SUFFIX", "amdt")
        assert REAL_ENSURE_OPENAI_API_KEY() == "file-key"


@pytest.mark.openai_provider
def test_subagent_summaries_send_google_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """With Luna primary the subagent path never calls Gemini either."""
    import gzip
    from unittest.mock import patch

    pytest.importorskip("google.genai")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    dest = tmp_path / "entry"
    (dest / "subagents").mkdir(parents=True)
    with gzip.open(dest / "subagents" / "agent-a1.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "user", "isSidechain": True, "agentId": "a1",
                             "message": {"role": "user", "content": "Go " * 100}}) + "\n")
        fh.write(json.dumps({"type": "assistant", "isSidechain": True, "agentId": "a1",
                             "message": {"role": "assistant",
                                         "content": [{"type": "text", "text": "Done " * 100}]}})
                 + "\n")
    fake = FakePost(_payload(json.dumps({"narrative": "It did the thing."})))
    monkeypatch.setattr(archive, "_openai_post", fake)

    with patch("google.genai.Client") as MockClient:
        gemini = MockClient.return_value
        summaries = archive.generate_subagent_summaries(
            dest_dir=dest,
            subagents=[{"agent_id": "a1", "archive_path": "subagents/agent-a1.jsonl.gz"}],
            parent_session_id="p",
        )

    assert summaries == [{"agent_id": "a1", "narrative": "It did the thing.",
                          "extractor_model_id": OPENAI_EXTRACTOR_MODEL_ID}]
    gemini.models.count_tokens.assert_not_called()
    gemini.models.generate_content.assert_not_called()


def test_carried_subagent_summaries_take_the_prior_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Unlabelled summaries carried into a Luna-written record keep their writer."""
    from tests.test_subagent_archive import _write_parent_jsonl, _write_subagent_jsonl

    sid = "abc12345-1234-5678-9abc-def012345678"
    proj = tmp_path / "-home-shawn-test-project"
    proj.mkdir()
    parent = proj / f"{sid}.jsonl"
    _write_parent_jsonl(parent, sid, [{"tool_use_id": "toolu_01AAAA",
                                       "subagent_type": "Explore", "prompt": "Audit it."}])
    sub = proj / sid / "subagents"
    sub.mkdir(parents=True)
    _write_subagent_jsonl(sub / "agent-a1a1a1a1a1.jsonl", "a1a1a1a1a1", sid,
                          first_prompt="Audit it.", parent_uuid="parent-user-0", extra_turns=2)

    luna = {"model_id": OPENAI_EXTRACTOR_MODEL_ID, "thinking_level": OPENAI_REASONING_EFFORT}
    monkeypatch.setattr(archive, "generate_auto_metadata",
                        lambda *_a, **_k: dict(PARENT_RESULT, _extractor=luna))
    monkeypatch.setattr(archive, "generate_subagent_summaries",
                        lambda **_k: [{"agent_id": "a1a1a1a1a1", "narrative": "old"}])
    root = tmp_path / "cc-archives"
    root.mkdir()
    first = archive.archive_session(parent, None, stats_only=True, archive_root=root,
                                    project_name_override="proj", auto_metadata=True,
                                    capture_type="session_end", session_id_override=sid)
    dest = Path(first["_archive_directory"])
    prior = json.loads((dest / "session.meta.json").read_text())
    prior["extractor_model_id"] = "gemini-3.5-flash"  # the old writer
    prior["subagent_summaries"] = [{"agent_id": "a1a1a1a1a1", "narrative": "old"}]

    archive.archive_session(parent, None, stats_only=True, archive_root=root,
                            project_name_override="proj", auto_metadata=True,
                            capture_type="session_end", session_id_override=sid,
                            existing_dest_dir=dest, prior_metadata=prior,
                            regenerate_metadata=True)

    after = json.loads((dest / "session.meta.json").read_text())
    assert after["extractor_model_id"] == OPENAI_EXTRACTOR_MODEL_ID
    assert after["subagent_summaries"] == [{"agent_id": "a1a1a1a1a1", "narrative": "old",
                                            "extractor_model_id": "gemini-3.5-flash"}]


def test_incomplete_response_logs_its_ends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The ends of a truncated output reach the log, for diagnosis."""
    import os

    partial = '{"title": "T", "key_exchanges": [' + "x" * 5000 + "TAILMARK"
    monkeypatch.setattr(archive, "_openai_post", FakePost(_payload(
        partial, status="incomplete", incomplete_details={"reason": "max_output_tokens"},
    )))
    with pytest.raises(RuntimeError):
        archive._call_openai_once("u", "s", None, api_key="k")
    log = Path(os.environ["CC_SESSION_LOG_DIR"]) / "auto-metadata.log"
    text = log.read_text(encoding="utf-8")
    assert "OpenAI incomplete response" in text and "TAILMARK" in text
