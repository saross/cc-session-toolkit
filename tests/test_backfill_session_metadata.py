"""Tests for scripts/backfill-session-metadata.py (v1.3 upgrade path)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

# -------------------------------------------------------------------------
# Dynamic import — script filename contains a hyphen.
# -------------------------------------------------------------------------

_BACKFILL_PATH = (
    Path(__file__).parent.parent / "scripts" / "backfill-session-metadata.py"
)


def _load_backfill_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "backfill_session_metadata_under_test", _BACKFILL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def backfill() -> ModuleType:
    return _load_backfill_module()


def _write_meta(
    archive_root: Path,
    rel: str,
    *,
    schema_version: str | None,
    purpose: str,
) -> Path:
    """Helper: write a minimal session.meta.json for finder tests."""
    archive_dir = archive_root / rel
    archive_dir.mkdir(parents=True, exist_ok=True)
    meta_path = archive_dir / "session.meta.json"
    payload: dict = {
        "auto_generated": {
            "title": "Test",
            "purpose": purpose,
            "tags": [],
        },
    }
    if schema_version is not None:
        payload["schema_version"] = schema_version
    meta_path.write_text(json.dumps(payload), encoding="utf-8")
    return meta_path


# -------------------------------------------------------------------------
# find_sessions_needing_v13_upgrade
# -------------------------------------------------------------------------


class TestFindSessionsNeedingV13Upgrade:
    """Partition the archive into upgrade-needed / already-current / empty."""

    def test_picks_up_v12_populated_session(
        self,
        backfill: ModuleType,
        tmp_path: Path,
    ) -> None:
        """A populated v1.2 session is in scope for the upgrade."""
        _write_meta(
            tmp_path, "proj/2026-01-01_a",
            schema_version="1.2", purpose="Build the parser",
        )
        results = backfill.find_sessions_needing_v13_upgrade(tmp_path)
        assert len(results) == 1
        assert results[0].name == "session.meta.json"

    def test_skips_v13_session(
        self,
        backfill: ModuleType,
        tmp_path: Path,
    ) -> None:
        """A session already on v1.3 is NOT in scope."""
        _write_meta(
            tmp_path, "proj/2026-01-01_a",
            schema_version="1.3", purpose="Build the parser",
        )
        assert backfill.find_sessions_needing_v13_upgrade(tmp_path) == []

    def test_skips_empty_metadata_session(
        self,
        backfill: ModuleType,
        tmp_path: Path,
    ) -> None:
        """The 'Auto-metadata unavailable' marker is the default-backfill
        population, NOT the upgrade population."""
        _write_meta(
            tmp_path, "proj/2026-01-01_a",
            schema_version="1.2", purpose="Auto-metadata unavailable",
        )
        assert backfill.find_sessions_needing_v13_upgrade(tmp_path) == []

    def test_skips_session_with_missing_purpose(
        self,
        backfill: ModuleType,
        tmp_path: Path,
    ) -> None:
        """A session with no purpose at all is treated as empty, not upgrade."""
        _write_meta(
            tmp_path, "proj/2026-01-01_a",
            schema_version="1.2", purpose="",
        )
        assert backfill.find_sessions_needing_v13_upgrade(tmp_path) == []

    def test_picks_up_session_with_no_schema_version(
        self,
        backfill: ModuleType,
        tmp_path: Path,
    ) -> None:
        """A pre-Phase-1 session with no schema_version key but populated
        metadata IS in scope — schema_version omitted means 'unknown old'."""
        _write_meta(
            tmp_path, "proj/2026-01-01_a",
            schema_version=None, purpose="Some legacy purpose",
        )
        results = backfill.find_sessions_needing_v13_upgrade(tmp_path)
        assert len(results) == 1

    def test_mixed_archive_partitions_correctly(
        self,
        backfill: ModuleType,
        tmp_path: Path,
    ) -> None:
        """In a mixed archive, only pre-v1.3 populated sessions are returned."""
        # Three populations side-by-side:
        # (a) v1.2 populated — upgrade target
        # (b) v1.3 populated — skip
        # (c) v1.2 empty marker — skip (default backfill instead)
        _write_meta(
            tmp_path, "proj/2026-01-01_a",
            schema_version="1.2", purpose="Real purpose A",
        )
        _write_meta(
            tmp_path, "proj/2026-01-02_b",
            schema_version="1.3", purpose="Real purpose B",
        )
        _write_meta(
            tmp_path, "proj/2026-01-03_c",
            schema_version="1.2", purpose="Auto-metadata unavailable",
        )
        results = backfill.find_sessions_needing_v13_upgrade(tmp_path)
        assert len(results) == 1
        assert "2026-01-01_a" in str(results[0])


# -------------------------------------------------------------------------
# backup_pre_v13_meta
# -------------------------------------------------------------------------


class TestBackupPreV13Meta:
    """Preserve the original meta side-by-side before rebuild."""

    def test_creates_v2_backup_alongside(
        self,
        backfill: ModuleType,
        tmp_path: Path,
    ) -> None:
        """Backup lands at session.meta.v2-backup.json with original content."""
        meta_path = _write_meta(
            tmp_path, "proj/2026-01-01_a",
            schema_version="1.2", purpose="Original purpose",
        )
        original_content = meta_path.read_text()
        backup_path = backfill.backup_pre_v13_meta(meta_path)
        assert backup_path is not None
        assert backup_path.name == "session.meta.v2-backup.json"
        assert backup_path.exists()
        assert backup_path.read_text() == original_content

    def test_does_not_overwrite_existing_backup(
        self,
        backfill: ModuleType,
        tmp_path: Path,
    ) -> None:
        """If a backup already exists, the function leaves it untouched.

        Prevents losing the true original on a second upgrade attempt — if
        a previous run already backed up a v1.2, a second backup would
        overwrite that with a v1.3 (or even the partially-upgraded state).
        """
        meta_path = _write_meta(
            tmp_path, "proj/2026-01-01_a",
            schema_version="1.2", purpose="Original purpose",
        )
        # Pre-existing backup from a prior run.
        prior_backup_content = '{"schema_version": "1.2", "marker": "prior"}'
        prior_backup = meta_path.with_name("session.meta.v2-backup.json")
        prior_backup.write_text(prior_backup_content)
        # Now mutate the canonical meta to simulate a v1.3 state.
        meta_path.write_text('{"schema_version": "1.3"}')
        # Attempt backup again.
        returned = backfill.backup_pre_v13_meta(meta_path)
        assert returned == prior_backup
        # Prior backup untouched.
        assert prior_backup.read_text() == prior_backup_content


# -------------------------------------------------------------------------
# Permanent-skip marker (added 2026-05-28)
# -------------------------------------------------------------------------


class TestMarkSessionPermanentlySkipped:
    """``mark_session_permanently_skipped`` adds the skip flag + reason."""

    def test_writes_flag_and_reason(
        self,
        backfill: ModuleType,
        tmp_path: Path,
    ) -> None:
        """Both fields land in the meta after the helper runs."""
        meta_path = _write_meta(
            tmp_path, "proj/2026-01-01_a",
            schema_version="1.2", purpose="x",
        )
        backfill.mark_session_permanently_skipped(
            meta_path, reason="Test empty session"
        )
        data = json.loads(meta_path.read_text())
        assert data["auto_metadata_skip_permanent"] is True
        assert data["auto_metadata_skip_reason"] == "Test empty session"
        # Original fields preserved.
        assert data["schema_version"] == "1.2"
        assert data["auto_generated"]["purpose"] == "x"


class TestFindersHonourSkipMarker:
    """Both finders skip sessions with auto_metadata_skip_permanent=True."""

    def test_v13_finder_skips_marked_session(
        self,
        backfill: ModuleType,
        tmp_path: Path,
    ) -> None:
        """A v1.2 populated session that would normally be picked up
        is excluded once the skip marker is set."""
        meta_path = _write_meta(
            tmp_path, "proj/2026-01-01_a",
            schema_version="1.2", purpose="Real purpose",
        )
        # Confirm baseline: finder DOES pick it up.
        assert len(backfill.find_sessions_needing_v13_upgrade(tmp_path)) == 1
        # Now mark and re-check.
        backfill.mark_session_permanently_skipped(meta_path, "test")
        assert backfill.find_sessions_needing_v13_upgrade(tmp_path) == []

    def test_default_finder_skips_marked_session(
        self,
        backfill: ModuleType,
        tmp_path: Path,
    ) -> None:
        """The default backfill finder also honours the marker."""
        meta_path = _write_meta(
            tmp_path, "proj/2026-01-01_a",
            schema_version="1.2", purpose="Auto-metadata unavailable",
        )
        assert len(backfill.find_sessions_needing_backfill(tmp_path)) == 1
        backfill.mark_session_permanently_skipped(meta_path, "test")
        assert backfill.find_sessions_needing_backfill(tmp_path) == []


# -------------------------------------------------------------------------
# Cost tracking (added 2026-05-28)
# -------------------------------------------------------------------------


class _MockUsageMetadata:
    """Minimal stand-in for genai's UsageMetadata response attribute."""

    def __init__(
        self,
        prompt_token_count: int | None = 1000,
        candidates_token_count: int | None = 500,
    ) -> None:
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count


class _MockGeminiResponse:
    """Minimal stand-in for client.models.generate_content's return."""

    def __init__(
        self,
        text: str = '{"narrative": "ok"}',
        usage_metadata: _MockUsageMetadata | None = None,
    ) -> None:
        self.text = text
        self.usage_metadata = (
            usage_metadata if usage_metadata is not None
            else _MockUsageMetadata()
        )


class _MockClient:
    """Records the most recent generate_content call for assertion."""

    def __init__(self, response: _MockGeminiResponse) -> None:
        self._response = response
        self.last_kwargs: dict | None = None

        class _Models:
            def __init__(_self) -> None:  # noqa: N805
                _self.parent = self

            def generate_content(_self, **kwargs):  # noqa: N805
                _self.parent.last_kwargs = kwargs
                return _self.parent._response

        self.models = _Models()


class TestInstrumentedCallGeminiOnce:
    """Wrapper records cost + forwards config + respects schema kwarg."""

    @pytest.fixture(autouse=True)
    def _clear_records(self, backfill: ModuleType) -> None:
        """Ensure module-level state is fresh for each test."""
        backfill._CALL_RECORDS.clear()
        backfill._CURRENT_CONTEXT["target"] = "test-target"
        backfill._CURRENT_CONTEXT["phase"] = "parent"

    def test_records_cost_from_usage_metadata(
        self, backfill: ModuleType
    ) -> None:
        """Happy path: cost computed from prompt/candidate token counts."""
        client = _MockClient(_MockGeminiResponse(
            usage_metadata=_MockUsageMetadata(
                prompt_token_count=1_000_000,  # exactly 1 MTok input
                candidates_token_count=100_000,  # 0.1 MTok output
            ),
        ))
        backfill._instrumented_call_gemini_once(
            client, "user msg", "system msg"
        )
        assert len(backfill._CALL_RECORDS) == 1
        rec = backfill._CALL_RECORDS[0]
        # At $0.75/MTok input + $4.50/MTok output:
        # cost = 1.0 * 0.75 + 0.1 * 4.50 = 0.75 + 0.45 = 1.20
        assert rec["cost_usd"] == pytest.approx(1.20, abs=1e-6)
        assert rec["cost_unknown_reason"] is None
        assert rec["input_tokens_charged"] == 1_000_000
        assert rec["output_tokens"] == 100_000
        assert rec["target"] == "test-target"
        assert rec["phase"] == "parent"
        assert rec["had_response_schema"] is False

    def test_missing_usage_metadata_yields_unknown_cost(
        self, backfill: ModuleType
    ) -> None:
        """A response with no usage_metadata records cost=None + reason."""
        response = _MockGeminiResponse()
        response.usage_metadata = None
        backfill._instrumented_call_gemini_once(
            _MockClient(response), "u", "s"
        )
        rec = backfill._CALL_RECORDS[0]
        assert rec["cost_usd"] is None
        assert "usage_metadata missing" in rec["cost_unknown_reason"]
        assert rec["input_tokens_charged"] is None
        assert rec["output_tokens"] is None

    def test_partial_usage_metadata_yields_unknown_cost(
        self, backfill: ModuleType
    ) -> None:
        """usage_metadata present but token counts None — cost=None."""
        backfill._instrumented_call_gemini_once(
            _MockClient(_MockGeminiResponse(
                usage_metadata=_MockUsageMetadata(
                    prompt_token_count=None, candidates_token_count=100,
                ),
            )),
            "u", "s",
        )
        rec = backfill._CALL_RECORDS[0]
        assert rec["cost_usd"] is None
        assert "token counts missing" in rec["cost_unknown_reason"]

    def test_response_schema_forwarded(self, backfill: ModuleType) -> None:
        """When supplied, response_schema lands in the Gemini config."""
        client = _MockClient(_MockGeminiResponse())
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        backfill._instrumented_call_gemini_once(
            client, "u", "s", response_schema=schema,
        )
        cfg = client.last_kwargs["config"]
        assert cfg["response_schema"] == schema
        assert backfill._CALL_RECORDS[0]["had_response_schema"] is True

    def test_schema_absent_by_default(self, backfill: ModuleType) -> None:
        """No response_schema kwarg → no schema in config."""
        client = _MockClient(_MockGeminiResponse())
        backfill._instrumented_call_gemini_once(client, "u", "s")
        cfg = client.last_kwargs["config"]
        assert "response_schema" not in cfg
        assert cfg["response_mime_type"] == "application/json"

    def test_response_text_none_raises_but_records_cost(
        self, backfill: ModuleType
    ) -> None:
        """A None text raises RuntimeError but the call WAS billed —
        record must land BEFORE the raise so the audit trail is honest."""
        response = _MockGeminiResponse(text=None)
        with pytest.raises(RuntimeError):
            backfill._instrumented_call_gemini_once(
                _MockClient(response), "u", "s"
            )
        # The record was appended before the raise.
        assert len(backfill._CALL_RECORDS) == 1


class TestSummariseCostRecords:
    """``_summarise_cost_records`` aggregates per-phase + flags lower-bound."""

    def test_empty_records(self, backfill: ModuleType) -> None:
        s = backfill._summarise_cost_records([])
        assert s["total_calls"] == 0
        assert s["total_cost_usd"] == 0.0
        assert s["total_cost_usd_is_lower_bound"] is False
        assert s["unknown_cost_calls"] == 0
        assert s["by_phase"] == {}

    def test_aggregates_by_phase(self, backfill: ModuleType) -> None:
        records = [
            {"phase": "parent", "cost_usd": 0.10},
            {"phase": "parent", "cost_usd": 0.20},
            {"phase": "subagent", "cost_usd": 0.05},
            {"phase": "subagent", "cost_usd": 0.05},
            {"phase": "subagent", "cost_usd": 0.05},
        ]
        s = backfill._summarise_cost_records(records)
        assert s["total_calls"] == 5
        assert s["total_cost_usd"] == pytest.approx(0.45)
        assert s["by_phase"]["parent"]["calls"] == 2
        assert s["by_phase"]["parent"]["total_cost_usd"] == pytest.approx(0.30)
        assert s["by_phase"]["subagent"]["calls"] == 3
        assert s["by_phase"]["subagent"]["total_cost_usd"] == pytest.approx(0.15)
        assert s["total_cost_usd_is_lower_bound"] is False

    def test_unknown_cost_flags_lower_bound(
        self, backfill: ModuleType
    ) -> None:
        """A single None cost flips total_cost_usd_is_lower_bound to True."""
        records = [
            {"phase": "parent", "cost_usd": 0.10},
            {"phase": "parent", "cost_usd": None},
            {"phase": "subagent", "cost_usd": 0.05},
        ]
        s = backfill._summarise_cost_records(records)
        assert s["total_cost_usd_is_lower_bound"] is True
        assert s["unknown_cost_calls"] == 1
        # Only measured costs contribute to the sum.
        assert s["total_cost_usd"] == pytest.approx(0.15)
        assert s["by_phase"]["parent"]["unknown_cost_calls"] == 1
        assert s["by_phase"]["parent"]["total_cost_usd"] == pytest.approx(0.10)
