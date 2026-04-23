# `session.meta.json` schema

Reference for the sidecar metadata file written alongside every archived
Claude Code session at `~/cc-archives/<project>/<datetime>_<slug>/session.meta.json`.

The archive directory layout:

```text
~/cc-archives/<project>/<datetime>_<slug>/
├── session.jsonl.gz       # parent session transcript (immutable after write)
├── session.meta.json      # this file
└── subagents/             # v1.2+, one file per sub-agent transcript
    ├── agent-<id>.jsonl.gz
    └── ...
```

## Version history

| Schema | Shipped | Change |
|--------|---------|--------|
| 1.0 | 2026-03-xx | Initial archive + stats + auto-metadata |
| 1.1 | 2026-03-xx | Thinking blocks, artifacts, relationships |
| 1.2 | 2026-04-23 | Sub-agent transcripts, `subagents` list, `statistics.subagents_summary` rollup, `depth_is_exact` |

Unknown top-level fields are ignored by conforming readers. `schema_version`
identifies which shape applies.

## Top-level fields

```json
{
  "schema_version": "1.2",
  "session": { ... },
  "project": { ... },
  "model": { ... },
  "thinking_blocks": { ... },
  "relationships": { ... },
  "artifacts": { ... },
  "statistics": { ... },
  "auto_generated": { ... },
  "three_ps": { ... },
  "archive": { ... },
  "subagents": [ ... ]
}
```

### `session`

```json
{ "id": "uuid", "started_at": "ISO", "ended_at": "ISO", "duration_minutes": int }
```

`duration_minutes` — whole minutes. Parent sessions are typically tens of
minutes to hours; minute resolution is appropriate.

### `project`

```json
{ "name": "string", "directory": "/abs/path" }
```

### `model`

```json
{ "provider": "anthropic", "model_id": "claude-opus-4-7", "access_method": "claude-code-cli" }
```

### `statistics`

```json
{
  "turns": int,
  "human_messages": int,
  "assistant_messages": int,
  "thinking_blocks": int,
  "tool_calls": { "total": int, "by_type": { "Bash": int, "Agent": int, ... } },
  "tokens": { "input": int, "output": int, "cache_read": int, "cache_creation": int },
  "estimated_cost_usd": float,
  "tool_outputs": { "total_bytes": int, "by_type": { ... }, "largest_single_output_bytes": int },
  "subagents_summary": { ... }      // v1.2+
}
```

`tool_calls.by_type.Agent` — count of Agent/Task tool invocations in the
parent JSONL. Current Claude Code emits the name `Agent`; historical traces
may use `Task`. Readers wanting to count sub-agent spawns should accept both
as synonyms.

### `statistics.subagents_summary` (v1.2+)

Aggregate rollup of the `subagents` list, suitable for "find sub-agent-heavy
sessions" queries without walking the list.

```json
{
  "count": int,
  "total_records": int,
  "total_tokens": { "input": int, "output": int },
  "estimated_cost_usd": float,                  // 2dp, summed from per-sub-agent 2dp estimates
  "by_type": { "<subagent_type>": int },        // e.g. "Explore": 3, "Plan": 1
  "by_kind": { "user_invoked": int, "acompact": int, ... },
  "capture_status_counts": { "complete": int, "in_flight": int, "source_missing": int }
}
```

Empty sub-agent list still emits the object with zero values, so consumers
can key off a known shape.

### `archive`

```json
{
  "jsonl_path": "session.jsonl.gz",
  "jsonl_sha256": "hex",                        // over the archived (possibly gzipped) bytes
  "jsonl_sha256_uncompressed": "hex",           // present when gzipped
  "jsonl_bytes": int,
  "jsonl_bytes_compressed": int,                // present when gzipped
  "jsonl_bytes_uncompressed": int,              // present when gzipped
  "jsonl_compression": "gzip",                  // present when gzipped
  "archived_at": "ISO",
  "capture_type": "session_end" | "pre_compact" | null
}
```

### `subagents` (v1.2+)

List of one object per sub-agent transcript archived alongside the parent.
Empty list when the session spawned no sub-agents or all of them failed to
discover.

```json
{
  "agent_id": "a1b2c3d4",                       // short hex id CC uses in filenames
  "agent_kind": "user_invoked",                 // or "acompact" | "aprompt_suggestion" | "unknown"
  "parent_session_id": "uuid",                  // matches session.id
  "parent_session_message_uuid": "uuid|null",   // which parent message spawned this agent

  "depth": 0,                                   // 0 = top-level; >= 1 = nested
  "depth_is_exact": true,                       // see "Depth semantics" below
  "parent_agent_id": null,                      // or the agent_id of the spawning sub-agent

  "parent_tool_use_id": "toolu_...",            // parent JSONL tool_use block id
  "parent_tool_use_index": 1,                   // 1-based ordinal among Agent tool_uses
  "parent_tool_use_linkage": "confirmed",       // "confirmed" | "inferred" | "unknown" | "not_applicable"
  "subagent_type": "Explore",                   // from parent tool_use input; null when unknown

  "archive_path": "subagents/agent-a1b2c3d4.jsonl.gz",   // relative to the archive dir
  "archive_compression": "gzip" | "none",
  "source_path": "/home/.../agent-a1b2c3d4.jsonl",      // pre-archive location
  "jsonl_sha256": "hex",                        // over archived bytes (may be gzipped)
  "jsonl_sha256_uncompressed": "hex",
  "jsonl_bytes_compressed": int | null,
  "jsonl_bytes_uncompressed": int,

  "records": int,                               // non-blank JSONL lines
  "user_messages": int,
  "assistant_messages": int,
  "tool_calls": { "total": int, "by_type": { ... } },
  "tokens": { "input": int, "output": int, "cache_read": int, "cache_creation": int },
  "estimated_cost_usd": 0.05,                   // 2dp

  "started_at": "ISO",
  "ended_at": "ISO",
  "duration_seconds": 44.7,                     // sub-agents are commonly <60s
  "first_prompt_preview": "first 200 chars of the prompt the parent invoked with",

  "capture_type": "session_end",                // or "pre_compact" | "backfill" | "manual"
  "capture_status": "complete",                 // or "in_flight" | "source_missing"
  "capture_notes": [                            // human-readable provenance trail
    "idempotent: target unchanged"
  ]
}
```

#### Depth semantics

- `depth == 0` means top-level (the parent message uuid belongs to the
  parent session's main JSONL). `depth_is_exact = true` — authoritative.
- `depth == 1` when a parent-agent match is found (`parent_agent_id != null`).
  This is a **minimum bound**, not an exact depth — a true depth-2 agent
  (A spawns B spawns C) would still report `depth = 1` because the
  single-hop lookup cannot distinguish. `depth_is_exact = false` flags this.

Readers interpreting `depth > 0` should honour `depth_is_exact`. A future
Claude Code version that exposes filesystem-nested sub-agents
(`subagents/.../subagents/...`) would let the archiver compute exact depth
by directory structure; until then the min-bound is honest.

Pre-2026-04-23 backfilled archives omit `depth_is_exact`. Treat absence as
`true` — all such entries have `depth = 0` and depth 0 is always exact.

#### `parent_tool_use_linkage` values

- `"confirmed"` — ordinal match AND the parent's tool_use `prompt` field
  starts with the child's `first_prompt_preview`. Highest confidence.
- `"inferred"` — ordinal match but prompt prefix did not verify (may still
  be correct; first prompt may have been truncated or transformed).
- `"unknown"` — more sub-agents than parent Agent tool_uses (e.g. a
  streamed-out sub-agent with no corresponding parent record).
- `"not_applicable"` — `agent_kind != "user_invoked"`; Claude Code internal
  agents (compact, prompt-suggestion) don't have a spawning tool_use in the
  parent transcript.

#### Ordering

Entries are sorted by the first record's `timestamp` (start time).
Parent-side ordinal matching consumes the parent's Agent/Task `tool_use`
blocks in the order they appear in the parent JSONL. For flat,
sequentially-launched sub-agents this produces the intuitive pairing.
Concurrent launches that resolve out-of-order may link incorrectly;
`parent_tool_use_linkage` will typically report `"inferred"` in that case.

## Units and conventions

- Timestamps: ISO 8601, timezone-aware. Backfilled entries use the source
  file's recorded timestamps; archival timestamps are wall-clock at write time.
- Durations: **parent session uses `duration_minutes`; sub-agent uses
  `duration_seconds`.** Deliberate — sub-agent runs are commonly under
  60 s and minute resolution would round most to zero. Readers comparing
  both fields should key off the unit in the field name.
- Cost: USD, 2 decimal places, using per-model Claude pricing current at
  archive time. See `src/cc_session_toolkit/extraction.py::estimate_cost`.
- Hashes: SHA-256, lowercase hex.
- Byte counts: raw on-disk bytes (after gzip when gzipped).

## Backwards compatibility

- Adding top-level fields is allowed without a schema bump when the field
  is optional and readers ignore unknowns. Older archives without the field
  remain valid.
- Changing the meaning or type of an existing field requires a schema bump
  and — if feasible — a migration script under `scripts/`.
- Renaming a field requires a schema bump and the old name should be
  accepted as a synonym by readers for at least one schema version.

## See also

- `saross/cc-session-toolkit` issue #1 — original sub-agent archiving
  motivation and layout correction.
- `saross/personal-assistant` issue #54 — 1-month review (2026-05-23) of
  whether the `sessions` postgres table warrants a child `session_subagents`
  table (option C).
- `scripts/backfill-subagents.py` — one-shot back-archival of existing
  archives to v1.2.
