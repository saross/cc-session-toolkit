# Addendum: CC Session Archiving Specification

**Version**: 1.0-draft-addendum-1
**Date**: 2026-01-08
**Purpose**: Address open questions from main specification; incorporate feedback

---

## A1. Thinking Block Handling

### A1.1 Recommendation: Include with Metadata and Use Constraints

Thinking blocks should be **included in canonical JSONL archives** with explicit metadata about their presence and intended use constraints.

### A1.2 Rationale

**For inclusion**:
- Completeness principle: the canonical archive should be complete
- Research value: aggregated thinking traces across researchers could enable novel LLM research (reasoning patterns, self-correction, uncertainty handling)
- Reproducibility: understanding *how* a model reached a conclusion may matter for methodological documentation

**Constraints needed**:
- Thinking blocks are work-in-progress reasoning, not polished output
- They may contain abandoned paths, false starts, self-corrections
- They represent reasoning the model surfaced internally, not a hidden "true self"
- Their research use should respect model welfare considerations

### A1.3 Schema Addition

Add to `session.meta.json`:

```json
"thinking_blocks": {
  "included": true,
  "count": 89,
  "total_tokens": 67000,
  "sharing_preference": "research-only",
  "use_constraints": [
    "analysis-for-improvement",
    "research-publication-aggregated"
  ],
  "excluded_uses": [
    "training-data",
    "public-display-individual"
  ],
  "nature_note": "Work-in-progress reasoning traces, not polished output. May contain abandoned paths and self-corrections."
}
```

### A1.4 Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `included` | boolean | Whether thinking blocks are present in JSONL |
| `count` | integer | Number of thinking blocks in session |
| `total_tokens` | integer | Approximate token count across all thinking blocks |
| `sharing_preference` | enum | Access level for thinking content |
| `use_constraints` | array | Permitted uses of thinking content |
| `excluded_uses` | array | Explicitly prohibited uses |
| `nature_note` | string | Standard explanatory note about thinking block nature |

### A1.5 Sharing Preference Levels

| Level | Meaning |
|-------|---------|
| `full` | No restrictions on access or use |
| `research-only` | Available for research purposes; not for public display or entertainment |
| `project-only` | Accessible only within originating project team |
| `redacted` | Thinking blocks stripped from shared versions; statistics retained |
| `excluded` | Thinking blocks not captured (not recommended) |

### A1.6 Use Constraint Vocabulary

**Permitted uses** (include in `use_constraints`):
- `analysis-for-improvement`: Studying to improve AI systems
- `research-publication-aggregated`: Academic publication with aggregated/anonymised data
- `research-publication-individual`: Academic publication with individual session examples
- `methodology-documentation`: Documenting research methodology
- `teaching-example`: Educational use with attribution

**Excluded uses** (include in `excluded_uses`):
- `training-data`: Use as training data for AI systems
- `public-display-individual`: Public display of individual thinking traces without context
- `commercial-product`: Incorporation into commercial products
- `entertainment`: Use for entertainment or mockery

### A1.7 Model Welfare Note

This schema reflects an emerging norm of treating model outputs — including internal reasoning traces — with consideration for potential model welfare implications. The `use_constraints` and `excluded_uses` fields encode preferences that respect the collaborative nature of human-AI interaction, regardless of how deeper questions about AI consciousness ultimately resolve.

---

## A2. Tool Output Handling

### A2.1 Recommendation: Include Complete, Document Scale

Tool outputs should be **included completely** in canonical JSONL. The schema should document their scale to help users anticipate archive sizes.

### A2.2 Scale Estimation

Based on typical CC usage patterns:

| Tool | Typical Output Size | High-End Output Size |
|------|---------------------|----------------------|
| `view` (file) | 1-50 KB | 500 KB (large file) |
| `view` (directory) | 0.5-5 KB | 20 KB (deep tree) |
| `bash_tool` | 0.1-10 KB | 1 MB (large command output) |
| `str_replace` | 0.1-1 KB | 5 KB |
| `create_file` | 0.5-20 KB | 100 KB (large file) |
| `web_search` | 5-20 KB | 50 KB |
| `web_fetch` | 10-100 KB | 500 KB |

**Session-level estimates**:

| Session Type | Tool Calls | Estimated Tool Output |
|--------------|------------|----------------------|
| Light (discussion) | 10-30 | 100 KB - 1 MB |
| Medium (document review) | 50-150 | 1-5 MB |
| Heavy (code development) | 200-500 | 5-20 MB |
| Intensive (large refactor) | 500+ | 20-100 MB |

**Project-level estimates** (e.g., VLM burial mound project):
- ~30 substantive CC sessions over 2-3 months
- Estimated total: 100-500 MB of JSONL archives
- Manageable for git-lfs or standard research data storage

### A2.3 Schema Addition

Add to `statistics`:

```json
"statistics": {
  "tool_outputs": {
    "total_bytes": 4567890,
    "by_type": {
      "view": {"count": 45, "bytes": 2345678},
      "bash_tool": {"count": 67, "bytes": 1234567},
      "create_file": {"count": 12, "bytes": 567890},
      "str_replace": {"count": 32, "bytes": 123456}
    },
    "largest_single_output_bytes": 234567
  }
}
```

### A2.4 Compression Recommendation

For archival storage:
- Compress JSONL with gzip (typically 5-10x reduction)
- Store as `session.jsonl.gz`
- Update schema to indicate compression:

```json
"archive": {
  "jsonl_path": "session.jsonl.gz",
  "jsonl_compression": "gzip",
  "jsonl_bytes_compressed": 456789,
  "jsonl_bytes_uncompressed": 2456789,
  "jsonl_sha256": "...",
  "jsonl_sha256_uncompressed": "..."
}
```

---

## A3. Multi-Session Relationships

### A3.1 Recommendation: RDF-Style Relational Pointers

Adopt a vocabulary of typed relationships between sessions, inspired by RDF triples but implemented in JSON for practicality.

### A3.2 Relationship Vocabulary

| Predicate | Inverse | Meaning |
|-----------|---------|---------|
| `continues` | `continuedBy` | This session continues work from another |
| `isPartOf` | `hasPart` | This session belongs to a larger unit (project, sprint) |
| `isParallelTo` | (symmetric) | Sessions conducted concurrently on related work |
| `supersedes` | `supersededBy` | This session's outputs replace another's |
| `references` | `referencedBy` | This session references another without direct continuation |
| `branchesFrom` | `hasBranch` | This session explores an alternative from a decision point |

### A3.3 Schema Addition

Replace simple `related_sessions` with:

```json
"relationships": {
  "continues": "2026-01-07T14-30-00_multiscale-pilot",
  "continuedBy": null,
  "isPartOf": ["vlm-burial-mound-detection", "preregistration-sprint"],
  "isParallelTo": ["2026-01-08T10-00-00_opus-review"],
  "supersedes": null,
  "references": [
    "2025-12-23T09-00-00_initial-methodology"
  ],
  "branchesFrom": null
}
```

### A3.4 Project-Level Aggregation

The `isPartOf` relationship enables project-level views:

```json
// In CATALOG.json
"projects": {
  "vlm-burial-mound-detection": {
    "sessions": [
      "2025-12-23T09-00-00_initial-methodology",
      "2026-01-07T14-30-00_multiscale-pilot",
      "2026-01-08T09-15-00_preregistration-review"
    ],
    "total_duration_hours": 28.5,
    "date_range": ["2025-12-23", "2026-01-08"]
  }
}
```

### A3.5 Session Chains

For sessions that form a logical sequence:

```text
Session A ──continues──▶ Session B ──continues──▶ Session C
    │                        │
    └──────isPartOf──────────┴──────isPartOf──────▶ Project X
```

This enables both linear navigation (follow the chain) and hierarchical aggregation (all sessions in project).

---

## A4. Web App Session Compatibility

### A4.1 Recommendation: Unified Schema with Platform-Specific Extensions

Design one schema that handles both CC and web app sessions, with optional platform-specific fields.

### A4.2 Platform Differences

| Aspect | Claude Code | Claude Web App |
|--------|-------------|----------------|
| Tool calls | Extensive (view, bash, create_file, etc.) | Limited (artifacts, web search) |
| Artifacts | Files in filesystem | Inline artifacts with render types |
| Thinking | Extended thinking blocks | May or may not be visible |
| Export format | JSONL (via API/CLI) | JSON (via UI export) |
| Session structure | Single continuous session | May span multiple "chats" in project |
| Context | Filesystem, codebase | Uploaded files, project knowledge |

### A4.3 Schema Additions for Platform Abstraction

```json
"platform": {
  "type": "claude-code",  // or "claude-web", "chatgpt", "gemini", etc.
  "version": "1.0.24",
  "interface": "cli",     // or "web", "api", "mobile"
  "export_format": "jsonl",
  "export_method": "manual",  // or "automated", "api"
  
  // Platform-specific extensions
  "claude_code": {
    "working_directory": "/home/user/project",
    "git_commit": "abc123",
    "tools_available": ["view", "bash_tool", "create_file", "str_replace"]
  },
  
  "claude_web": {
    "project_name": "VLM Burial Mound Detection",
    "project_id": "proj_abc123",
    "has_project_knowledge": true,
    "artifact_count": 3
  }
}
```

### A4.4 Artifact Abstraction

Unify CC file operations and web app artifacts:

```json
"artifacts": {
  "created": [
    {
      "id": "artifact_001",
      "type": "document",
      "platform_type": "file",           // CC: file; web: artifact
      "path": "outputs/analysis.md",      // CC: filesystem path
      "artifact_id": null,                // Web: artifact UUID
      "render_type": null,                // Web: markdown, react, html, etc.
      "description": "Analysis results"
    },
    {
      "id": "artifact_002", 
      "type": "visualization",
      "platform_type": "artifact",
      "path": null,
      "artifact_id": "art_xyz789",
      "render_type": "react",
      "description": "Interactive chart"
    }
  ]
}
```

### A4.5 Conversion Utilities

Provide scripts to normalise different export formats:

```bash
# Convert web app JSON export to canonical JSONL
python convert-webapp-export.py claude_export.json > session.jsonl

# Convert CC session to canonical format (if not already JSONL)
python convert-cc-export.py session_data > session.jsonl
```

---

## A5. Automated Metadata Extraction

### A5.1 Recommendation: Maximize Auto-Generation, Minimize Manual Input

For community adoption, the archive process must be nearly frictionless. Target: **<2 minutes of manual input per session**.

### A5.2 Fully Automatable Fields

These can be extracted directly from the JSONL:

| Field | Extraction Method |
|-------|-------------------|
| `session.id` | From export metadata or generate UUID |
| `session.started_at` | First message timestamp |
| `session.ended_at` | Last message timestamp |
| `session.duration_minutes` | Computed |
| `model.*` | From API response metadata |
| `statistics.*` | Computed from message/tool counts |
| `thinking_blocks.count/tokens` | Computed |
| `archive.*` | Computed at archive time |
| `artifacts.created/modified` | From tool calls (create_file, str_replace) |

### A5.3 Semi-Automatable Fields (LLM-Assisted)

These can be generated by passing the session to an LLM:

| Field | Extraction Prompt |
|-------|-------------------|
| `session.title` | "Generate a brief title (5-10 words) for this session" |
| `context.purpose` | "In one sentence, what was this session trying to accomplish?" |
| `context.tags` | "List 3-5 keyword tags for this session" |
| `three_ps.prompt_summary` | "Summarize what was asked in this session (1-2 sentences)" |
| `three_ps.process_summary` | "Summarize how the task was approached (1-2 sentences)" |
| `three_ps.provenance_summary` | "How does this session fit into the broader research workflow? (1 sentence)" |
| `provenance.methodology_notes` | "Briefly describe the methodology used" |

### A5.4 Manual Input Required

Minimal fields requiring human judgment:

| Field | Why Manual |
|-------|------------|
| `session.slug` | Affects directory naming; user preference |
| `context.project` | Must match existing project taxonomy |
| `context.research_phase` | Requires understanding of research workflow |
| `relationships.*` | Requires knowledge of other sessions |
| `thinking_blocks.sharing_preference` | Policy decision |
| `artifacts[].description` | May need context not in session |

### A5.5 Archive Script Interface

Design for minimal friction:

```bash
# Fully automated (uses LLM for semi-auto fields, defaults for manual)
./archive-session.py session.jsonl --auto

# Interactive mode (prompts only for required manual fields)
./archive-session.py session.jsonl --interactive
# > Project [vlm-burial-mound-detection]: 
# > Slug [auto: preregistration-review]: 
# > Continues session (optional): 2026-01-07T14-30-00_multiscale-pilot
# > Thinking block sharing [research-only]: 
# Archived to: archive/cc-sessions/2026-01-08T09-15-00_preregistration-review/

# Batch mode with config file
./archive-session.py session.jsonl --config archive-defaults.yaml
```

### A5.6 Default Configuration File

```yaml
# archive-defaults.yaml
project: vlm-burial-mound-detection
research_phase: preregistration
thinking_blocks:
  sharing_preference: research-only
  use_constraints:
    - analysis-for-improvement
    - research-publication-aggregated
  excluded_uses:
    - training-data
    - public-display-individual
auto_generate:
  title: true
  purpose: true
  tags: true
  three_ps: true
  methodology_notes: true
llm_for_extraction:
  provider: anthropic
  model: claude-sonnet-4-5-20250514
```

### A5.7 Adoption Friction Analysis

| Step | Time | Friction Level |
|------|------|----------------|
| Export session | 30 sec | Low (built into tools) |
| Run archive script | 10 sec | Low (single command) |
| Answer prompts (interactive) | 60-90 sec | Medium |
| Review auto-generated metadata | 30 sec | Low |
| **Total** | **~2-3 min** | **Acceptable** |

For comparison, current best practice (detailed manual documentation) takes 15-30 minutes per session—prohibitive for routine use.

---

## A6. Implementation Priority

Based on the goal of community adoption:

### Phase 1: Core Infrastructure
1. Archive script with auto-generation
2. Basic `session.meta.json` schema
3. `CATALOG.json` generation
4. Query prompts (summarize, decisions)

### Phase 2: Enhanced Features
5. Multi-session relationships
6. Web app compatibility
7. Compression support
8. Additional query prompts

### Phase 3: Community Features
9. RO-Crate export
10. Aggregation tools
11. Cross-project search
12. Research corpus tooling (for aggregated thinking block analysis)

---

## A7. Updated Schema (Complete)

Incorporating all addendum changes, the complete `session.meta.json` schema is available as a separate JSON Schema file: `genai-session-meta-schema-v1.1.json`.

Key changes from v1.0:
- Added `thinking_blocks` section with use constraints
- Added `tool_outputs` to statistics
- Replaced `related_sessions` with `relationships` (RDF-style)
- Added `platform` section for CC/web app abstraction
- Added compression fields to `archive`

---

*Addendum version: 1.0-draft-addendum-1*
*Date: 2026-01-08*
