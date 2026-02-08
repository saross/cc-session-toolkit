# Specification: Archiving Claude Code Sessions

**Version**: 1.0-draft
**Date**: 2026-01-08
**Author**: Shawn Ross (with Claude)
**Context**: VLM Burial Mound Detection Project; RDA IG on Documenting GenAI Interactions

---

## 1. Introduction

### 1.1 Purpose

This document specifies an approach to archiving Claude Code (CC) sessions that balances completeness, findability, and practical usability. It is designed to serve as:

1. A working specification for the VLM burial mound detection project
2. A model implementation for the RDA Interest Group on Documenting GenAI Interactions
3. A testbed for the "Three Ps" framework (Prompt, Process, Provenance)

### 1.2 The "Write-Once-Read-Never" Problem

Traditional documentation approaches often fail because elaborate derivative views:

- Impose maintenance burden that exceeds their value
- Drift from source truth over time
- Represent speculative effort for views that may never be read

The alternative—**store complete, query on demand**—aligns better with actual usage patterns: archives are accessed rarely, but intensively when needed.

### 1.3 Design Philosophy

This specification adopts three core principles:

1. **Canonical completeness**: The JSONL archive is the source of truth; nothing is discarded
2. **Metadata for finding, not reading**: Structured metadata enables discovery; content is accessed from the canonical source
3. **LLM-intermediated access**: Rather than pre-generating human-readable views, provide query tools (prompts and scripts) that extract views on demand

This approach treats LLM session archives the same way we treat other research data: preserve the complete record, describe it with FAIR-aligned metadata, and provide tools for access.

---

## 2. Architecture

### 2.1 Directory Structure

```text
archive/cc-sessions/
├── CATALOG.json                    # Machine-readable index of all sessions
├── CATALOG.md                      # Auto-generated from CATALOG.json
├── queries/                        # Extraction prompts and scripts
│   ├── summarize-session.md        # Prompt: executive summary
│   ├── extract-decisions.md        # Prompt: decisions and conclusions
│   ├── extract-artifacts.md        # Prompt: files created/modified
│   ├── extract-methodology.md      # Prompt: methods documentation
│   ├── convert-to-markdown.py      # Script: deterministic MD conversion
│   └── README.md                   # Query tool documentation
├── 2026-01-08T09-15-00_preregistration-review/
│   ├── session.jsonl               # Complete, canonical archive
│   └── session.meta.json           # Minimal findability metadata
├── 2026-01-07T14-30-00_multiscale-pilot/
│   ├── session.jsonl
│   └── session.meta.json
└── ...
```

### 2.2 Component Roles

| Component | Role | Access Pattern |
|-----------|------|----------------|
| `session.jsonl` | Canonical archive | Query on demand |
| `session.meta.json` | Findability metadata | Catalogue/search |
| `CATALOG.json` | Session index | Browse/filter |
| `queries/*.md` | LLM extraction prompts | Interactive analysis |
| `queries/*.py` | Deterministic extraction | Batch processing |

### 2.3 Rationale for LLM-Intermediated Access

Pre-generating comprehensive human-readable transcripts has several problems:

1. **Scale**: A 2-hour CC session can produce 50,000+ lines of JSONL; any "complete" MD version would be similarly unwieldy
2. **Purpose mismatch**: Different readers need different views (summary vs. decisions vs. methodology)
3. **Maintenance**: Schema changes require regenerating all derived views

LLM-intermediated access solves these by:

1. **Generating views on demand**: Only create what's actually needed
2. **Adapting to reader needs**: Same archive, different queries for different purposes
3. **Leveraging LLM comprehension**: LLMs can extract meaning, not just reformat

The queries themselves become part of the documentation infrastructure—reusable, versionable, improvable.

---

## 3. Session Metadata Schema

### 3.1 Schema Definition

```json
{
  "$schema": "https://example.org/schemas/genai-session-meta/v1.0",
  "schema_version": "1.0",

  "session": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "slug": "preregistration-review",
    "title": "Preregistration Final Review",
    "started_at": "2026-01-08T09:15:00Z",
    "ended_at": "2026-01-08T11:45:00Z",
    "duration_minutes": 150
  },

  "model": {
    "provider": "anthropic",
    "model_id": "claude-code",
    "model_version": "claude-sonnet-4-5-20250929",
    "access_method": "claude-code-cli",
    "access_version": "1.0.24"
  },

  "context": {
    "project": "vlm-burial-mound-detection",
    "purpose": "preregistration-consistency-review",
    "research_phase": "preregistration",
    "workflow_step": "document-review",
    "tags": ["preregistration", "document-review", "corrections"],
    "related_sessions": {
      "continues": "2026-01-07T14-30-00_multiscale-pilot",
      "continued_by": null,
      "related": []
    }
  },

  "statistics": {
    "turns": 47,
    "human_messages": 24,
    "assistant_messages": 23,
    "thinking_blocks": 89,
    "tokens": {
      "input": 125000,
      "output": 45000,
      "thinking": 67000,
      "cache_read": 80000
    },
    "tool_calls": {
      "total": 156,
      "by_type": {
        "view": 45,
        "bash_tool": 67,
        "create_file": 12,
        "str_replace": 32
      }
    },
    "estimated_cost_usd": 0.85
  },

  "artifacts": {
    "created": [
      {
        "path": "outputs/pilot/multiscale_analysis.json",
        "type": "data",
        "description": "Multi-scale voting analysis results"
      },
      {
        "path": "archive/corrections/execution-plan-corrections.md",
        "type": "document",
        "description": "Corrections for execution-plan.md"
      }
    ],
    "modified": [
      {
        "path": "preregistration.md",
        "type": "document",
        "changes": "v4.1 → v4.2",
        "description": "Added pilot notes, H5 clarifications"
      }
    ],
    "referenced": [
      {
        "path": "preregistration.md",
        "type": "document",
        "access": "read"
      }
    ]
  },

  "provenance": {
    "inputs": [
      {
        "path": "preregistration.md",
        "version": "v4.1",
        "role": "primary-document"
      },
      {
        "path": "execution-plan.md",
        "version": "v2.3",
        "role": "review-target"
      }
    ],
    "outputs": [
      {
        "path": "preregistration.md",
        "version": "v4.2",
        "role": "revised-document"
      },
      {
        "path": "execution-plan-corrections.md",
        "version": "v1.0",
        "role": "correction-instructions"
      }
    ],
    "methodology_notes": "Systematic section-by-section review against preregistration requirements"
  },

  "archive": {
    "jsonl_path": "session.jsonl",
    "jsonl_sha256": "a1b2c3d4e5f6...",
    "jsonl_bytes": 2456789,
    "jsonl_lines": 47,
    "archived_at": "2026-01-08T12:00:00Z",
    "archived_by": "archive-sessions.py v1.2",
    "archive_notes": null
  },

  "three_ps": {
    "prompt_summary": "Review preregistration and related documents for consistency; generate correction documents",
    "process_summary": "Iterative document review with section-by-section comparison; correction document generation for CC implementation",
    "provenance_summary": "Part of preregistration finalisation workflow; follows pilot analysis sessions"
  }
}
```

### 3.2 Field Definitions

#### 3.2.1 Session Identification

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `session.id` | UUID | Yes | Unique identifier (from CC if available, else generated) |
| `session.slug` | string | Yes | Human-readable identifier for directory naming |
| `session.title` | string | No | Descriptive title for catalogue display |
| `session.started_at` | ISO 8601 | Yes | Session start timestamp |
| `session.ended_at` | ISO 8601 | Yes | Session end timestamp |
| `session.duration_minutes` | integer | Yes | Computed duration |

#### 3.2.2 Model Information

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model.provider` | string | Yes | anthropic, openai, google, etc. |
| `model.model_id` | string | Yes | Product name (claude-code, chatgpt, etc.) |
| `model.model_version` | string | Yes | Specific model version string |
| `model.access_method` | string | Yes | CLI, web, API, etc. |
| `model.access_version` | string | No | Version of access tool |

#### 3.2.3 Context

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `context.project` | string | Yes | Project identifier |
| `context.purpose` | string | Yes | Brief purpose statement |
| `context.research_phase` | string | No | Phase of research (exploration, analysis, writing, etc.) |
| `context.workflow_step` | string | No | Step within workflow |
| `context.tags` | array | No | Searchable tags |
| `context.related_sessions` | object | No | Links to related sessions |

#### 3.2.4 Statistics

Statistics enable quick assessment without parsing the full JSONL:

| Field | Type | Description |
|-------|------|-------------|
| `statistics.turns` | integer | Total conversation turns |
| `statistics.human_messages` | integer | Human message count |
| `statistics.assistant_messages` | integer | Assistant message count |
| `statistics.thinking_blocks` | integer | Extended thinking block count |
| `statistics.tokens.*` | integers | Token counts by type |
| `statistics.tool_calls.*` | integers | Tool call counts by type |
| `statistics.estimated_cost_usd` | number | Estimated API cost |

#### 3.2.5 Artifacts

Track files created, modified, and referenced:

| Field | Type | Description |
|-------|------|-------------|
| `artifacts.created[]` | array | Files created during session |
| `artifacts.modified[]` | array | Files modified during session |
| `artifacts.referenced[]` | array | Files read but not modified |

Each artifact entry includes:
- `path`: Relative path from project root
- `type`: data, document, code, config, etc.
- `description`: Brief description of content/purpose

#### 3.2.6 Provenance

Research workflow integration:

| Field | Type | Description |
|-------|------|-------------|
| `provenance.inputs[]` | array | Input files with versions and roles |
| `provenance.outputs[]` | array | Output files with versions and roles |
| `provenance.methodology_notes` | string | Brief methodology description |

#### 3.2.7 Archive

Integrity and verification:

| Field | Type | Description |
|-------|------|-------------|
| `archive.jsonl_path` | string | Path to JSONL file |
| `archive.jsonl_sha256` | string | SHA-256 hash for integrity |
| `archive.jsonl_bytes` | integer | File size |
| `archive.jsonl_lines` | integer | Line count |
| `archive.archived_at` | ISO 8601 | Archive timestamp |
| `archive.archived_by` | string | Archive tool and version |

#### 3.2.8 Three Ps Summary

High-level documentation aligned with RDA IG framework:

| Field | Type | Description |
|-------|------|-------------|
| `three_ps.prompt_summary` | string | What was asked (Prompt) |
| `three_ps.process_summary` | string | How the tool was used (Process) |
| `three_ps.provenance_summary` | string | Role in research workflow (Provenance) |

### 3.3 FAIR Alignment

| Principle | Implementation |
|-----------|----------------|
| **Findable** | UUID, slug, tags, catalogue index |
| **Accessible** | Standard JSON format, documented schema |
| **Interoperable** | ISO 8601 timestamps, standard field names |
| **Reusable** | Complete provenance, methodology notes, Three Ps summary |

---

## 4. Query Prompts for LLM-Intermediated Access

### 4.1 Session Summary

**File**: `queries/summarize-session.md`

```markdown
# Session Summary Query

You are analysing a Claude Code session transcript in JSONL format. Each line
is a JSON object representing one event in the conversation (human messages,
assistant messages, tool calls, tool results, thinking blocks).

## Task

Provide a structured summary including:

### 1. Purpose
What was the user trying to accomplish? (1-2 sentences)

### 2. Key Activities
What major tasks were performed? (bullet list, 3-7 items, in rough
chronological order)

### 3. Decisions Made
What significant decisions or conclusions were reached? Include:
- Technical decisions (e.g., "chose Option A over Option B")
- Findings (e.g., "identified 5 issues in document X")
- Agreements (e.g., "will proceed with approach Y")

### 4. Artifacts Produced
What files were created or significantly modified? List with:
- File path
- Brief description of content/purpose

### 5. Open Items
What was left unfinished or flagged for follow-up? (if any)

### 6. Session Statistics
- Duration (from first to last timestamp)
- Approximate turn count
- Notable tool usage patterns

## Output Format

Use markdown with the headers above. Be concise—this is a reference summary,
not a complete transcript. Aim for 300-500 words total.

## Session Data

[Paste or attach session.jsonl content]
```

### 4.2 Decision Extraction

**File**: `queries/extract-decisions.md`

```markdown
# Decision Extraction Query

You are analysing a Claude Code session to identify decisions, conclusions,
and commitments made during the conversation.

## Task

Extract all instances where:
- A **decision** was made ("we'll go with Option A", "let's use X approach")
- A **conclusion** was reached ("this confirms that...", "the issue is...")
- A **commitment** was made ("I'll do X before Y", "next step is...")
- A **problem** was identified and resolved
- A **question** was answered definitively

## Output Format

For each item extracted:

### [Brief descriptive title]

- **Type**: Decision | Conclusion | Commitment | Resolution | Answer
- **Context**: What prompted this (1-2 sentences)
- **Outcome**: What was decided/concluded (1-2 sentences)
- **Confidence**: High | Medium | Low (based on how definitive the statement was)
- **Location**: Early | Middle | Late in conversation

## Guidance

- Focus on substantive decisions, not trivial ones ("let's use markdown" is
  trivial; "let's use Option A for the experimental design" is substantive)
- Include decisions made by both human and assistant
- Note if a decision was revisited or changed later in the conversation
- Group related decisions if they form a coherent thread

## Session Data

[Paste or attach session.jsonl content]
```

### 4.3 Artifact Extraction

**File**: `queries/extract-artifacts.md`

```markdown
# Artifact Extraction Query

You are analysing a Claude Code session to identify all files that were
created, modified, or significantly referenced.

## Task

Identify all artifacts (files) involved in the session:

### 1. Files Created
Files that did not exist before and were created during the session.
Look for: `create_file` tool calls, `write` operations in bash

### 2. Files Modified
Existing files that were changed during the session.
Look for: `str_replace` tool calls, `sed`/`echo >>` in bash, explicit
mentions of updating files

### 3. Files Read/Referenced
Files that were examined but not modified.
Look for: `view` tool calls, `cat`/`head`/`tail` in bash, file content
appearing in conversation

## Output Format

For each artifact:

| File | Action | Description | Tool Used |
|------|--------|-------------|-----------|
| path/to/file.md | Created | Brief description | create_file |
| other/file.json | Modified | What changed | str_replace |
| input/doc.md | Read | Why it was read | view |

## Additional Information

After the table, note:
- Any files that were created then deleted (intermediate artifacts)
- Any failed file operations (attempted but failed)
- File relationships (e.g., "X was created based on template Y")

## Session Data

[Paste or attach session.jsonl content]
```

### 4.4 Methodology Extraction

**File**: `queries/extract-methodology.md`

```markdown
# Methodology Extraction Query

You are analysing a Claude Code session to extract methodology documentation
suitable for a research methods section or supplementary materials.

## Task

Document the methodology used in this session as if writing for a
methods section of a paper. Include:

### 1. Objective
What was the session trying to accomplish? Frame in research terms.

### 2. Approach
What approach or workflow was used? Describe the logical steps.

### 3. Tools and Parameters
- Model used (extract from conversation or metadata)
- Key parameters or settings
- External tools or scripts invoked

### 4. Data Inputs
What data or documents were used as inputs? Note versions if mentioned.

### 5. Process
Describe the actual process followed:
- Was it iterative? How many iterations?
- Were there decision points? What drove the decisions?
- Were there corrections or backtracking?

### 6. Validation
How were outputs validated or checked?
- Manual review?
- Automated checks?
- Cross-referencing?

### 7. Outputs
What were the final outputs? How do they relate to the research objectives?

## Output Format

Write in third person, past tense, suitable for inclusion in a methods
section. Aim for 200-400 words. Be specific about what was done, not
what could be done.

## Session Data

[Paste or attach session.jsonl content]
```

### 4.5 Error and Issue Extraction

**File**: `queries/extract-issues.md`

```markdown
# Error and Issue Extraction Query

You are analysing a Claude Code session to identify errors, issues,
problems, and their resolutions.

## Task

Extract all instances of:

### 1. Errors Encountered
- Tool failures
- Code errors
- Parsing failures
- API errors

### 2. Issues Identified
- Problems found in documents/code being reviewed
- Inconsistencies discovered
- Gaps or omissions noted

### 3. Misunderstandings
- Cases where the assistant misunderstood the request
- Cases where clarification was needed
- Incorrect assumptions that were corrected

### 4. Resolutions
For each error/issue, note how it was resolved (if it was).

## Output Format

### Errors Encountered

| Error | Context | Resolution | Resolved? |
|-------|---------|------------|-----------|
| Brief description | What was being attempted | How it was fixed | Yes/No/Partial |

### Issues Identified

| Issue | Severity | Location | Resolution |
|-------|----------|----------|------------|
| Brief description | High/Medium/Low | Where found | How addressed |

### Misunderstandings

| Misunderstanding | Clarification | Impact |
|------------------|---------------|--------|
| What was misunderstood | How it was corrected | Effect on session |

## Session Data

[Paste or attach session.jsonl content]
```

---

## 5. Deterministic Conversion Script

### 5.1 Purpose

While LLM-intermediated access is preferred for analysis, a deterministic
script provides:

1. Consistent, reproducible output
2. Batch processing capability
3. Fallback when LLM access is unavailable
4. Baseline for comparison with LLM-generated summaries

### 5.2 Script: `convert-to-markdown.py`

```python
#!/usr/bin/env python3
"""
Convert Claude Code session JSONL to human-readable Markdown.

Usage:
    python convert-to-markdown.py session.jsonl > session.md
    python convert-to-markdown.py session.jsonl --output session.md

Options:
    --include-thinking     Include thinking blocks (default: collapsed)
    --include-tool-output  Include full tool output (default: truncated)
    --max-tool-output N    Max chars for tool output (default: 500)
    --no-collapse          Don't use <details> tags for collapsible sections
"""

import json
import argparse
import hashlib
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert CC session JSONL to Markdown'
    )
    parser.add_argument('jsonl_file', type=Path, help='Input JSONL file')
    parser.add_argument('--output', '-o', type=Path, help='Output file')
    parser.add_argument('--include-thinking', action='store_true',
                        help='Include thinking blocks expanded')
    parser.add_argument('--include-tool-output', action='store_true',
                        help='Include full tool output')
    parser.add_argument('--max-tool-output', type=int, default=500,
                        help='Max chars for tool output')
    parser.add_argument('--no-collapse', action='store_true',
                        help="Don't use collapsible sections")
    return parser.parse_args()


def format_timestamp(ts_str):
    """Format ISO timestamp for display."""
    if not ts_str:
        return "Unknown"
    try:
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S UTC')
    except:
        return ts_str


def truncate(text, max_len=500):
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n... [{len(text) - max_len} chars truncated]"


def collapsible(summary, content, collapse=True):
    """Wrap content in collapsible details tag."""
    if not collapse:
        return f"**{summary}**\n\n{content}"
    return f"""<details>
<summary>{summary}</summary>

{content}

</details>"""


def extract_messages(jsonl_path):
    """Extract messages from JSONL file."""
    messages = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                messages.append(json.loads(line))
    return messages


def format_tool_call(tool_name, tool_input, tool_output, args):
    """Format a tool call for display."""
    input_preview = truncate(str(tool_input), 200)

    if args.include_tool_output:
        output_text = str(tool_output)
    else:
        output_text = truncate(str(tool_output), args.max_tool_output)

    content = f"""**Input:**
```
{input_preview}
```

**Output:**
```
{output_text}
```"""

    return collapsible(f"🔧 Tool: {tool_name}", content, not args.no_collapse)


def format_thinking(thinking_text, args):
    """Format thinking block for display."""
    if not args.include_thinking:
        preview = truncate(thinking_text, 200)
        return collapsible("💭 Thinking", f"```\n{preview}\n```", True)
    return collapsible("💭 Thinking", f"```\n{thinking_text}\n```",
                       not args.no_collapse)


def convert_session(messages, args):
    """Convert messages to Markdown."""
    lines = []

    # Header
    lines.append("# Claude Code Session Transcript\n")

    # Try to extract metadata
    first_ts = None
    last_ts = None
    turn_count = 0
    tool_calls = {}

    for msg in messages:
        ts = msg.get('timestamp')
        if ts:
            if not first_ts:
                first_ts = ts
            last_ts = ts
        if msg.get('role') == 'human':
            turn_count += 1
        if msg.get('type') == 'tool_use':
            tool_name = msg.get('name', 'unknown')
            tool_calls[tool_name] = tool_calls.get(tool_name, 0) + 1

    # Metadata section
    lines.append("## Session Metadata\n")
    lines.append(f"- **Start**: {format_timestamp(first_ts)}")
    lines.append(f"- **End**: {format_timestamp(last_ts)}")
    lines.append(f"- **Turns**: {turn_count}")
    if tool_calls:
        tools_str = ", ".join(f"{k}: {v}" for k, v in sorted(tool_calls.items()))
        lines.append(f"- **Tool calls**: {tools_str}")
    lines.append("")

    # Conversation
    lines.append("## Conversation\n")
    lines.append("---\n")

    current_turn = 0
    for msg in messages:
        role = msg.get('role', msg.get('type', 'unknown'))

        if role == 'human':
            current_turn += 1
            content = msg.get('content', '')
            if isinstance(content, list):
                content = '\n'.join(
                    c.get('text', str(c)) for c in content if isinstance(c, dict)
                )
            lines.append(f"### 👤 Human (Turn {current_turn})\n")
            lines.append(content)
            lines.append("\n---\n")

        elif role == 'assistant':
            content = msg.get('content', '')
            if isinstance(content, list):
                # Handle structured content (text, tool_use, thinking)
                for block in content:
                    if isinstance(block, dict):
                        if block.get('type') == 'text':
                            lines.append(f"### 🤖 Assistant\n")
                            lines.append(block.get('text', ''))
                        elif block.get('type') == 'thinking':
                            lines.append(format_thinking(block.get('thinking', ''), args))
                        elif block.get('type') == 'tool_use':
                            lines.append(format_tool_call(
                                block.get('name', 'unknown'),
                                block.get('input', {}),
                                '[See tool_result]',
                                args
                            ))
            else:
                lines.append(f"### 🤖 Assistant\n")
                lines.append(str(content))
            lines.append("\n---\n")

        elif role == 'tool_result':
            # Tool results are usually shown inline with tool calls
            pass

    return '\n'.join(lines)


def main():
    args = parse_args()

    if not args.jsonl_file.exists():
        print(f"Error: File not found: {args.jsonl_file}")
        return 1

    messages = extract_messages(args.jsonl_file)
    markdown = convert_session(messages, args)

    if args.output:
        args.output.write_text(markdown)
        print(f"Written to {args.output}")
    else:
        print(markdown)

    return 0


if __name__ == '__main__':
    exit(main())
```

### 5.3 Usage Examples

```bash
# Basic conversion (thinking collapsed, tool output truncated)
python convert-to-markdown.py session.jsonl > session.md

# Full transcript with all details
python convert-to-markdown.py session.jsonl \
    --include-thinking \
    --include-tool-output \
    > session-full.md

# For platforms that don't support <details> tags
python convert-to-markdown.py session.jsonl --no-collapse > session.md
```

---

## 6. Catalog Schema

### 6.1 CATALOG.json

The catalog provides a searchable index of all archived sessions:

```json
{
  "$schema": "https://example.org/schemas/genai-session-catalog/v1.0",
  "schema_version": "1.0",
  "generated_at": "2026-01-08T12:00:00Z",
  "project": "vlm-burial-mound-detection",

  "sessions": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "slug": "preregistration-review",
      "title": "Preregistration Final Review",
      "directory": "2026-01-08T09-15-00_preregistration-review",
      "started_at": "2026-01-08T09:15:00Z",
      "duration_minutes": 150,
      "tags": ["preregistration", "document-review"],
      "purpose": "Review preregistration and related documents for consistency",
      "artifacts_created": 2,
      "artifacts_modified": 1,
      "continues": "2026-01-07T14-30-00_multiscale-pilot"
    },
    {
      "id": "...",
      "slug": "multiscale-pilot",
      "title": "Multi-Scale Voting Pilot Analysis",
      "directory": "2026-01-07T14-30-00_multiscale-pilot",
      "started_at": "2026-01-07T14:30:00Z",
      "duration_minutes": 180,
      "tags": ["pilot", "analysis", "multi-scale"],
      "purpose": "Analyse multi-scale voting strategies for tile size pilot",
      "artifacts_created": 5,
      "artifacts_modified": 2,
      "continues": null
    }
  ],

  "summary": {
    "total_sessions": 15,
    "total_duration_hours": 28.5,
    "date_range": {
      "first": "2025-12-23",
      "last": "2026-01-08"
    },
    "tags": {
      "preregistration": 8,
      "pilot": 4,
      "analysis": 6,
      "document-review": 5
    }
  }
}
```

### 6.2 CATALOG.md (Auto-Generated)

```markdown
# CC Session Archive: VLM Burial Mound Detection

*Generated: 2026-01-08 12:00 UTC*

## Summary

- **Total sessions**: 15
- **Total duration**: 28.5 hours
- **Date range**: 2025-12-23 to 2026-01-08

## Sessions by Date

### January 2026

#### 2026-01-08: Preregistration Final Review
- **Duration**: 2h 30m
- **Tags**: preregistration, document-review
- **Purpose**: Review preregistration and related documents for consistency
- **Artifacts**: 2 created, 1 modified
- **Continues**: [Multi-Scale Voting Pilot](#2026-01-07-multi-scale-voting-pilot-analysis)
- **Files**: [session.jsonl](./2026-01-08T09-15-00_preregistration-review/session.jsonl) | [metadata](./2026-01-08T09-15-00_preregistration-review/session.meta.json)

#### 2026-01-07: Multi-Scale Voting Pilot Analysis
- **Duration**: 3h 00m
- **Tags**: pilot, analysis, multi-scale
- **Purpose**: Analyse multi-scale voting strategies for tile size pilot
- **Artifacts**: 5 created, 2 modified
- **Files**: [session.jsonl](./2026-01-07T14-30-00_multiscale-pilot/session.jsonl) | [metadata](./2026-01-07T14-30-00_multiscale-pilot/session.meta.json)

...

## Tags Index

- **preregistration** (8 sessions): [list of links]
- **pilot** (4 sessions): [list of links]
- **analysis** (6 sessions): [list of links]
```

---

## 7. Implementation Guide

### 7.1 Archive Workflow

```text
┌─────────────────────────────────────────────────────────────────┐
│                     CC Session Ends                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Export JSONL                                                 │
│     - Use CC's built-in export or API                           │
│     - Capture complete conversation including thinking blocks    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. Generate session.meta.json                                   │
│     - Automated: statistics, timestamps, tool counts            │
│     - Manual: purpose, tags, provenance, Three Ps summary       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. Create session directory                                     │
│     - Name: {date}T{time}_{slug}/                               │
│     - Contents: session.jsonl, session.meta.json                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. Update CATALOG.json                                          │
│     - Add session entry                                          │
│     - Update summary statistics                                  │
│     - Regenerate CATALOG.md                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Archive Script

A script to automate steps 2-4:

```bash
# Archive current or specified session
./scripts/archive-session.py [session_id] \
    --slug "preregistration-review" \
    --purpose "Review preregistration for consistency" \
    --tags preregistration,document-review \
    --continues 2026-01-07T14-30-00_multiscale-pilot
```

### 7.3 Query Workflow

```text
┌─────────────────────────────────────────────────────────────────┐
│  Need to access session content                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Find session                                                 │
│     - Browse CATALOG.md                                          │
│     - Search CATALOG.json by tags, date, purpose                │
│     - Check session.meta.json for specific sessions             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. Choose access method                                         │
│     ┌──────────────────┬──────────────────┬──────────────────┐  │
│     │ Quick overview   │ Specific extract │ Full transcript  │  │
│     │ → summarize      │ → targeted query │ → convert script │  │
│     │   prompt         │   prompt         │                  │  │
│     └──────────────────┴──────────────────┴──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. Execute query                                                │
│     - LLM: Paste JSONL + query prompt into capable LLM          │
│     - Script: Run convert-to-markdown.py for deterministic MD   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Alignment with Standards

### 8.1 FAIR Principles

| Principle | Implementation |
|-----------|----------------|
| **F1**: Globally unique identifier | Session UUID in metadata |
| **F2**: Rich metadata | Comprehensive session.meta.json |
| **F3**: Metadata includes identifier | `session.id` field |
| **F4**: Registered/indexed | CATALOG.json provides searchable index |
| **A1**: Retrievable by identifier | Directory structure uses slug |
| **A2**: Metadata accessible | JSON format, documented schema |
| **I1**: Formal language | JSON with defined schema |
| **I2**: FAIR vocabularies | ISO 8601 dates, standard field names |
| **I3**: Qualified references | `related_sessions` with explicit relationships |
| **R1**: Plurality of attributes | Extensive metadata fields |
| **R1.1**: Clear usage license | Documented in project |
| **R1.2**: Provenance | `provenance` section, Three Ps |
| **R1.3**: Domain standards | Aligned with RDA IG Three Ps framework |

### 8.2 RDA IG Three Ps Framework

This specification implements the Three Ps framework:

| P | Implementation |
|---|----------------|
| **Prompt** | `three_ps.prompt_summary` + full conversation in JSONL |
| **Process** | `three_ps.process_summary` + `statistics` + `artifacts` |
| **Provenance** | `provenance` section + `context.related_sessions` |

### 8.3 Research Object Crate (RO-Crate) Compatibility

The session directory structure is compatible with RO-Crate packaging:

```json
{
  "@context": "https://w3id.org/ro/crate/1.1/context",
  "@graph": [
    {
      "@id": "./",
      "@type": "Dataset",
      "name": "CC Session: Preregistration Review",
      "description": "Claude Code session for preregistration consistency review",
      "hasPart": [
        {"@id": "session.jsonl"},
        {"@id": "session.meta.json"}
      ]
    },
    {
      "@id": "session.jsonl",
      "@type": "File",
      "name": "Session transcript",
      "encodingFormat": "application/jsonl"
    },
    {
      "@id": "session.meta.json",
      "@type": "File",
      "name": "Session metadata",
      "encodingFormat": "application/json",
      "conformsTo": "https://example.org/schemas/genai-session-meta/v1.0"
    }
  ]
}
```

---

## 9. Open Questions

### 9.1 For Discussion

1. **Thinking block handling**: Should thinking blocks be included in the canonical JSONL even though they may contain internal reasoning that wasn't intended for the user?

2. **Tool output completeness**: Some tool outputs (e.g., viewing large files) can be very large. Should there be a size threshold for inclusion, or should everything be preserved?

3. **Multi-session threads**: How should we represent sessions that span multiple CC invocations but form a logical unit of work?

4. **Web app sessions**: This specification focuses on CC. How should it be adapted for Claude web app sessions (which have different export formats)?

5. **Automated metadata extraction**: How much of session.meta.json can be reliably auto-generated vs. requiring manual input?

### 9.2 Future Extensions

- Integration with version control (linking sessions to git commits)
- Automated artifact tracking (detecting created/modified files)
- Session comparison tools (diff between related sessions)
- Aggregated project-level documentation generation

---

## 10. References

- RDA Interest Group: Documenting Generative AI Interactions in Research (draft)
- FAIR Principles: https://www.go-fair.org/fair-principles/
- RO-Crate: https://www.researchobject.org/ro-crate/
- CodeMeta: https://codemeta.github.io/

---

*Document version: 1.0-draft*
*Created: 2026-01-08*
