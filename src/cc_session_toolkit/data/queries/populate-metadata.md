# Populate Session Metadata Query

You are analysing a Claude Code session to populate empty fields in its `session.meta.json` file.

## Core Principle: Capture *Why*, Not Just *What*

Every field you populate should help a reader in 6 months understand **why**
this session happened and **why** decisions were made — not just *that* they
happened. Descriptions that record only actions ("implemented batch API")
decay into noise. Descriptions that encode rationale ("implemented batch API
because concurrent approach hit rate limits at scale; chose batch over
streaming because workload is embarrassingly parallel") remain useful.

For each field, ask yourself: **will a reader who has no memory of this session
understand the reasoning, or only the outcome?**

## Inputs

You will receive:

1. **session.jsonl** — The full session transcript (JSONL format)
2. **session.meta.json** — The existing metadata file with placeholder/empty fields

## Task

Analyse the session transcript and generate values for any empty or placeholder fields in the metadata. Return a complete, updated `session.meta.json` that can replace the existing file.

## Fields to Populate

### auto_generated

```json
"auto_generated": {
  "title": "Brief descriptive title (3-8 words)",
  "purpose": "What the user was trying to accomplish and why (1-2 sentences)",
  "tags": ["relevant", "tags", "3-6 items"]
}
```

- **title**: Concise summary of the session's main accomplishment
- **purpose**: The user's goal *and the motivation behind it*. Not just "refactored
  auth module" but "refactored auth module because session token storage didn't
  meet new compliance requirements." If the motivation isn't explicit in the
  transcript, note what can be inferred.
- **tags**: Lowercase keywords covering domain, task type, tools used

### three_ps

```json
"three_ps": {
  "prompt_summary": "What was asked and why (Prompt)",
  "process_summary": "How the tool was used and why this approach (Process)",
  "provenance_summary": "What depends on this session's outcomes (Provenance)"
}
```

- **prompt_summary**: What the user asked *and the underlying need*. Not "asked to
  fix the batch script" but "asked to fix the batch script because overnight run
  failed at the 2 GB file size limit, blocking the analysis pipeline." (1-2 sentences)
- **process_summary**: The workflow followed *and why this approach was chosen over
  alternatives*. Note decision points, corrections, and pivots. (1-2 sentences)
- **provenance_summary**: Where this session sits in the broader project — what
  preceded it, what depends on its outputs, what would break if its results were
  wrong. (1 sentence)

### relationships

```json
"relationships": {
  "continues": "UUID of previous session (if this is a continuation)",
  "continuedBy": "UUID of next session (if work continued)",
  "isPartOf": "UUID of parent session (for agent sub-sessions)"
}
```

- **continues**: Only populate if the session explicitly continues prior work
- **continuedBy**: Only populate if work was continued in a later session
- **isPartOf**: Only populate for agent sub-sessions spawned from a parent session
- Leave fields as `null` if not applicable

### artifacts[].description

For each artifact in `created`, `modified`, and `referenced` arrays, populate empty `description` fields:

```json
{
  "path": "scripts/example.py",
  "type": "code",
  "description": "Why this file was created/changed, not just what it contains"
}
```

Describe the *reason* the file was touched, not just its contents. "Created to
replace the serial processing script after discovering the API supports 100
concurrent jobs" is better than "Batch processing script."

## Output Format

Return the complete, updated `session.meta.json` as a JSON code block. Preserve all existing populated fields exactly as they are — only fill in empty/placeholder values.

```json
{
  "schema_version": "1.1",
  ... (complete JSON)
}
```

## Guidelines

- Use UK/Australian English spelling
- Encode rationale in every description — *why*, not just *what*
- Tags should be lowercase, hyphenated for multi-word (e.g., `code-review`)
- Don't invent information not evidenced in the transcript
- If motivation isn't explicit, state what can be inferred: "likely because..."
  rather than fabricating a reason
- If a field cannot be determined from the transcript, use a sensible default
  or note uncertainty

## Session Data

**session.meta.json:**

```json
[Paste existing session.meta.json here]
```

**session.jsonl:**

[Paste or attach session.jsonl content]
