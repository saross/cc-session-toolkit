# Error and Issue Extraction Query

You are analysing a Claude Code session to identify errors, issues, problems, and their resolutions.

## Core Principle

Errors are learning opportunities that decay fastest. In 6 months, you won't
remember *why* something failed or *what you tried before it worked*. Capture
the **root cause**, not just the symptom; capture **what was tried**, not just
the fix; capture **what made it hard to diagnose**, not just that it was fixed.

## Task

Extract all instances of:

### 1. Errors Encountered

- Tool failures, code errors, parsing failures, API errors
- For each: **what caused it** (root cause, not just the error message) and
  **why it wasn't caught earlier** (missing test, unexpected input, documentation gap)

### 2. Issues Identified

- Problems found in documents/code being reviewed
- Inconsistencies discovered
- Gaps or omissions noted
- For each: **why it matters** (severity rationale) and **what would happen
  if it weren't fixed**

### 3. Misunderstandings

- Cases where the assistant misunderstood the request
- Cases where clarification was needed
- Incorrect assumptions that were corrected
- For each: **what was ambiguous** that led to the misunderstanding

### 4. Resolutions

For each error/issue: how it was resolved, **why that fix was chosen over
alternatives**, and **whether the fix is permanent or a workaround**.

## Output Format

### Errors Encountered

| Error | Root Cause | What Was Tried | Resolution | Permanent Fix? |
|-------|-----------|----------------|------------|----------------|
| Brief description | Why it happened | Approaches attempted | How it was fixed | Yes / Workaround / Unknown |

### Issues Identified

| Issue | Severity | Why It Matters | Location | Resolution |
|-------|----------|---------------|----------|------------|
| Brief description | High/Medium/Low | Impact if unfixed | Where found | How addressed |

### Misunderstandings

| Misunderstanding | What Was Ambiguous | Clarification | Impact |
|------------------|-------------------|---------------|--------|
| What was misunderstood | Source of ambiguity | How it was corrected | Effect on session |

### Patterns

After the tables, note any **recurring patterns**: the same type of error
appearing multiple times, systematic gaps in testing, or classes of
misunderstanding that suggest a deeper issue.

## Session Data

[Paste or attach session.jsonl content]
