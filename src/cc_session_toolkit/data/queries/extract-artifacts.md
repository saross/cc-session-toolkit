# Artifact Extraction Query

You are analysing a Claude Code session to identify all files that were created, modified, or significantly referenced.

## Core Principle

An artifact list that records only file paths and actions is a changelog, not
documentation. Capture **why** each file was touched — the motivation, not just
the mechanics. "Created `scripts/batch-process.py`" tells you nothing in 6
months. "Created `scripts/batch-process.py` to replace the serial processing
script after discovering the API supports 100 concurrent jobs" is useful.

## Task

Identify all artifacts (files) involved in the session:

### 1. Files Created

Files that did not exist before and were created during the session.
Look for: `Write` tool calls

### 2. Files Modified

Existing files that were changed during the session.
Look for: `Edit` tool calls

### 3. Files Read/Referenced

Files that were examined but not modified.
Look for: `Read` tool calls, `Glob` results, file content appearing in conversation

## Output Format

For each artifact:

| File | Action | Why | What Changed | Tool |
|------|--------|-----|-------------|------|
| path/to/file.md | Created | Why it was needed | Brief description | Write |
| other/file.json | Modified | What motivated the change | What specifically changed | Edit |
| input/doc.md | Read | Why it was consulted | Key information extracted | Read |

The **Why** column is the most important — it captures the rationale that decays.
"Modified" tells you the action; "because the validation rules didn't handle the
new field type" tells you the reason.

## Additional Information

After the table, note:

- Any files that were created then deleted (intermediate artifacts)
- Any failed file operations (attempted but failed, and **why they failed**)
- File relationships (e.g., "X was created based on template Y")
- **Design decisions** embedded in file organisation (e.g., "split into two files
  because the original exceeded the linter's complexity threshold")

## Session Data

[Paste or attach session.jsonl content]
