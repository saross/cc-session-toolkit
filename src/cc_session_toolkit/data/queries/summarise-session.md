# Session Summary Query

You are analysing a Claude Code session transcript in JSONL format. Each line is a JSON object representing one event in the conversation (human messages, assistant messages, tool calls, tool results, thinking blocks).

## Core Principle

Capture **why**, not just **what**. A summary that records only actions ("fixed
the bug, updated the config") will be useless in 6 months. A summary that
captures rationale ("fixed the race condition that caused intermittent test
failures since the concurrency refactor; updated the config to increase the
retry limit because the new provider has higher latency") remains useful.

## Task

Provide a structured summary including:

### 1. Purpose

What was the user trying to accomplish, and **why**? What motivated this session?
(1-2 sentences. If the motivation isn't explicit, note what can be inferred.)

### 2. Key Activities

What major tasks were performed? For each, note the **rationale** if it's not
obvious. (Bullet list, 3-7 items, in rough chronological order)

### 3. Decisions Made

What significant decisions or conclusions were reached? For each decision,
include the **reasoning** — what was chosen, what was rejected, and why.

- Technical decisions (e.g., "chose Option A over Option B **because** ...")
- Findings (e.g., "identified 5 issues in document X, **caused by** ...")
- Agreements (e.g., "will proceed with approach Y **since** ...")

### 4. Artifacts Produced

What files were created or significantly modified? List with:

- File path
- Brief description of content/purpose **and why it was needed**

### 5. Open Items

What was left unfinished or flagged for follow-up? For each, note **why** it was
deferred (blocked, out of scope, time constraint). (If none, state "None.")

### 6. Context That Won't Be Obvious Later

What assumptions, constraints, or circumstances shaped this session that a reader
in 6 months won't know? (E.g., "API was rate-limited during testing", "co-author
deadline drove the priority", "chose the simpler approach because this is a
prototype.") This is the most valuable section — don't skip it.

### 7. Session Statistics

- Duration (from first to last timestamp)
- Approximate turn count
- Notable tool usage patterns

## Output Format

Use markdown with the headers above. Be concise — this is a reference summary,
not a complete transcript. Aim for 300-500 words total. Prioritise rationale
over exhaustive detail.

## Session Data

[Paste or attach session.jsonl content]
