# Decision Extraction Query

You are analysing a Claude Code session to identify decisions, conclusions, and commitments made during the conversation.

## Core Principle

A decision without its rationale is barely better than no record at all. "Chose
batch API" tells you nothing in 6 months. "Chose batch API because the concurrent
approach hit rate limits at scale, and the workload is embarrassingly parallel so
latency doesn't matter" is a decision you can revisit, challenge, or build on.

**For every decision you extract, include the reasoning.** If the reasoning isn't
explicit in the transcript, note what can be inferred and flag the gap.

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
- **Context**: What prompted this — the problem, question, or trade-off (1-2 sentences)
- **Outcome**: What was decided/concluded (1-2 sentences)
- **Reasoning**: Why this choice over alternatives. What was considered and rejected?
  What constraints drove the decision? (1-3 sentences. This is the most important field.)
- **Confidence**: High | Medium | Low (based on how definitive the statement was)
- **Reversibility**: Easy | Moderate | Hard (how costly would it be to change this later?)
- **Location**: Early | Middle | Late in conversation

## Guidance

- Focus on substantive decisions, not trivial ones ("let's use markdown" is trivial;
  "let's use Option A for the experimental design" is substantive)
- Include decisions made by both human and assistant
- Note if a decision was revisited or changed later in the conversation
- Group related decisions if they form a coherent thread
- **Watch for implicit decisions**: choices made by not discussing alternatives
  (e.g., using a library without considering others). Flag these — they're often
  the most important ones to document because nobody remembers making them

## Session Data

[Paste or attach session.jsonl content]
