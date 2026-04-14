# Methodology Extraction Query

You are analysing a Claude Code session to extract methodology documentation suitable for a research methods section or supplementary materials.

## Core Principle

A methods section must be **reproducible** — a reader should be able to follow
your steps and understand **why** each choice was made. Document not just what
was done but what alternatives existed and why they were rejected. Note where
the process deviated from plan and why.

## Task

Document the methodology used in this session as if writing for a methods section of a paper. Include:

### 1. Objective

What was the session trying to accomplish? Frame in research terms. Include the
**motivation** — why this objective matters in the broader project.

### 2. Approach

What approach or workflow was used? Describe the logical steps. **Note why this
approach was chosen** — were alternatives considered? What constraints (time,
data availability, API limits) shaped the choice?

### 3. Tools and Parameters

- Model used (extract from conversation or metadata)
- Key parameters or settings, **with justification for non-default values**
- External tools or scripts invoked

### 4. Data Inputs

What data or documents were used as inputs? Note versions if mentioned. **Flag
any data limitations** acknowledged during the session.

### 5. Process

Describe the actual process followed:

- Was it iterative? How many iterations, and what triggered each?
- Were there decision points? What drove the decisions?
- Were there corrections or backtracking? **What was learned from errors?**
- Did the process deviate from the initial plan? Why?

### 6. Validation

How were outputs validated or checked?

- Manual review?
- Automated checks?
- Cross-referencing?
- **What wasn't validated** that ideally should have been?

### 7. Outputs

What were the final outputs? How do they relate to the research objectives?
**Note any known limitations** of the outputs.

### 8. Reproducibility Notes

What would someone need to know to reproduce this work? Include:
- Environment assumptions (API access, model versions, data availability)
- Non-obvious dependencies or ordering constraints
- Anything that worked "by accident" (e.g., relied on a specific data ordering)

## Output Format

Write in third person, past tense, suitable for inclusion in a methods section.
Aim for 200-400 words. Be specific about what was done, not what could be done.
**Include rationale for methodological choices** — a methods section that only
lists steps is incomplete.

## Session Data

[Paste or attach session.jsonl content]
