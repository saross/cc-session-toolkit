# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Purpose

`cc-session-toolkit` is a pip-installable package for archiving, cataloguing,
and reflecting on Claude Code research sessions. It consolidates tooling
previously duplicated across multiple research projects into a single
reusable package.

## Repository Structure

```text
src/cc_session_toolkit/
├── __init__.py          # Package version
├── cli.py               # Subcommand-based argparse CLI
├── project.py           # CWD-upward project root detection
├── config.py            # Constants, file type mappings, defaults
├── extraction.py        # Session stats, thinking blocks, artifacts
├── naming.py            # slugify(), archive directory naming
├── archive.py           # Archive operations, metadata creation
├── catalogue.py         # Catalogue CRUD, rebuild, markdown
├── summarise.py         # Session analysis for metadata
├── init.py              # Project scaffolding (init command)
└── data/                # Package data (queries, templates, specs)
    ├── queries/          # LLM extraction prompts
    ├── templates/        # Parameterised templates
    ├── reflections/      # Reflection document stubs
    └── specs/            # Formal specification documents
```

## Key Architecture Decisions

### CWD-based project detection

All functions accept an explicit `project_root: Path` parameter. The
`find_project_root()` function searches upward from CWD for `.git/`,
`CLAUDE.md`, or `pyproject.toml`. This replaces `__file__`-based paths
that break when pip-installed.

### Document-discovery reflection protocol

SKILL.md defines the protocol and is regenerable from the package.
Reflection documents are discovered via YAML frontmatter in a project
directory. Adding a new document type = creating a new `.md` file with
frontmatter. Protocol updates propagate via `cc-session init --update`.

## Development

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Testing

```bash
pytest tests/ -v
```

All tests use `tmp_path` fixtures — no side effects on the real filesystem.

## Code Standards

- Python 3.11+, type hints, pathlib
- UK/Australian English (analyse, catalogue, summarise)
- PEP 8, max 100 characters per line
- Verbose docstrings (Google style)
