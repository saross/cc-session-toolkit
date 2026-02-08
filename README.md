# cc-session-toolkit

Session archiving, catalogue management, and reflection protocol tooling for
[Claude Code](https://docs.anthropic.com/en/docs/claude-code) research
sessions. Designed for FAIR-aligned research transparency.

## Features

- **Session archiving** — copy CC session transcripts to structured project
  archives with rich metadata (statistics, artifacts, relationships,
  thinking-block ethics)
- **Catalogue management** — build and maintain a searchable index of
  archived sessions with tag indices and relationship graphs
- **Reflection protocol** — end-of-session reflection documents with
  document-discovery via YAML frontmatter
- **Project scaffolding** — `cc-session init` sets up archive directories,
  queries, templates, and the `/reflect` skill in one command

## Installation

```bash
# From GitHub
pip install git+https://github.com/saross/cc-session-toolkit.git

# With YAML support (recommended)
pip install "cc-session-toolkit[yaml] @ git+https://github.com/saross/cc-session-toolkit.git"

# For development
git clone https://github.com/saross/cc-session-toolkit.git
cd cc-session-toolkit
pip install -e ".[dev]"
```

Requires Python 3.11+.

## Quick Start

```bash
# Scaffold a project
cd ~/Code/my-research-project
cc-session init --project-name my-project

# List available sessions
cc-session list

# Archive the latest session
cc-session archive --title "Pipeline Development"

# Archive all unarchived sessions
cc-session archive --all

# Rebuild the catalogue
cc-session catalogue --rebuild --markdown
```

## CLI Reference

```text
cc-session init [--project-name NAME] [--update]
    Scaffold project files. --update regenerates SKILL.md and queries.

cc-session archive [--all] [--session-id ID] [--title TITLE]
                   [--gzip] [--dry-run] [--force] [--stats-only]
    Archive sessions. Default: latest unarchived session.

cc-session list
    List sessions and their archive status.

cc-session list-archives
    Show archived sessions with metadata completion status.

cc-session summarise SESSION_ID
    Analyse an archived session to help generate metadata.

cc-session update SESSION_ID [-m FILE]
    Update metadata for an existing archive.

cc-session catalogue [--rebuild] [--markdown]
    Regenerate catalogue from archived session metadata.
```

## Project Structure After `init`

```text
my-project/
├── archive/cc-sessions/
│   ├── queries/              # LLM extraction prompts
│   └── archive-defaults.yaml # Project-specific defaults
├── docs/notes/reflections/   # Reflection documents (with frontmatter)
│   ├── session-reflection.md
│   ├── llm-observations.md
│   ├── working-notes.md
│   ├── abductive-reasoning.md
│   └── session-log.md
└── .claude/skills/reflect/
    └── SKILL.md              # Reflection protocol (regenerable)
```

## Migration from Existing Projects

```bash
# For projects with existing archive scripts
cd ~/Code/existing-project
cc-session init --project-name my-project

# Existing archive/cc-sessions/ data is preserved
# Delete the old script after verifying cc-session works
```

## Licence

Apache 2.0. See [LICENCE](LICENCE).
