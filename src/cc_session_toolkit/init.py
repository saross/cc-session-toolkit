"""
Project scaffolding — the ``cc-session init`` command.

Creates the directory structure and copies package data files into a
project, using ``importlib.resources`` to read assets from the installed
package.
"""

from __future__ import annotations

import importlib.resources
import shutil
from pathlib import Path

from cc_session_toolkit.project import find_project_root, get_project_name


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _pkg_data_path() -> Path:
    """Return the on-disk path to the ``data/`` package directory."""
    ref = importlib.resources.files("cc_session_toolkit") / "data"
    # files() returns a Traversable; for on-disk packages this is a Path
    return Path(str(ref))


def _copy_if_missing(src: Path, dest: Path, *, label: str) -> None:
    """Copy *src* to *dest* if *dest* does not already exist."""
    if dest.exists():
        print(f"  [skip] {label} (already exists)")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        print(f"  [create] {label}")


def _render_template(src: Path, dest: Path, context: dict[str, str],
                     *, label: str, force: bool = False) -> None:
    """
    Read *src*, replace ``{{key}}`` placeholders, write to *dest*.

    Skips if *dest* exists unless *force* is True.
    """
    if dest.exists() and not force:
        print(f"  [skip] {label} (already exists)")
        return

    content = src.read_text(encoding="utf-8")
    for key, value in context.items():
        content = content.replace(f"{{{{{key}}}}}", value)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    action = "update" if force else "create"
    print(f"  [{action}] {label}")


# -------------------------------------------------------------------------
# Main init function
# -------------------------------------------------------------------------

def initialise_project(
    project_root: Path | None = None,
    project_name: str | None = None,
    *,
    update: bool = False,
    reflections_dir: str = "wiki/reflections",
) -> None:
    """
    Scaffold project files for cc-session-toolkit.

    Creates:

    1. ``archive/cc-sessions/queries/`` with LLM extraction prompts
    2. ``archive/cc-sessions/archive-defaults.yaml`` from template
    3. Reflection document directory with stub templates
    4. ``working-notes.md`` as a *sibling* of the reflections directory
    5. ``.claude/skills/reflect/SKILL.md`` from protocol template

    ``working-notes.md`` is the research-notes layer (owned by ``/observe``)
    and is deliberately placed *beside* the reflections directory, never
    inside it: ``reflections/`` is the meta-research layer (owned by
    ``/reflect``). A historical version of this scaffold shipped the
    template inside ``data/reflections/``, so it was copied into the
    ``reflections/`` directory and the misplacement regenerated in every
    newly-scaffolded project. The template now lives at
    ``data/working-notes.md`` and is placed at the parent of
    ``reflections_dir``.

    With ``--update``, regenerates SKILL.md and query prompts from the
    current package version without touching reflection content or
    archive data.

    Args:
        project_root: Explicit project root.  Defaults to auto-detection.
        project_name: Explicit project name.  Defaults to auto-detection.
        update: If True, overwrite SKILL.md and queries (not content).
        reflections_dir: Relative path for reflection documents
            (default: ``wiki/reflections`` — the canonical four-artefact
            wiki layout; working-notes lands at its parent, ``wiki/``).
    """
    root = project_root or find_project_root()
    name = project_name or get_project_name(root)
    data = _pkg_data_path()

    context = {
        "project_name": name,
        "reflections_dir": reflections_dir,
    }

    print(f"Initialising cc-session for project: {name}")
    print(f"  Root: {root}")
    print()

    # 1. Archive directory and queries
    queries_dest = root / "archive" / "cc-sessions" / "queries"
    queries_src = data / "queries"

    if queries_src.exists():
        for src_file in sorted(queries_src.glob("*.md")):
            dest_file = queries_dest / src_file.name
            if update:
                _render_template(
                    src_file, dest_file, context,
                    label=f"queries/{src_file.name}", force=True,
                )
            else:
                _copy_if_missing(
                    src_file, dest_file,
                    label=f"queries/{src_file.name}",
                )

    # 2. Archive defaults
    defaults_src = data / "templates" / "archive-defaults.yaml"
    defaults_dest = root / "archive" / "cc-sessions" / "archive-defaults.yaml"
    _render_template(
        defaults_src, defaults_dest, context,
        label="archive-defaults.yaml",
    )

    # 3. Reflection document stubs
    reflections_dest = root / reflections_dir
    reflections_src = data / "reflections"

    if reflections_src.exists():
        for src_file in sorted(reflections_src.glob("*.md")):
            dest_file = reflections_dest / src_file.name
            # Never overwrite reflection content, even with --update
            _copy_if_missing(
                src_file, dest_file,
                label=f"{reflections_dir}/{src_file.name}",
            )

    # 4. Working notes — sibling of reflections/, NOT inside it.
    #    working-notes.md is the research-notes layer (owned by /observe);
    #    reflections/ is the meta-research layer (owned by /reflect). The
    #    template lives at data/working-notes.md (outside data/reflections/)
    #    precisely so the section-3 loop above cannot sweep it into
    #    reflections_dest. The notes root is the parent of reflections_dir
    #    (e.g. "wiki" when reflections_dir is "wiki/reflections").
    notes_dir = (root / reflections_dir).parent
    working_notes_src = data / "working-notes.md"
    if working_notes_src.exists():
        _copy_if_missing(
            working_notes_src,
            notes_dir / "working-notes.md",
            label=str(
                (Path(reflections_dir).parent / "working-notes.md")
            ),
        )

    # 5. SKILL.md
    skill_src = data / "templates" / "reflect-skill.md"
    skill_dest = root / ".claude" / "skills" / "reflect" / "SKILL.md"
    _render_template(
        skill_src, skill_dest, context,
        label=".claude/skills/reflect/SKILL.md",
        force=update,
    )

    print()
    print("Done. Project is ready for cc-session archiving and reflection.")
    if not update:
        print(
            f"\nReflection documents: {root / reflections_dir}")
        print(
            f"Working notes: {notes_dir / 'working-notes.md'}")
        print(
            f"Archive queries: {queries_dest}")
        print(
            f"SKILL.md: {skill_dest}")
