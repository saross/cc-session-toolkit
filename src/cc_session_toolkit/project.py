"""
Project root detection and path utilities.

All project root detection uses CWD-upward marker search instead of
``__file__``-based paths, so functions work correctly when the package
is pip-installed rather than run as a local script.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any


# Markers that indicate a project root directory (checked in order)
PROJECT_ROOT_MARKERS = (".git", "CLAUDE.md", "pyproject.toml")

# Standard Claude Code projects directory
CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"


def find_project_root(start: Path | None = None) -> Path:
    """
    Search upward from *start* (default: CWD) for a project root directory.

    A directory is considered a project root if it contains any of the
    marker files/directories: ``.git/``, ``CLAUDE.md``, or
    ``pyproject.toml``.

    Args:
        start: Directory to begin searching from.  Defaults to the
            current working directory.

    Returns:
        The first ancestor directory (inclusive) containing a marker.

    Raises:
        FileNotFoundError: If no marker is found before reaching the
            filesystem root.
    """
    current = (start or Path.cwd()).resolve()

    while True:
        for marker in PROJECT_ROOT_MARKERS:
            if (current / marker).exists():
                return current

        parent = current.parent
        if parent == current:
            # Reached filesystem root without finding a marker
            raise FileNotFoundError(
                f"No project root found (searched upward from {start or Path.cwd()} "
                f"for markers: {', '.join(PROJECT_ROOT_MARKERS)})"
            )
        current = parent


def get_project_name(project_root: Path | None = None) -> str:
    """
    Detect the project name using a cascading fallback strategy.

    Priority:
        1. ``CLAUDE.md`` containing a ``# Project: <name>`` line
        2. Git remote origin URL (extracts repository name)
        3. Directory name of the project root

    Args:
        project_root: Explicit project root path.  If *None*,
            :func:`find_project_root` is called.

    Returns:
        Human-readable project name string.
    """
    root = project_root or find_project_root()

    # 1. Try CLAUDE.md
    claude_md = root / "CLAUDE.md"
    if claude_md.exists():
        content = claude_md.read_text(encoding="utf-8")
        match = re.search(
            r"^#\s*Project:\s*(.+)$", content, re.MULTILINE | re.IGNORECASE
        )
        if match:
            return match.group(1).strip()

    # 2. Try git remote
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True,
        )
        remote_url = result.stdout.strip()
        # Handles both SSH (git@host:owner/repo.git) and HTTPS URLs
        match = re.search(r"[/:]([^/]+)/([^/]+?)(?:\.git)?$", remote_url)
        if match:
            return match.group(2)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # 3. Fallback to directory name
    return root.name


def get_cc_project_path(project_root: Path | None = None) -> str:
    """
    Return the path-encoded project identifier used by Claude Code.

    Claude Code encodes project paths by replacing ``/`` with ``-``,
    e.g. ``/home/shawn/Code/map-reader-llm`` becomes
    ``-home-shawn-Code-map-reader-llm``.

    Args:
        project_root: Explicit project root.  Defaults to auto-detection.

    Returns:
        Dash-separated path string used as directory name under
        ``~/.claude/projects/``.
    """
    root = (project_root or find_project_root()).resolve()
    return str(root).replace("/", "-")


def get_archive_dir(project_root: Path | None = None) -> Path:
    """
    Return the archive directory for the current project.

    Convention: ``<project_root>/archive/cc-sessions/``

    Args:
        project_root: Explicit project root.  Defaults to auto-detection.

    Returns:
        Path to the ``archive/cc-sessions/`` directory (may not exist yet).
    """
    root = project_root or find_project_root()
    return root / "archive" / "cc-sessions"


def get_defaults_file(project_root: Path | None = None) -> Path:
    """
    Return the path to the project's ``archive-defaults.yaml``.

    Args:
        project_root: Explicit project root.  Defaults to auto-detection.

    Returns:
        Path to the defaults file (may not exist).
    """
    return get_archive_dir(project_root) / "archive-defaults.yaml"


def get_catalogue_file(project_root: Path | None = None) -> Path:
    """
    Return the path to the project's ``CATALOG.json``.

    Args:
        project_root: Explicit project root.  Defaults to auto-detection.

    Returns:
        Path to the catalogue file (may not exist).
    """
    return get_archive_dir(project_root) / "CATALOG.json"
