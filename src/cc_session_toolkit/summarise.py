"""
Session analysis for metadata population (from llm-reproducibility v1.2).

Analyses an archived session to extract key information that helps
generate metadata (title, purpose, tags, three_ps).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path


def summarise_session(archive_dir: Path) -> None:
    """
    Extract summary information from an archived session.

    Prints first/last user messages, tools used, and files modified to
    help generate metadata interactively.

    Args:
        archive_dir: Path to the specific session archive directory
            (e.g. ``archive/cc-sessions/project/2026-01-15T10-30_xyz/``).
    """
    # Find the session file (jsonl or jsonl.gz)
    session_file = archive_dir / "session.jsonl"
    if not session_file.exists():
        session_file = archive_dir / "session.jsonl.gz"
        if not session_file.exists():
            print(f"Session file not found in: {archive_dir}")
            return

    print(f"Analysing: {archive_dir.name}")
    print("=" * 60)

    # Read lines
    if session_file.suffix == ".gz":
        with gzip.open(session_file, "rt", encoding="utf-8") as fh:
            lines = fh.readlines()
    else:
        lines = session_file.read_text(encoding="utf-8").splitlines()

    # Extract key information
    first_human_msg: str | None = None
    last_human_msg: str | None = None
    human_messages: list[str] = []
    tool_types: set[str] = set()
    files_modified: set[str] = set()

    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            msg_type = entry.get("type")

            if msg_type == "user" and entry.get("userType") == "external":
                message = entry.get("message", {})
                content = (
                    message.get("content", "")
                    if isinstance(message, dict)
                    else ""
                )

                if isinstance(content, str) and content.strip():
                    if first_human_msg is None:
                        first_human_msg = content[:500]
                    last_human_msg = content[:500]
                    human_messages.append(content[:200])
                elif isinstance(content, list):
                    for block in content:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "text"
                        ):
                            text = block.get("text", "")
                            if text.strip():
                                if first_human_msg is None:
                                    first_human_msg = text[:500]
                                last_human_msg = text[:500]
                                human_messages.append(text[:200])
                                break

            elif msg_type == "assistant":
                message = entry.get("message", {})
                for content_block in message.get("content", []):
                    if content_block.get("type") == "tool_use":
                        tool_name = content_block.get("name", "")
                        tool_types.add(tool_name)
                        if tool_name in ("Edit", "Write"):
                            inp = content_block.get("input", {})
                            if "file_path" in inp:
                                files_modified.add(inp["file_path"])

        except json.JSONDecodeError:
            continue

    # Print summary
    print(f"\n**First user message:**\n{first_human_msg}\n")
    print(f"**Last user message:**\n{last_human_msg}\n")
    print(
        f"**Tools used:** "
        f"{', '.join(sorted(tool_types)) or 'None'}"
    )
    print(f"**Files modified:** {len(files_modified)}")
    if files_modified and len(files_modified) <= 20:
        for f in sorted(files_modified)[:20]:
            print(f"  - {f}")

    # Load existing metadata for context
    meta_file = archive_dir / "session.meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        stats = meta.get("statistics", {})
        print("\n**Statistics:**")
        print(f"  Turns: {stats.get('turns', 'N/A')}")
        print(
            f"  Duration: "
            f"{meta.get('session', {}).get('duration_minutes', 'N/A')} "
            f"minutes"
        )
        print(
            f"  Tool calls: "
            f"{stats.get('tool_calls', {}).get('total', 'N/A')}"
        )

    # Suggest metadata template
    print("\n" + "=" * 60)
    print("**Suggested metadata template:**")
    print("=" * 60)
    print("""
{
  "title": "[TODO: Brief title based on above]",
  "purpose": "[TODO: 1-2 sentence purpose]",
  "tags": ["TODO"],
  "three_ps": {
    "prompt_summary": "[TODO: What was asked]",
    "process_summary": "[TODO: How the tool was used]",
    "provenance_summary": "[TODO: Role in research workflow]"
  }
}
""")
