"""Quiet helpers for optional Git metadata collection."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Sequence


def git_repository_root(path: str | Path) -> Path | None:
    """Return the containing Git worktree root, or None outside a worktree."""
    try:
        output = subprocess.check_output(
            ['git', '-C', str(Path(path).resolve()), 'rev-parse', '--show-toplevel'],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return None
    return Path(output) if output else None


def git_output(command: Sequence[str], repository_root: Path | None) -> str | None:
    """Run a Git command only when a containing repository was found."""
    if repository_root is None:
        return None
    try:
        return subprocess.check_output(
            command,
            cwd=repository_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return None
