# -*- coding: utf-8 -*-
"""Private filesystem helpers shared by API and fusion data stores."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def ensure_private_dir(path: str | Path) -> Path:
    """Create a directory and restrict it to the current user on POSIX."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(directory, 0o700)
    except OSError:
        pass
    return directory


def write_private_text(path: str | Path, text: str) -> None:
    """Atomically replace a text file with a mode-0600 temporary file."""
    target = Path(path)
    ensure_private_dir(target.parent)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        os.replace(temporary, target)
        try:
            os.chmod(target, 0o600)
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def append_private_text(path: str | Path, text: str) -> None:
    """Append text using an exclusive mode-0600 file creation."""
    target = Path(path)
    ensure_private_dir(target.parent)
    fd = os.open(target, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        handle = os.fdopen(fd, "a", encoding="utf-8")
    except BaseException:
        os.close(fd)
        raise
    try:
        os.chmod(target, 0o600)
    except OSError:
        pass
    with handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
