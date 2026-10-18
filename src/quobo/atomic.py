"""Atomic filesystem publish primitives.

Two semantics, one shared mechanism (temp file -> fsync -> publish):

- ``publish_replace``: overwrites the destination (for checkpoint updates that
  legitimately replace a prior partial state with a more-complete one).
- ``publish_exclusive``: fails if the destination already exists (for
  experiment results and caches that must never silently overwrite a valid
  prior artifact).

Both serialize content to bytes *before* touching the filesystem, so a crash
or serialization error cannot leave a truncated final file. Temp litter is
cleaned by the caller's ``finally``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def publish_replace(dest: Path, payload: bytes, *, suffix: str = ".tmp") -> None:
    """Atomically overwrite ``dest`` with ``payload`` via temp + os.replace.

    The destination's parent is created. A crash after the temp write but
    before ``os.replace`` leaves the prior destination intact and the temp
    file as litter (cleaned by the caller).
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=dest.parent, prefix=f".{dest.name}.", suffix=suffix, delete=False
        ) as stream:
            temp = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, dest)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def publish_exclusive(dest: Path, payload: bytes, *, suffix: str = ".tmp") -> None:
    """Atomically create ``dest`` with ``payload``; fail if it already exists.

    Uses a hard link so the publish is exclusive — another writer that already
    published the same path causes ``FileExistsError`` rather than a silent
    overwrite. Raises ``FileExistsError`` if ``dest`` exists.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=dest.parent, prefix=f".{dest.name}.", suffix=suffix, delete=False
        ) as stream:
            temp = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, dest)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
