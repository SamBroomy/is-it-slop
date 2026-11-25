"""A memory watchdog: stop the pipeline if it uses too much RAM, instead of taking the machine down.

`ulimit -v` is not a good guard for this workload: it limits *virtual* address space, and a
multi-threaded Rust process reserves far more of that than it ever uses. The kernel OOM killer is
worse: on WSL it can take the whole VM and every tool running in it with it. This watches the
process's anonymous memory (what an OOM kill is about) and exits cleanly well before that.
Memory-mapped file pages, such as Arrow datasets read from disk, are not counted: the kernel can
drop those whenever it needs the space.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path


def anonymous_gb() -> float:
    """Return the anonymous resident memory of this process in GB (Linux; 0.0 elsewhere)."""
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) / 1e6  # kB -> GB
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def watch(limit_gb: float, *, interval: float = 0.5) -> threading.Thread:
    """Start a daemon thread that exits the process (status 137) if anonymous memory passes `limit_gb`."""

    def loop() -> None:
        while True:
            used = anonymous_gb()
            if used > limit_gb:
                sys.stderr.write(f"\ncuration: using {used:.1f} GB, over the {limit_gb:.1f} GB limit; stopping\n")
                sys.stderr.flush()
                os._exit(137)
            time.sleep(interval)

    thread = threading.Thread(target=loop, name="memory-watchdog", daemon=True)
    thread.start()
    return thread
