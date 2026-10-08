"""Shared BLED112 path resolution and process-level transport locking."""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


class PortBusyError(RuntimeError):
    """Raised when another process already owns the BLED112 transport."""


def tty_is_in_use(tty: str) -> bool:
    """Best-effort check for older tools that do not use the lock file."""
    check_path = os.path.realpath(tty)
    try:
        result = subprocess.run(
            ["fuser", "-s", check_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def normalize_mac(text: str) -> str:
    """Validate and normalize a Myo MAC in either colon or dash notation."""
    value = str(text).strip().upper().replace("-", ":")
    if value.startswith("<") or value.endswith(">"):
        raise ValueError("replace the MAC placeholder with six hexadecimal octets")
    if not re.fullmatch(r"[0-9A-F]{2}(?::[0-9A-F]{2}){5}", value):
        raise ValueError("MAC must contain exactly six hexadecimal octets")
    return value


def resolve_bled112_tty(configured: str, *, by_id_dir: str = "/dev/serial/by-id") -> str:
    """Return a usable BLED112 path, tolerating ttyACM renumbering.

    An explicit non-numeric path always wins when it exists. For a numeric
    ``ttyACM``/``ttyUSB`` path, prefer a unique Bluegiga by-id alias so USB
    renumbering does not change the transport. Choosing among multiple
    dongles is intentionally refused.
    """
    configured_path = Path(configured).expanduser()
    # A ttyACM/ttyUSB number is not stable across USB re-enumeration. Prefer
    # the unique Bluegiga udev alias even when that numeric node still exists.
    numeric_tty = configured_path.name.startswith(("ttyACM", "ttyUSB"))

    by_id = Path(by_id_dir).expanduser()
    if not by_id.is_dir():
        return str(configured_path)
    candidates = sorted(
        path
        for path in by_id.glob("usb-Bluegiga_Low_Energy_Dongle*-if*")
        if path.exists()
    )
    # Some udev rules expose more than one interface symlink for a single
    # dongle. Treat those aliases as one candidate, but never collapse two
    # distinct physical dongles.
    unique_targets: dict[str, Path] = {}
    for path in candidates:
        unique_targets.setdefault(os.path.realpath(path), path)
    candidates = sorted(unique_targets.values())
    if len(candidates) == 1 and (numeric_tty or not configured_path.exists()):
        return str(candidates[0])
    if configured_path.exists():
        return str(configured_path)
    return str(configured_path)


def lock_path(lock_name: str = "bled112", lock_dir: str | None = None) -> Path:
    """Build a stable lock path shared by tty aliases and reconnects."""
    directory = Path(lock_dir or os.environ.get("WUJI_MYO_LOCK_DIR", "/tmp"))
    # Keep the filename safe even when a caller supplies a custom label.
    digest = hashlib.sha256(lock_name.encode("utf-8")).hexdigest()[:16]
    return directory / f"wuji-myo-{digest}.lock"


def lock_owner_description() -> str:
    """Return best-effort metadata for the process holding the lock."""
    try:
        text = lock_path().read_text(encoding="utf-8").strip().replace("\n", ", ")
    except OSError:
        return "unknown owner"
    return text or "unknown owner"


@contextmanager
def acquire_bled112_lock(timeout_s: float = 0.0) -> Iterator[TextIO]:
    """Hold the BLED112 lock for the lifetime of a transport operation."""
    if timeout_s < 0:
        raise ValueError("timeout_s must be non-negative")
    # Use one host-wide BLED112 lock. The device path can change from ttyACM0
    # to ttyACM1 during re-enumeration, so a path-derived lock would allow two
    # processes to overlap during exactly the failure we need to recover from.
    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    deadline = time.monotonic() + timeout_s
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                handle.seek(0)
                handle.truncate()
                cmdline = ""
                try:
                    raw_cmdline = Path(f"/proc/{os.getpid()}/cmdline").read_bytes()
                    cmdline = " ".join(part.decode(errors="replace") for part in raw_cmdline.split(b"\0") if part)
                except OSError:
                    pass
                handle.write(f"pid={os.getpid()}\ncmd={cmdline or 'unknown'}\n")
                handle.flush()
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise PortBusyError(
                        "BLED112 is already in use by another process; "
                        f"stop the collector/scan before retrying ({lock_owner_description()})"
                    ) from None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise PortBusyError(
                        "BLED112 is already in use by another process; "
                        f"stop the collector/scan before retrying ({lock_owner_description()})"
                    ) from None
                time.sleep(min(0.05, remaining))
        yield handle
    finally:
        if acquired:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
