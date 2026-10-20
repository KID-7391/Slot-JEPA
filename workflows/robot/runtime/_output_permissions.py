"""Make container-generated planning outputs manageable by the host user."""

from __future__ import annotations

import os
from pathlib import Path


def configure_output_umask() -> int:
    """Apply a group-writable umask inherited by simulator subprocesses."""
    raw = os.environ.get("REALM_OUTPUT_UMASK", "0002").strip()
    try:
        value = int(raw, 8)
    except ValueError as exc:
        raise ValueError(f"REALM_OUTPUT_UMASK must be an octal value, got {raw!r}") from exc
    if not 0 <= value <= 0o777:
        raise ValueError(f"REALM_OUTPUT_UMASK is outside 0000..0777: {raw!r}")
    os.umask(value)
    return value


def _host_ids() -> tuple[int, int] | None:
    raw_uid = os.environ.get("REALM_HOST_UID")
    raw_gid = os.environ.get("REALM_HOST_GID")
    if raw_uid is None or raw_gid is None:
        return None
    try:
        uid = int(raw_uid)
        gid = int(raw_gid)
    except ValueError as exc:
        raise ValueError("REALM_HOST_UID and REALM_HOST_GID must be integers") from exc
    if uid < 0 or gid < 0:
        raise ValueError("REALM_HOST_UID and REALM_HOST_GID must be non-negative")
    return uid, gid


def handoff_output_ownership(path, output_root) -> bool:
    """Recursively chown one completed output, refusing paths outside the output root."""
    ids = _host_ids()
    if ids is None or path in (None, ""):
        return False
    uid, gid = ids
    target = Path(path).resolve()
    root = Path(output_root).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Refusing to chown output outside {root}: {target}") from exc
    if not target.exists():
        return False

    paths = [target, *target.rglob("*")]
    for item in reversed(paths):
        try:
            os.chown(item, uid, gid, follow_symlinks=False)
        except PermissionError as exc:
            # Host-side/non-root planning already creates correctly owned files.
            print(
                f"[output_permissions] warning: could not chown {item} to "
                f"{uid}:{gid}: {exc}",
                flush=True,
            )
            return False
    return True


def handoff_result_outputs(result, output_root) -> list[str]:
    """Find output_dir values in a result payload and hand each one to the host."""
    candidates = []

    def visit(value):
        if isinstance(value, dict):
            output_dir = value.get("output_dir")
            if isinstance(output_dir, str) and output_dir:
                candidates.append(output_dir)
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(result)
    handed_off = []
    for candidate in dict.fromkeys(candidates):
        if handoff_output_ownership(candidate, output_root):
            handed_off.append(str(Path(candidate).resolve()))
    return handed_off
