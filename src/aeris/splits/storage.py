"""Freeze and verify a split against the exact audited manifest snapshot."""

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from aeris.data.manifest import parse_manifest
from aeris.splits.planner import SplitError, build_split_plan

_SCHEMA_VERSION = 1


def plan_from_manifest(
    path: Path,
    fractions: Mapping[str, object],
    *,
    ood_event: str | None = None,
    id_evaluation_groups: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    manifest = path.read_bytes()
    try:
        records = parse_manifest(manifest.decode("utf-8").split("\n"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SplitError(f"{path}: invalid JSONL manifest — {error}") from error
    plan = {
        "schema_version": _SCHEMA_VERSION,
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        **build_split_plan(
            records, fractions, ood_event=ood_event, id_evaluation_groups=id_evaluation_groups
        ),
    }
    canonical = json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {**plan, "split_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}


def freeze_split(path: Path, plan: dict[str, Any]) -> None:
    """Publish once. Repeating the same split is safe and keeps the existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary = NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False)
    pending = Path(temporary.name)
    try:
        with temporary:
            temporary.write(encoded)

        try:
            os.link(pending, path)
        except FileExistsError:
            verify_split(path, plan)
    finally:
        pending.unlink(missing_ok=True)


def verify_split(path: Path, expected: dict[str, Any]) -> None:
    """Check that the stored split is the deterministic result for this manifest."""
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SplitError(f"{path}: invalid frozen split — {error}") from error
    if not isinstance(stored, dict):
        raise SplitError(f"{path}: frozen split must be a JSON object")
    if stored.get("manifest_sha256") != expected["manifest_sha256"]:
        raise SplitError(f"{path}: manifest SHA-256 has changed; use a new split output path")
    if stored != expected:
        raise SplitError(f"{path}: frozen split differs from the current plan; use a new output path")
