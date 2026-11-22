"""Discover xBD image pairs, audit them and write a JSONL manifest.

Expected package layout, to be confirmed on the real download::

    <raw_dir>/**/labels/<event>_<index>_<pre|post>_disaster.json
    <raw_dir>/**/images/<event>_<index>_<pre|post>_disaster.png
    <raw_dir>/**/targets/<event>_<index>_<pre|post>_disaster_target.png  (optional)

Nothing is skipped silently: every label file becomes a manifest record or is
listed as unrecognized, and every image without labels is listed in the audit.
"""

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from aeris.data.grouping import assign_groups
from aeris.data.labels import DAMAGE_LABELS, LabelError, LabelFile, Phase, parse_label
from aeris.data.raster import rasterize_damage, rasterize_localization
from aeris.data.targets import compare_target, target_path

MANIFEST_SCHEMA_VERSION = 1
PHASES: tuple[Phase, Phase] = ("pre", "post")
STEM_PATTERN = re.compile(r"^(?P<event>.+)_(?P<index>\d+)_(?P<phase>pre|post)_disaster$")
METADATA_FIELDS = (
    "disaster",
    "disaster_type",
    "catalog_id",
    "capture_date",
    "sensor",
    "gsd",
    "off_nadir_angle",
    "sun_elevation",
    "sun_azimuth",
    "width",
    "height",
)

VALID = "valid"
WARNING = "warning"
INVALID = "invalid"
STATUSES = (VALID, WARNING, INVALID)

# Sample-level issue codes that make a sample unusable.
DUPLICATE_SAMPLE_ID = "duplicate_sample_id"
IMAGE_SIZE_MISMATCH = "image_size_mismatch"
# Warnings. Label-file issues are recorded as "<phase>:<code>".
EVENT_TYPE_MISMATCH = "event_type_mismatch"
PRE_POST_UID_MISMATCH = "pre_post_uid_mismatch"
PRE_POST_GEOMETRY_MISMATCH = "pre_post_geometry_mismatch"

Record = dict[str, Any]


class ManifestVerificationError(ValueError):
    """The saved audit does not describe the current source files."""


@dataclass(slots=True)
class _SampleFiles:
    sample_id: str
    event_id: str
    labels_dir: Path
    labels: dict[str, Path] = field(default_factory=dict)

    def image(self, phase: Phase) -> Path:
        return self.labels_dir.parent / "images" / f"{self.sample_id}_{phase}_disaster.png"


@dataclass(frozen=True, slots=True)
class AuditResult:
    records: list[Record]
    summary: dict[str, Any]


def build_manifest(
    raw_dir: Path,
    source_package: str,
    *,
    compute_hashes: bool = True,
    check_targets: bool = False,
) -> AuditResult:
    """Audit every image pair under ``raw_dir``. Records are sorted by sample ID."""
    samples, unrecognized, unlabeled_images = _discover(raw_dir)
    sample_id_counts = Counter(sample.sample_id for sample in samples)
    records = [
        _audit_sample(
            sample,
            raw_dir,
            source_package,
            duplicate=sample_id_counts[sample.sample_id] > 1,
            compute_hashes=compute_hashes,
            check_targets=check_targets,
        )
        for sample in samples
    ]
    assign_groups(records)
    summary = _summarize(
        records,
        raw_dir,
        source_package,
        [_relative(path, raw_dir) for path in unrecognized],
        [_relative(path, raw_dir) for path in unlabeled_images],
        check_targets=check_targets,
    )
    return AuditResult(records, summary)


def _encode_manifest(result: AuditResult, manifest_path: Path) -> tuple[bytes, dict[str, Any]]:
    manifest = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in result.records
    ).encode("utf-8")
    summary = {
        **result.summary,
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }
    return manifest, summary


def write_manifest(result: AuditResult, manifest_path: Path, audit_path: Path) -> None:
    """Write the JSONL manifest and the audit summary, which records the manifest hash."""
    manifest, summary = _encode_manifest(result, manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(manifest)
    audit_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def verify_manifest(
    raw_dir: Path, source_package: str, manifest_path: Path, audit_path: Path
) -> dict[str, Any]:
    """Re-audit the raw package and compare the saved manifest and report."""
    if not raw_dir.is_dir():
        raise ManifestVerificationError(f"Veri dizini bulunamadı — {raw_dir}")
    stored_manifest = manifest_path.read_bytes()
    try:
        stored_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ManifestVerificationError(f"Denetim raporu okunamadı — {audit_path}") from error

    if not isinstance(stored_audit, dict) or stored_audit.get("target_check") is None:
        raise ManifestVerificationError("Maske denetimi eksik. data audit --check-targets çalıştır.")
    if stored_audit.get("manifest_sha256") != hashlib.sha256(stored_manifest).hexdigest():
        raise ManifestVerificationError("Denetim raporundaki manifest SHA-256 değeri uyuşmuyor.")

    fresh = build_manifest(raw_dir, source_package, compute_hashes=True, check_targets=True)
    if not fresh.records or fresh.summary["status_counts"][INVALID] == len(fresh.records):
        raise ManifestVerificationError("Veri dizininde kullanılabilir xBD görüntü çifti yok.")
    if source_package == "challenge_train":
        _check_training_package(fresh.records)

    manifest, audit = _encode_manifest(fresh, manifest_path)
    if stored_manifest != manifest:
        raise ManifestVerificationError(
            "Ham veri veya dosya hash’leri kaydedilen manifestle uyuşmuyor. "
            "Yeni bir denetim çıktısı üret."
        )
    if stored_audit != audit:
        raise ManifestVerificationError("Denetim raporu güncel kaynak dosyalarıyla uyuşmuyor.")
    return audit


def _check_training_package(records: Sequence[Record]) -> None:
    for record in records:
        for field in ("pre_image_path", "post_image_path", "pre_label_path", "post_label_path"):
            path = record[field]
            if path and {"test", "hold", "holdout"} & set(Path(path).parts[:-1]):
                raise ManifestVerificationError(f"Challenge test veya holdout dosyası bulundu — {path}")


def read_manifest(path: Path) -> list[Record]:
    with path.open(encoding="utf-8") as lines:
        return parse_manifest(lines)


def parse_manifest(lines: Iterable[str]) -> list[Record]:
    return [json.loads(line) for line in lines if line.strip()]


def _discover(raw_dir: Path) -> tuple[list[_SampleFiles], list[Path], list[Path]]:
    samples: dict[tuple[Path, str], _SampleFiles] = {}
    unrecognized: list[Path] = []
    for path in sorted(raw_dir.rglob("*.json")):
        if path.parent.name != "labels":
            continue
        match = STEM_PATTERN.match(path.stem)
        if match is None:
            unrecognized.append(path)
            continue
        sample_id = f"{match['event']}_{match['index']}"
        sample = samples.setdefault(
            (path.parent, sample_id), _SampleFiles(sample_id, match["event"], path.parent)
        )
        sample.labels[match["phase"]] = path

    expected_images = {sample.image(phase) for sample in samples.values() for phase in PHASES}
    unlabeled_images = [
        path
        for path in sorted(raw_dir.rglob("*.png"))
        if path.parent.name == "images" and path not in expected_images
    ]
    ordered = sorted(samples.values(), key=lambda s: (s.sample_id, s.labels_dir.as_posix()))
    return ordered, unrecognized, unlabeled_images


def _audit_sample(
    sample: _SampleFiles,
    raw_dir: Path,
    source_package: str,
    *,
    duplicate: bool,
    compute_hashes: bool,
    check_targets: bool,
) -> Record:
    issues: Counter[str] = Counter()
    fatal: set[str] = set()

    def fail(code: str) -> None:
        issues[code] += 1
        fatal.add(code)

    if duplicate:
        fail(DUPLICATE_SAMPLE_ID)

    sizes: dict[str, tuple[int, int]] = {}
    for phase in PHASES:
        if phase not in sample.labels:
            fail(f"missing_{phase}_label")
        image = sample.image(phase)
        if not image.is_file():
            fail(f"missing_{phase}_image")
            continue
        try:
            with Image.open(image) as opened:
                sizes[phase], mode = opened.size, opened.mode
        except OSError:
            fail(f"unreadable_{phase}_image")
            continue
        if mode != "RGB":
            issues[f"{phase}:non_rgb_image"] += 1
    if len(set(sizes.values())) > 1:
        fail(IMAGE_SIZE_MISMATCH)
    size = sizes.get("pre") or sizes.get("post")

    labels: dict[str, LabelFile] = {}
    for phase in PHASES:
        if phase not in sample.labels:
            continue
        try:
            labels[phase] = parse_label(sample.labels[phase], phase, sizes.get(phase))
        except LabelError as error:
            fail(f"{phase}:{error.code}")
            continue
        for code, count in labels[phase].issue_counts().items():
            issues[f"{phase}:{code}"] += count

    _check_consistency(sample, labels, sizes, issues)
    overlap_pixels: dict[str, int] = {}
    target_check: dict[str, Any] | None = {} if check_targets else None
    for phase, label in labels.items():
        if phase not in sizes:
            continue
        if phase == "pre":
            raster = rasterize_localization(label.buildings, *sizes[phase])
        else:
            raster = rasterize_damage(label.buildings, *sizes[phase])
        if raster.empty_buildings:
            issues[f"{phase}:no_pixel_building"] += len(raster.empty_buildings)
        overlap_pixels[phase] = raster.overlap_pixels
        if target_check is not None:
            target_check[phase] = _check_target(
                raster.mask, sample.image(phase), binary=phase == "pre", issues=issues
            )

    post = labels.get("post")
    metadata = {phase: _metadata(labels[phase].metadata) for phase in PHASES if phase in labels}
    event_type = next(
        (m["disaster_type"] for m in reversed(metadata.values()) if m["disaster_type"]), None
    )
    if fatal:
        status = INVALID
    elif issues:
        status = WARNING
    else:
        status = VALID

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "sample_id": sample.sample_id,
        "event_id": sample.event_id,
        "event_type": event_type,
        "scene_id": None,
        "scene_id_source": None,
        "spatial_group_id": None,
        "spatial_group_source": None,
        "source_package": source_package,
        "pre_image_path": _relative(sample.image("pre"), raw_dir),
        "post_image_path": _relative(sample.image("post"), raw_dir),
        "pre_label_path": _optional_relative(sample.labels.get("pre"), raw_dir),
        "post_label_path": _optional_relative(sample.labels.get("post"), raw_dir),
        "width": size[0] if size else None,
        "height": size[1] if size else None,
        "building_count": {
            phase: len(labels[phase].buildings) if phase in labels else None for phase in PHASES
        },
        "damage_class_counts": _damage_counts(post) if post is not None else None,
        "lng_lat_bounds": _union_bounds(label.lng_lat_bounds for label in labels.values()),
        "metadata": metadata,
        "overlap_pixels": overlap_pixels,
        "target_check": target_check,
        "sha256": _hashes(sample, include_targets=check_targets) if compute_hashes else None,
        "validation_status": status,
        "issues": dict(sorted(issues.items())),
    }


def _check_consistency(
    sample: _SampleFiles,
    labels: Mapping[str, LabelFile],
    sizes: Mapping[str, tuple[int, int]],
    issues: Counter[str],
) -> None:
    for phase, label in labels.items():
        width, height = label.metadata.get("width"), label.metadata.get("height")
        if phase in sizes and width is not None and height is not None:
            if (width, height) != sizes[phase]:
                issues[f"{phase}:metadata_size_mismatch"] += 1
        disaster = label.metadata.get("disaster")
        if disaster is not None and disaster != sample.event_id:
            issues[f"{phase}:event_name_mismatch"] += 1

    event_types = {label.metadata.get("disaster_type") for label in labels.values()}
    if len(event_types - {None}) > 1:
        issues[EVENT_TYPE_MISMATCH] += 1

    if "pre" not in labels or "post" not in labels:
        return
    pre = {b.uid: b.polygon for b in labels["pre"].buildings if b.uid is not None}
    post = {b.uid: b.polygon for b in labels["post"].buildings if b.uid is not None}
    if pre.keys() != post.keys():
        issues[PRE_POST_UID_MISMATCH] += len(pre.keys() ^ post.keys())
    changed = sum(
        not pre[uid].equals_exact(post[uid], 1e-6) for uid in pre.keys() & post.keys()
    )
    if changed:
        issues[PRE_POST_GEOMETRY_MISMATCH] += changed


def _check_target(
    mask: Any, image: Path, *, binary: bool, issues: Counter[str]
) -> dict[str, Any] | str:
    phase = "pre" if binary else "post"
    path = target_path(image)
    if not path.is_file():
        return "missing"
    try:
        comparison = compare_target(mask, path, binary=binary)
    except (OSError, ValueError):
        issues[f"{phase}:unreadable_target"] += 1
        return "unreadable"
    return {
        "compared_pixels": comparison.compared_pixels,
        "mismatched_pixels": comparison.mismatched_pixels,
        "ignore_pixels": comparison.ignore_pixels,
        "target_values_on_ignore": {
            str(value): count for value, count in comparison.target_values_on_ignore.items()
        },
    }


def _metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    return {name: metadata.get(name) for name in METADATA_FIELDS}


def _damage_counts(label: LabelFile) -> dict[str, int]:
    counts = label.damage_label_counts()
    return {name: counts[name] for name in DAMAGE_LABELS}


def _union_bounds(
    bounds: Iterable[tuple[float, float, float, float] | None],
) -> list[float] | None:
    present = [b for b in bounds if b is not None]
    if not present:
        return None
    return [
        min(b[0] for b in present),
        min(b[1] for b in present),
        max(b[2] for b in present),
        max(b[3] for b in present),
    ]


def _hashes(sample: _SampleFiles, *, include_targets: bool) -> dict[str, str | None]:
    files = {
        "pre_image": sample.image("pre"),
        "post_image": sample.image("post"),
        "pre_label": sample.labels.get("pre"),
        "post_label": sample.labels.get("post"),
    }
    if include_targets:
        files.update({f"{phase}_target": target_path(sample.image(phase)) for phase in PHASES})
    return {name: _sha256(path) for name, path in files.items()}


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _optional_relative(path: Path | None, root: Path) -> str | None:
    return None if path is None else _relative(path, root)


def _summarize(
    records: Sequence[Record],
    raw_dir: Path,
    source_package: str,
    unrecognized: list[str],
    unlabeled_images: list[str],
    *,
    check_targets: bool,
) -> dict[str, Any]:
    issue_totals: Counter[str] = Counter()
    issue_samples: Counter[str] = Counter()
    for record in records:
        issue_totals.update(record["issues"])
        issue_samples.update(record["issues"].keys())

    by_event: defaultdict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_event[record["event_id"]].append(record)

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_package": source_package,
        "raw_dir": raw_dir.as_posix(),
        "samples": len(records),
        "status_counts": _status_counts(records),
        "damage_label_counts": _usable_damage_counts(records),
        "events": {event: _event_summary(rs) for event, rs in sorted(by_event.items())},
        "issues": {
            code: {"occurrences": issue_totals[code], "samples": issue_samples[code]}
            for code in sorted(issue_totals)
        },
        "metadata_availability": _metadata_availability(records),
        "grouping": _grouping_summary(records),
        "overlap_pixels": {
            phase: sum(r["overlap_pixels"].get(phase, 0) for r in records) for phase in PHASES
        },
        "target_check": _target_summary(records) if check_targets else None,
        "unrecognized_label_files": _listing(unrecognized),
        "images_without_labels": _listing(unlabeled_images),
    }


def _status_counts(records: Sequence[Record]) -> dict[str, int]:
    counts = Counter(record["validation_status"] for record in records)
    return {status: counts[status] for status in STATUSES}


def _usable_damage_counts(records: Sequence[Record]) -> dict[str, int]:
    """Building damage labels in samples that are not invalid."""
    counts: Counter[str] = Counter()
    for record in records:
        if record["validation_status"] != INVALID and record["damage_class_counts"]:
            counts.update(record["damage_class_counts"])
    return {name: counts[name] for name in DAMAGE_LABELS}


def _event_summary(records: Sequence[Record]) -> dict[str, Any]:
    scenes = Counter(record["scene_id"] for record in records)
    spatial_groups = Counter(record["spatial_group_id"] for record in records)
    return {
        "event_types": sorted({r["event_type"] for r in records if r["event_type"]}),
        "samples": len(records),
        "status_counts": _status_counts(records),
        "pre_buildings": sum(
            r["building_count"]["pre"] or 0 for r in records if r["validation_status"] != INVALID
        ),
        "damage_label_counts": _usable_damage_counts(records),
        "scenes": len(scenes),
        "largest_scene_samples": max(scenes.values()),
        "spatial_groups": len(spatial_groups),
        "largest_spatial_group_samples": max(spatial_groups.values()),
    }


def _metadata_availability(records: Sequence[Record]) -> dict[str, dict[str, int]]:
    availability: dict[str, dict[str, int]] = {}
    for phase in PHASES:
        parsed = [r["metadata"][phase] for r in records if phase in r["metadata"]]
        availability[phase] = {
            "labels": len(parsed),
            **{name: sum(m[name] is not None for m in parsed) for name in METADATA_FIELDS},
        }
    return availability


def _grouping_summary(records: Sequence[Record]) -> dict[str, Any]:
    def cross_event(key: str) -> list[dict[str, Any]]:
        events: defaultdict[str, set[str]] = defaultdict(set)
        for record in records:
            events[record[key]].add(record["event_id"])
        return [
            {"group": group, "events": sorted(names)}
            for group, names in sorted(events.items())
            if len(names) > 1
        ]

    return {
        "scene_rule": "union of pre and post catalog_id",
        "spatial_rule": "union of scenes whose lng_lat bounds overlap or share an edge",
        "scenes": len({r["scene_id"] for r in records}),
        "samples_without_catalog_id": sum(r["scene_id_source"] == "sample" for r in records),
        "spatial_groups": len({r["spatial_group_id"] for r in records}),
        "samples_without_lng_lat": sum(r["lng_lat_bounds"] is None for r in records),
        "cross_event_scenes": cross_event("scene_id"),
        "cross_event_spatial_groups": cross_event("spatial_group_id"),
    }


def _target_summary(records: Sequence[Record]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for phase in PHASES:
        checks = [
            r["target_check"][phase]
            for r in records
            if r["target_check"] and isinstance(r["target_check"].get(phase), dict)
        ]
        statuses = Counter(
            r["target_check"].get(phase, "not_checked")
            for r in records
            if r["target_check"] is not None
            and not isinstance(r["target_check"].get(phase), dict)
        )
        values_on_ignore: Counter[str] = Counter()
        for check in checks:
            values_on_ignore.update(check["target_values_on_ignore"])
        compared = sum(c["compared_pixels"] for c in checks)
        mismatched = sum(c["mismatched_pixels"] for c in checks)
        summary[phase] = {
            "samples_compared": len(checks),
            "samples_exact": sum(c["mismatched_pixels"] == 0 for c in checks),
            "compared_pixels": compared,
            "mismatched_pixels": mismatched,
            "mismatch_fraction": mismatched / compared if compared else None,
            "max_sample_mismatch_fraction": max(
                (c["mismatched_pixels"] / c["compared_pixels"] for c in checks if c["compared_pixels"]),
                default=None,
            ),
            "target_values_on_ignore": dict(sorted(values_on_ignore.items())),
            "not_compared": dict(sorted(statuses.items())),
        }
    return summary


def _listing(paths: list[str], limit: int = 20) -> dict[str, Any]:
    return {"count": len(paths), "examples": paths[:limit]}
