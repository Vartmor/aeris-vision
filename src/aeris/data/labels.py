"""Parse xBD/xView2 label JSON files without assuming the schema is clean.

Expected layout, to be confirmed by the data audit on the real package::

    {"features": {"xy": [feature, ...], "lng_lat": [feature, ...]},
     "metadata": {"disaster": ..., "disaster_type": ..., "catalog_id": ..., ...}}

    feature = {"properties": {"feature_type": "building", "uid": ...,
                              "subtype": "no-damage"},  # post-disaster only
               "wkt": "POLYGON ((x y, ...))"}

Every deviation becomes a :class:`LabelIssue` so the audit can count it. Buildings
are never relabeled: a missing, unknown or ``un-classified`` damage subtype maps
to :data:`IGNORE`, never to ``no damage``.
"""

import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import shapely
from shapely import wkt as shapely_wkt
from shapely.errors import ShapelyError
from shapely.geometry import Polygon

Phase = Literal["pre", "post"]

BACKGROUND = 0
IGNORE = 255
DAMAGE_CLASSES: Mapping[str, int] = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
}
DAMAGE_CLASS_NAMES: Mapping[int, str] = {
    1: "no_damage",
    2: "minor_damage",
    3: "major_damage",
    4: "destroyed",
}
UNCLASSIFIED_SUBTYPE = "un-classified"

# Damage label categories counted by the audit, in report order.
DAMAGE_LABELS = (
    *DAMAGE_CLASS_NAMES.values(),
    "un_classified",
    "missing_subtype",
    "unknown_subtype",
)

# Issue codes. Features with a "skipped" issue are not returned as buildings.
UNKNOWN_FEATURE_TYPE = "unknown_feature_type"  # skipped
MISSING_WKT = "missing_wkt"  # skipped
UNPARSEABLE_WKT = "unparseable_wkt"  # skipped
UNSUPPORTED_GEOMETRY = "unsupported_geometry"  # skipped
EMPTY_GEOMETRY = "empty_geometry"  # skipped, zero area
OUTSIDE_IMAGE = "outside_image"  # skipped, no overlap with the image
INVALID_GEOMETRY = "invalid_geometry"  # kept, e.g. self-intersection
OUT_OF_BOUNDS = "out_of_bounds"  # kept, clipped by rasterization
MISSING_UID = "missing_uid"
DUPLICATE_UID = "duplicate_uid"
MISSING_SUBTYPE = "missing_subtype"  # post only, damage ignored
UNKNOWN_SUBTYPE = "unknown_subtype"  # post only, damage ignored
MISSING_METADATA = "missing_metadata"
INVALID_DISASTER_TYPE = "invalid_disaster_type"
INVALID_CATALOG_ID = "invalid_catalog_id"
UNPARSEABLE_LNG_LAT = "unparseable_lng_lat"
LNG_LAT_COUNT_MISMATCH = "lng_lat_count_mismatch"


class LabelError(ValueError):
    """Raised when a label file cannot be used at all."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code


@dataclass(frozen=True, slots=True)
class LabelIssue:
    code: str
    uid: str | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class Building:
    """One building polygon in pixel coordinates (x = column, y = row).

    ``damage_class`` is 1-4 for the four damage classes, :data:`IGNORE` when the
    post-disaster damage label is unusable and ``None`` in pre-disaster labels.
    ``damage_label`` is one of :data:`DAMAGE_LABELS`, or ``None`` before the disaster.
    """

    uid: str | None
    polygon: Polygon
    damage_class: int | None
    damage_label: str | None


@dataclass(frozen=True, slots=True)
class LabelFile:
    phase: Phase
    metadata: Mapping[str, object]
    buildings: tuple[Building, ...]
    issues: tuple[LabelIssue, ...]
    lng_lat_bounds: tuple[float, float, float, float] | None
    """(min longitude, min latitude, max longitude, max latitude) of all buildings."""

    def issue_counts(self) -> Counter[str]:
        return Counter(issue.code for issue in self.issues)

    def damage_label_counts(self) -> Counter[str]:
        return Counter(
            building.damage_label
            for building in self.buildings
            if building.damage_label is not None
        )


def parse_label(
    path: Path, phase: Phase, image_size: tuple[int, int] | None = None
) -> LabelFile:
    """Read one label file. ``image_size`` is (width, height) for bounds checks."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LabelError("unreadable_label", f"{path}: {error}") from error

    features = document.get("features") if isinstance(document, dict) else None
    xy_features = features.get("xy") if isinstance(features, dict) else None
    if not isinstance(xy_features, list):
        raise LabelError("unexpected_structure", f"{path}: features.xy list not found")

    issues: list[LabelIssue] = []
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        issues.append(LabelIssue(MISSING_METADATA))
        metadata = {}
    disaster_type = metadata.get("disaster_type")
    if disaster_type is not None and (not isinstance(disaster_type, str) or not disaster_type.strip()):
        issues.append(LabelIssue(INVALID_DISASTER_TYPE))
        metadata["disaster_type"] = None
    catalog_id = metadata.get("catalog_id")
    if catalog_id is not None and type(catalog_id) is not int and (
        not isinstance(catalog_id, str) or not catalog_id.strip()
    ):
        issues.append(LabelIssue(INVALID_CATALOG_ID))
        metadata["catalog_id"] = None

    buildings: list[Building] = []
    for feature in xy_features:
        building = _parse_xy_feature(feature, phase, image_size, issues)
        if building is not None:
            buildings.append(building)

    uid_counts = Counter(b.uid for b in buildings if b.uid is not None)
    issues.extend(
        LabelIssue(DUPLICATE_UID, uid) for uid, count in uid_counts.items() if count > 1
    )

    lng_lat_features = features.get("lng_lat", [])
    if not isinstance(lng_lat_features, list):
        issues.append(LabelIssue(UNPARSEABLE_LNG_LAT, detail="features.lng_lat is not a list"))
        lng_lat_features = []
    bounds = _lng_lat_bounds(lng_lat_features, issues)
    xy_building_count = sum(_is_building(feature) for feature in xy_features)
    lng_lat_building_count = sum(_is_building(feature) for feature in lng_lat_features)
    if lng_lat_building_count != xy_building_count:
        issues.append(
            LabelIssue(
                LNG_LAT_COUNT_MISMATCH,
                detail=f"xy={xy_building_count}, lng_lat={lng_lat_building_count}",
            )
        )

    return LabelFile(phase, metadata, tuple(buildings), tuple(issues), bounds)


def _properties(feature: object) -> Mapping[str, object]:
    properties = feature.get("properties") if isinstance(feature, dict) else None
    return properties if isinstance(properties, dict) else {}


def _is_building(feature: object) -> bool:
    return _properties(feature).get("feature_type") == "building"


def _parse_xy_feature(
    feature: object,
    phase: Phase,
    image_size: tuple[int, int] | None,
    issues: list[LabelIssue],
) -> Building | None:
    properties = _properties(feature)
    uid = properties.get("uid")
    uid = uid if isinstance(uid, str) and uid else None

    feature_type = properties.get("feature_type")
    if feature_type != "building":
        issues.append(LabelIssue(UNKNOWN_FEATURE_TYPE, uid, repr(feature_type)))
        return None
    if uid is None:
        issues.append(LabelIssue(MISSING_UID))

    text = feature.get("wkt") if isinstance(feature, dict) else None
    if not isinstance(text, str):
        issues.append(LabelIssue(MISSING_WKT, uid))
        return None
    try:
        geometry = shapely_wkt.loads(text)
    except (ShapelyError, TypeError, ValueError) as error:
        issues.append(LabelIssue(UNPARSEABLE_WKT, uid, str(error)))
        return None

    if not isinstance(geometry, Polygon):
        issues.append(LabelIssue(UNSUPPORTED_GEOMETRY, uid, geometry.geom_type))
        return None
    if geometry.is_empty or not all(math.isfinite(v) for v in geometry.bounds):
        issues.append(LabelIssue(EMPTY_GEOMETRY if geometry.is_empty else UNPARSEABLE_WKT, uid))
        return None
    if geometry.area <= 0:
        issues.append(LabelIssue(EMPTY_GEOMETRY, uid))
        return None
    if not geometry.is_valid:
        issues.append(LabelIssue(INVALID_GEOMETRY, uid, shapely.is_valid_reason(geometry)))
    if image_size is not None:
        # Bounds, not GEOS predicates, so that invalid polygons are judged reliably.
        width, height = image_size
        min_x, min_y, max_x, max_y = geometry.bounds
        if max_x <= 0 or max_y <= 0 or min_x >= width or min_y >= height:
            issues.append(LabelIssue(OUTSIDE_IMAGE, uid))
            return None
        if min_x < 0 or min_y < 0 or max_x > width or max_y > height:
            issues.append(LabelIssue(OUT_OF_BOUNDS, uid))

    if phase == "pre":
        return Building(uid, geometry, None, None)

    subtype = properties.get("subtype")
    if isinstance(subtype, str) and subtype in DAMAGE_CLASSES:
        damage_class = DAMAGE_CLASSES[subtype]
        return Building(uid, geometry, damage_class, DAMAGE_CLASS_NAMES[damage_class])
    if subtype == UNCLASSIFIED_SUBTYPE:
        return Building(uid, geometry, IGNORE, "un_classified")
    if subtype is None:
        issues.append(LabelIssue(MISSING_SUBTYPE, uid))
        return Building(uid, geometry, IGNORE, "missing_subtype")
    issues.append(LabelIssue(UNKNOWN_SUBTYPE, uid, repr(subtype)))
    return Building(uid, geometry, IGNORE, "unknown_subtype")


def _lng_lat_bounds(
    features: list[object], issues: list[LabelIssue]
) -> tuple[float, float, float, float] | None:
    bounds: list[tuple[float, float, float, float]] = []
    for feature in features:
        if not _is_building(feature):
            continue
        text = feature.get("wkt") if isinstance(feature, dict) else None
        if not isinstance(text, str):
            issues.append(LabelIssue(UNPARSEABLE_LNG_LAT, detail="missing wkt"))
            continue
        try:
            geometry = shapely_wkt.loads(text)
        except (ShapelyError, TypeError, ValueError) as error:
            issues.append(LabelIssue(UNPARSEABLE_LNG_LAT, detail=str(error)))
            continue
        if geometry.is_empty or not all(math.isfinite(v) for v in geometry.bounds):
            issues.append(LabelIssue(UNPARSEABLE_LNG_LAT, detail="empty or non-finite"))
            continue
        bounds.append(geometry.bounds)

    if not bounds:
        return None
    return (
        min(b[0] for b in bounds),
        min(b[1] for b in bounds),
        max(b[2] for b in bounds),
        max(b[3] for b in bounds),
    )
