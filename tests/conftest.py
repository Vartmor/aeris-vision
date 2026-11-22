"""Synthetic xBD-style packages for tests that must run without the real dataset."""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

PHASES = ("pre", "post")


def rectangle(
    uid: str | None,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    subtype: str | None = "no-damage",
) -> dict[str, Any]:
    return {"uid": uid, "coords": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], "subtype": subtype}


def polygon_wkt(coords: Iterable[tuple[float, float]]) -> str:
    points = list(coords)
    ring = ", ".join(f"{x} {y}" for x, y in [*points, points[0]])
    return f"POLYGON (({ring}))"


def label_document(
    buildings: Iterable[Mapping[str, Any]],
    phase: str,
    metadata: Mapping[str, object] | None = None,
    lng_lat_origin: tuple[float, float] = (10.0, 20.0),
) -> dict[str, Any]:
    """Build a label JSON. A building may set "wkt" or "properties" to override defaults."""
    lon0, lat0 = lng_lat_origin
    xy, lng_lat = [], []
    for building in buildings:
        properties = {"feature_type": "building", "uid": building.get("uid")}
        if phase == "post" and building.get("subtype") is not None:
            properties["subtype"] = building["subtype"]
        properties.update(building.get("properties", {}))
        coords = building.get("coords")
        xy_wkt = building.get("wkt") or polygon_wkt(coords)
        geo_wkt = building.get("wkt") or polygon_wkt(
            (lon0 + x * 1e-5, lat0 - y * 1e-5) for x, y in coords
        )
        xy.append({"properties": properties, "wkt": xy_wkt})
        lng_lat.append({"properties": dict(properties), "wkt": geo_wkt})
    return {"features": {"lng_lat": lng_lat, "xy": xy}, "metadata": dict(metadata or {})}


class SyntheticXbd:
    rectangle = staticmethod(rectangle)
    polygon_wkt = staticmethod(polygon_wkt)
    label_document = staticmethod(label_document)

    def __init__(self, root: Path) -> None:
        self.root = root

    def write_label(
        self,
        buildings: Iterable[Mapping[str, Any]],
        phase: str,
        metadata: Mapping[str, object] | None = None,
        *,
        document: object | None = None,
        name: str = "label.json",
    ) -> Path:
        """Write one label file outside any package layout."""
        path = self.root / name
        if document is None:
            document = label_document(buildings, phase, metadata)
        path.write_text(
            document if isinstance(document, str) else json.dumps(document), encoding="utf-8"
        )
        return path

    def metadata(
        self, event: str, phase: str, size: tuple[int, int], **overrides: object
    ) -> dict[str, object]:
        return {
            "disaster": event,
            "disaster_type": "wind",
            "catalog_id": f"{event}-{phase}-catalog",
            "capture_date": "2018-10-10T16:00:00.000Z",
            "sensor": "WORLDVIEW02",
            "gsd": 2.0,
            "off_nadir_angle": 20.0,
            "sun_elevation": 50.0,
            "sun_azimuth": 140.0,
            "width": size[0],
            "height": size[1],
            **overrides,
        }

    def write_pair(
        self,
        sample_id: str,
        post_buildings: Iterable[Mapping[str, Any]],
        *,
        pre_buildings: Iterable[Mapping[str, Any]] | None = None,
        size: tuple[int, int] = (16, 16),
        post_size: tuple[int, int] | None = None,
        metadata: Mapping[str, Mapping[str, object]] | None = None,
        lng_lat_origin: tuple[float, float] = (10.0, 20.0),
        package: str = "train",
        skip: Iterable[str] = (),
        image_mode: str = "RGB",
    ) -> Path:
        """Write images and labels; ``skip`` names files to leave out, e.g. "post_label"."""
        event = sample_id.rsplit("_", 1)[0]
        post_buildings = list(post_buildings)
        if pre_buildings is None:
            pre_buildings = [{**b, "subtype": None} for b in post_buildings]
        buildings = {"pre": list(pre_buildings), "post": post_buildings}
        sizes = {"pre": size, "post": post_size or size}
        skip = set(skip)
        package_dir = self.root / package
        for phase in PHASES:
            stem = f"{sample_id}_{phase}_disaster"
            if f"{phase}_image" not in skip:
                image_path = package_dir / "images" / f"{stem}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new(image_mode, sizes[phase]).save(image_path)
            if f"{phase}_label" not in skip:
                label_path = package_dir / "labels" / f"{stem}.json"
                label_path.parent.mkdir(parents=True, exist_ok=True)
                phase_metadata = self.metadata(
                    event, phase, sizes[phase], **(metadata or {}).get(phase, {})
                )
                document = label_document(
                    buildings[phase], phase, phase_metadata, lng_lat_origin
                )
                label_path.write_text(json.dumps(document), encoding="utf-8")
        return package_dir

    def write_target(
        self, sample_id: str, phase: str, mask: np.ndarray, package: str = "train"
    ) -> Path:
        path = self.root / package / "targets" / f"{sample_id}_{phase}_disaster_target.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8)).save(path)
        return path


@pytest.fixture
def synthetic_xbd(tmp_path: Path) -> SyntheticXbd:
    root = tmp_path / "xbd"
    root.mkdir()
    return SyntheticXbd(root)
