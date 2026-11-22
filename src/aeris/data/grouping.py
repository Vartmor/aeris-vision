"""Group manifest records by source scene and geographic overlap.

Scenes: every sample is joined with the Maxar ``catalog_id`` of its pre- and
post-disaster images, so tiles cut from the same capture land in one group.

Spatial groups: scenes whose building ``lng_lat`` bounds overlap or share an
edge are joined, so repeated and adjoining captures land in one group.

Both rules only merge, never split, so the groups are conservative for leakage.
A sample without catalog IDs forms its own scene; a sample without ``lng_lat``
bounds keeps its scene as its spatial group. The audit reports how often each
fallback happens.
"""

from collections.abc import Hashable, Iterable, MutableMapping
from typing import Any

from shapely import STRtree, box


class UnionFind:
    """Track connected groups with path compression."""

    def __init__(self) -> None:
        self._parent: dict[Hashable, Hashable] = {}

    def add(self, node: Hashable) -> None:
        self._parent.setdefault(node, node)

    def find(self, node: Hashable) -> Hashable:
        self.add(node)
        root = node
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[node] != root:
            self._parent[node], node = root, self._parent[node]
        return root

    def union(self, first: Hashable, second: Hashable) -> None:
        self._parent[self.find(first)] = self.find(second)


Record = MutableMapping[str, Any]


def assign_groups(records: Iterable[Record]) -> None:
    """Set ``scene_id``, ``scene_id_source``, ``spatial_group_id`` and
    ``spatial_group_source`` on every record in place."""
    records = list(records)
    _assign_scenes(records)
    _assign_spatial_groups(records)


def catalog_id(record: Record, phase: str) -> str | None:
    metadata = (record.get("metadata") or {}).get(phase) or {}
    value = metadata.get("catalog_id")
    if value is None or isinstance(value, bool):
        return None
    value = str(value).strip()
    return value or None


def _assign_scenes(records: list[Record]) -> None:
    scenes = UnionFind()
    for record in records:
        sample = ("sample", record["sample_id"])
        scenes.add(sample)
        for phase in ("pre", "post"):
            value = catalog_id(record, phase)
            if value is not None:
                scenes.union(sample, ("catalog", value))

    smallest_catalog: dict[Hashable, str] = {}
    for record in records:
        for phase in ("pre", "post"):
            value = catalog_id(record, phase)
            if value is not None:
                root = scenes.find(("catalog", value))
                smallest_catalog[root] = min(value, smallest_catalog.get(root, value))

    for record in records:
        root = scenes.find(("sample", record["sample_id"]))
        if root in smallest_catalog:
            record["scene_id"] = f"catalog:{smallest_catalog[root]}"
            record["scene_id_source"] = "catalog_id"
        else:
            record["scene_id"] = f"sample:{record['sample_id']}"
            record["scene_id_source"] = "sample"


def _assign_spatial_groups(records: list[Record]) -> None:
    groups = UnionFind()
    for record in records:
        groups.add(record["scene_id"])

    bounded = [record for record in records if record.get("lng_lat_bounds") is not None]
    envelopes = [box(*record["lng_lat_bounds"]) for record in bounded]
    index = STRtree(envelopes)
    for row, (record, envelope) in enumerate(zip(bounded, envelopes)):
        min_x, min_y, max_x, max_y = record["lng_lat_bounds"]
        for column in index.query(envelope):
            other = bounded[column]
            if column <= row or record["scene_id"] == other["scene_id"]:
                continue
            other_min_x, other_min_y, other_max_x, other_max_y = other["lng_lat_bounds"]
            shared_width = min(max_x, other_max_x) - max(min_x, other_min_x)
            shared_height = min(max_y, other_max_y) - max(min_y, other_min_y)
            if shared_width > 0 or shared_height > 0:
                groups.union(record["scene_id"], other["scene_id"])

    smallest_sample: dict[Hashable, str] = {}
    for record in records:
        root = groups.find(record["scene_id"])
        sample_id = record["sample_id"]
        smallest_sample[root] = min(sample_id, smallest_sample.get(root, sample_id))

    for record in records:
        root = groups.find(record["scene_id"])
        record["spatial_group_id"] = f"spatial:{smallest_sample[root]}"
        record["spatial_group_source"] = (
            "lng_lat_bounds" if record.get("lng_lat_bounds") is not None else "scene"
        )
