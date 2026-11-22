from aeris.data.grouping import assign_groups


def record(sample_id: str, pre: str | None, post: str | None, bounds=None) -> dict:
    return {
        "sample_id": sample_id,
        "metadata": {"pre": {"catalog_id": pre}, "post": {"catalog_id": post}},
        "lng_lat_bounds": bounds,
    }


def by_id(records: list[dict], key: str) -> dict[str, str]:
    return {r["sample_id"]: r[key] for r in records}


def test_shared_catalog_ids_chain_samples_into_one_scene() -> None:
    records = [
        record("a", "PRE-1", "POST-1"),
        record("b", "PRE-1", "POST-2"),
        record("c", "PRE-3", "POST-2"),
        record("d", "PRE-4", "POST-4"),
    ]

    assign_groups(records)

    assert by_id(records, "scene_id") == {
        "a": "catalog:POST-1",
        "b": "catalog:POST-1",
        "c": "catalog:POST-1",
        "d": "catalog:POST-4",
    }
    assert set(by_id(records, "scene_id_source").values()) == {"catalog_id"}


def test_sample_without_catalog_id_is_its_own_scene() -> None:
    records = [record("a", None, " "), {"sample_id": "b", "metadata": {}, "lng_lat_bounds": None}]

    assign_groups(records)

    assert by_id(records, "scene_id") == {"a": "sample:a", "b": "sample:b"}
    assert set(by_id(records, "scene_id_source").values()) == {"sample"}


def test_overlapping_bounds_join_scenes_but_corner_contact_does_not() -> None:
    records = [
        record("a", "A", "A", [0.0, 0.0, 2.0, 2.0]),
        record("b", "B", "B", [1.0, 1.0, 3.0, 3.0]),
        record("c", "C", "C", [3.0, 0.0, 4.0, 1.0]),
        record("d", "A", "D", None),
        record("e", "E", "E", None),
    ]

    assign_groups(records)

    assert by_id(records, "spatial_group_id") == {
        "a": "spatial:a",
        "b": "spatial:a",
        "c": "spatial:c",
        "d": "spatial:a",
        "e": "spatial:e",
    }
    assert by_id(records, "spatial_group_source") == {
        "a": "lng_lat_bounds",
        "b": "lng_lat_bounds",
        "c": "lng_lat_bounds",
        "d": "scene",
        "e": "scene",
    }


def test_bounds_sharing_an_edge_join_neighboring_scenes() -> None:
    records = [
        record("a", "A", "A", [0.0, 0.0, 1.0, 1.0]),
        record("b", "B", "B", [1.0, 0.25, 2.0, 0.75]),
        record("c", "C", "C", [2.01, 0.25, 3.0, 0.75]),
    ]

    assign_groups(records)

    assert by_id(records, "spatial_group_id") == {
        "a": "spatial:a",
        "b": "spatial:a",
        "c": "spatial:c",
    }


def test_group_ids_do_not_depend_on_record_order() -> None:
    def groups(records: list[dict]) -> dict[str, tuple[str, str]]:
        assign_groups(records)
        return {r["sample_id"]: (r["scene_id"], r["spatial_group_id"]) for r in records}

    def records() -> list[dict]:
        return [
            record("a", "X", "Y", [0.0, 0.0, 1.0, 1.0]),
            record("b", "Z", "Y", [5.0, 5.0, 6.0, 6.0]),
            record("c", "W", "W", [0.5, 0.5, 2.0, 2.0]),
        ]

    assert groups(records()) == groups(list(reversed(records())))
