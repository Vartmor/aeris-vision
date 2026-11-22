import pytest

from aeris.data import labels
from aeris.data.labels import IGNORE, LabelError, parse_label


def issue_codes(label: labels.LabelFile) -> dict[str, int]:
    return dict(label.issue_counts())


def test_post_subtypes_map_to_project_classes(synthetic_xbd) -> None:
    rectangle = synthetic_xbd.rectangle
    path = synthetic_xbd.write_label(
        [
            rectangle("a", 0, 0, 2, 2, "no-damage"),
            rectangle("b", 2, 0, 4, 2, "minor-damage"),
            rectangle("c", 4, 0, 6, 2, "major-damage"),
            rectangle("d", 6, 0, 8, 2, "destroyed"),
            rectangle("e", 8, 0, 10, 2, "un-classified"),
        ],
        "post",
        {"disaster": "x"},
    )

    label = parse_label(path, "post", (16, 16))

    assert [(b.uid, b.damage_class, b.damage_label) for b in label.buildings] == [
        ("a", 1, "no_damage"),
        ("b", 2, "minor_damage"),
        ("c", 3, "major_damage"),
        ("d", 4, "destroyed"),
        ("e", IGNORE, "un_classified"),
    ]
    assert label.issues == ()
    assert label.metadata == {"disaster": "x"}


def test_missing_or_unknown_subtype_is_ignored_and_never_no_damage(synthetic_xbd) -> None:
    rectangle = synthetic_xbd.rectangle
    path = synthetic_xbd.write_label(
        [
            rectangle("missing", 0, 0, 2, 2, None),
            rectangle("odd", 2, 0, 4, 2, "partial"),
            {**rectangle("malformed", 4, 0, 6, 2), "subtype": ["destroyed"]},
        ],
        "post",
        {},
    )

    label = parse_label(path, "post")

    assert [(b.damage_class, b.damage_label) for b in label.buildings] == [
        (IGNORE, "missing_subtype"),
        (IGNORE, "unknown_subtype"),
        (IGNORE, "unknown_subtype"),
    ]
    assert issue_codes(label) == {labels.MISSING_SUBTYPE: 1, labels.UNKNOWN_SUBTYPE: 2}
    assert label.damage_label_counts() == {"missing_subtype": 1, "unknown_subtype": 2}


def test_pre_disaster_buildings_have_no_damage_label(synthetic_xbd) -> None:
    path = synthetic_xbd.write_label(
        [synthetic_xbd.rectangle("a", 0, 0, 2, 2, None)], "pre", {}
    )

    (building,) = parse_label(path, "pre").buildings

    assert (building.damage_class, building.damage_label) == (None, None)


def test_geometry_problems_are_reported_and_unusable_shapes_skipped(synthetic_xbd) -> None:
    rectangle = synthetic_xbd.rectangle
    document = synthetic_xbd.label_document(
        [
            {"uid": "garbage", "subtype": "destroyed", "wkt": "POLYGON ((0 0, 1"},
            {"uid": "multi", "subtype": "destroyed", "wkt": "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 0)))"},
            {"uid": "flat", "subtype": "destroyed", "wkt": "POLYGON ((0 0, 2 0, 4 0, 0 0))"},
            {"uid": "bowtie", "subtype": "destroyed", "wkt": "POLYGON ((0 0, 6 4, 6 0, 0 2, 0 0))"},
            rectangle("outside", 20, 20, 22, 22),
            rectangle("edge", 14, 14, 18, 18),
            rectangle("no-wkt", 0, 0, 2, 2),
        ],
        "post",
    )
    del document["features"]["xy"][-1]["wkt"]
    path = synthetic_xbd.write_label([], "post", document=document)

    label = parse_label(path, "post", (16, 16))

    assert [b.uid for b in label.buildings] == ["bowtie", "edge"]
    assert issue_codes(label) == {
        labels.UNPARSEABLE_WKT: 1,
        labels.UNSUPPORTED_GEOMETRY: 1,
        labels.EMPTY_GEOMETRY: 1,
        labels.INVALID_GEOMETRY: 1,
        labels.OUTSIDE_IMAGE: 1,
        labels.OUT_OF_BOUNDS: 1,
        labels.MISSING_WKT: 1,
        labels.UNPARSEABLE_LNG_LAT: 1,
    }


def test_feature_type_and_uid_problems_are_reported(synthetic_xbd) -> None:
    rectangle = synthetic_xbd.rectangle
    path = synthetic_xbd.write_label(
        [
            {**rectangle("road", 0, 0, 2, 2), "properties": {"feature_type": "road"}},
            rectangle(None, 2, 0, 4, 2),
            rectangle("twin", 4, 0, 6, 2),
            rectangle("twin", 6, 0, 8, 2),
        ],
        "post",
        {},
    )

    label = parse_label(path, "post")

    assert [b.uid for b in label.buildings] == [None, "twin", "twin"]
    assert issue_codes(label) == {
        labels.UNKNOWN_FEATURE_TYPE: 1,
        labels.MISSING_UID: 1,
        labels.DUPLICATE_UID: 1,
    }


def test_lng_lat_bounds_cover_all_buildings(synthetic_xbd) -> None:
    rectangle = synthetic_xbd.rectangle
    path = synthetic_xbd.write_label(
        [rectangle("a", 0, 0, 2, 2), rectangle("b", 10, 10, 12, 14)], "post", {}
    )

    bounds = parse_label(path, "post").lng_lat_bounds

    assert bounds == pytest.approx((10.0, 20.0 - 14e-5, 10.0 + 12e-5, 20.0))


def test_lng_lat_count_mismatch_and_missing_metadata_are_reported(synthetic_xbd) -> None:
    document = synthetic_xbd.label_document(
        [synthetic_xbd.rectangle("a", 0, 0, 2, 2)], "post"
    )
    document["features"]["lng_lat"] = []
    del document["metadata"]
    path = synthetic_xbd.write_label([], "post", document=document)

    label = parse_label(path, "post")

    assert label.lng_lat_bounds is None
    assert issue_codes(label) == {
        labels.MISSING_METADATA: 1,
        labels.LNG_LAT_COUNT_MISMATCH: 1,
    }


@pytest.mark.parametrize(
    ("document", "code"),
    [
        ("{not json", "unreadable_label"),
        ({"features": {"lng_lat": []}}, "unexpected_structure"),
        ([], "unexpected_structure"),
    ],
)
def test_unusable_label_files_raise_label_error(synthetic_xbd, document, code) -> None:
    path = synthetic_xbd.write_label([], "post", document=document)

    with pytest.raises(LabelError) as error:
        parse_label(path, "post")

    assert error.value.code == code
