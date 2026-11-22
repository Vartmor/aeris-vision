import hashlib
import json

import numpy as np

from aeris.data.manifest import (
    INVALID,
    VALID,
    WARNING,
    build_manifest,
    read_manifest,
    write_manifest,
)


def by_id(result) -> dict[str, dict]:
    return {record["sample_id"]: record for record in result.records}


def test_valid_pair_produces_a_complete_record(synthetic_xbd) -> None:
    rectangle = synthetic_xbd.rectangle
    package = synthetic_xbd.write_pair(
        "hurricane-a_00000001",
        [
            rectangle("b1", 0, 0, 4, 4, "no-damage"),
            rectangle("b2", 6, 6, 10, 10, "major-damage"),
            rectangle("b3", 12, 0, 14, 2, "un-classified"),
        ],
    )

    result = build_manifest(synthetic_xbd.root, "challenge_train")

    (record,) = result.records
    assert record["validation_status"] == VALID
    assert record["issues"] == {}
    assert record["sample_id"] == "hurricane-a_00000001"
    assert record["event_id"] == "hurricane-a"
    assert record["event_type"] == "wind"
    assert record["source_package"] == "challenge_train"
    assert record["pre_image_path"] == "train/images/hurricane-a_00000001_pre_disaster.png"
    assert record["post_label_path"] == "train/labels/hurricane-a_00000001_post_disaster.json"
    assert (record["width"], record["height"]) == (16, 16)
    assert record["building_count"] == {"pre": 3, "post": 3}
    assert record["damage_class_counts"] == {
        "no_damage": 1,
        "minor_damage": 0,
        "major_damage": 1,
        "destroyed": 0,
        "un_classified": 1,
        "missing_subtype": 0,
        "unknown_subtype": 0,
    }
    assert record["scene_id"] == "catalog:hurricane-a-post-catalog"
    assert record["spatial_group_id"] == "spatial:hurricane-a_00000001"
    assert record["metadata"]["post"]["off_nadir_angle"] == 20.0
    assert record["lng_lat_bounds"] is not None
    image = package / "images" / "hurricane-a_00000001_post_disaster.png"
    assert record["sha256"]["post_image"] == hashlib.sha256(image.read_bytes()).hexdigest()


def test_missing_and_unrecognized_files_are_reported_not_skipped(synthetic_xbd) -> None:
    rectangle = synthetic_xbd.rectangle
    package = synthetic_xbd.write_pair(
        "flood-b_00000001", [rectangle("b", 0, 0, 2, 2)], skip={"post_label", "pre_image"}
    )
    synthetic_xbd.write_pair("flood-b_00000002", [], skip={"pre_label", "post_label"})
    (package / "labels" / "notes.json").write_text("{}", encoding="utf-8")

    result = build_manifest(synthetic_xbd.root, "challenge_train", compute_hashes=False)

    (record,) = result.records
    assert record["validation_status"] == INVALID
    assert record["issues"] == {"missing_post_label": 1, "missing_pre_image": 1}
    assert record["post_label_path"] is None
    assert record["damage_class_counts"] is None
    assert result.summary["unrecognized_label_files"] == {
        "count": 1,
        "examples": ["train/labels/notes.json"],
    }
    assert result.summary["images_without_labels"]["examples"] == [
        "train/images/flood-b_00000002_post_disaster.png",
        "train/images/flood-b_00000002_pre_disaster.png",
    ]


def test_pair_consistency_problems(synthetic_xbd) -> None:
    rectangle = synthetic_xbd.rectangle
    synthetic_xbd.write_pair("fire-c_00000001", [], size=(16, 16), post_size=(8, 8))
    synthetic_xbd.write_pair(
        "fire-c_00000002",
        [rectangle("same", 0, 0, 2, 2), rectangle("moved", 4, 4, 6, 6), rectangle("new", 8, 8, 9, 9)],
        pre_buildings=[rectangle("same", 0, 0, 2, 2, None), rectangle("moved", 4, 4, 7, 7, None)],
        metadata={"post": {"disaster": "fire-x", "disaster_type": "flood", "width": 32}},
        image_mode="L",
    )

    records = by_id(build_manifest(synthetic_xbd.root, "challenge_train", compute_hashes=False))

    assert records["fire-c_00000001"]["validation_status"] == INVALID
    assert records["fire-c_00000001"]["issues"] == {"image_size_mismatch": 1}
    assert records["fire-c_00000002"]["validation_status"] == WARNING
    assert records["fire-c_00000002"]["issues"] == {
        "event_type_mismatch": 1,
        "post:event_name_mismatch": 1,
        "post:metadata_size_mismatch": 1,
        "post:non_rgb_image": 1,
        "pre:non_rgb_image": 1,
        "pre_post_geometry_mismatch": 1,
        "pre_post_uid_mismatch": 1,
    }


def test_duplicate_sample_ids_across_packages_are_invalid(synthetic_xbd) -> None:
    rectangle = synthetic_xbd.rectangle
    synthetic_xbd.write_pair("quake-d_00000001", [rectangle("b", 0, 0, 2, 2)], package="train")
    synthetic_xbd.write_pair("quake-d_00000001", [rectangle("b", 0, 0, 2, 2)], package="copy")

    result = build_manifest(synthetic_xbd.root, "challenge_train", compute_hashes=False)

    assert [r["validation_status"] for r in result.records] == [INVALID, INVALID]
    assert [r["pre_label_path"].split("/")[0] for r in result.records] == ["copy", "train"]


def test_malformed_grouping_metadata_is_reported_and_excluded(synthetic_xbd) -> None:
    synthetic_xbd.write_pair(
        "hurricane-a_00000001",
        [],
        metadata={"post": {"disaster_type": ["flood"], "catalog_id": {"id": "post"}}},
    )

    result = build_manifest(synthetic_xbd.root, "challenge_train", compute_hashes=False)

    (record,) = result.records
    assert record["validation_status"] == WARNING
    assert record["event_type"] == "wind"
    assert record["scene_id"] == "catalog:hurricane-a-pre-catalog"
    assert record["metadata"]["post"]["disaster_type"] is None
    assert record["metadata"]["post"]["catalog_id"] is None
    assert record["issues"] == {"post:invalid_catalog_id": 1, "post:invalid_disaster_type": 1}


def test_summary_reports_event_support_from_usable_samples(synthetic_xbd) -> None:
    rectangle = synthetic_xbd.rectangle
    synthetic_xbd.write_pair(
        "hurricane-a_00000001",
        [rectangle("a", 0, 0, 2, 2, "destroyed"), rectangle("b", 4, 4, 6, 6, "minor-damage")],
    )
    synthetic_xbd.write_pair(
        "hurricane-a_00000002",
        [rectangle("c", 0, 0, 2, 2, "destroyed")],
        lng_lat_origin=(50.0, 50.0),
        metadata={"post": {"catalog_id": "other"}, "pre": {"catalog_id": "other-pre"}},
    )
    synthetic_xbd.write_pair(
        "hurricane-a_00000003", [rectangle("d", 0, 0, 2, 2, "destroyed")], skip={"post_image"}
    )
    synthetic_xbd.write_pair(
        "volcano-e_00000001",
        [rectangle("e", 0, 0, 2, 2, "no-damage"), rectangle("f", 3, 3, 4, 4, "not-a-class")],
        metadata={"post": {"disaster_type": "volcano"}, "pre": {"disaster_type": "volcano"}},
        lng_lat_origin=(80.0, 80.0),
    )

    summary = build_manifest(synthetic_xbd.root, "challenge_train", compute_hashes=False).summary

    assert summary["samples"] == 4
    assert summary["status_counts"] == {VALID: 2, WARNING: 1, INVALID: 1}
    hurricane = summary["events"]["hurricane-a"]
    assert hurricane["samples"] == 3
    assert hurricane["pre_buildings"] == 3
    assert hurricane["damage_label_counts"]["destroyed"] == 2
    assert hurricane["damage_label_counts"]["minor_damage"] == 1
    assert hurricane["scenes"] == 2
    assert hurricane["largest_scene_samples"] == 2
    assert summary["events"]["volcano-e"]["event_types"] == ["volcano"]
    assert summary["damage_label_counts"]["unknown_subtype"] == 1
    assert summary["issues"]["post:unknown_subtype"] == {"occurrences": 1, "samples": 1}
    assert summary["metadata_availability"]["post"]["labels"] == 4
    assert summary["metadata_availability"]["post"]["gsd"] == 4
    assert summary["grouping"]["cross_event_scenes"] == []


def test_written_manifest_round_trips_and_is_deterministic(synthetic_xbd, tmp_path) -> None:
    rectangle = synthetic_xbd.rectangle
    synthetic_xbd.write_pair("hurricane-a_00000002", [rectangle("a", 0, 0, 2, 2)])
    synthetic_xbd.write_pair("hurricane-a_00000001", [rectangle("b", 0, 0, 2, 2)])
    manifest, audit = tmp_path / "out" / "manifest.jsonl", tmp_path / "out" / "audit.json"

    write_manifest(build_manifest(synthetic_xbd.root, "challenge_train"), manifest, audit)
    first = manifest.read_bytes()
    write_manifest(build_manifest(synthetic_xbd.root, "challenge_train"), manifest, audit)

    assert manifest.read_bytes() == first
    records = read_manifest(manifest)
    assert [r["sample_id"] for r in records] == ["hurricane-a_00000001", "hurricane-a_00000002"]
    summary = json.loads(audit.read_text(encoding="utf-8"))
    assert summary["manifest_sha256"] == hashlib.sha256(first).hexdigest()


def test_target_check_compares_masks_with_official_targets(synthetic_xbd) -> None:
    rectangle = synthetic_xbd.rectangle
    sample = "hurricane-a_00000001"
    synthetic_xbd.write_pair(
        sample,
        [rectangle("a", 0, 0, 2, 2, "major-damage"), rectangle("u", 4, 0, 6, 2, "un-classified")],
        size=(8, 4),
    )
    pre_target = np.zeros((4, 8), np.uint8)
    pre_target[0:2, 0:2] = 1
    pre_target[0:2, 4:6] = 1
    post_target = np.zeros((4, 8), np.uint8)
    post_target[0:2, 0:2] = 3
    post_target[0:2, 4:6] = 1
    post_target[3, 7] = 2
    synthetic_xbd.write_target(sample, "pre", pre_target)
    synthetic_xbd.write_target(sample, "post", post_target)

    result = build_manifest(
        synthetic_xbd.root, "challenge_train", compute_hashes=False, check_targets=True
    )

    (record,) = result.records
    assert record["target_check"]["pre"]["mismatched_pixels"] == 0
    assert record["target_check"]["post"] == {
        "compared_pixels": 28,
        "mismatched_pixels": 1,
        "ignore_pixels": 4,
        "target_values_on_ignore": {"1": 4},
    }
    summary = result.summary["target_check"]
    assert summary["pre"]["samples_exact"] == 1
    assert summary["post"]["mismatch_fraction"] == 1 / 28
    assert summary["post"]["target_values_on_ignore"] == {"1": 4}


def test_target_check_reports_missing_targets(synthetic_xbd) -> None:
    synthetic_xbd.write_pair("hurricane-a_00000001", [synthetic_xbd.rectangle("a", 0, 0, 2, 2)])

    result = build_manifest(
        synthetic_xbd.root, "challenge_train", compute_hashes=False, check_targets=True
    )

    assert result.records[0]["target_check"] == {"pre": "missing", "post": "missing"}
    assert result.summary["target_check"]["post"]["not_compared"] == {"missing": 1}
