import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from aeris.data.labels import DAMAGE_CLASS_NAMES
from aeris.splits.planner import SplitError, build_split_plan
from aeris.splits.storage import freeze_split, plan_from_manifest, verify_split

_FRACTIONS = {"train": 0.70, "validation": 0.10, "calibration": 0.10, "test": 0.10}


def _sample(
    name: str,
    event: str,
    *,
    event_type: str = "wind",
    counts: tuple[int, ...] = (1, 1, 1, 1),
    **changes: object,
) -> dict:
    return {
        "sample_id": name,
        "event_id": event,
        "event_type": event_type,
        "scene_id": f"scene-{name}",
        "spatial_group_id": f"spatial-{name}",
        "damage_class_counts": dict(zip(DAMAGE_CLASS_NAMES.values(), counts)),
        "validation_status": "valid",
        **changes,
    }


def test_uniform_id_groups_hit_target_fractions_reproducibly() -> None:
    records = [
        *(_sample(f"wind-{index:02}", "training-wind") for index in range(40)),
        _sample("held-out", "held-wind"),
    ]

    plan = build_split_plan(records, _FRACTIONS, ood_event="held-wind")

    assert plan == build_split_plan(reversed(records), _FRACTIONS, ood_event="held-wind")
    assert {
        name: len(samples) for name, samples in plan["partitions"].items()
    } == {"train": 28, "validation": 4, "calibration": 4, "test": 4, "ood_test": 1}
    assert plan["same_event_type_in_train"] is True


def test_ood_quarantine_is_transitive_across_scene_spatial_and_image_hash() -> None:
    records = [
        _sample("ood", "held-wind", scene_id="shared-scene"),
        _sample("via-scene", "id-wind", scene_id="shared-scene", spatial_group_id="shared-bounds"),
        _sample(
            "via-bounds", "id-wind", spatial_group_id="shared-bounds",
            sha256={"post_image": "shared-image"},
        ),
        _sample("via-image", "id-wind", sha256={"pre_image": "shared-image"}),
        *(_sample(f"id-{index}", "id-wind") for index in range(8)),
    ]

    plan = build_split_plan(records, _FRACTIONS, ood_event="held-wind")

    assert plan["partitions"]["ood_test"] == ["ood"]
    assert plan["excluded"]["groups_linked_to_ood"] == ["via-bounds", "via-image", "via-scene"]
    assert len({plan["group_by_sample"][sample] for sample in (
        "ood", "via-scene", "via-bounds", "via-image"
    )}) == 1
    assert sum(info["samples"] for info in plan["summary"].values()) == 9


def test_id_groups_remain_intact_and_have_disjoint_coverage_anchors() -> None:
    records = [
        _sample("id-a0", "wind", counts=(1, 1, 0, 0), scene_id="a"),
        _sample("id-a1", "wind", counts=(1, 0, 0, 0), scene_id="a"),
        _sample("id-b", "wind", counts=(0, 0, 1, 1)),
        _sample("id-c", "wind", counts=(1, 0, 1, 0)),
        _sample("id-d", "wind", counts=(0, 1, 0, 1)),
        _sample("id-e", "wind", counts=(0, 0, 0, 0)),
        _sample("id-f", "wind", counts=(0, 0, 0, 0)),
        _sample("ood", "held-wind"),
    ]

    plan = build_split_plan(records, _FRACTIONS, ood_event="held-wind")

    partition_by_sample = {
        sample: name for name, samples in plan["partitions"].items() for sample in samples
    }
    assert partition_by_sample["id-a0"] == partition_by_sample["id-a1"]
    for name in ("train", "validation"):
        assert all(plan["summary"][name]["damage_class_counts"].values())
    assert plan["partitions"]["calibration"]
    assert plan["partitions"]["test"]


def test_automatic_holdout_prefers_a_type_that_remains_in_train() -> None:
    records = [
        *(
            _sample(f"fire-{index}", "fire", event_type="fire", counts=(8, 8, 8, 8))
            for index in range(8)
        ),
        *(_sample(f"flood-{index}", "flood", event_type="flood") for index in range(4)),
        _sample("wind-a", "wind-a", counts=(1, 0, 0, 0)),
        _sample("wind-b", "wind-b", counts=(1, 1, 0, 0)),
    ]

    plan = build_split_plan(records, _FRACTIONS)

    other_type = build_split_plan(records, _FRACTIONS, ood_event="fire")
    assert other_type["same_event_type_in_train"] is False
    assert plan["ood_event"] == "wind-b"
    assert "wind-a" in plan["summary"]["train"]["events"]
    assert plan["same_event_type_in_train"] is True
    assert plan["summary"]["ood_test"]["damage_class_counts"]["major_damage"] == 0


def test_unseen_type_and_duplicate_invalid_rows_are_audited() -> None:
    records = [
        *(_sample(f"id-{index}", "wind") for index in range(4)),
        _sample("ood", "quake", event_type="earthquake"),
        _sample("broken", "quake", validation_status="invalid"),
        _sample("broken", "quake", validation_status="invalid"),
    ]

    plan = build_split_plan(records, _FRACTIONS, ood_event="quake")

    assert plan["ood_event_type"] == "earthquake"
    assert plan["same_event_type_in_train"] is False
    assert plan["partitions"]["ood_test"] == ["ood"]
    assert plan["excluded"]["invalid_manifest_records"] == {"broken": 2}


def test_insufficient_independent_damage_support_fails() -> None:
    records = [
        *(_sample(f"id-{index}", "wind", counts=(1, 1, 1, 0)) for index in range(7)),
        _sample("only-destroyed-group", "wind", counts=(0, 0, 0, 7)),
        _sample("ood", "held-wind"),
    ]

    with pytest.raises(SplitError, match="cannot cover all four classes"):
        build_split_plan(records, _FRACTIONS, ood_event="held-wind")


def test_frozen_split_publication_is_idempotent_and_detects_changes(tmp_path: Path) -> None:
    records = [
        *(_sample(f"id-{index}", "wind") for index in range(4)),
        _sample("ood", "held-wind"),
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    artifact = tmp_path / "splits" / "split.json"
    expected = plan_from_manifest(manifest, _FRACTIONS, ood_event="held-wind")

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: freeze_split(artifact, expected), range(4)))
    verify_split(artifact, expected)
    original = artifact.read_bytes()
    assert sorted(artifact.parent.iterdir()) == [artifact]

    damaged = {**expected, "split_sha256": "changed"}
    with pytest.raises(SplitError, match="frozen split differs"):
        freeze_split(artifact, damaged)
    assert artifact.read_bytes() == original

    with manifest.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    stale = plan_from_manifest(manifest, _FRACTIONS, ood_event="held-wind")
    with pytest.raises(SplitError, match="manifest SHA-256 has changed"):
        verify_split(artifact, stale)


def test_reviewed_id_groups_reject_unseen_events_in_id_test() -> None:
    records = [
        *(_sample(f"id-{index}", "wind") for index in range(4)),
        _sample("that-event", "other-wind"),
        _sample("ood", "held-wind"),
    ]

    with pytest.raises(SplitError, match="test contains events absent from train: other-wind"):
        build_split_plan(
            records,
            _FRACTIONS,
            ood_event="held-wind",
            id_evaluation_groups={
                "validation": ["id-1"],
                "calibration": ["id-2"],
                "test": ["that-event"],
            },
        )
