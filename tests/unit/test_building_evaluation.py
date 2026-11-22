import pytest

from aeris.eval.buildings import (
    BuildingMask,
    BuildingMatch,
    MergedCandidate,
    SevereDamageCounts,
    evaluate_buildings,
)


def building(building_id: str, damage_class: int | None, *pixels: int) -> BuildingMask:
    return BuildingMask(building_id, frozenset(pixels), damage_class)


def test_matching_includes_iou_half_and_excludes_lower_iou() -> None:
    report = evaluate_buildings(
        [building("ref-half", 3, 0, 1, 2, 3), building("ref-below", 4, 4, 5, 6)],
        [building("half", 1, 0, 1), building("below", 2, 4)],
        image_width=8,
        image_height=1,
    )

    assert report.matches == (BuildingMatch("ref-half", "half", 0.5),)
    assert report.unmatched_reference_ids == ("ref-below",)
    assert report.unmatched_candidate_ids == ("below",)
    assert report.localization_recall == 0.5


def test_greedy_matching_uses_highest_iou_before_building_ids() -> None:
    references = [building("a", 3, 0, 1), building("z", 4, 0, 1, 2, 3)]
    candidates = [building("a", 1, 2, 3), building("z", 2, 0, 1, 2, 3)]

    report = evaluate_buildings(references, candidates, image_width=4, image_height=1)

    assert report.matches == (BuildingMatch("z", "z", 1.0),)
    assert report.unmatched_reference_ids == ("a",)
    assert report.unmatched_candidate_ids == ("a",)


def test_equal_iou_is_resolved_by_reference_then_candidate_id() -> None:
    references = [building("r2", 4, 2, 3), building("r1", 3, 0, 1)]
    candidates = [building("p2", 1, 0, 1, 2, 3), building("p1", 2, 0, 1, 2, 3)]

    report = evaluate_buildings(references, candidates, image_width=4, image_height=1)

    assert report.matches == (
        BuildingMatch("r1", "p1", 0.5),
        BuildingMatch("r2", "p2", 0.5),
    )
    assert report == evaluate_buildings(
        reversed(references), reversed(candidates), image_width=4, image_height=1
    )
    assert len(report.merged_candidates) == 2
    assert report.merged_reference_ids == ("r1", "r2")
    assert report.severe_damage == SevereDamageCounts(0, 2, 0, 0, 0, 0)


def test_exactly_half_reference_coverage_is_not_a_merge() -> None:
    report = evaluate_buildings(
        [building("r1", 3, 0, 1), building("r2", 4, 2, 3)],
        [building("p", 3, 0, 1, 2)],
        image_width=4,
        image_height=1,
    )

    assert report.merged_candidates == ()
    assert report.severe_damage == SevereDamageCounts(1, 0, 0, 0, 1, 0)


def test_merge_and_class_errors_count_each_severe_reference_once() -> None:
    references = [
        building("merged-major", 3, 0, 1),
        building("merged-destroyed", 4, 2, 3),
        building("missing", 4, 4),
        building("predicted-no-damage", 3, 5),
        building("predicted-minor", 4, 6),
        building("predicted-major", 3, 7),
        building("predicted-destroyed", 4, 8),
        building("unclassified", None, 9),
    ]
    candidates = [
        building("merge", 1, 0, 1, 2, 3),
        building("no-damage", 1, 5),
        building("minor", 2, 6),
        building("major", 3, 7),
        building("destroyed", 4, 8),
        building("covers-unclassified", 4, 9),
        building("false-candidate", 3, 10),
    ]

    report = evaluate_buildings(references, candidates, image_width=12, image_height=1)

    assert report.merged_candidates == (
        MergedCandidate("merge", ("merged-destroyed", "merged-major")),
    )
    assert report.unmatched_reference_ids == ("merged-major", "missing")
    assert report.unmatched_candidate_ids == ("false-candidate",)
    assert report.severe_damage == SevereDamageCounts(1, 2, 1, 1, 1, 1)
    assert report.severe_damage.total == 7
    assert report.localization_recall == 6 / 8


def test_image_without_reference_buildings_has_undefined_recall() -> None:
    report = evaluate_buildings(
        [], [building("false-candidate", 1, 0)], image_width=1, image_height=1
    )

    assert report.localization_recall is None
    assert report.unmatched_candidate_ids == ("false-candidate",)
    assert report.severe_damage.total == 0


def test_image_without_candidates_counts_all_severe_references_as_missing() -> None:
    report = evaluate_buildings(
        [building("major", 3, 0), building("minor", 2, 1)],
        [],
        image_width=2,
        image_height=1,
    )

    assert report.unmatched_reference_ids == ("major", "minor")
    assert report.severe_damage.not_detected == 1
    assert report.severe_damage.total == 1
    assert report.localization_recall == 0.0


@pytest.mark.parametrize(
    ("references", "candidates", "message"),
    [
        ([building("r", 3)], [], "empty mask"),
        ([building("r", 3, 4)], [], "outside the raster grid"),
        ([building("r", 3, 0), building("r", 4, 1)], [], "Duplicate building ID"),
        ([building("r", 255, 0)], [], "invalid damage class"),
        ([], [building("p", None, 0)], "invalid damage class"),
    ],
)
def test_invalid_input_fails_before_evaluation(
    references: list[BuildingMask], candidates: list[BuildingMask], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_buildings(references, candidates, image_width=2, image_height=2)
