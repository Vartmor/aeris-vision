import numpy as np
import pytest
from shapely import wkt

from aeris.data.labels import IGNORE, Building
from aeris.data.raster import building_pixels, rasterize_damage, rasterize_localization


def building(text: str, damage_class: int | None = 1, uid: str = "b") -> Building:
    return Building(uid, wkt.loads(text), damage_class, None)


def pixels(mask: np.ndarray) -> set[tuple[int, int]]:
    return {(int(row), int(column)) for row, column in zip(*np.nonzero(mask))}


def test_pixel_belongs_to_polygon_when_its_center_is_inside() -> None:
    raster = rasterize_localization(
        [building("POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))")], width=4, height=4
    )

    assert pixels(raster.mask) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert raster.mask.dtype == np.uint8
    assert raster.mask.max() == 1


def test_pixel_centers_on_the_boundary_belong_to_the_polygon() -> None:
    rows, columns = building_pixels(
        wkt.loads("POLYGON ((0.5 0.5, 2.5 0.5, 2.5 1.5, 0.5 1.5, 0.5 0.5))"), 4, 4
    )

    assert set(zip(rows.tolist(), columns.tolist())) == {
        (row, column) for row in (0, 1) for column in (0, 1, 2)
    }


def test_holes_are_left_empty() -> None:
    raster = rasterize_localization(
        [building("POLYGON ((0 0, 3 0, 3 3, 0 3, 0 0), (1 1, 2 1, 2 2, 1 2, 1 1))")],
        width=3,
        height=3,
    )

    assert pixels(raster.mask) == {(r, c) for r in range(3) for c in range(3)} - {(1, 1)}


def test_polygons_are_clipped_to_the_image() -> None:
    raster = rasterize_localization(
        [building("POLYGON ((-5 -5, 2 -5, 2 1, -5 1, -5 -5))")], width=3, height=3
    )

    assert pixels(raster.mask) == {(0, 0), (0, 1)}


def test_building_without_a_pixel_center_is_reported_empty() -> None:
    raster = rasterize_localization(
        [
            building("POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))"),
            building("POLYGON ((2.6 2.6, 2.9 2.6, 2.9 2.9, 2.6 2.6))"),
        ],
        width=4,
        height=4,
    )

    assert raster.empty_buildings == (1,)


def test_overlapping_damage_takes_most_severe_class_and_classified_beats_ignore() -> None:
    raster = rasterize_damage(
        [
            building("POLYGON ((0 0, 3 0, 3 1, 0 1, 0 0))", 4, "destroyed"),
            building("POLYGON ((0 0, 2 0, 2 1, 0 1, 0 0))", 2, "minor"),
            building("POLYGON ((1 0, 4 0, 4 1, 1 1, 1 0))", IGNORE, "unclassified"),
        ],
        width=5,
        height=1,
    )

    assert raster.mask.tolist() == [[4, 4, 4, IGNORE, 0]]
    assert raster.overlap_pixels == 3


def test_damage_masks_reject_pre_disaster_buildings() -> None:
    with pytest.raises(ValueError, match="post-disaster"):
        rasterize_damage([building("POLYGON ((0 0, 1 0, 1 1, 0 0))", None)], 2, 2)
