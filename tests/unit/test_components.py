import numpy as np

from aeris.infer.components import candidate_footprints


def test_components_keep_diagonal_contact_and_original_pixels():
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[0, 0] = 1
    mask[2:4, 2:4] = 1
    mask[4, 4] = 1
    mask[9, 6:11] = 1

    candidates, discarded = candidate_footprints(mask, minimum_area=5, minimum_side=2)

    assert discarded == 2
    (component,) = candidates
    assert component.building_id == "c00002"
    assert component.bbox == (2, 2, 5, 5)
    assert component.pixels.tolist() == [26, 27, 38, 39, 52]
