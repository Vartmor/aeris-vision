"""Building-level evaluation on a shared raster grid."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Set
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral


@dataclass(frozen=True, slots=True)
class BuildingMask:
    """One building's occupied pixels, indexed as row * image_width + column.

    Damage classes use the project IDs 1 (no damage) through 4 (destroyed).
    A reference with unclassified damage uses None.
    """

    building_id: str
    pixels: Set[int]
    damage_class: int | None


@dataclass(frozen=True, slots=True)
class BuildingMatch:
    reference_id: str
    candidate_id: str
    iou: float


@dataclass(frozen=True, slots=True)
class MergedCandidate:
    candidate_id: str
    reference_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SevereDamageCounts:
    not_detected: int
    in_merged_candidate: int
    no_damage: int
    minor_damage: int
    major_damage: int
    destroyed: int

    @property
    def total(self) -> int:
        """Number of reference buildings labeled major or destroyed."""
        return (
            self.not_detected
            + self.in_merged_candidate
            + self.no_damage
            + self.minor_damage
            + self.major_damage
            + self.destroyed
        )


@dataclass(frozen=True, slots=True)
class BuildingEvaluation:
    matches: tuple[BuildingMatch, ...]
    unmatched_reference_ids: tuple[str, ...]
    unmatched_candidate_ids: tuple[str, ...]
    merged_candidates: tuple[MergedCandidate, ...]
    merged_reference_ids: tuple[str, ...]
    severe_damage: SevereDamageCounts

    @property
    def localization_recall(self) -> float | None:
        """Matched references divided by all labeled building geometries."""
        reference_count = len(self.matches) + len(self.unmatched_reference_ids)
        if reference_count == 0:
            return None
        return len(self.matches) / reference_count


def _index_masks(
    masks: Iterable[BuildingMask], pixel_count: int, *, candidate: bool
) -> dict[str, BuildingMask]:
    indexed: dict[str, BuildingMask] = {}
    for mask in masks:
        building_id = mask.building_id
        if not isinstance(building_id, str) or not building_id:
            raise ValueError("Building IDs must be nonempty strings")
        if building_id in indexed:
            raise ValueError(f"Duplicate building ID {building_id!r}")
        if not mask.pixels:
            raise ValueError(f"Building {building_id!r} has an empty mask")
        if any(
            not isinstance(pixel, Integral)
            or isinstance(pixel, bool)
            or not 0 <= pixel < pixel_count
            for pixel in mask.pixels
        ):
            raise ValueError(f"Building {building_id!r} has pixels outside the raster grid")

        damage_class = mask.damage_class
        if (candidate or damage_class is not None) and (
            not isinstance(damage_class, Integral)
            or isinstance(damage_class, bool)
            or not 1 <= damage_class <= 4
        ):
            raise ValueError(f"Building {building_id!r} has an invalid damage class")
        indexed[building_id] = mask
    return indexed


def _summarize_severe_damage(
    references: Iterable[BuildingMask],
    matches: Iterable[BuildingMatch],
    candidates: dict[str, BuildingMask],
    merged_references: set[str],
) -> SevereDamageCounts:
    matched_classes = {
        match.reference_id: candidates[match.candidate_id].damage_class
        for match in matches
    }
    not_detected = 0
    in_merged_candidate = 0
    by_class: Counter[int] = Counter()

    for reference in references:
        if reference.damage_class not in (3, 4):
            continue

        reference_id = reference.building_id
        if reference_id in merged_references:
            in_merged_candidate += 1
            continue
        predicted_class = matched_classes.get(reference_id)
        if predicted_class is None:
            not_detected += 1
        else:
            by_class[predicted_class] += 1

    return SevereDamageCounts(
        not_detected=not_detected,
        in_merged_candidate=in_merged_candidate,
        no_damage=by_class[1],
        minor_damage=by_class[2],
        major_damage=by_class[3],
        destroyed=by_class[4],
    )


def evaluate_buildings(
    references: Iterable[BuildingMask],
    candidates: Iterable[BuildingMask],
    *,
    image_width: int,
    image_height: int,
) -> BuildingEvaluation:
    """Evaluate one image using the v1.1 building matching protocol.

    Matching is greedy at IoU >= 1/2. A candidate is a merge if it covers
    more than half of the area of at least two references. References
    inside a merge take precedence in the severe damage summary.
    """
    if (
        not isinstance(image_width, Integral)
        or isinstance(image_width, bool)
        or image_width < 1
        or not isinstance(image_height, Integral)
        or isinstance(image_height, bool)
        or image_height < 1
    ):
        raise ValueError("Image width and height must be positive integers")

    pixel_count = int(image_width) * int(image_height)
    reference_by_id = _index_masks(references, pixel_count, candidate=False)
    candidate_by_id = _index_masks(candidates, pixel_count, candidate=True)

    candidates_by_pixel: dict[int, list[str]] = defaultdict(list)
    for candidate_id, mask in candidate_by_id.items():
        for pixel in mask.pixels:
            candidates_by_pixel[pixel].append(candidate_id)

    eligible_pairs: list[tuple[Fraction, str, str]] = []
    covered_references: dict[str, list[str]] = defaultdict(list)
    for reference_id, reference in reference_by_id.items():
        intersections: Counter[str] = Counter()
        for pixel in reference.pixels:
            intersections.update(candidates_by_pixel.get(pixel, ()))

        reference_area = len(reference.pixels)
        for candidate_id, intersection in intersections.items():
            if 2 * intersection > reference_area:
                covered_references[candidate_id].append(reference_id)

            candidate_area = len(candidate_by_id[candidate_id].pixels)
            union = reference_area + candidate_area - intersection
            if 2 * intersection >= union:
                eligible_pairs.append(
                    (Fraction(intersection, union), reference_id, candidate_id)
                )

    eligible_pairs.sort(key=lambda pair: (-pair[0], pair[1], pair[2]))
    matched_references: set[str] = set()
    matched_candidates: set[str] = set()
    matches: list[BuildingMatch] = []
    for iou, reference_id, candidate_id in eligible_pairs:
        if reference_id in matched_references or candidate_id in matched_candidates:
            continue
        matches.append(BuildingMatch(reference_id, candidate_id, float(iou)))
        matched_references.add(reference_id)
        matched_candidates.add(candidate_id)

    merged_candidates = tuple(
        MergedCandidate(candidate_id, tuple(sorted(reference_ids)))
        for candidate_id, reference_ids in sorted(covered_references.items())
        if len(reference_ids) >= 2
    )
    merged_references = {
        reference_id
        for candidate in merged_candidates
        for reference_id in candidate.reference_ids
    }

    return BuildingEvaluation(
        matches=tuple(matches),
        unmatched_reference_ids=tuple(
            sorted(reference_by_id.keys() - matched_references)
        ),
        unmatched_candidate_ids=tuple(
            sorted(candidate_by_id.keys() - matched_candidates)
        ),
        merged_candidates=merged_candidates,
        merged_reference_ids=tuple(sorted(merged_references)),
        severe_damage=_summarize_severe_damage(
            reference_by_id.values(), matches, candidate_by_id, merged_references
        ),
    )
