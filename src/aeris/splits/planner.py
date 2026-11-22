"""Deterministic scene-aware ID and held-out-event partitions."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isclose, isfinite
from typing import Any

from aeris.data.grouping import UnionFind
from aeris.data.labels import DAMAGE_CLASS_NAMES
from aeris.data.manifest import INVALID, STATUSES, Record

ID_SPLITS = ("train", "validation", "calibration", "test")
OOD_SPLIT = "ood_test"
_CLASSES = tuple(DAMAGE_CLASS_NAMES.values())
_ALL_CLASSES = (1 << len(_CLASSES)) - 1
_EVENT_TYPE_BIT = 1 << len(_CLASSES)


class SplitError(ValueError):
    """The manifest or requested partition cannot satisfy the split protocol."""


@dataclass(frozen=True, slots=True)
class _Sample:
    sample_id: str
    event_id: str
    event_type: str | None
    damage: tuple[int, ...]
    links: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _Group:
    members: tuple[_Sample, ...]
    damage: tuple[int, ...]
    events: frozenset[str]

    @property
    def group_id(self) -> str:
        return self.members[0].sample_id


@dataclass(frozen=True, slots=True)
class _Holdout:
    event_id: str
    event_type: str | None
    ood: list[_Sample]
    groups: list[_Group]
    excluded: list[_Sample]
    anchors: tuple[tuple[str, str], ...]
    rank: tuple[bool, int, int, int]
    matching_events: frozenset[str]


def build_split_plan(
    records: Iterable[Record],
    fractions: Mapping[str, object],
    *,
    ood_event: str | None = None,
    id_evaluation_groups: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Build a group-disjoint split and its class-support audit from xBD records."""
    id_fractions = _validate_fractions(fractions)
    samples, invalid = _manifest_samples(records)
    if not samples:
        raise SplitError("Manifest contains no usable samples")

    groups = _group_samples(samples)
    if id_evaluation_groups is not None and ood_event is None:
        raise SplitError("Set ood_event explicitly when pinning ID evaluation groups")
    holdout = _select_holdout(samples, groups, ood_event)
    if id_evaluation_groups is None:
        assigned_groups = _assign_id(holdout.groups, id_fractions, holdout.anchors)
    else:
        assigned_groups = _assign_reviewed_id(holdout.groups, id_evaluation_groups)
    partitions = {
        name: sorted(
            (sample for group in assigned_groups[name] for sample in group.members),
            key=lambda sample: sample.sample_id,
        )
        for name in ID_SPLITS
    }
    partitions[OOD_SPLIT] = holdout.ood
    group_by_sample = {
        sample.sample_id: group.group_id for group in groups for sample in group.members
    }
    _check_partition(partitions, group_by_sample, samples, holdout)
    training_events = {sample.event_id for sample in partitions["train"]}

    return {
        "ood_event": holdout.event_id,
        "ood_event_type": holdout.event_type,
        "same_event_type_in_train": bool(training_events & holdout.matching_events),
        "id_fractions": id_fractions,
        "partitions": {
            name: [sample.sample_id for sample in members]
            for name, members in partitions.items()
        },
        "group_by_sample": group_by_sample,
        "excluded": {
            "invalid_manifest_records": dict(sorted(invalid.items())),
            "groups_linked_to_ood": sorted(sample.sample_id for sample in holdout.excluded),
        },
        "summary": {
            name: {
                "samples": len(members),
                "groups": len({group_by_sample[sample.sample_id] for sample in members}),
                "events": dict(sorted(Counter(sample.event_id for sample in members).items())),
                "damage_class_counts": dict(zip(_CLASSES, _damage_totals(members))),
            }
            for name, members in partitions.items()
        },
    }


def _validate_fractions(values: Mapping[str, object]) -> dict[str, float]:
    if set(values) != set(ID_SPLITS):
        raise SplitError("ID fractions must contain train, validation, calibration and test")
    fractions = {}
    for name in ID_SPLITS:
        value = values[name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0 < value <= 1
            or not isfinite(value)
        ):
            raise SplitError(f"ID fraction for {name} must be a number in (0, 1]")
        fractions[name] = float(value)
    if not isclose(sum(fractions.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise SplitError("ID fractions must add up to 1")
    return fractions


def _manifest_samples(records: Iterable[Record]) -> tuple[list[_Sample], Counter[str]]:
    samples = {}
    invalid: Counter[str] = Counter()
    for line, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise SplitError(f"Manifest record {line} must be a JSON object")
        sample_id = record.get("sample_id")
        status = record.get("validation_status")
        if not isinstance(sample_id, str) or not sample_id.strip() or status not in STATUSES:
            raise SplitError(f"Manifest record {line} has an invalid sample_id or status")
        if status == INVALID:
            invalid[sample_id] += 1
            continue
        if sample_id in samples:
            raise SplitError(f"Duplicate usable sample ID: {sample_id}")

        event_id = record.get("event_id")
        event_type = record.get("event_type")
        if not isinstance(event_id, str) or not event_id.strip():
            raise SplitError(f"{sample_id}: event_id must be a nonempty string")
        if event_type is not None and not isinstance(event_type, str):
            raise SplitError(f"{sample_id}: event_type must be a string or null")

        damage = record.get("damage_class_counts")
        if not isinstance(damage, dict) or any(
            type(damage.get(name)) is not int or damage[name] < 0 for name in _CLASSES
        ):
            raise SplitError(f"{sample_id}: all four damage counts must be nonnegative integers")

        if event_type is not None:
            event_type = event_type.strip().casefold() or None
            if event_type in ("unknown", "n/a"):
                event_type = None
        links = _manifest_links(record)
        samples[sample_id] = _Sample(
            sample_id,
            event_id,
            event_type,
            tuple(damage[name] for name in _CLASSES),
            links,
        )
    conflict = samples.keys() & invalid.keys()
    if conflict:
        raise SplitError(f"Sample ID occurs in both usable and invalid records: {min(conflict)}")
    return [samples[sample_id] for sample_id in sorted(samples)], invalid


def _manifest_links(record: Record) -> tuple[tuple[str, str], ...]:
    hashes = record.get("sha256")
    if hashes is None:
        hashes = {}
    if not isinstance(hashes, dict):
        raise SplitError(f"{record['sample_id']}: sha256 must be an object or null")

    links = {
        "scene": (record.get("scene_id"),),
        "spatial": (record.get("spatial_group_id"),),
        "image_path": (record.get("pre_image_path"), record.get("post_image_path")),
        "image_hash": (hashes.get("pre_image"), hashes.get("post_image")),
    }
    result = []
    for kind, values in links.items():
        for value in values:
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise SplitError(f"{record['sample_id']}: invalid {kind} group key")
            result.append((kind, value))
    return tuple(result)


def _damage_totals(samples: Sequence[_Sample]) -> tuple[int, ...]:
    return tuple(sum(sample.damage[index] for sample in samples) for index in range(len(_CLASSES)))


def _group_samples(samples: Sequence[_Sample]) -> list[_Group]:
    forest = UnionFind()
    for sample in samples:
        forest.add(sample.sample_id)
        for link in sample.links:
            forest.union(sample.sample_id, link)

    grouped = defaultdict(list)
    for sample in samples:
        grouped[forest.find(sample.sample_id)].append(sample)
    return [
        _Group(tuple(members), _damage_totals(members), frozenset(s.event_id for s in members))
        for members in grouped.values()
    ]


def _select_holdout(
    samples: Sequence[_Sample], groups: Sequence[_Group], requested: str | None
) -> _Holdout:
    by_event = defaultdict(list)
    for sample in samples:
        by_event[sample.event_id].append(sample)
    if requested is not None and requested not in by_event:
        raise SplitError(f"OOD event not found among usable samples: {requested!r}")

    event_types = {}
    for event, members in by_event.items():
        known_types = {sample.event_type for sample in members if sample.event_type}
        event_types[event] = known_types.pop() if len(known_types) == 1 else None

    best = None
    rejected = {}
    for event in (sorted(by_event) if requested is None else [requested]):
        ood = by_event[event]
        counts = _damage_totals(ood)
        supported = [count for count in counts if count > 0]
        if not supported:
            rejected[event] = "no classified damage labels"
            continue

        remaining, excluded = [], []
        for group in groups:
            if event in group.events:
                excluded.extend(sample for sample in group.members if sample.event_id != event)
            else:
                remaining.append(group)
        if len(remaining) < len(ID_SPLITS):
            rejected[event] = "fewer than four independent ID groups"
            continue

        event_type = event_types[event]
        matching_events = {
            name for name, kind in event_types.items()
            if event_type is not None and kind == event_type
        }
        anchors, same_type = _coverage_anchors(remaining, matching_events)
        if anchors is None:
            rejected[event] = "cannot cover all four classes in both train and validation"
            continue

        rank = (same_type, len(supported), min(supported), -len(excluded))
        if best is None or rank > best.rank:
            best = _Holdout(
                event, event_type, ood, remaining, excluded, anchors, rank,
                frozenset(matching_events),
            )

    if best is None:
        details = "; ".join(f"{event}: {reason}" for event, reason in rejected.items())
        raise SplitError(f"No feasible held-out event ({details})")
    return best


def _coverage_anchors(
    groups: Sequence[_Group], matching_events: set[str]
) -> tuple[tuple[tuple[str, str], ...] | None, bool]:
    """Find disjoint class-covering train and validation groups.

    A bitmask tracks four damage classes per partition, plus whether train has
    the OOD event type. For each state, keep the solution using the fewest
    groups so there is still room for calibration and ID test.
    """
    anchors = {(0, 0): ()}
    seen_signatures: Counter[tuple[int, int]] = Counter()
    for group in sorted(groups, key=lambda group: (len(group.members), group.group_id)):
        damage = sum(1 << index for index, count in enumerate(group.damage) if count)
        train = damage | (_EVENT_TYPE_BIT if group.events & matching_events else 0)
        signature = (train, damage)
        # Identical coverage can contribute at most once to each partition.
        if seen_signatures[signature] == 2:
            continue
        seen_signatures[signature] += 1
        updated = anchors.copy()
        for (train_mask, validation_mask), selected in anchors.items():
            choices = (
                ((train_mask | train, validation_mask), "train"),
                ((train_mask, validation_mask | damage), "validation"),
            )
            for state, partition in choices:
                previous = updated.get(state)
                if previous is None or len(selected) + 1 < len(previous):
                    updated[state] = (*selected, (group.group_id, partition))
        anchors = updated

    for same_type in (True, False):
        train_mask = _ALL_CLASSES | (_EVENT_TYPE_BIT if same_type else 0)
        selected = anchors.get((train_mask, _ALL_CLASSES))
        if selected is not None and len(selected) <= len(groups) - 2:
            return selected, same_type
    return None, False


def _assign_reviewed_id(
    groups: Sequence[_Group], requested: Mapping[str, object]
) -> dict[str, list[_Group]]:
    """Use reviewed group IDs and keep every evaluation event represented in train."""
    evaluation = ID_SPLITS[1:]
    if set(requested) != set(evaluation):
        raise SplitError("Pinned ID groups must cover validation, calibration and test")

    remaining = {group.group_id: group for group in groups}
    assigned: dict[str, list[_Group]] = {name: [] for name in ID_SPLITS}
    for name in evaluation:
        group_ids = requested[name]
        if not isinstance(group_ids, list) or not group_ids or not all(
            isinstance(group_id, str) and group_id.strip() for group_id in group_ids
        ):
            raise SplitError(f"{name} must name a nonempty list of group IDs")
        for group_id in group_ids:
            group = remaining.pop(group_id, None)
            if group is None:
                raise SplitError(f"ID group unavailable or already assigned: {group_id}")
            assigned[name].append(group)
    assigned["train"] = list(remaining.values())

    train_events = {event for group in assigned["train"] for event in group.events}
    for name in evaluation:
        unseen = {event for group in assigned[name] for event in group.events} - train_events
        if unseen:
            raise SplitError(f"{name} contains events absent from train: {', '.join(sorted(unseen))}")
    return assigned


def _assign_id(
    groups: Sequence[_Group],
    fractions: Mapping[str, float],
    anchors: Sequence[tuple[str, str]],
) -> dict[str, list[_Group]]:
    assigned: dict[str, list[_Group]] = {name: [] for name in ID_SPLITS}
    by_id = {group.group_id: group for group in groups}
    dimensions = {group.group_id: (len(group.members), *group.damage) for group in groups}
    totals = tuple(map(sum, zip(*dimensions.values())))
    counts = {name: [0] * len(totals) for name in ID_SPLITS}

    def add(name: str, group: _Group) -> None:
        assigned[name].append(group)
        counts[name] = [
            current + amount
            for current, amount in zip(counts[name], dimensions[group.group_id])
        ]

    for group_id, name in anchors:
        add(name, by_id.pop(group_id))

    remaining = sorted(by_id.values(), key=lambda group: (-len(group.members), group.group_id))
    for index, group in enumerate(remaining):
        empty = [name for name in ID_SPLITS if not assigned[name]]
        candidates = empty if len(remaining) - index == len(empty) else ID_SPLITS
        amount = dimensions[group.group_id]
        name = min(
            candidates,
            key=lambda name: _balance_delta(counts[name], amount, totals, fractions[name]),
        )
        add(name, group)
    return assigned


def _balance_delta(
    current: Sequence[int], added: Sequence[int], totals: Sequence[int], fraction: float
) -> float:
    changes = [
        ((before + amount) / total - fraction) ** 2 - (before / total - fraction) ** 2
        for before, amount, total in zip(current, added, totals)
    ]
    return changes[0] + sum(changes[1:]) / len(_CLASSES)


def _check_partition(
    partitions: Mapping[str, Sequence[_Sample]],
    groups: Mapping[str, str],
    samples: Sequence[_Sample],
    holdout: _Holdout,
) -> None:
    assignment = {}
    group_assignment = {}
    for name, members in partitions.items():
        if not members:
            raise SplitError(f"Empty partition: {name}")
        for sample in members:
            sample_id = sample.sample_id
            if sample_id in assignment:
                raise SplitError(f"Sample belongs to multiple partitions: {sample_id}")
            assignment[sample_id] = name
            group = groups[sample_id]
            if group_assignment.setdefault(group, name) != name:
                raise SplitError(f"Group belongs to multiple partitions: {group}")
            if (name == OOD_SPLIT) != (sample.event_id == holdout.event_id):
                raise SplitError(f"Incorrect held-out event assignment: {sample_id}")

    expected = {sample.sample_id for sample in samples}
    excluded = {sample.sample_id for sample in holdout.excluded}
    if expected - excluded != assignment.keys():
        raise SplitError("Partition does not account for all usable samples")
    for name in ("train", "validation"):
        if not all(_damage_totals(partitions[name])):
            raise SplitError(f"Missing damage class in {name}")
