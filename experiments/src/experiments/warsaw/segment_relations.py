"""
Measure how a Warsaw scene's labelled objects stand to one another.

A scene labels its faces per class, and those labels overlap: the same faces can be a
cabinet and its door, a drawer and the handle on it. Deciding what that means -- a part,
a duplicate label, a boundary error -- is left to the ontology and to a model. This
module only measures, and reports what it measured:

- how many faces two segments share, and what share that is of each of them,
- how many edges they touch along, which is exact and needs no tolerance,
- how far apart they are, for the parts that touch their whole nowhere at all.

Nothing here decides anything. There is no threshold to tune: a pair is reported when it
shares faces, touches, or is among a segment's nearest, and the numbers go to whoever
does the deciding.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import trimesh
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from experiments.warsaw.world_loader import WarsawWorldLoader


@dataclass
class SegmentDescriptor:
    """
    What one labelled object is, measured rather than judged.
    """

    name: str
    """
    The segment's name, which is also the name of the body it will become.
    """

    class_name: str
    """
    The class the scene labels it with.
    """

    instance: int
    """
    Which object of that class it is.
    """

    faces: int
    """
    How many of the scene's faces it is made of.
    """

    exclusive_faces: int
    """
    How many of those no other segment also claims.
    """

    area: float
    """
    Its surface area, in square metres.
    """

    exclusive_area: float
    """
    How much of that area no other segment also claims.

    This is what makes an instance worth showing as an example of its label: a segment
    can be wholly unclaimed by anything else because it is a stray sliver of a scan, so
    the share alone picks fragments over objects.
    """

    centroid: Tuple[float, float, float]
    """
    The middle of its surface, in the world's frame.
    """

    minimum_corner: Tuple[float, float, float]
    """
    The lowest corner its geometry reaches, in the world's frame.
    """

    maximum_corner: Tuple[float, float, float]
    """
    The highest corner its geometry reaches, in the world's frame.
    """

    components: int
    """
    How many connected pieces it falls into.

    One piece is an object; a dozen scattered pieces is usually an annotation that
    caught fragments of something else.
    """

    height: float
    """
    How high its middle sits above the lowest point of the scene.

    Measured from the scene rather than from the world's origin, which after the loader
    recentres the scene body sits in the middle of the room rather than on its floor.
    """

    @property
    def exclusive_share(self) -> float:
        """
        :return: The share of its faces no other segment claims, between 0 and 1.
        """
        return self.exclusive_faces / self.faces if self.faces else 0.0

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> SegmentDescriptor:
        """
        :param payload: A descriptor as it was written.
        :return: It, as it was measured.
        """
        minimum_corner, maximum_corner = payload["bounding_box"]
        return cls(
            name=payload["name"],
            class_name=payload["class"],
            instance=payload["instance"],
            faces=payload["faces"],
            exclusive_faces=payload["exclusive_faces"],
            area=payload["area"],
            exclusive_area=payload["exclusive_area"],
            centroid=tuple(payload["centroid"]),
            minimum_corner=tuple(minimum_corner),
            maximum_corner=tuple(maximum_corner),
            components=payload["components"],
            height=payload["height"],
        )

    def to_json(self) -> Dict[str, Any]:
        """
        :return: The descriptor as JSON-ready data.
        """
        return {
            "name": self.name,
            "class": self.class_name,
            "instance": self.instance,
            "faces": self.faces,
            "exclusive_faces": self.exclusive_faces,
            "exclusive_share": round(self.exclusive_share, 4),
            "area": round(self.area, 4),
            "exclusive_area": round(self.exclusive_area, 4),
            "centroid": [round(value, 4) for value in self.centroid],
            "bounding_box": [
                [round(value, 4) for value in self.minimum_corner],
                [round(value, 4) for value in self.maximum_corner],
            ],
            "components": self.components,
            "height": round(self.height, 4),
        }


@dataclass
class PairEvidence:
    """
    What was measured about two labelled objects that may stand in some relation.
    """

    one: str
    """
    One segment's name.
    """

    other: str
    """
    The other segment's name.
    """

    shared_faces: int
    """
    How many faces both of them claim.
    """

    share_of_one: float
    """
    What share of the first segment those faces are.
    """

    share_of_other: float
    """
    What share of the second segment those faces are.
    """

    touching_edges: int
    """
    Along how many edges one's faces meet the other's.

    This counts edges of the mesh itself, so it is exact: two segments either share an
    edge or they do not, with no distance to choose.
    """

    distance: float
    """
    How far apart their surfaces are, in metres, measured between face centres.

    ..note:: Measuring between face centres slightly overstates the distance between the
        surfaces themselves, which is immaterial for telling "touching" from "across the
        room" but should not be read as a precise clearance.
    """

    rank_from_one: Optional[int] = None
    """
    Where the second segment ranks among the first's nearest, or None if further away.
    """

    rank_from_other: Optional[int] = None
    """
    Where the first ranks among the second's nearest, or None if further away.
    """

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> PairEvidence:
        """
        :param payload: A pair as it was written, possibly carrying more than the
            measurement -- what the ontology made of it is written beside it.
        :return: The measurement alone.
        """
        return cls(
            one=payload["one"],
            other=payload["other"],
            shared_faces=payload["shared_faces"],
            share_of_one=payload["share_of_one"],
            share_of_other=payload["share_of_other"],
            touching_edges=payload["touching_edges"],
            distance=payload["distance"],
            rank_from_one=payload.get("rank_from_one"),
            rank_from_other=payload.get("rank_from_other"),
        )

    def to_json(self) -> Dict[str, Any]:
        """
        :return: The evidence as JSON-ready data.
        """
        return {
            "one": self.one,
            "other": self.other,
            "shared_faces": self.shared_faces,
            "share_of_one": round(self.share_of_one, 4),
            "share_of_other": round(self.share_of_other, 4),
            "touching_edges": self.touching_edges,
            "distance": round(self.distance, 4),
            "rank_from_one": self.rank_from_one,
            "rank_from_other": self.rank_from_other,
        }

    def as_prompt_block(self, descriptors: Dict[str, SegmentDescriptor]) -> str:
        """
        State the measurements in the words a model reads them in.

        :param descriptors: The scene's segments by name, for their sizes.
        :return: One line per segment, saying what it is and how it meets the other.
        """
        one, other = descriptors[self.one], descriptors[self.other]
        if self.shared_faces:
            overlap = (
                f"they share {self.shared_faces} faces, "
                f"{self.share_of_one:.0%} of {self.one} and "
                f"{self.share_of_other:.0%} of {self.other}"
            )
        elif self.touching_edges:
            overlap = f"they share no faces but touch along {self.touching_edges} edges"
        else:
            overlap = (
                f"they share no faces and touch nowhere; "
                f"their surfaces are {self.distance:.3f} m apart"
            )
        return (
            f"{self.one}: {one.faces} faces, {one.area:.3f} m2, "
            f"middle {one.height:.2f} m up, {one.components} piece(s), "
            f"{one.exclusive_share:.0%} of it claimed by nothing else.\n"
            f"{self.other}: {other.faces} faces, {other.area:.3f} m2, "
            f"middle {other.height:.2f} m up, {other.components} piece(s), "
            f"{other.exclusive_share:.0%} of it claimed by nothing else.\n"
            f"Between them: {overlap}."
        )


@dataclass
class SegmentRelations:
    """
    Everything measured about a scene's labelled objects and how they meet.
    """

    descriptors: Dict[str, SegmentDescriptor]
    """
    Each segment by name.
    """

    pairs: List[PairEvidence]
    """
    Every pair that shares faces, touches, or is among a segment's nearest.
    """

    def pairs_of(self, name: str) -> List[PairEvidence]:
        """
        :param name: The segment to look up.
        :return: Every measured pair it takes part in.
        """
        return [pair for pair in self.pairs if name in (pair.one, pair.other)]

    def to_json(self) -> Dict[str, Any]:
        """
        :return: The whole table as JSON-ready data.
        """
        return {
            "segments": [
                descriptor.to_json() for descriptor in self.descriptors.values()
            ],
            "pairs": [pair.to_json() for pair in self.pairs],
        }


@dataclass
class ClaimantGroup:
    """
    A set of faces claimed by exactly the same segments.
    """

    names: Tuple[str, ...]
    """
    The segments claiming them, by name, in alphabetical order.
    """

    faces: np.ndarray
    """
    The faces every one of them claims.
    """

    def to_json(self) -> Dict[str, Any]:
        """
        :return: The group as JSON-ready data, without the faces themselves.
        """
        return {"claimants": list(self.names), "faces": int(len(self.faces))}


def claimant_groups(
    segment_faces: List[np.ndarray], names: List[str], face_count: int
) -> List[ClaimantGroup]:
    """
    Gather the contested faces by who claims them.

    Which object a face belongs to is one question per set of claimants, not one per pair
    of them: a face claimed by a cabinet, a door and an island is a single question with
    three answers to choose from, where asking it as three pairs invites three answers
    that need not agree. There are far fewer such sets than pairs, and each is asked once.

    :param segment_faces: Per segment, the faces it is made of.
    :param names: The segments' names, in the same order.
    :param face_count: How many faces the scene's mesh has.
    :return: One group per set of claimants, the largest first.
    """
    slots, counts = _claim_slots(segment_faces, face_count)
    contested = np.flatnonzero(counts > 1)
    if not len(contested):
        return []

    rows = np.sort(slots[contested], axis=1)
    sets, inverse = np.unique(rows, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()

    groups = [
        ClaimantGroup(
            names=tuple(sorted(names[int(index)] for index in claimants if index >= 0)),
            faces=contested[inverse == position],
        )
        for position, claimants in enumerate(sets)
    ]
    return sorted(groups, key=lambda group: -len(group.faces))


def _claim_slots(
    segment_faces: List[np.ndarray], face_count: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Record, per face, which segments claim it.

    A face is claimed by only a handful of segments, so the claims fit in a few columns
    and every question about them -- who overlaps whom, who touches whom, which faces
    are claimed once -- becomes an array operation rather than a walk over a million
    faces.

    :param segment_faces: Per segment, the faces it is made of.
    :param face_count: How many faces the scene's mesh has.
    :return: The claiming segments per face, one per column and -1 where unclaimed, and
        how many segments claim each face.
    """
    counts = np.zeros(face_count, dtype=np.int32)
    for faces in segment_faces:
        counts[faces] += 1

    slots = np.full((face_count, max(int(counts.max()), 1)), -1, dtype=np.int32)
    filled = np.zeros(face_count, dtype=np.int32)
    for index, faces in enumerate(segment_faces):
        slots[faces, filled[faces]] = index
        filled[faces] += 1
    return slots, counts


def _shared_faces(slots: np.ndarray, counts: np.ndarray) -> Dict[Tuple[int, int], int]:
    """
    :param slots: The claiming segments per face.
    :param counts: How many segments claim each face.
    :return: Per pair of segments, how many faces both claim.
    """
    shared: Dict[Tuple[int, int], int] = {}
    contested = slots[counts > 1]
    for one, other in combinations(range(slots.shape[1]), 2):
        left, right = contested[:, one], contested[:, other]
        both = (left >= 0) & (right >= 0)
        if not both.any():
            continue
        pairs, occurrences = np.unique(
            np.sort(np.stack([left[both], right[both]], axis=1), axis=1),
            axis=0,
            return_counts=True,
        )
        for (low, high), occurrence in zip(pairs, occurrences):
            key = (int(low), int(high))
            shared[key] = shared.get(key, 0) + int(occurrence)
    return shared


def _edges_between_segments(
    adjacency: np.ndarray, slots: np.ndarray
) -> Tuple[Dict[Tuple[int, int], int], Dict[int, np.ndarray]]:
    """
    Walk the mesh's edges once, sorting each into the segments it runs between or
    within.

    :param adjacency: The pairs of faces sharing an edge.
    :param slots: The claiming segments per face.
    :return: Per pair of segments, how many edges they meet along; and per segment, the
        edges running inside it.
    """
    crossing = []
    internal: Dict[int, List[np.ndarray]] = defaultdict(list)
    columns = slots.shape[1]
    for one in range(columns):
        left = slots[adjacency[:, 0], one]
        for other in range(columns):
            right = slots[adjacency[:, 1], other]
            both = (left >= 0) & (right >= 0)
            if not both.any():
                continue
            edges = np.flatnonzero(both)
            same = left[edges] == right[edges]
            for segment in np.unique(left[edges][same]):
                internal[int(segment)].append(edges[same][left[edges][same] == segment])
            differing = edges[~same]
            if len(differing):
                crossing.append(
                    np.column_stack(
                        [
                            differing,
                            np.sort(
                                np.stack([left[differing], right[differing]], axis=1),
                                axis=1,
                            ),
                        ]
                    )
                )

    touching: Dict[Tuple[int, int], int] = {}
    if crossing:
        # An edge whose faces each carry both segments is found twice, once per way of
        # reading it, so it is counted once per edge rather than once per reading.
        once = np.unique(np.concatenate(crossing), axis=0)
        pairs, occurrences = np.unique(once[:, 1:], axis=0, return_counts=True)
        touching = {
            (int(low), int(high)): int(occurrence)
            for (low, high), occurrence in zip(pairs, occurrences)
        }

    return touching, {
        segment: np.concatenate(found) for segment, found in internal.items()
    }


def _distance_between(
    one: int, other: int, trees: List[cKDTree], points: List[np.ndarray]
) -> float:
    """
    Measure how far apart two segments' face centres are.

    The distance is symmetric, so the smaller set of points is thrown at the larger
    segment's tree: a handle against a door is a hundred queries rather than a hundred
    thousand.

    :param one: One segment's index.
    :param other: The other segment's index.
    :param trees: Per segment, a search tree over its face centres.
    :param points: Per segment, its face centres.
    :return: The distance between their nearest face centres, in metres.
    """
    if len(points[one]) <= len(points[other]):
        return float(trees[other].query(points[one])[0].min())
    return float(trees[one].query(points[other])[0].min())


def _nearest_neighbours(
    trees: List[cKDTree],
    points: List[np.ndarray],
    center_bounds: np.ndarray,
    how_many: int,
) -> List[List[Tuple[float, int]]]:
    """
    Find each segment's nearest others, exactly.

    The boxes around two segments are never further apart than the segments themselves,
    so candidates are visited in order of that lower bound and the search stops once the
    bound passes the worst neighbour already found. That prunes almost every candidate
    without ever discarding one that could have been nearer.

    :param trees: Per segment, a search tree over its face centres.
    :param points: Per segment, its face centres.
    :param center_bounds: Per segment, the lowest and highest corner of those centres.
    :param how_many: How many neighbours to keep.
    :return: Per segment, its nearest others as (distance, segment index), nearest
        first.
    """
    minimum_corners, maximum_corners = center_bounds[:, 0], center_bounds[:, 1]
    neighbours: List[List[Tuple[float, int]]] = []
    for index in range(len(trees)):
        gaps = np.maximum(
            0.0,
            np.maximum(
                minimum_corners - maximum_corners[index],
                minimum_corners[index] - maximum_corners,
            ),
        )
        lower_bounds = np.linalg.norm(gaps, axis=1)
        lower_bounds[index] = np.inf

        found: List[Tuple[float, int]] = []
        for candidate in np.argsort(lower_bounds):
            if not np.isfinite(lower_bounds[candidate]):
                break
            if len(found) >= how_many and lower_bounds[candidate] >= found[-1][0]:
                break
            distance = _distance_between(index, int(candidate), trees, points)
            found.append((distance, int(candidate)))
            found.sort()
            del found[how_many:]
        neighbours.append(found)
    return neighbours


def segment_evidence(loader: "WarsawWorldLoader", nearest: int = 5) -> SegmentRelations:
    """
    Measure a scene's labelled objects and how they meet.

    A pair is reported when the two segments share faces, touch along an edge, or are
    among each other's *nearest* nearest neighbours. Everything else in the scene stands
    in no measurable relation and is left out.

    :param loader: The loader holding the scene, whose world-frame mesh is measured so
        that heights and distances are the ones the world uses.
    :param nearest: How many nearest neighbours each segment reports.
    :return: Every segment's description and every measured pair.
    """
    segments = loader.label_segments
    mesh = loader.scene_mesh
    centres = mesh.triangles_center
    areas = mesh.area_faces
    floor = float(mesh.vertices[:, 2].min())

    slots, counts = _claim_slots(
        [segment.faces for segment in segments], len(mesh.faces)
    )
    claimed_once = counts == 1

    # Only edges between labelled faces can tell two segments apart, and they are a
    # fraction of the mesh's own.
    labelled = counts > 0
    adjacency = mesh.face_adjacency
    labelled_adjacency = adjacency[
        labelled[adjacency[:, 0]] & labelled[adjacency[:, 1]]
    ]

    shared = _shared_faces(slots, counts)
    touching, internal_edges = _edges_between_segments(labelled_adjacency, slots)

    descriptors: Dict[str, SegmentDescriptor] = {}
    points: List[np.ndarray] = []
    trees: List[cKDTree] = []
    center_bounds = np.zeros((len(segments), 2, 3))
    for index, segment in enumerate(segments):
        faces = segment.faces
        segment_centres = centres[faces]
        points.append(segment_centres)
        trees.append(cKDTree(segment_centres))
        center_bounds[index] = [
            segment_centres.min(axis=0),
            segment_centres.max(axis=0),
        ]

        vertices = mesh.vertices[mesh.faces[faces].ravel()]
        own_edges = labelled_adjacency[
            internal_edges.get(index, np.empty(0, dtype=np.int64))
        ]
        pieces = trimesh.graph.connected_components(own_edges, nodes=faces)

        descriptors[str(segment.name)] = SegmentDescriptor(
            name=str(segment.name),
            class_name=segment.class_name,
            instance=segment.instance,
            faces=len(faces),
            exclusive_faces=int(claimed_once[faces].sum()),
            area=float(areas[faces].sum()),
            exclusive_area=float(areas[faces][claimed_once[faces]].sum()),
            centroid=tuple(segment_centres.mean(axis=0)),
            minimum_corner=tuple(vertices.min(axis=0)),
            maximum_corner=tuple(vertices.max(axis=0)),
            components=len(pieces),
            height=float(segment_centres[:, 2].mean()) - floor,
        )

    neighbours = _nearest_neighbours(trees, points, center_bounds, nearest)
    ranks: Dict[Tuple[int, int], int] = {}
    distances: Dict[Tuple[int, int], float] = {}
    for index, found in enumerate(neighbours):
        for rank, (distance, other) in enumerate(found, start=1):
            ranks[(index, other)] = rank
            key = (min(index, other), max(index, other))
            distances[key] = min(distances.get(key, np.inf), distance)

    names = [str(segment.name) for segment in segments]
    sizes = [len(segment.faces) for segment in segments]
    pairs = []
    for one, other in sorted(set(shared) | set(touching) | set(distances)):
        overlap = shared.get((one, other), 0)
        pairs.append(
            PairEvidence(
                one=names[one],
                other=names[other],
                shared_faces=overlap,
                share_of_one=overlap / sizes[one] if sizes[one] else 0.0,
                share_of_other=overlap / sizes[other] if sizes[other] else 0.0,
                touching_edges=touching.get((one, other), 0),
                distance=distances.get(
                    (one, other),
                    (
                        0.0
                        if overlap or touching.get((one, other))
                        else _distance_between(one, other, trees, points)
                    ),
                ),
                rank_from_one=ranks.get((one, other)),
                rank_from_other=ranks.get((other, one)),
            )
        )

    return SegmentRelations(descriptors=descriptors, pairs=pairs)
