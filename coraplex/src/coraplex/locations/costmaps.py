from __future__ import annotations

import logging
import random
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from skimage.measure import label
from typing_extensions import Tuple, List, Optional, Iterator, Callable, TYPE_CHECKING

from coraplex.locations.base import PoseGeneratorBackend
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Quaternion,
    RotationMatrix,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose, Point3, Vector3
from semantic_digital_twin.world import World

if TYPE_CHECKING:
    from coraplex.datastructures.dataclasses import Context

logger = logging.getLogger("coraplex")


class OrientationGenerator:
    """
    Provides methods to generate orientations for pose candidates.
    """

    @staticmethod
    def generate_origin_orientation(
        position: Point3, origin: Pose, rotate_by_angle: float = 0
    ) -> Quaternion:
        """
        Generates an orientation such that the robot faces the origin of the locations.

        :param position: The position in the locations, already converted to the world coordinate frame.
        :param origin: The origin of the locations, the point which the robot should face.
        :param rotate_by_angle: Angle to rotate the orientation.
        :return: A quaternion of the calculated orientation.
        """
        rotation_R_new_rotation = RotationMatrix.from_rpy(0, 0, rotate_by_angle)
        angle = (
            np.arctan2(
                position.y - origin.y,
                position.x - origin.x,
            )
            + np.pi
        )[0]
        world_R_rotation = RotationMatrix.from_rpy(0, 0, angle)
        world_R_new_rotation = world_R_rotation @ rotation_R_new_rotation
        return world_R_new_rotation.to_quaternion()

    @staticmethod
    def orientation_generator_for_axis(
        axis: Vector3,
    ) -> Callable[[Point3, Pose], Quaternion]:
        """
        Creates an orientation generator where the given axis is facing the target.

        :param axis: The axis which should be facing the target
        :return: A callable orientation generator
        """
        rotation = axis[1] * (np.pi / 2) * -1
        return partial(
            OrientationGenerator.generate_origin_orientation, rotate_by_angle=rotation
        )

    @staticmethod
    def generate_random_orientation(
        *_, random_number_generator: random.Random = random.Random(42)
    ) -> Quaternion:
        """
        Generates a random orientation rotated around the z-axis (yaw).
        A random angle is sampled using a provided RNG instance to ensure reproducibility.

        :param _: Ignored parameters to maintain compatibility with other orientation generators.
        :param random_number_generator: Random number generator instance for reproducible sampling.

        :return: A quaternion of the randomly generated orientation.
        """
        return Quaternion.from_rpy(0, 0, random_number_generator.uniform(0, 2 * np.pi))


@dataclass
class Rectangle:
    """
    A rectangle that is described by a lower and upper x and y value.
    """

    x_lower: float
    x_upper: float
    y_lower: float
    y_upper: float

    def translate(self, x: float, y: float):
        """Translate the rectangle by x and y"""
        self.x_lower += x
        self.x_upper += x
        self.y_lower += y
        self.y_upper += y

    def scale(self, x_factor: float, y_factor: float):
        """Scale the rectangle by x_factor and y_factor"""
        self.x_lower *= x_factor
        self.x_upper *= x_factor
        self.y_lower *= y_factor
        self.y_upper *= y_factor


@dataclass
class Costmap(PoseGeneratorBackend):
    """
    The base class of all Costmaps.
    Costmaps describe regions in the world that are suitable for a certaint task.
    """

    resolution: float
    """
    The distance in metre in the real-world which is represented by a single entry in the locations. 
    """
    height: Optional[int] = field(kw_only=True, default=None)
    """
    Height of the locations.
    """
    width: Optional[int] = field(kw_only=True, default=None)
    """
    Width of the locations.
    """
    origin: Pose = field(kw_only=True, default_factory=Pose)
    """
    Origin pose of the locations.
    """
    map: np.ndarray = field(default_factory=lambda: np.zeros((10, 10)), kw_only=True)
    """
    Numpy array to save the locations distribution
    
    Costmaps represent the 2D distribution in a numpy array where axis 0 is the X-Axis of the coordinate system and axis 1 
    is the Y-Axis of the coordinate system. An increase in the index of the axis of the numpy array corresponds to an increase in the 
    value of the spatial axis. The factor by how the value of the index of the numpy corresponds to the spatial coordinate 
    system is given by the resolution. 

    Furthermore, there is a difference in the origin of the two representations while the numpy arrays start from the top left 
    corner, the origin given as Pose is placed in the middle of the array. The locations is build around the origin and 
    since the array start from 0, 0 in the corner this conversion is necessary. 

                y-axis      0, 10
        0,0 ------------------
            ------------------
            ------------------
    x-axis  ------------------
            ------------------
            ------------------
      10, 0 ------------------
    """

    world: World
    """
    The world from which this locations was created.
    """

    number_of_samples: int = field(kw_only=True, default=200)
    """
    Number of samples to draw from each region of the map, at most.

    Every region is a place of its own to stand in, and how many of them a map falls
    apart into says nothing about how many candidates any one of them owes, so the budget
    is per region rather than shared out between them.
    """

    sample_randomly: bool = field(kw_only=True, default=False)
    """
    If the sampling should randomly pick valid entries
    """

    orientation_generator: Optional[Callable[[Point3, Pose], Quaternion]] = field(
        kw_only=True, default=None
    )
    """
    An optional orientatoin generator to use to generate the orientation for a sampled pose
    """

    def _find_consecutive_line_length(
        self, start: Tuple[int, int], occupancy_map: np.ndarray
    ) -> int:
        """
        Finds the number of consecutive entries in the locations which are greater
        than zero.

        :param start: The indices in the locations from which the consecutive line should be found.
        :param occupancy_map: The locations in which the line should be found.
        :return: The length of the consecutive line of entries greater than zero.
        """
        width = occupancy_map.shape[1]
        length = 0
        for column in range(start[1], width):
            if occupancy_map[start[0]][column] > 0:
                length += 1
            else:
                return length
        return length

    def _find_maximal_box_height(
        self, start: Tuple[int, int], length: int, occupancy_map: np.ndarray
    ) -> int:
        """
        Finds the maximal height for a rectangle with a given width in a locations.
        The method traverses one row at a time and checks if all entries for the
        given width are greater than zero. If an entry is less or equal than zero
        the height is returned.

        :param start: The indices in the locations from which the method should start.
        :param length: The given width for the rectangle
        :param occupancy_map: The locations in which should be searched.
        :return: The height of the rectangle.
        """
        height, width = occupancy_map.shape
        box_height = 1
        for row in range(start[0], height):
            for column in range(start[1], start[1] + length):
                if occupancy_map[row][column] <= 0:
                    return box_height
            box_height += 1
        return box_height

    def merge(self, other: Costmap) -> Costmap:
        """
        Merges the values of two locations and returns a new locations that has for
        every cell the merged values of both inputs. To merge two locations they
        need to fulfill 3 constrains:

        1. They need to have the same size
        2. They need to have the same x and y coordinates in the origin
        3. They need to have the same resolution

        If any of these constrains is not fulfilled a ValueError will be raised.

        :param other: The other locations with which this locations should be merged.
        :return: A new locations that contains the merged values
        """
        if self.width != other.width or self.height != other.height:
            raise ValueError("You can only merge locations of the same size.")
        elif (
            not np.allclose(self.origin.x, other.origin.x)
            or not np.allclose(self.origin.y, other.origin.y)
            or not np.allclose(
                self.origin.to_rotation_matrix(), other.origin.to_rotation_matrix()
            )
        ):
            raise ValueError(
                "To merge locations, the x and y coordinate as well as the orientation must be equal."
            )
        elif self.resolution != other.resolution:
            raise ValueError("To merge two locations their resolution must be equal.")
        elif self.world != other.world:
            raise ValueError(
                "To merge two locations they must belong to the same world."
            )
        new_map = np.zeros((self.height, self.width))
        # A numpy array of the positions where both locations are greater than 0
        overlap = np.logical_and(self.map > 0, other.map > 0)
        new_map[overlap] = self.map[overlap] * other.map[overlap]
        maximum_value = np.max(new_map)
        if maximum_value != 0:
            new_map = (new_map / np.max(new_map)).reshape((self.height, self.width))
        else:
            new_map = new_map.reshape((self.height, self.width))
            logger.warning("Merged locations is empty.")
        return Costmap(
            resolution=self.resolution,
            height=self.height,
            width=self.width,
            origin=self.origin,
            map=new_map,
            world=self.world,
        )

    def __add__(self, other: Costmap) -> Costmap:
        """
        Overloading of the "+" operator for merging of Costmaps. Furthermore, checks if 'other' is actual a Costmap and
        raises a ValueError if this is not the case. Please check :func:`~Costmap.merge` for further information of merging.

        :param other: Another Costmap
        :return: A new Costmap that contains the merged values from this Costmap and the other Costmap
        """
        if isinstance(other, Costmap):
            return self.merge(other)
        else:
            raise ValueError(
                f"Can only combine two locations other type was {type(other)}"
            )

    def __and__(self, other):
        return self.merge(other)

    def partitioning_rectangles(self) -> List[Rectangle]:
        """
        Partition the map attached to this locations into rectangles. The rectangles are axis aligned, exhaustive and
        disjoint sets.

        :return: A list containing the partitioning rectangles
        """
        unassigned_map = np.copy(self.map)
        origin = np.array([self.height / 2, self.width / 2]) * -1
        rectangles = []

        # for every index pair (row, column) in the occupancy locations
        for row in range(0, self.map.shape[0]):
            for column in range(0, self.map.shape[1]):

                # if this index has not been used yet
                if unassigned_map[row][column] > 0:
                    line_width = self._find_consecutive_line_length(
                        (row, column), unassigned_map
                    )
                    start_index = (row, column)
                    box_height = self._find_maximal_box_height(
                        (row, column), line_width, unassigned_map
                    )

                    # calculate the rectangle in the locations
                    x_lower = start_index[0]
                    x_upper = start_index[0] + box_height
                    y_lower = start_index[1]
                    y_upper = start_index[1] + line_width

                    # mark the found rectangle as occupied
                    unassigned_map[
                        row : row + box_height, column : column + line_width
                    ] = 0

                    # transform rectangle to map space
                    rectangle = Rectangle(x_lower, x_upper, y_lower, y_upper)
                    rectangle.translate(*origin)
                    rectangle.scale(self.resolution, self.resolution)
                    rectangles.append(rectangle)

        return rectangles

    def __iter__(self) -> Iterator[Pose]:
        """
        A generator that crates pose candidates from a given locations. The generator
        selects the highest 100 values and returns the corresponding positions.
        Orientations are calculated such that the Robot faces the center of the locations.

        :Yield: A tuple of position and orientation
        """

        orientation_generator = (
            self.orientation_generator
            or OrientationGenerator.generate_origin_orientation
        )

        # Determines how many positions should be sampled from the locations
        if (
            self.number_of_samples == -1
            or self.number_of_samples > self.map.flatten().shape[0]
        ):
            self.number_of_samples = self.map.flatten().shape[0]

        for segment in self.segment_map():
            weights = segment.flatten()
            sample_count = min(self.number_of_samples, np.count_nonzero(weights))
            if sample_count == 0:
                continue

            if self.sample_randomly:
                # A costmap is a distribution, so a random draw follows its values: an
                # even draw over the whole segment would spend most samples on the cells
                # the map rates worst, and on the zeroed ones it rates unusable.
                indices = np.random.choice(
                    segment.size,
                    sample_count,
                    replace=False,
                    p=weights / weights.sum(),
                )
            else:
                indices = np.argpartition(weights, -sample_count)[-sample_count:]

            # Best first, so a consumer that stops at its first usable candidate spends
            # its time on the places the map rates highest.
            indices = indices[np.argsort(weights[indices])[::-1]]
            indices = np.dstack(np.unravel_index(indices, self.map.shape)).reshape(
                -1, 2
            )

            height = segment.shape[0]
            width = segment.shape[1]
            center = np.array([height // 2, width // 2])
            for index in indices:
                if segment[index[0]][index[1]] == 0:
                    continue
                # Compute world position independent of origin orientation:
                # map indices increase with world axes; origin is at the center.
                offset = (index - center) * self.resolution
                position = self.origin.to_position() + Vector3(offset[0], offset[1], 0)

                orientation: Quaternion = orientation_generator(position, self.origin)
                yield Pose(
                    position,
                    orientation,
                    self.world.root,
                )

    def segment_map(self) -> List[np.ndarray]:
        """
        Finds partitions in the locations and isolates them, a partition is a number of entries in the locations which are
        neighbours. Returns a list of numpy arrays with one partition per array.

        :return: A list of numpy arrays with one partition per array
        """
        # In case the map is empty we just return the map
        if np.sum(self.map) == 0:
            return [self.map]

        discrete_map = np.copy(self.map)
        # Label only works on integer arrays
        discrete_map[discrete_map != 0] = 1

        labeled_map, number_of_labels = label(
            discrete_map, return_num=True, connectivity=2
        )
        segments = []
        # We don't want the maps for value 0
        for label_value in range(1, number_of_labels + 1):
            isolated_segment = deepcopy(self.map)
            isolated_segment[labeled_map != label_value] = 0
            segments.append(isolated_segment)
        # Maps with the highest values go first
        segments.sort(key=lambda segment: np.max(segment), reverse=True)
        return segments


@dataclass
class OccupancyCostmap(Costmap):
    """
    The occupancy Costmap represents a map of the environment where obstacles or
    positions which are inaccessible for a robot have a value of -1.
    """

    distance_to_obstacle: float
    """
    The distance by which obstacles in the occupancy map are inflated and are therefore not valid positions, in meter
    """

    robot_view: AbstractRobot
    """
    Robot semantic annotation which is used to create the map
    """

    base_clearance: float = field(kw_only=True, default=0.1)
    """
    How much room, in meters, a robot base is given beyond its own radius when obstacles
    are inflated, so that standing on a free cell leaves it clear of what is next to it.
    """

    _distance_to_obstacle_index: int = field(init=False, default=None)
    """
    Conversion of the distance_to_obstacle to index range for the internal representation.
    """

    def __post_init__(self):
        self._distance_to_obstacle_index = max(
            int(self.distance_to_obstacle / self.resolution), 1
        )
        self.map = self._create_from_world()

    def create_ray_mask_around_origin(self):
        """
        Determines the occupied space around the origin position using ray testing. A ray is cast from the ground
        straight up 10m and if it hits something the position is considered occupied.

        :return: A 2d numpy array of the occupied space
        """
        origin_position = self.origin.to_position().to_list()
        # Generate 2d grid with indices
        indices = np.concatenate(
            np.dstack(
                np.mgrid[
                    int(-self.width / 2) : int(self.width / 2),
                    int(-self.width / 2) : int(self.width / 2),
                ]
            ),
            axis=0,
        ) * self.resolution + np.array(origin_position[:2])

        # base height of the robot plus a safty offset
        base_height = self.robot_view.mobile_base.bounding_box.height + 0.1
        # Add the z-coordinate to the grid, which is either 0 or 10
        ray_starts = np.pad(
            indices, (0, 1), mode="constant", constant_values=base_height
        )[:-1]
        ray_ends = np.pad(indices, (0, 1), mode="constant", constant_values=0)[:-1]
        # Zips both arrays such that there are tuples for every coordinate that
        # only differ in the z-coordinate
        rays = np.dstack(np.dstack((ray_starts, ray_ends))).T

        free_cells = np.ones(len(rays))

        ray_tracer = RayTracer(self.world)
        ray_hits = ray_tracer.ray_test(rays[:, 0], rays[:, 1])
        if self.robot_view:
            free_cells[ray_hits[1]] = [
                (
                    1
                    if ray_hits[2][hit_index]
                    in self.world.get_kinematic_structure_entities_of_branch(
                        self.robot_view.root
                    )
                    else 0
                )
                for hit_index in range(len(ray_hits[1]))
            ]
        else:
            free_cells[ray_hits[1]] = 0

        free_cells = np.flip(np.reshape(np.array(free_cells), (self.width, self.width)))
        return free_cells

    def inflate_obstacles(self, occupancy_map: np.ndarray):
        """
        Inflates found obstacles in the environment by the distance_to_obstacle factor.

        :param occupancy_map: Map of obstacles to inflate.
        :return: The map with inflated obstacles.
        """
        neighbourhood_shape = (
            self._distance_to_obstacle_index * 2,
            self._distance_to_obstacle_index * 2,
        )
        view_shape = (
            tuple(np.subtract(occupancy_map.shape, neighbourhood_shape) + 1)
            + neighbourhood_shape
        )
        strides = occupancy_map.strides + occupancy_map.strides

        neighbourhoods = np.lib.stride_tricks.as_strided(
            occupancy_map, view_shape, strides
        )
        neighbourhoods = neighbourhoods.reshape(neighbourhoods.shape[:-2] + (-1,))

        neighbourhood_sum = np.sum(neighbourhoods, axis=2)
        occupancy_map = (
            neighbourhood_sum == (self._distance_to_obstacle_index * 2) ** 2
        ).astype("int16")
        return occupancy_map

    def _create_from_world(self) -> np.ndarray:
        """
        Creates an Occupancy Costmap for the specified World.
        This map marks every position as valid that has no object above it. After
        creating the locations the distance to obstacle parameter is applied.
        """

        ray_mask = self.create_ray_mask_around_origin()

        occupancy_map = np.pad(
            ray_mask,
            (
                int(self._distance_to_obstacle_index / 2),
                int(self._distance_to_obstacle_index / 2),
            ),
        )

        occupancy_map = self.inflate_obstacles(occupancy_map)
        # The map loses some size due to the strides and because I dont want to
        # deal with indices outside of the index range
        offset = self.width - occupancy_map.shape[0]
        odd = 0 if offset % 2 == 0 else 1
        occupancy_map = np.pad(occupancy_map, (offset // 2, offset // 2 + odd))

        return np.flip(occupancy_map)

    @classmethod
    def default_distance_to_obstacle(cls, robot: AbstractRobot) -> float:
        """
        :param robot: The robot that has to stand on the free cells.
        :return: How far obstacles are inflated for that robot: the radius of its base
            plus :attr:`base_clearance`.
        """
        base_bounding_box = robot.mobile_base.bounding_box
        return (
            base_bounding_box.depth / 2 + base_bounding_box.width / 2
        ) / 2 + cls.base_clearance

    @classmethod
    def default_map(cls, context: Context, target: Pose) -> OccupancyCostmap:
        """
        Creates an occupancy costmap with some default values, the most important one being that the distance_to_obstacle
        is set to the radius of the robot base.

        :param context: The context to create the occupancy cost map.
        :param target: The target pose for the occupancy cost map.
        :returns: A occupancy cost map with default values.
        """
        ground_pose = deepcopy(target)
        ground_pose.z = 0

        return OccupancyCostmap(
            resolution=0.02,
            width=200,
            height=200,
            world=context.world,
            distance_to_obstacle=cls.default_distance_to_obstacle(context.robot),
            robot_view=context.robot,
            origin=ground_pose,
        )


@dataclass
class VisibilityCostmap(Costmap):
    """
    A locations that represents the visibility of a specific point for every position around
    this point. For a detailed explanation on how the creation of the locations works
    please look here: `PhD Thesis (page 173) <https://mediatum.ub.tum.de/doc/1239461/1239461.pdf>`_
    """

    minimal_height: float
    """
    The lower bound of the height range the target has to be visible in.
    """

    maximal_height: float
    """
    The upper bound of the height range the target has to be visible in.
    """

    def __post_init__(self):
        self.origin: Pose = (
            Pose(reference_frame=self.world.root) if not self.origin else self.origin
        )
        self._generate_map()

    def _create_images(self) -> List[np.ndarray]:
        """
        Creates four depth images in every direction around the point
        for which the locations should be created. The depth images are converted
        to metre, meaning that every entry in the depth images represents the
        distance to the next object in metre.

        :return: A list of four depth images, the images are represented as 2D arrays.
        """
        images = []

        ray_tracer = RayTracer(self.world)

        origin_copy = deepcopy(self.origin).to_homogeneous_matrix()

        for _ in range(4):
            origin_copy = origin_copy @ HomogeneousTransformationMatrix.from_xyz_rpy(
                yaw=np.pi / 2
            )
            images.append(
                ray_tracer.create_depth_map(
                    origin_copy, resolution=self.width, min_distance=0.1
                )
            )

        return images

    def _generate_map(self):
        """
        This method generates the resulting density map by using the algorithm explained
        in Lorenz Mösenlechners `PhD Thesis (page 178) <https://mediatum.ub.tum.de/doc/1239461/1239461.pdf>`_
        The resulting map is then saved to :py:attr:`self.map`
        """
        depth_images = self._create_images()
        # A 2D array where every cell contains the arctan2 value with respect to
        # the middle of the array. Additionally, the interval is shifted such that
        # it is between 0 and 2pi
        angles = (
            np.arctan2(
                np.mgrid[
                    -int(self.width / 2) : int(self.width / 2),
                    -int(self.width / 2) : int(self.width / 2),
                ][0],
                np.mgrid[
                    -int(self.width / 2) : int(self.width / 2),
                    -int(self.width / 2) : int(self.width / 2),
                ][1],
            )
            + np.pi
        )
        depth_image_index = np.zeros(angles.shape)

        # Just for completion, since the depth image index array has zeros in every
        # position this operation is not necessary.

        # Creates a 2D array which contains the index of the depth image for every
        # coordinate
        depth_image_index[
            np.logical_and(angles >= np.pi * 1.25, angles <= np.pi * 1.75)
        ] = 3
        depth_image_index[
            np.logical_and(angles >= np.pi * 0.75, angles < np.pi * 1.25)
        ] = 2
        depth_image_index[
            np.logical_and(angles >= np.pi * 0.25, angles < np.pi * 0.75)
        ] = 1

        indices = np.dstack(np.mgrid[0 : self.width, 0 : self.width])
        depth_indices = np.zeros(indices.shape)
        # x-value of index: depth_image_index == n, :1
        # y-value of index: depth_image_index == n, 1:2

        # (y, size-x-1) for index between 1.25 pi and 1.75 pi
        depth_indices[depth_image_index == 3, :1] = indices[depth_image_index == 3, 1:2]
        depth_indices[depth_image_index == 3, 1:2] = (
            self.width - indices[depth_image_index == 3, :1] - 1
        )

        # (size-x-1, y) for index between 0.75 pi and 1.25 pi
        depth_indices[depth_image_index == 2, :1] = (
            self.width - indices[depth_image_index == 2, :1] - 1
        )
        depth_indices[depth_image_index == 2, 1:2] = indices[
            depth_image_index == 2, 1:2
        ]

        # (size-y-1, x) for index between 0.25 pi and 0.75 pi
        depth_indices[depth_image_index == 1, :1] = (
            self.width - indices[depth_image_index == 1, 1:2] - 1
        )
        depth_indices[depth_image_index == 1, 1:2] = indices[depth_image_index == 1, :1]

        # (x, y) for index between 0.25 pi and 1.75 pi
        depth_indices[depth_image_index == 0, :1] = indices[depth_image_index == 0, :1]
        depth_indices[depth_image_index == 0, 1:2] = indices[
            depth_image_index == 0, 1:2
        ]

        # Convert back to origin in the middle of the locations
        depth_indices[:, :, :1] -= self.width / 2
        depth_indices[:, :, 1:2] = np.absolute(
            self.width / 2 - depth_indices[:, :, 1:2]
        )

        # Sets the y index for the coordinates of the middle of the locations to 1,
        # the computed value is 0 which would cause an error in the next step where
        # the calculation divides the x coordinates by the y coordinates
        depth_indices[int(self.width / 2), int(self.width / 2), 1] = 1

        # Calculate columns for the respective position in the locations
        columns = (
            np.around(
                (
                    (depth_indices[:, :, :1] / depth_indices[:, :, 1:2])
                    * (self.width / 2)
                )
                + self.width / 2
            )
            .reshape((self.width, self.width))
            .astype("int16")
        )

        # An array with size * size that contains the euclidean distance to the
        # origin (in the middle of the locations) from every cell
        distances = np.maximum(
            np.linalg.norm(
                np.dstack(
                    np.mgrid[
                        -int(self.width / 2) : int(self.width / 2),
                        -int(self.width / 2) : int(self.width / 2),
                    ]
                ),
                axis=2,
            ),
            0.001,
        )

        # Row ranges
        # Calculation of the ranges of coordinates in the row which have to be
        # taken into account. The range is from r_min to r_max.
        # These are two arrays with shape: size*size, the row minimum constrains the
        # beginning of the range for every coordinate and the row maximum contains the
        # end for each coordinate
        row_minimum = (
            np.arctan((self.minimal_height - self.origin.z) / distances) * self.width
        ) + self.width / 2
        row_maximum = (
            np.arctan((self.maximal_height - self.origin.z) / distances) * self.width
        ) + self.width / 2

        row_minimum = np.minimum(np.around(row_minimum), self.width - 1).astype("int16")
        row_maximum = np.minimum(np.around(row_maximum), self.width - 1).astype("int16")

        row_ranges = np.dstack((row_minimum, row_maximum + 1)).reshape(
            (self.width**2, 2)
        )
        rows = np.arange(self.width)
        # Calculates a mask from the row minimum and row maximum values. This mask is
        # for every coordinate respectively and determines which values from the column
        # of the depth image should be taken into account for the locations.
        # A Mask of a single coordinate has the length of the column of the depth image
        # and together with the computed column at this coordinate determines which
        # values of the depth image make up the value of the visibility locations at this
        # point.
        mask = (
            (row_ranges[:, 0, None] <= rows) & (row_ranges[:, 1, None] > rows)
        ).reshape((self.width, self.width, self.width))

        values = np.zeros((self.width, self.width))
        visibility_map = np.zeros((self.width, self.width))
        # This is done to iterate over the depth images one at a time
        for image_index in range(4):
            row_masks = mask[depth_image_index == image_index].T
            # This statement does several things, first it takes the values from
            # the depth image for this quarter of the locations. The values taken are
            # the complete columns of the depth image (which where computed beforehand)
            # and checks if the values in them are greater than the distance to the
            # respective coordinates. This does not take the row ranges into account.
            values = (
                depth_images[image_index][
                    :, columns[depth_image_index == image_index].flatten()
                ]
                < np.tile(
                    distances[depth_image_index == image_index][:, None],
                    (1, self.width),
                ).T
                * self.resolution
            )
            # This applies the created mask of the row ranges to the values of
            # the columns which are compared in the previous statement
            masked = np.ma.masked_array(values, mask=~row_masks)
            # The calculated values are added to the locations
            visibility_map[depth_image_index == image_index] = np.sum(masked, axis=0)
        visibility_map /= np.max(visibility_map)
        # Weird flipping shit so that the map fits the orientation of the visualization.
        # the locations in itself is consistent and just needs to be flipped to fit the world coordinate system
        visibility_map = np.flip(visibility_map, axis=0)
        visibility_map = np.flip(visibility_map)

        # Invert the map
        inverted_map = np.zeros(visibility_map.shape)
        inverted_map[visibility_map == 0] = 1
        inverted_map[visibility_map != 0] = 0

        self.map = inverted_map


@dataclass
class GaussianCostmap(Costmap):
    """
    Gaussian Costmaps are 2D gaussian distributions around the origin with the given mean and sigma
    """

    mean: int
    """
    The mean input for the gaussian distribution, this also specifies 
    the length of the side of the resulting locations. The locations is Created
    as a square.
    """

    sigma: float
    """
    The sigma input for the gaussian distribution.
    """

    world: World
    """
    The world to use.
    """

    def __post_init__(self):
        gaussian_window = self._gaussian_window(self.mean, self.sigma)
        self.map: np.ndarray = np.outer(gaussian_window, gaussian_window)
        cut_distance = int(0.05 * self.mean)
        center = int(self.mean / 2)
        # Cuts out the middle 5% of the gaussian to avoid the robot being too close to the target since this is usually
        # bad for reaching the target with a end_effector. 15% is a magic number that might need some tuning in the future
        self.map[
            center - cut_distance : center + cut_distance,
            center - cut_distance : center + cut_distance,
        ] = 0
        self.size: float = self.mean
        self.width = int(self.size)
        self.height = int(self.size)

    def _gaussian_window(self, mean: int, standard_deviation: float) -> np.ndarray:
        """
        This method creates a window of values with a gaussian distribution of the
        given size and standard deviation.
        Code from `Scipy <https://github.com/scipy/scipy/blob/v0.14.0/scipy/signal/windows.py#L976>`_
        """
        offsets = np.arange(0, mean) - (mean - 1.0) / 2.0
        doubled_variance = 2 * standard_deviation * standard_deviation
        window = np.exp(-(offsets**2) / doubled_variance)
        return window


@dataclass
class RingCostmap(Costmap):
    """
    Creates a ring locations, similar to the gaussian locations but this looks more like a donut. Can be used to create poses
    for reaching a point for the robot.
    """

    standard_deviation: int
    """
    Standard deviation of the gaussian distribution that makes up the ring.
    """

    distance: float
    """
    Distance between the center of the locations and the center of the ring. A distance of 0 results in a gaussian locations
    """

    def __post_init__(self):
        self.map = self.ring()

    def ring(self) -> np.ndarray:
        radius_in_pixels = self.distance / self.resolution

        y, x = np.ogrid[: self.width, : self.height]
        center_x = (self.height - int(self.height % 2 == 0)) / 2.0
        center_y = (self.width - int(self.width % 2 == 0)) / 2.0

        distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        ring_costmap = np.exp(
            -((distance_from_center - radius_in_pixels) ** 2)
            / (2 * self.standard_deviation**2)
        )
        return ring_costmap


# Mainly used for debugging
# Data is 2d array
def plot_grid(data: np.ndarray) -> None:
    """
    An auxiliary method only used for debugging, it will plot a 2D numpy array using MatplotLib.
    """
    grid_colors = colors.ListedColormap(["white", "black", "green", "red", "blue"])
    rows = data.shape[0]
    columns = data.shape[1]
    figure, axes = plt.subplots()
    axes.imshow(data, cmap=grid_colors)
    # draw gridlines
    # ax.grid(which='major', axis='both', linestyle='-', rgba_color='k', linewidth=1)
    axes.set_xticks(np.arange(0.5, rows, 1))
    axes.set_yticks(np.arange(0.5, columns, 1))
    plt.tick_params(axis="both", labelsize=0, length=0)
    # fig.set_size_inches((8.5, 11), forward=False)
    # plt.savefig(saveImageName + ".png", dpi=500)
    plt.show()
