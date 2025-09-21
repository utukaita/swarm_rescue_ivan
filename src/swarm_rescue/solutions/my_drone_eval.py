import random
import heapq
import math
from typing import Optional, List, Type
from enum import Enum

import arcade
import numpy as np

from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.utils import normalize_angle, circular_mean

import os
import sys

import cv2

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spg_overlay.utils.constants import MAX_RANGE_SEMANTIC_SENSOR, MAX_RANGE_LIDAR_SENSOR
from spg_overlay.utils.grid import Grid

from spg_overlay.utils.pose import Pose
from spg_overlay.entities.drone_abstract import DroneAbstract

class Utility:
    @staticmethod
    def distance(end, start=(0, 0)):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        return (dx ** 2 + dy ** 2) ** 0.5

class TimeGrid(Grid):
    """Simple time grid"""

    def __init__(self,
                 size_area_world,
                 resolution: float):
        super().__init__(size_area_world=size_area_world,
                         resolution=resolution)

        self.size_area_world = size_area_world
        self.resolution = resolution

        self.x_max_grid = int(self.size_area_world[0] / self.resolution
                                   + 0.5)
        self.y_max_grid = int(self.size_area_world[1] / self.resolution
                                   + 0.5)
        self.grid = np.zeros((self.x_max_grid, self.y_max_grid))


class SemanticGrid(Grid):
    """Simple semantic grid"""
    class Entity(Enum):
        """
        Different entities marked in the semantic grid
        """
        NONE = 0
        RESCUE_CENTER = 1
        WOUNDED_PERSON = 2
        NO_GPS_ZONE = 3 # Implementation later
        KILL_ZONE = 4 # Implementation later
        NO_COM_ZONE = 5 # Implementation later
        WOUNDED_PERSON_TARGET = 6

    def __init__(self,
                 size_area_world,
                 resolution: float):
        super().__init__(size_area_world=size_area_world,
                         resolution=resolution)

        self.size_area_world = size_area_world
        self.resolution = resolution

        self.x_max_grid = int(self.size_area_world[0] / self.resolution
                                   + 0.5)
        self.y_max_grid = int(self.size_area_world[1] / self.resolution
                                   + 0.5)

        self.zoomed_grid = np.full((self.x_max_grid, self.y_max_grid), self.Entity.NONE.value)

        self.time_grid = TimeGrid(size_area_world=size_area_world,
                                 resolution=resolution)

    def update_grid(self, pose: Pose, semantic_values, iteration):
        """
        Bayesian map update with new observation
        semantic_values : semantic observations
        pose : corrected pose in world coordinates
        """
        EVERY_N = 3
        LIDAR_DIST_CLIP = 40.0
        EMPTY_ZONE_VALUE = -0.602
        OBSTACLE_ZONE_VALUE = 2.0
        FREE_ZONE_VALUE = -4.0
        THRESHOLD_MIN = -40
        THRESHOLD_MAX = 40
        pos_x, pos_y = self._conv_world_to_grid(pose.position[0], pose.position[1])
        for x in range(max(pos_x-MAX_RANGE_SEMANTIC_SENSOR, 0), min(pos_y+MAX_RANGE_SEMANTIC_SENSOR, self.x_max_grid-1)):
            for y in range(max(pos_y-MAX_RANGE_SEMANTIC_SENSOR, 0), min(pos_y+MAX_RANGE_SEMANTIC_SENSOR, self.y_max_grid-1)):
                if (pos_x-x)**2 + (pos_y-y)**2 < MAX_RANGE_SEMANTIC_SENSOR**2 \
                        and self.grid[x, y] == self.Entity.WOUNDED_PERSON.value:
                    self.grid[x, y] = self.Entity.NONE.value
        detection_semantic = semantic_values
        if (detection_semantic is not
                None):
            for data in detection_semantic:
                # If the wounded person detected is held by nobody
                entity = self.Entity.NONE.value
                if (data.entity_type ==
                        DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and
                        not data.grasped):  # And not grasped by another drone
                    entity = self.Entity.WOUNDED_PERSON.value
                elif (data.entity_type ==
                        DroneSemanticSensor.TypeEntity.RESCUE_CENTER):
                    entity = self.Entity.RESCUE_CENTER.value

                cos_rays = math.cos(data.angle + pose.orientation)
                sin_rays = math.sin(data.angle + pose.orientation)
                x = int(pose.position[0] + cos_rays * data.distance)
                y = int(pose.position[1] + sin_rays * data.distance)
                grid_x, grid_y = self._conv_world_to_grid(x, y)
                capped_grid_x = min(grid_x, self.x_max_grid - 1)
                capped_grid_y = min(grid_y, self.y_max_grid - 1)
                self.grid[capped_grid_x, capped_grid_y] = entity
                self.time_grid.grid[capped_grid_x, capped_grid_y] = iteration

        # compute zoomed grid for displaying
        self.zoomed_grid = self.grid.copy()
        new_zoomed_size = (int(self.size_area_world[1] * 0.5),
                           int(self.size_area_world[0] * 0.5))
        self.zoomed_grid = cv2.resize(self.zoomed_grid, new_zoomed_size,
                                      interpolation=cv2.INTER_NEAREST)

    def merge_grid(self, other):
        """
        Merge two semantic grids
        """
        for x in range(self.x_max_grid):
            for y in range(self.y_max_grid):
                if other.time_grid.grid[x, y] > self.time_grid.grid[x, y]:
                    self.grid[x, y] = other.grid[x, y]
                    self.time_grid.grid[x, y] = other.time_grid.grid[x, y]


class OccupancyGrid(Grid):
    """Simple occupancy grid"""

    def __init__(self,
                 size_area_world,
                 resolution: float,
                 lidar):
        super().__init__(size_area_world=size_area_world,
                         resolution=resolution)

        self.size_area_world = size_area_world
        self.resolution = resolution

        self.lidar = lidar

        self.x_max_grid = int(self.size_area_world[0] / self.resolution
                                   + 0.5)
        self.y_max_grid = int(self.size_area_world[1] / self.resolution
                                   + 0.5)

        self.grid = np.zeros((self.x_max_grid, self.y_max_grid))
        self.zoomed_grid = np.empty((self.x_max_grid, self.y_max_grid))

        self.time_grid = TimeGrid(size_area_world=size_area_world,
                                 resolution=resolution)

    def update_grid(self, pose: Pose, iteration):
        """
        Bayesian map update with new observation
        lidar : lidar data
        pose : corrected pose in world coordinates
        """
        EVERY_N = 3
        LIDAR_DIST_CLIP = 40.0
        EMPTY_ZONE_VALUE = -0.602
        OBSTACLE_ZONE_VALUE = 2.0
        FREE_ZONE_VALUE = -4.0
        THRESHOLD_MIN = -40
        THRESHOLD_MAX = 40

        lidar_dist = self.lidar.get_sensor_values()[::EVERY_N].copy()
        lidar_angles = self.lidar.ray_angles[::EVERY_N].copy()

        # Compute cos and sin of the absolute angle of the lidar
        cos_rays = np.cos(lidar_angles + pose.orientation)
        sin_rays = np.sin(lidar_angles + pose.orientation)

        max_range = MAX_RANGE_LIDAR_SENSOR * 0.9

        # For empty zones
        # points_x and point_y contains the border of detected empty zone
        # We use a value a little bit less than LIDAR_DIST_CLIP because of the
        # noise in lidar
        lidar_dist_empty = np.maximum(lidar_dist - LIDAR_DIST_CLIP, 0.0)
        # All values of lidar_dist_empty_clip are now <= max_range
        lidar_dist_empty_clip = np.minimum(lidar_dist_empty, max_range)
        points_x = pose.position[0] + np.multiply(lidar_dist_empty_clip,
                                                  cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist_empty_clip,
                                                  sin_rays)

        for pt_x, pt_y in zip(points_x, points_y):
            self.add_value_along_line(pose.position[0], pose.position[1],
                                      pt_x, pt_y,
                                      EMPTY_ZONE_VALUE)
            self.time_grid.add_value_along_line(pose.position[0], pose.position[1],
                                                pt_x, pt_y,
                                                iteration)

        # For obstacle zones, all values of lidar_dist are < max_range
        select_collision = lidar_dist < max_range

        points_x = pose.position[0] + np.multiply(lidar_dist, cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist, sin_rays)

        points_x = points_x[select_collision]
        points_y = points_y[select_collision]

        self.add_points(points_x, points_y, OBSTACLE_ZONE_VALUE)
        self.time_grid.add_points(points_x, points_y, iteration)

        # the current position of the drone is free !
        self.add_points(pose.position[0], pose.position[1], FREE_ZONE_VALUE)
        self.time_grid.add_points(pose.position[0], pose.position[1], iteration)

        # threshold values
        self.grid = np.clip(self.grid, THRESHOLD_MIN, THRESHOLD_MAX)

        # compute zoomed grid for displaying
        self.zoomed_grid = self.grid.copy()
        new_zoomed_size = (int(self.size_area_world[1] * 0.5),
                           int(self.size_area_world[0] * 0.5))
        self.zoomed_grid = cv2.resize(self.zoomed_grid, new_zoomed_size,
                                      interpolation=cv2.INTER_NEAREST)

    def merge_grid(self, other):
        """
        Merge two occupancy grids
        """
        for x in range(self.x_max_grid):
            for y in range(self.y_max_grid):
                if other.time_grid.grid[x, y] > self.time_grid.grid[x, y]:
                    self.grid[x, y] = other.grid[x, y]
                    self.time_grid.grid[x, y] = other.time_grid.grid[x, y]


class MyDroneEval(DroneAbstract):
    class Activity(Enum):
        """
        All the states of the drone as a state machine
        """
        EXPLORING = 0
        SEARCHING_WOUNDED = 1
        SEARCHING_RESCUE_CENTER = 2


    # to calculate map
    OCCUPIED_CERTAINTY_THRESHOLD = 0.80  # probability of being occupied (99% occupied)
    FREE_CERTAINTY_THRESHOLD = 1 - OCCUPIED_CERTAINTY_THRESHOLD  # probablity of not being occupied (99% free = 1% occupied)
    GRID_OCCUPIED_THRESHOLD = np.log(OCCUPIED_CERTAINTY_THRESHOLD) - np.log(
        1 - OCCUPIED_CERTAINTY_THRESHOLD)  # using log-odds probability
    GRID_FREE_THRESHOLD = -GRID_OCCUPIED_THRESHOLD

    # to calculate utility
    NUM_REGIONS = 5

    # to calculate distance
    DISTANCE_THRESHOLD = 2
    LIDAR_RANGE = 40

    # to calculate speed
    MAX_SPEED = 1
    GRASP_SPEED = 2

    # Control parameters
    ROTATION_COEFF = 1
    FORWARD_COEFF = 1

    # to calculate next target point
    NUM_PARTITIONS = 10
    PARTITION_DIST_THRESHOLD = 3
    TARGET_SEARCH_RADIUS = 20

    # to calculate path
    GRASP_HITBOX_SIZE = 6
    NORMAL_HITBOX_SIZE = 6

    ANGULAR_THRESHOLD = np.deg2rad(2)

    # gain constant
    GAIN_CONSTANT = 1
    PROP_GAIN = 1 / 6
    ROT_GAIN = 1 / (6 * np.pi)

    # to determine whether a drone has died
    TIME_TO_DEAD = 100

    def __init__(self,
                 identifier: Optional[int] = None, **kwargs):
        super().__init__(identifier=identifier,
                         display_lidar_graph=False,
                         **kwargs)


        self.movement_log = []
        self.iter_path = 0

        # Information to broadcast
        self.identifier = identifier
        self.drone_dict = {identifier: 0}

        self.wounded_persons = []


        self.state = self.Activity.EXPLORING
        self.stateChanged = True

        # Values associated with movement in general
        self.speed = self.MAX_SPEED

        self.iteration = 0

        self.estimated_pose = Pose(np.asarray(self.measured_gps_position()), self.measured_compass_angle())

        resolution = 8
        self.occupancy_grid = OccupancyGrid(size_area_world=self.size_area,
                                            resolution=resolution,
                                            lidar=self.lidar())
        self.semantic_grid = SemanticGrid(size_area_world=self.size_area,
                                          resolution=resolution)

        #iterations since target wounded person disappeared
        self.iterations_wounded_disappeared = 0
        self.target_position = None

        # last command sent, used to estimate position when gps is not available
        self.prev_command = None

        # meshgrid to evaluate utility of a possible target
        self.MESH_XX, self.MESH_YY = np.meshgrid(np.arange(self.occupancy_grid.y_max_grid), np.arange(self.occupancy_grid.x_max_grid))

        # for path finding
        self.path = []
        self.target = None
        self.final_target = None
        self.prev_final_target = None
        self.hitbox_size = self.NORMAL_HITBOX_SIZE

        # displacement used to calculate PID
        self.prev_diff_angle = 0
        self.prev_diff_position = 0

    def define_message_for_all(self):
        """
        Sharing the map
        Sharing positions of wounded people

        Test different messaging periods
        """
        if self.iteration % 15 == 0:
            msg_data = (self.drone_dict,
                        self.semantic_grid,
                        self.occupancy_grid)
            return msg_data

    def process_communication_sensor(self):
        if self.communicator:
            received_messages = self.communicator.received_messages

            for message in received_messages:
                content = message[1]
                for drone, time in content[0].items():
                    if drone not in self.drone_dict:
                        self.drone_dict[drone] = time
                    else:
                        self.drone_dict[drone] = max(self.drone_dict[drone], time)
                if content[1] is not None and self.semantic_grid.grid is not None:
                    self.semantic_grid.merge_grid(content[1])
                if content[2] is not None and self.occupancy_grid.grid is not None:
                    self.occupancy_grid.merge_grid(content[2])

    def control(self):
        command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 1}

        # Find the position and orientation of the drone
        self.estimated_pose, self.speed = self.get_estimated_pose(self.estimated_pose, self.odometer_values(), self.speed)

        if self.iteration == 0:
            self.prev_final_target = self.occupancy_grid._conv_world_to_grid(*self.measured_gps_position())

        # Update the occupancy grid and the semantic grid
        self.occupancy_grid.update_grid(pose=self.estimated_pose,
                                        iteration=self.iteration)
        self.semantic_grid.update_grid(pose=self.estimated_pose,
                                       semantic_values=self.semantic_values(),
                                       iteration=self.iteration)
        # Update iteration counter
        self.iteration += 1

        # Communicate with other drones
        self.process_communication_sensor()

        # Update drone dictionary
        self.drone_dict[self.identifier] = self.iteration
        self.drone_dict = {drone: time for drone, time in self.drone_dict.items() if self.iteration - time < self.TIME_TO_DEAD}

        # Display the occupancy grid and the semantic grid
        if self.iteration % 5 == 0 and self.identifier == 0:
            self.occupancy_grid.display(self.occupancy_grid.zoomed_grid,
                                        self.estimated_pose,
                                        title="zoomed occupancy grid")
            self.semantic_grid.display(self.semantic_grid.zoomed_grid,
                                       self.estimated_pose,
                                       title="zoomed semantic grid")
        #print(self.drone_dict)

        occupancy_grid = self.occupancy_grid.grid
        drone_pos = self.occupancy_grid._conv_world_to_grid(*self.estimated_pose.position)

        # Test if state has been changed

        if self.stateChanged:
            #print(' -- State changed to', self.state)
            self.path = []
            self.target = None
            if self.final_target is not None:
                self.path = self.map_path_from_to(occupancy_grid,
                                                  drone_pos, self.final_target,
                                                  self.hitbox_size)[0]
                self.target = self.path[0]

        self.stateChanged = False

        #############
        # TRANSITIONS OF THE STATE MACHINE
        #############

        if self.state is self.Activity.EXPLORING:
            semantic_values = self.process_semantic_sensor()
            if semantic_values[0]:
                self.state = self.Activity.SEARCHING_WOUNDED
                print("state is now SEARCHING_WOUNDED")
                return semantic_values[2]
            wounded_position = self.decide_wounded()
            if wounded_position is not None:
                self.prev_final_target = drone_pos
                self.final_target = wounded_position
                self.state = self.Activity.SEARCHING_WOUNDED
                return command

            new_path_needed = self.new_path_needed(self.path, drone_pos, self.target)
            if new_path_needed:
                self.final_target = self.determine_target(occupancy_grid, drone_pos)
                #print('Mapping from', self.prev_final_target, 'to', self.final_target)
                self.path = self.map_path_from_to(occupancy_grid, drone_pos, self.final_target,
                                                  self.hitbox_size)[0]
                self.target = self.path[0]
                return command


        elif self.state is self.Activity.SEARCHING_WOUNDED:
            is_grasping = self.base.grasper.grasped_entities
            print("grasping: ", is_grasping)
            if is_grasping:
                print("is grasping")
                self.hitbox_size = self.GRASP_HITBOX_SIZE
                self.speed = self.GRASP_SPEED
                self.final_target = self.determine_center(occupancy_grid)
                self.state = self.Activity.SEARCHING_RESCUE_CENTER
                print("state is now SEARCHING_RESCUE_CENTER")
                return {'forward': 0, 'lateral': 0, 'rotation': 0, 'grasper': 1}

            semantic_values = self.process_semantic_sensor()
            if semantic_values[0]:
                print("moving to wounded person")
                return semantic_values[2]
            if self.final_target is None:
                self.state = self.Activity.EXPLORING
                return command
            wounded_disappeared = True
            pos_x, pos_y = int(self.final_target[0]), int(self.final_target[1])
            search_radius = 10
            missing_limit = 5

            x_min = max(pos_x - search_radius, 0)
            x_max = min(pos_x + search_radius, self.semantic_grid.x_max_grid - 1) + 1
            y_min = max(pos_y - search_radius, 0)
            y_max = min(pos_y + search_radius, self.semantic_grid.y_max_grid - 1) + 1

            # Extract the region of interest
            roi = self.semantic_grid.grid[x_min:x_max, y_min:y_max]

            # Check if there is any WOUNDED_PERSON in the region of interest
            if np.any(roi == self.semantic_grid.Entity.WOUNDED_PERSON):
                self.wounded_disappeared = False
                # print("Wounded person still there")
                self.iterations_wounded_disappeared = 0
            if wounded_disappeared:
                if self.iterations_wounded_disappeared==missing_limit-1:
                    #self.semantic_grid.grid[int(pos_x), int(pos_y)] = self.semantic_grid.Entity.NONE.value
                    self.iterations_wounded_disappeared = 0
                    self.prev_final_target = self.final_target
                    self.final_target = self.determine_target(occupancy_grid, drone_pos)
                    self.state = self.Activity.EXPLORING
                    return command
                else:
                    self.iterations_wounded_disappeared += 1

        elif self.state == self.Activity.SEARCHING_RESCUE_CENTER:
            semantic_values = self.process_semantic_sensor()
            if semantic_values[1]:
                return semantic_values[2]

            if not self.base.grasper.grasped_entities:
                self.hitbox_size = self.NORMAL_HITBOX_SIZE
                self.speed = self.MAX_SPEED
                self.prev_final_target = self.final_target
                self.final_target = self.determine_target(occupancy_grid, drone_pos)
                self.state = self.Activity.EXPLORING
                return command

        # The basic case of movement
        if len(self.path) > 0 and self.path[0] is not None:
            self.target = self.get_next_target(self.path, drone_pos)

        # grasp is 1
        if self.target is None:
            return command
        return self.go_to(self.occupancy_grid._conv_grid_to_world(*self.target)) | {'grasper': 1}


        # found_wounded = self.determine_wounded()
        # print(self.semantic_values())

        # if self.state is self.Activity.EXPLORING and found_wounded:
        #     self.state = self.Activity.SEARCHING_WOUNDED

        # elif self.state is self.Activity.SEARCHING_WOUNDED and self.base.grasper.grasped_entities:
        #     self.state = self.Activity.SEARCHING_RESCUE_CENTER

        # elif self.state is self.Activity.SEARCHING_RESCUE_CENTER and not self.base.grasper.grasped_entities:
        #     self.state = self.Activity.EXPLORING

        # ##########
        # # COMMANDS FOR EACH STATE
        # ##########

        # drone_pos = self.occupancy_grid._conv_world_to_grid(*self.estimated_pose.position)
        # occupancy_grid = self.occupancy_grid.grid

        # if self.stateChanged:
        #     print('state changed to', self.state)
        #     self.path = []
        #     self.target = None
        #     self.prev_final_target = self.final_target
        #     self.final_target = None

        # stateGrasping = {self.Activity.EXPLORING: 0, self.Activity.EXPLORING: 1, self.Activity.SEARCHING_RESCUE_CENTER: 1}

        # if self.new_path_needed(self.path, drone_pos, self.target):
        #     self.prev_final_target = self.final_target
        #     print('\niteration', self.iteration, '\n')

        #     if self.state is self.Activity.EXPLORING:
        #         self.final_target = self.determine_target(occupancy_grid, drone_pos)
        #         print('found target at', self.final_target)

        #     elif self.state is self.Activity.SEARCHING_WOUNDED:
        #         self.final_target = found_wounded

        #         self.get_neighborhood(occupancy_grid, found_wounded, self.HITBOX_SIZE)
        #         print('found wounded at', found_wounded)

        #     elif self.state is self.Activity.SEARCHING_RESCUE_CENTER:
        #         self.final_target = self.determine_center(occupancy_grid)
        #         print('found center at ', self.final_target)

        #     if self.prev_final_target is None:
        #         self.prev_final_target = self.occupancy_grid._conv_world_to_grid(*self.estimated_pose.position)

        #     hitbox_size = self.HITBOX_SIZE
        #     print('--STATE', self.state, 'target locked in', self.final_target)
        #     print('mapping from', self.prev_final_target, 'to', self.final_target)
        #     self.path = self.map_path_from_to(occupancy_grid, self.prev_final_target, self.final_target, hitbox_size)

        #     """
        #     two things:
        #         basically the wounded people counts as obstacles in the occupancy grid,
        #             so the is_valid_function needs to be adapted to it to make sure it is only walls
        #             that invalidate the position
        #         the prev final target keeps switching even if the end position is invalid,
        #             as the self final target is already set (even if it may be invalid due to mapping
        #             error with is_valid_function as described above for rescue centers and wounded people)
        #     """

        #     print('found path', self.path)
        #     self.target = self.path[0]

        # else:
        #     if self.path:
        #         self.target = self.get_next_target(self.path, drone_pos)
        #     command =  self.go_to(self.occupancy_grid._conv_grid_to_world(*self.target))

        # command['grasper'] = stateGrasping[self.state]

        # # increment the iteration counter
        # self.iteration += 1

        # # Display the occupancy grid and the semantic grid
        # if self.iteration % 5 == 0:
        #     self.occupancy_grid.display(self.occupancy_grid.zoomed_grid,
        #                                 self.estimated_pose,
        #                                 title="zoomed occupancy grid")
        #     self.semantic_grid.display(self.semantic_grid.zoomed_grid,
        #                                self.estimated_pose,
        #                                title="zoomed semantic grid")

        #return command

    def __setattr__(self, name, value):
        if name == 'state':
            self.stateChanged = True
        super().__setattr__(name, value)

    def get_estimated_pose(self, estimated_pose, odometer_values, speed):
            #print('using odometer values')
            if not self.gps_is_disabled():
                return Pose(np.asarray(self.measured_gps_position()), self.measured_compass_angle()), speed
            dist, alpha, theta = odometer_values
            x, y = estimated_pose.position
            orient = estimated_pose.orientation

            new_x = x + dist * math.cos(alpha + orient)
            new_y = y + dist * math.sin(alpha + orient)
            new_orient = normalize_angle(orient + theta)
            return Pose(np.array([new_x, new_y]), new_orient), speed/2

    def draw_bottom_layer(self):
         if self.target is not None:
              # Use real world positions
             position = self.measured_gps_position()
             position += self._half_size_array
             target = self.occupancy_grid._conv_grid_to_world(*self.target)
             target += self._half_size_array
             #arcade.draw_line(*position, *target, [128, 128, 128], 2)

    def decide_wounded(self):
        target_position = None
        pose = self.estimated_pose

        # Find the closest wounded person by comparing distances to all visible ones
        """
        closest_wounded = None
        min_distance = float('inf')
        for data in self.semantic_values():
            if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
                if data.distance < min_distance:
                    min_distance = data.distance
                    cos_rays = math.cos(data.angle + pose.orientation)
                    sin_rays = math.sin(data.angle + pose.orientation)
                    x = int(pose.position[0] + cos_rays * data.distance)
                    y = int(pose.position[1] + sin_rays * data.distance)
                    grid_x, grid_y = self.semantic_grid._conv_world_to_grid(x, y)
                    capped_grid_x = min(grid_x, self.semantic_grid.x_max_grid - 1)
                    capped_grid_y = min(grid_y, self.semantic_grid.y_max_grid - 1)
                    target_position = np.array([capped_grid_x, capped_grid_y])
        if target_position is not None:
            return target_position
        """
        # If not any wounded person seen, find the closest wounded person by comparing a* distances to all known wounded persons
        pos = self.semantic_grid._conv_world_to_grid(*pose.position)
        min_distance = float('inf')
        wounded_persons = np.column_stack(
            np.nonzero(self.semantic_grid.grid == self.semantic_grid.Entity.WOUNDED_PERSON.value))
        if np.any(wounded_persons):
            for wounded_person in wounded_persons:
                _, path_length = self.map_path_from_to(self.occupancy_grid.grid, self.prev_final_target, wounded_person,
                                                       self.hitbox_size)
                if path_length < min_distance:
                    min_distance = path_length
                    target_position = wounded_person
        vicinity = self.vicinity(target_position, self.hitbox_size)
        return vicinity

    def determine_target(self, occupancy_grid, drone_pos):
        """
        Determine the optimal target point in the occupancy grid for exploration.

        This function searches for a point within the occupancy grid that has a high
        probability of being free and is surrounded by the maximum number of unexplored
        points within a defined search radius. The function iterates over potential
        frontier points and evaluates how many unexplored points are present in their
        vicinity, ultimately returning the coordinates of the best target point.


        Parameters:
        occupancy_grid (np.ndarray): A 2D array representing the occupancy grid,
                                        where values indicate:
                                        - > GRID_FREE_THRESHOLD: free space of uncertain probability
                                        - < 0: free space of probable probability
                                        - == 0: unexplored

        Returns:
        tuple: The (row, column) coordinates of the best target point for exploration,
            or None if no suitable point is found.
        """
        # Map variable initializations
        border_grid = (occupancy_grid > self.GRID_FREE_THRESHOLD) & (occupancy_grid < 0)
        obstacles_grid = occupancy_grid > self.GRID_OCCUPIED_THRESHOLD
        candidate_points = np.column_stack(np.nonzero(border_grid))
        # print(candidate_points.shape)

        # Remove all points outside the search radius
        dist_to_drone = np.linalg.norm((candidate_points - drone_pos), axis=1)
        candidate_points = candidate_points[dist_to_drone < self.TARGET_SEARCH_RADIUS, :]
        # print(candidate_points.shape)

        # Remove all points within range of an obstacle
        valid_points = self.is_valid_position(obstacles_grid, candidate_points, self.hitbox_size)
        candidate_points = candidate_points[valid_points]
        # print(candidate_points.shape)

        # Find point with most unexplored points in its radius
        unexplored_points = occupancy_grid == 0
        unexplored_submatrix = self.get_neighborhood(unexplored_points, candidate_points, self.TARGET_SEARCH_RADIUS)
        num_unexplored_points = np.sum(unexplored_submatrix, axis=(1, 2))
        # print(num_unexplored_points.shape)

        return tuple(candidate_points[np.argmax(num_unexplored_points)])

    def is_valid_position(self, obstacles_grid, points, hitbox_size):
        """Check if the drone can fit at position (x, y)."""
        points = np.array(points)
        print('Points:', points)

        half_hitbox = hitbox_size // 2
        n, m = obstacles_grid.shape
        x, y = points[:, 0], points[:, 1]

        hitbox_within_bounds = (0 <= x - half_hitbox) & (x + half_hitbox <= n - 1) & (0 <= y - half_hitbox) & (
                    y + half_hitbox <= m - 1)

        x, y = np.clip(x, half_hitbox, n - 1 - half_hitbox), np.clip(y, half_hitbox, m - 1 - half_hitbox)

        occupancy_hitbox_submatrix = self.get_neighborhood(obstacles_grid, np.column_stack((x, y)), hitbox_size)
        obstacles_within_hitbox = np.any(occupancy_hitbox_submatrix, axis=(1, 2))

        return hitbox_within_bounds & ~obstacles_within_hitbox

    def get_neighborhood(self, grid, points, hitbox_size):
        half_hitbox = hitbox_size // 2
        n, m = grid.shape
        x, y = points[:, 0], points[:, 1]

        x, y = np.clip(x, half_hitbox, n - 1 - half_hitbox), np.clip(y, half_hitbox, m - 1 - half_hitbox)

        row_offsets = np.arange(- half_hitbox, half_hitbox + 1)
        col_offsets = np.arange(- half_hitbox, half_hitbox + 1)

        row_indices = x[:, np.newaxis] + row_offsets[np.newaxis,
                                         :]  # (len(points), 1) + (1, hitbox_size) = (len(points), hitbox_size)
        col_indices = y[:, np.newaxis] + col_offsets[np.newaxis,
                                         :]  # (len(points), 1) + (1, hitbox_size) = (len(points), hitbox_size)

        return grid[row_indices[:, :, np.newaxis], col_indices[:, np.newaxis,
                                                   :]]  # (len(points), hitbox_size, 1) + (len(points), 1, hitbox_size) = (len(points), hitbox_size, hitbox_size)

    def determine_center(self, occupancy_grid):
            # obstacles_grid = occupancy_grid >= self.GRID_OCCUPIED_THRESHOLD
            print("determining rescue center...")
            rescue_centers = np.column_stack(
                np.nonzero(self.semantic_grid.grid == self.semantic_grid.Entity.RESCUE_CENTER.value))
            if np.any(rescue_centers):
                for rescue_center in rescue_centers:
                    self.get_neighborhood(occupancy_grid, rescue_center, self.hitbox_size)
                    pos = int(rescue_center[0]), int(rescue_center[1])
                    vicinity = self.vicinity(pos, self.hitbox_size)
                    print('vicinity', vicinity)
                    print('pos', pos)
                    return vicinity

            print('no centers that can match hitbox have been found')
            return None
            # for x in range(self.semantic_grid.x_max_grid):
            #     for y in range(self.semantic_grid.y_max_grid):
            #         if self.semantic_grid.grid[x, y] == self.semantic_grid.Entity.RESCUE_CENTER.value:
            #             return np.array([x, y])

    def line_of_sight(self, start, end, occupancy_grid):
        """Check if a straight line between p1 and p2 is obstacle-free using Bresenham's algorithm."""
        obstacles_grid = occupancy_grid >= self.GRID_OCCUPIED_THRESHOLD
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            if not self.is_valid_position(obstacles_grid, (x0, y0), self.hitbox_size):
                return False
            if (x0, y0) == (x1, y1):
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return True

    def map_path_from_to(self, occupancy_grid, start, end, hitbox_size=1):
            """
            Calculate a path from a start position to an end position on a given occupancy grid.

            Modified to use Murphy Modified Bresenham Line Algorithm (MMBLA) for path smoothing.
            """
            obstacles_grid = occupancy_grid >= self.GRID_OCCUPIED_THRESHOLD

            if not self.is_valid_position(obstacles_grid, end, hitbox_size):
                print('invalid end position', end)
                return [start], 1
            if not self.is_valid_position(obstacles_grid, start, hitbox_size):
                print('invalid start position', start)
                return [start], 1

            # Priority queue
            priority_queue = []
            heapq.heappush(priority_queue, (0, start[0], start[1]))  # (f score, x, y)

            # Distance array (g scores)
            g_score = np.ones(obstacles_grid.shape) * np.inf
            g_score[start] = 0

            # Parent dictionary to reconstruct the path
            parent = {}
            parent[start] = None

            # Directions for neighbors (8-directional movement)
            directions = (
            (0, 1, 1), (1, 0, 1), (0, -1, 1), (-1, 0, 1), (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)),
            (-1, -1, np.sqrt(2)))

            # While priority queue is not empty
            while priority_queue:
                f_score, candidate_x, candidate_y = heapq.heappop(priority_queue)

                # If we reached the end, reconstruct the path
                if (candidate_x, candidate_y) == end:
                    path = []
                    while (candidate_x, candidate_y) is not None:
                        path.append((candidate_x, candidate_y))
                        if (candidate_x, candidate_y) == start:  # Exit loop when reaching the start
                            break
                        candidate_x, candidate_y = parent[(candidate_x, candidate_y)]
                    path.reverse()

                    # Apply MMBLA to smooth the path
                    smoothed_path = [path[0]]  # Start with the first point
                    for i in range(2, len(path)):
                        if not self.line_of_sight(smoothed_path[-1], path[i], occupancy_grid):
                            smoothed_path.append(path[i - 1])
                    smoothed_path.append(path[-1])  # Ensure the last point is added
                    print(f"smoothed path: {smoothed_path}")
                    return smoothed_path, len(path)

                # Explore neighbors
                for dx, dy, move_cost in directions:
                    neighbor_x, neighbor_y = candidate_x + dx, candidate_y + dy

                    # Check if new position is valid for the drone's hitbox
                    if self.is_valid_position(obstacles_grid, (neighbor_x, neighbor_y), hitbox_size):
                        # Compute new cost to neighbor
                        new_cost = g_score[candidate_x][candidate_y] + move_cost
                        if new_cost < g_score[neighbor_x, neighbor_y]:
                            g_score[neighbor_x, neighbor_y] = new_cost
                            parent[(neighbor_x, neighbor_y)] = (candidate_x, candidate_y)
                            f_score = new_cost + Utility.distance((neighbor_x, neighbor_y), (end[0], end[1]))
                            heapq.heappush(priority_queue, (f_score, neighbor_x, neighbor_y))

            return [start], 0  # No path found

    def new_path_needed(self, path, drone_pos, target):
            return not path and (target is None or Utility.distance(drone_pos, target) < self.DISTANCE_THRESHOLD)

    def get_next_target(self, path, drone_pos):
            target = path[0]
            #print("path: ", path)
            #print("target: ", target)
            #print("drone_pos: ", drone_pos)
            while Utility.distance(drone_pos, target) < self.DISTANCE_THRESHOLD and len(path) > 0:
                target = path.pop(0)
            return target

    def go_to(self, target):
            drone_position = self.estimated_pose.position
            drone_angle = self.estimated_pose.orientation

            diff_position = target - drone_position
            target_angle = np.atan2(diff_position[1], diff_position[0])
            diff_angle = normalize_angle(target_angle - drone_angle)

            # Use angular threshold for switching between turning and moving
            if abs(diff_angle) > self.ANGULAR_THRESHOLD:
                return self.turn_to_angle(diff_angle)

            # # Compute forward command directly from the distance
            return self.move_straight(diff_position, drone_angle)

    def turn_to_angle(self, diff_angle):
            # Smooth derivative term to reduce noise
            deriv_diff_angle = normalize_angle(diff_angle - self.prev_diff_angle)

            # PD control for angular correction
            Kp = 0.8
            Kd = 2.75
            rotation = Kp * diff_angle + Kd * deriv_diff_angle

            # Clamp rotation motion according to set speed
            rotation = np.clip(rotation, -self.speed, self.speed)

            # Update previous state
            self.prev_diff_angle = diff_angle

            # Return command
            return {"forward": 0, "rotation": rotation}

    def move_straight(self, diff_position, drone_angle):
            # Derivative term for position difference
            deriv_diff_position = diff_position - self.prev_diff_position

            # Calculate distance and derivative components along the direction of motion
            oriented_vector = np.array([np.cos(drone_angle), np.sin(drone_angle)])
            signed_dist = np.dot(diff_position, oriented_vector)
            signed_deriv_dist = np.dot(deriv_diff_position, oriented_vector)

            # PD control for forward motion
            Kp = 0.8  # Proportional gain for smoother response
            Kd = 5.5  # Lower derivative gain to reduce oscillations
            forward = Kp * signed_dist + Kd * signed_deriv_dist

            # Clamp forward motion according to set speed
            forward = np.clip(forward, -self.speed, self.speed)

            # Update previous state
            self.prev_diff_position = diff_position

            # Return command
            return {"forward": forward, "rotation": 0}

    def vicinity(self, target, hitbox, loops=10000):
        """ Finds a point in the vicinity of the target that is free and has line of sight to the target """
        if target is None:
            return None
        for i in range (loops):
            for j in range(loops):
                x = random.randint(target[0]-hitbox*5, target[0]+ hitbox*5)
                if abs(x-target[0])>hitbox/2 and 0<x<self.occupancy_grid.x_max_grid-1:
                    break
            for k in range(loops):
                y = random.randint(target[1]-hitbox*5, target[1]+hitbox*5)
                if abs(y-target[1])>hitbox/2 and 0<y<self.occupancy_grid.y_max_grid-1:
                    break
        
            if (self.occupancy_grid.grid[x, y] < self.GRID_FREE_THRESHOLD,
                    self.line_of_sight(target, (x, y), occupancy_grid=self.occupancy_grid.grid)):
                return x, y
        return 0,0

    def process_semantic_sensor(self):
            """
            According to his state in the state machine, the Drone will move
            towards a wounded person or the rescue center
            """
            command = {"forward": 0.5,
                       "lateral": 0.0,
                       "rotation": 0.0,
                       "grasper": 1}
            angular_vel_controller_max = 1.0

            detection_semantic = self.semantic_values()
            best_angle = 0

            found_wounded = False
            if (self.state is self.Activity.EXPLORING or self.state is self.Activity.SEARCHING_WOUNDED
                    and detection_semantic is not None):
                scores = []
                for data in detection_semantic:
                    # If the wounded person detected is held by nobody
                    if (data.entity_type ==
                            DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and
                            not data.grasped):
                        found_wounded = True
                        v = (data.angle * data.angle) + \
                            (data.distance * data.distance / 10 ** 5)
                        scores.append((v, data.angle, data.distance))

                # Select the best one among wounded persons detected
                best_score = 10000
                for score in scores:
                    if score[0] < best_score:
                        best_score = score[0]
                        best_angle = score[1]

            found_rescue_center = False
            is_near = False
            angles_list = []
            if self.state is self.Activity.SEARCHING_RESCUE_CENTER and detection_semantic:
                for data in detection_semantic:
                    if (data.entity_type ==
                            DroneSemanticSensor.TypeEntity.RESCUE_CENTER):
                        found_rescue_center = True
                        angles_list.append(data.angle)
                        is_near = (data.distance < 30)

                if found_rescue_center:
                    best_angle = circular_mean(np.array(angles_list))

            if found_rescue_center or found_wounded:
                # simple P controller
                # The robot will turn until best_angle is 0
                kp = 2.0
                a = kp * best_angle
                a = min(a, 1.0)
                a = max(a, -1.0)
                command["rotation"] = a * angular_vel_controller_max

                # reduce speed if we need to turn a lot
                if abs(a) == 1:
                    command["forward"] = 0.2

            if found_rescue_center and is_near:
                command["forward"] = 0.0
                command["rotation"] = -1.0

            return found_wounded, found_rescue_center, command



# LEGACY FUNCTIONS
    def control2(self):
        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}

        # Find the position and orientation of the drone

        if isinstance(self.measured_gps_position(), np.ndarray):
            self.estimated_pose = Pose(np.asarray(self.measured_gps_position()),
                                       self.measured_compass_angle())
            self.speed = self.MAX_SPEED

        else:
            dist, alpha, theta = 0.0, 0.0, 0.0
            if not self.odometer_is_disabled():
                dist, alpha, theta = tuple(self.odometer_values())
            x, y = self.estimated_pose.position
            orient = self.estimated_pose.orientation
            new_x = x + dist * math.cos(alpha + orient)
            new_y = y + dist * math.sin(alpha + orient)
            new_orient = orient + theta
            self.estimated_pose.position = new_x, new_y
            self.estimated_pose.orientation = new_orient
            self.speed = self.MAX_SPEED/2

        # Update the occupancy grid and the semantic grid
        self.occupancy_grid.update_grid(pose=self.estimated_pose,
                                        iteration=self.iteration)
        self.semantic_grid.update_grid(pose=self.estimated_pose,
                                       semantic_values=self.semantic_values(),
                                       iteration=self.iteration)

        # Communicate with other drones

        self.process_communication_sensor()

        #############
        # TRANSITIONS OF THE STATE MACHINE
        #############

        if self.state is self.Activity.EXPLORING and self.decide_target():
            self.state = self.Activity.SEARCHING_WOUNDED
            print(f"{self.identifier}going to wounded person")


        elif self.state is self.Activity.SEARCHING_WOUNDED:
            wounded_disappeared = True
            pos_x, pos_y = int(self.target_position[0]), int(self.target_position[1])
            search_radius = 10
            missing_limit = 5
            for x in range(max(pos_x - search_radius, 0), min(pos_x + search_radius, self.semantic_grid.x_max_grid-1)):
                for y in range(max(pos_y - search_radius, 0), min(pos_y + search_radius, self.semantic_grid.y_max_grid-1)):
                    if self.semantic_grid.grid[x, y] == self.semantic_grid.Entity.WOUNDED_PERSON.value:
                        wounded_disappeared = False
                        #print("Wounded person still there")
                        self.wounded_disappeared = 0
                        break
            if wounded_disappeared:
                if self.wounded_disappeared==missing_limit:
                    self.semantic_grid.grid[int(pos_x), int(pos_y)] = self.semantic_grid.Entity.NONE.value
                    self.state = self.Activity.EXPLORING
                    self.wounded_disappeared = 0
                else:
                    self.wounded_disappeared += 1

        # Here also assigning the target position to the rescue center
        elif (self.state is self.Activity.SEARCHING_WOUNDED and
              self.base.grasper.grasped_entities):
            self.state = self.Activity.SEARCHING_RESCUE_CENTER
            print("Going to rescue center")
            for x in range(self.semantic_grid.x_max_grid):
                for y in range(self.semantic_grid.y_max_grid):
                    if self.semantic_grid.grid[x, y] == self.semantic_grid.Entity.RESCUE_CENTER.value:
                        self.target_position = np.array([x, y])
                        self.path = None
                        break

        elif (self.state is self.Activity.SEARCHING_RESCUE_CENTER and
              not self.base.grasper.grasped_entities):
            print("Lost the wounded person")
            self.state = self.Activity.EXPLORING

        # print("state: {}, can_grasp: {}, grasped entities: {}"
        #       .format(self.state.name,
        #               self.base.grasper.can_grasp,
        #               self.base.grasper.grasped_entities))

        ##########
        # COMMANDS FOR EACH STATE
        ##########

        if self.state is self.Activity.EXPLORING:
            command = self.control_explore(self.target_position)
            command["grasper"] = 0

        elif self.state is self.Activity.SEARCHING_WOUNDED or self.state is self.Activity.SEARCHING_RESCUE_CENTER:
            command = self.control_explore(self.target_position)
            command["grasper"] = 1

        # increment the iteration counter
        self.iteration += 1

        # update drone dictionary
        self.drone_dict[self.identifier] = self.iteration
        self.drone_dict = {drone: time for drone, time in self.drone_dict.items() if self.iteration - time < self.TIME_TO_DEAD}

        # Display the occupancy grid and the semantic grid
        if self.iteration % 5 == 0 and self.identifier == 0:
            self.occupancy_grid.display(self.occupancy_grid.zoomed_grid,
                                        self.estimated_pose,
                                        title="zoomed occupancy grid")
            self.semantic_grid.display(self.semantic_grid.zoomed_grid,
                                       self.estimated_pose,
                                       title="zoomed semantic grid")
            #print(self.drone_dict)

        return command
    def determine_wounded(self):
            """
            Deprecated.
            """
            detection_semantic = self.semantic_values()
            best_data = None

            if (self.state is self.Activity.EXPLORING
                or self.state is self.Activity.GRASPING_WOUNDED) \
                    and detection_semantic is not None:
                scores = []
                for data in detection_semantic:
                    # If the wounded person detected is held by nobody
                    if (data.entity_type ==
                            DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and
                            not data.grasped):
                        v = (data.angle * data.angle) + \
                            (data.distance * data.distance / 10 ** 5)
                        scores.append((v, data.angle, data.distance))

                # Select the best one among wounded persons detected
                best_score = 10000
                for score in scores:
                    if score[0] < best_score:
                        best_score = score[0]
                        best_data = score[1:]
            return best_data