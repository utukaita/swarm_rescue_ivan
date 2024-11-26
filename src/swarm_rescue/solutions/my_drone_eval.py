import random
import math
from typing import Optional, List, Type
from enum import Enum
import numpy as np

from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.utils import normalize_angle, circular_mean

import os
import sys

import cv2

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spg_overlay.utils.constants import MAX_RANGE_LIDAR_SENSOR
from spg_overlay.utils.grid import Grid

from spg_overlay.utils.pose import Pose
from spg_overlay.entities.drone_abstract import DroneAbstract


class SemanticGrid(Grid):
    def __init__(self,
                 size_area_world,
                 resolution: float,
                 semantic_sensor):  # semantic sensor
        super().__init__(size_area_world=size_area_world,
                         resolution=resolution)

        self.size_area_world = size_area_world
        self.resolution = resolution

        self.semantic_sensor = semantic_sensor  # Semantic sensor instead

        self.x_max_grid: int = int(self.size_area_world[0] / self.resolution
                                   + 0.5)
        self.y_max_grid: int = int(self.size_area_world[1] / self.resolution
                                   + 0.5)

        self.grid = np.zeros((self.x_max_grid, self.y_max_grid))
        self.zoomed_grid = np.empty((self.x_max_grid, self.y_max_grid))

    def update_grid(self, pose: Pose):
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

        # For obstacle zones, all values of lidar_dist are < max_range
        select_collision = lidar_dist < max_range

        points_x = pose.position[0] + np.multiply(lidar_dist, cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist, sin_rays)

        points_x = points_x[select_collision]
        points_y = points_y[select_collision]

        self.add_points(points_x, points_y, OBSTACLE_ZONE_VALUE)

        # the current position of the drone is free !
        self.add_points(pose.position[0], pose.position[1], FREE_ZONE_VALUE)

        # threshold values
        self.grid = np.clip(self.grid, THRESHOLD_MIN, THRESHOLD_MAX)

        # compute zoomed grid for displaying
        self.zoomed_grid = self.grid.copy()
        new_zoomed_size = (int(self.size_area_world[1] * 0.5),
                           int(self.size_area_world[0] * 0.5))
        self.zoomed_grid = cv2.resize(self.zoomed_grid, new_zoomed_size,
                                      interpolation=cv2.INTER_NEAREST)


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

        self.x_max_grid: int = int(self.size_area_world[0] / self.resolution
                                   + 0.5)
        self.y_max_grid: int = int(self.size_area_world[1] / self.resolution
                                   + 0.5)

        self.grid = np.zeros((self.x_max_grid, self.y_max_grid))
        self.zoomed_grid = np.empty((self.x_max_grid, self.y_max_grid))

    def update_grid(self, pose: Pose):
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

        # For obstacle zones, all values of lidar_dist are < max_range
        select_collision = lidar_dist < max_range

        points_x = pose.position[0] + np.multiply(lidar_dist, cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist, sin_rays)

        points_x = points_x[select_collision]
        points_y = points_y[select_collision]

        self.add_points(points_x, points_y, OBSTACLE_ZONE_VALUE)

        # the current position of the drone is free !
        self.add_points(pose.position[0], pose.position[1], FREE_ZONE_VALUE)

        # threshold values
        self.grid = np.clip(self.grid, THRESHOLD_MIN, THRESHOLD_MAX)

        # compute zoomed grid for displaying
        self.zoomed_grid = self.grid.copy()
        new_zoomed_size = (int(self.size_area_world[1] * 0.5),
                           int(self.size_area_world[0] * 0.5))
        self.zoomed_grid = cv2.resize(self.zoomed_grid, new_zoomed_size,
                                      interpolation=cv2.INTER_NEAREST)


class MyDroneEval(DroneAbstract):
    class Activity(Enum):
        """
        All the states of the drone as a state machine
        """
        EXPLORING = 0
        SEARCHING_WOUNDED = 1
        GRASPING_WOUNDED = 2
        SEARCHING_RESCUE_CENTER = 3
        DROPPING_AT_RESCUE_CENTER = 4

    # to calculate map
    OCCUPIED_CERTAINTY_THRESHOLD = 0.99  # probability of being occupied (99% occupied)
    FREE_CERTAINTY_THRESHOLD = 1 - OCCUPIED_CERTAINTY_THRESHOLD  # probablity of not being occupied (99% free = 1% occupied)
    GRID_OCCUPIED_THRESHOLD = math.log(OCCUPIED_CERTAINTY_THRESHOLD) - math.log(
        1 - OCCUPIED_CERTAINTY_THRESHOLD)  # using log-odds probability
    GRID_FREE_THRESHOLD = -GRID_OCCUPIED_THRESHOLD

    # to calculate utility
    NUM_REGIONS = 5

    # to calculate distance
    DISTANCE_THRESHOLD = 10

    # to calculate speed
    MAX_SPEED = 1

    def __init__(self,
                 identifier: Optional[int] = None, **kwargs):
        super().__init__(identifier=identifier,
                         display_lidar_graph=False,
                         **kwargs)
        # The state is initialized to searching wounded person
        self.state = self.Activity.EXPLORING

        # Values associated with movement in general
        self.speed = self.MAX_SPEED

        # Those values are used by the random control function
        self.counterStraight = 0
        self.angleStopTurning = 0
        self.isTurning = False

        self.iteration: int = 0

        self.estimated_pose = Pose()

        resolution = 8
        self.grid = OccupancyGrid(size_area_world=self.size_area,
                                  resolution=resolution,
                                  lidar=self.lidar())

        self.target = None

    def define_message_for_all(self):
        """
        Sharing the map
        Sharing positions of wounded people
        Sharing if carrying a wounded person

        Test different messaging periods
        """
        pass

    def control(self):
        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}

        found_wounded, found_rescue_center, command_semantic = (
            self.process_semantic_sensor())

        #############
        # TRANSITIONS OF THE STATE MACHINE
        #############

        if self.state is self.Activity.EXPLORING or self.Activity.SEARCHING_WOUNDED and found_wounded:
            self.state = self.Activity.GRASPING_WOUNDED

        elif (self.state is self.Activity.GRASPING_WOUNDED and
              self.base.grasper.grasped_entities):
            self.state = self.Activity.SEARCHING_RESCUE_CENTER

        elif (self.state is self.Activity.GRASPING_WOUNDED and
              not found_wounded):
            self.state = self.Activity.SEARCHING_WOUNDED

        elif (self.state is self.Activity.SEARCHING_RESCUE_CENTER and
              found_rescue_center):
            self.state = self.Activity.DROPPING_AT_RESCUE_CENTER

        elif (self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and
              not self.base.grasper.grasped_entities):
            self.state = self.Activity.EXPLORING

        elif (self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and
              not found_rescue_center):
            self.state = self.Activity.SEARCHING_RESCUE_CENTER

        # print("state: {}, can_grasp: {}, grasped entities: {}"
        #       .format(self.state.name,
        #               self.base.grasper.can_grasp,
        #               self.base.grasper.grasped_entities))

        ##########
        # COMMANDS FOR EACH STATE
        # Searching randomly, but when a rescue center or wounded person is
        # detected, we use a special command
        ##########
        if self.state is self.Activity.EXPLORING:
            command = self.control_explore()
            command["grasper"] = 0

        if self.state is self.Activity.SEARCHING_WOUNDED:
            command = self.control_explore()
            command["grasper"] = 0

        elif self.state is self.Activity.GRASPING_WOUNDED:
            command = command_semantic
            command["grasper"] = 1

        elif self.state is self.Activity.SEARCHING_RESCUE_CENTER:
            command = self.control_explore()
            command["grasper"] = 1

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER:
            command = command_semantic
            command["grasper"] = 1

        # increment the iteration counter
        self.iteration += 1

        self.estimated_pose = Pose(np.asarray(self.measured_gps_position()),
                                   self.measured_compass_angle())
        # self.estimated_pose = Pose(np.asarray(self.true_position()),
        #                            self.true_angle())
        if isinstance(self.measured_gps_position(), np.ndarray):
            self.estimated_pose = Pose(np.asarray(self.measured_gps_position()),
                                       self.measured_compass_angle())
            self.speed = self.MAX_SPEED

        else:
            dist, alpha, theta = (0.0, 0.0, 0.0)
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

        self.grid.update_grid(pose=self.estimated_pose)
        if self.iteration % 5 == 0:
            # self.grid.display(self.grid.grid,
            #                       self.estimated_pose,
            #                       title="occupancy grid")
            self.grid.display(self.grid.zoomed_grid,
                              self.estimated_pose,
                              title="zoomed occupancy grid")
            # pass
        # Update semantics on the map

        return command

    def process_lidar_sensor(self):
        """
        Returns True if the drone collided an obstacle
        """
        if self.lidar_values() is None:
            return False

        collided = False
        dist = min(self.lidar_values())

        if dist < 40:
            collided = True

        return collided

    def control_explore(self):
        """
        The Drone will move forward and turn for a random angle when an
        obstacle is hit
        """
        pos = self.grid._conv_world_to_grid(*self.measured_gps_position())
        angle = self.measured_compass_angle()
        forward = 0
        rotation = 0

        if self.target is None or np.linalg.norm(self.target - pos) < self.DISTANCE_THRESHOLD:
            # convert probabilities grid to binary grid
            bin_grid = self.grid.grid.copy()
            bin_grid[bin_grid <= self.GRID_FREE_THRESHOLD] = 0  # free
            bin_grid[bin_grid >= self.GRID_OCCUPIED_THRESHOLD] = 1  # occupied
            frontier = np.logical_and(bin_grid != 1, bin_grid != 0)  # not explored yet as not sure if free or occupied
            frontier_indices = np.nonzero(frontier)

            rand_point = random.choice(np.column_stack(frontier_indices))
            rand_point[0] -= bin_grid.shape[1]/2
            rand_point[1] -= bin_grid.shape[0]/2
            self.target = rand_point
            print(" NEW TARGET \n\n\n")
            print(f"target {self.target}")

        else:
            # Calculate target vector
            target_vector = self.target - pos

            # Target angle and angular error
            theta_t = np.arctan2(*target_vector)
            delta_theta = (theta_t - angle + np.pi) % (2 * np.pi) - np.pi

            # Distance to the target
            distance = np.linalg.norm(target_vector)

            # Control parameters
            k_angular = 1.0
            k_forward = 0.5
            epsilon = 0.1  # Angular error threshold for moving forward

            # Compute actuator values
            rotation = np.clip(k_angular * delta_theta, -1, 1)
            forward = np.clip(k_forward * distance, -1, 1) if abs(delta_theta) < epsilon else 0
            lateral_controller = 0  # No lateral movement

        return {"forward": forward,
                "rotation": rotation}

        #     target = rand_point
        #     pos_u = pos / np.linalg.norm(pos)
        #     target_u = target / np.linalg.norm(target)
        #     target_angle = np.arccos(np.clip(np.dot(pos_u, target_u), -1.0, 1.0))

        #     self.target = target
        #     self.target_u = target_u
        #     self.target_angle = target_angle

        # else:
        #     pos_u = pos / np.linalg.norm(pos)
        #     target_vector = self.target - pos
        #     target_angle = np.arctan2(target_vector[1], target_vector[0])  # (dy, dx)

        #     rotation = target_angle / 10 * np.pi
        #     forward = min(np.linalg.norm(self.target - pos) / 50, self.speed)

        # command = {"forward": forward,
        #            "rotation": rotation}

        # # command = {"forward": forward,
        # #            "rotation": rotation}
        # # if self.iteration % 75 == 0:
        # #     np.savetxt('test.txt', frontier, fmt='%d')
        # #     print(0/0)

        # return command

    def process_semantic_sensor(self):
        """
        According to his state in the state machine, the Drone will move
        towards a wound person or the rescue center
        """
        command = {"forward": 0.5,
                   "lateral": 0.0,
                   "rotation": 0.0}
        angular_vel_controller_max = 1.0

        detection_semantic = self.semantic_values()
        best_angle = 0

        found_wounded = False
        if (self.state is self.Activity.EXPLORING
            or self.state is self.Activity.SEARCHING_WOUNDED
            or self.state is self.Activity.GRASPING_WOUNDED) \
                and detection_semantic is not None:
            scores = []
            for data in detection_semantic:
                # If the wounded person detected is held by nobody
                if (data.entity_type ==
                        DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and
                        not data.grasped):  # And not grasped by another drone
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
        if (self.state is self.Activity.SEARCHING_RESCUE_CENTER
            or self.state is self.Activity.DROPPING_AT_RESCUE_CENTER) \
                and detection_semantic:
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