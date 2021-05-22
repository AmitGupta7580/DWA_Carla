#!/usr/bin/env python

from __future__ import print_function

import glob
import sys

try:
    sys.path.append(glob.glob('.\\Windows\\CARLA_0.9.11\\PythonAPI\\carla\\dist\\carla-0.9.11-py3.7-win-amd64.egg')[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import threading
import time

# ==============================================================================
# -- Dynamic Window Approach ---------------------------------------------------
# ==============================================================================

class Config:
    """
    DWA parameter class
    """
    def __init__(self):
        # robot parameter
        self.max_speed = 5.0  # [m/s]
        self.min_speed = 0  # [m/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 4  # [m/ss]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.2  # [m/s]
        self.yaw_rate_resolution = 0.4 * math.pi / 180.0  # [rad/s]
        self.dt = 0.2  # [s] Time tick for motion prediction
        self.predict_time = 3.0  # [s]
        self.to_goal_cost_gain = 0.2
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 7.0
        self.lane_cost_gain = 3.0
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
        # Robot Dimension information
        self.robot_radius = 1.0  # [m] for collision check

class DWA:
    def __init__(self, goal):
        #######   Parameters for DWA  ########
        self._config = Config()
        self._goal = goal
        self._obj_dis = None
        self._lane_points = None
        ######################################
        self._front_image = None
        self._depth_image = None
        ######################################

    def dwa_control(self, state, image, dep_image):
        """
        Dynamic Window Approach control
        """
        # self._front_image = image
        # self._depth_image = dep_image
        self._obj_dis = image
        self._lane_points = dep_image
        # image proccesing 
        # self._image_processing()

        #  DWA
        dw = self._calc_dynamic_window(state)  # dynamic window
        u, trajectory = self._calc_control_and_trajectory(dw, state)

        return u, trajectory

    def _calc_dynamic_window(self, state):
        """
        calculation dynamic window based on current state x
        """
        # Dynamic window from robot specification
        Vs = [self._config.min_speed, self._config.max_speed, -self._config.max_yaw_rate, self._config.max_yaw_rate]
        # Dynamic window from motion model
        Vd = [state[3] - self._config.max_accel * self._config.dt,
                state[3] + self._config.max_accel * self._config.dt,
                state[4] - self._config.max_delta_yaw_rate * self._config.dt,
                state[4] + self._config.max_delta_yaw_rate * self._config.dt]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
                max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
        return dw

    def _calc_control_and_trajectory(self, dw, state):
        """
        Calculation final input with dynamic window
        """
        x_init = state[:]
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([state])

        # evaluate all trajectory with sampled input in dynamic window
        for v in np.arange(dw[0], dw[1], self._config.v_resolution):
            for y in np.arange(dw[2], dw[3], self._config.yaw_rate_resolution):
                trajectory = self._predict_trajectory(x_init, v, y)
                # calc cost
                to_goal_cost = self._config.to_goal_cost_gain * self._calc_to_goal_cost(trajectory)
                speed_cost = self._config.speed_cost_gain * (self._config.max_speed - trajectory[-1, 3])
                ob_cost = self._config.obstacle_cost_gain * self._calc_obstacle_cost(trajectory, x_init)
                lane_cost = self._config.lane_cost_gain * self._calc_lane_cost(trajectory, x_init)
                final_cost = to_goal_cost + speed_cost + ob_cost + lane_cost
                if min_cost > final_cost :
                    min_cost = final_cost
                    best_u = [v, y]
                    best_trajectory = trajectory
                    if abs(best_u[0]) < self._config.robot_stuck_flag_cons and abs(state[3]) < self._config.robot_stuck_flag_cons:
                        best_u[1] = -self._config.max_delta_yaw_rate
        return best_u, best_trajectory

    def _predict_trajectory(self, x_init, v, y):
        """
        Predict trajectory with an input
        """
        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        while time <= self._config.predict_time:
            x = self.motion(x, [v, y], self._config.dt)
            trajectory = np.vstack((trajectory, x))
            time += self._config.dt
        return trajectory

    def motion(self, x, u, dt):
        """
        Motion 
        """
        x[2] += u[1] * dt
        x[0] += u[0] * math.cos(x[2]) * dt
        x[1] += u[0] * math.sin(x[2]) * dt
        x[3] = u[0]
        x[4] = u[1]

        return x

    def _calc_obstacle_cost(self, trajectory, x_init):
        '''
        To be implemented using image fetched from camera
        '''
        if len(self._obj_dis) == 0:
            return 0
        ox = self._obj_dis[:, 0]
        oy = self._obj_dis[:, 1]
        dx = trajectory[:, 0] - ox[:, None]
        dy = trajectory[:, 1] - oy[:, None]
        r = np.hypot(dx, dy)

        if np.array(r <= self._config.robot_radius).any():
            return float("Inf")

        min_r = np.min(r)
        return 1.0 / min_r

    def _calc_to_goal_cost(self, trajectory):
        """
        Calculate to goal cost with angle difference
        """
        dx = self._goal[0] - trajectory[-1, 0]
        dy = self._goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1, 2]
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))
        return math.sqrt((dx*dx) + (dy*dy))

    def _calc_lane_cost(self, trajectory, x_init):
        """
        Calculate to Lane cost
        """
        if len(self._lane_points) == 0:
            return 0
        r = []
        for lane in self._lane_points:
            rm = []
            for t in trajectory:
                rm.append(abs(self._perpendicular_distance_from_line(t, lane)))
            r.append(rm)
        r = np.array(r)

        # print(r)
        if np.array(r <= self._config.robot_radius).any():
            return float("Inf")

        min_r = np.min(r)
        return 1.0 / min_r

    def _perpendicular_distance_from_line(self, state, lane_points):
        """
        Calculate to Distance of a position from the lane
        """

        # points of lane and position
        x1 = lane_points[0][0]
        x2 = lane_points[1][0]
        y1 = lane_points[0][1]
        y2 = lane_points[1][1]
        px = state[0]
        py = state[1]

        # Edge-Cases
        if (y2-y1) == 0: # horizontal line
            if px > max(x1, x2) or px < min(x1, x2):
                return float("Inf")
        elif (x2-x1) == 0: # vertical line
            if py > max(y1, y2) or py < min(y1, y2):
                return float("Inf")
        else:
            m_per = (x1-x2)/(y2-y1)
            if not self._on_same_side(lane_points[0], [px, py], m_per, lane_points[1]) and\
                 not self._on_same_side(lane_points[1], [px, py], m_per, lane_points[0]):
                return float("Inf")

        # vertical line
        if (x2-x1) == 0:                      
            return px-x1

        # lane parameters
        m = (y2-y1)/(x2-x1)                   # slope
        c = y1-(x1*m)                         # constant
        d = (m*px + c - py)/(math.sqrt(m*m + 1)) # d = m*x1+c-y1/(1+m^2)

        return d

    def _on_same_side(self, point_1, point_2, m, point):
        c = point[1] - m*point[0]  # constant of line
        if (m*point_1[0] + c - point_1[1])*(m*point_2[0] + c - point_2[1]) < 0:
            return False
        return True
