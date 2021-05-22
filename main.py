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

import argparse
import random
import math
import weakref
import numpy as np
import matplotlib.pyplot as plt
import cv2
import threading
import time

# ==============================================================================
# -- Other Helper Classes ------------------------------------------------------
# ==============================================================================

from pid_control import VehiclePIDController
from DWA import DWA, Config

# ==============================================================================
# -- Global variables ----------------------------------------------------------
# ==============================================================================

initial_state = {"x": -7.530000, "y": 121.209999, 
                "z": 0.500000, "pitch": 0.000000, 
                "yaw": 89.999954, "roll": 0.000000}

HEIGHT = 280
WIDTH = 400
# main_image = None
# cam_image = None
# dep_image = None
world = None

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class World(object):
    def __init__(self, carla_world, args):
        self.world = carla_world
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            sys.exit(1)
        self.vehicle = None
        self.camera_manager = None
        self._gamma = args.gamma
        self.players = [None, None]
        self.players_location = [[0, 45, 0], [5, 70, 0]]
        self.restart()

    def restart(self):

        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter('vehicle.audi.a2'))
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            blueprint.set_attribute('color', '224,0,0')
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')

        # Spawn the vehicle.
        while self.vehicle is None :#or self.player2 is None or self.player3 is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_point = carla.Transform(
                carla.Location(x=initial_state["x"], y=initial_state["y"], z=initial_state["z"]), 
                carla.Rotation(yaw=initial_state["yaw"]),
            )
            self.vehicle = self.world.try_spawn_actor(blueprint, spawn_point)
        # spawing objects
        for player_id in range(0):
            while self.players[player_id] is None:
                spawn_point = carla.Transform(
                    carla.Location(
                        x=initial_state["x"] + self.players_location[player_id][0], 
                        y=initial_state["y"] + self.players_location[player_id][1],
                        z=initial_state["z"],
                    ), 
                    carla.Rotation(yaw=initial_state["yaw"] + self.players_location[player_id][2]),
                )
                self.players[player_id] = self.world.try_spawn_actor(blueprint, spawn_point)

        # Set up the sensors.
        self.camera_manager = CameraManager(self.vehicle, self._gamma)
        # self.camera_manager.set_sensor(0, 0, notify=False)    # main camera
        self.camera_manager.set_sensor(0, 1, notify=False)    # front camera
        self.camera_manager.set_sensor(1, 1, notify=False)    # depth camera
        print("[+] World setup completed")

    def destroy_sensors(self):
        if self.camera_manager is not None:
            sen = self.camera_manager.sen
            for s in sen:
                s.destroy()
            self.camera_manager.sensor = None
            self.camera_manager.index = None

    def destroy(self):
        self.destroy_sensors()
        if self.vehicle is not None:
            self.vehicle.destroy()
        actors = self.players
        for actor in actors:
            if actor is not None:
                actor.destroy()

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================

class CameraManager(object):
    def __init__(self, parent_actor, gamma_correction):
        self.sen = []
        self.sensor = None
        self._parent = parent_actor
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), carla.AttachmentType.SpringArm),
            (carla.Transform(carla.Location(x=1.6, z=3.0), carla.Rotation(pitch=-10.0)), carla.AttachmentType.Rigid),
            (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), carla.AttachmentType.SpringArm)]
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)', {}]]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(WIDTH))
                bp.set_attribute('image_size_y', str(HEIGHT))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            item.append(bp)
        self.index = None
        self.transform_index = None
        self.main_image = None
        self.cam_image = None
        self.dep_image = None

    def set_sensor(self, index, pos_index, notify=True):
        self.transform_index = pos_index
        self.sensor = self._parent.get_world().spawn_actor(
            self.sensors[index][-1],
            self._camera_transforms[pos_index][0],
            attach_to=self._parent,
            attachment_type=self._camera_transforms[pos_index][1])
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image, index, pos_index))
        self.sen.append(self.sensor)
        self.index = index

    @staticmethod
    def _parse_image(weak_self, image, sen_index, pos_index):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            return
        else:
            image.convert(self.sensors[sen_index][1]) # (280, 400, 4)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            # depth image
            if sen_index == 1:
                if pos_index == 1:
                    array = array.astype(np.float32)
                    # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
                    dep_array = np.dot(array[:, :, :3], [65536.0, 256.0, 1.0])
                    dep_array /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
                    # global dep_image
                    self.dep_image = dep_array

            array = array[:, :, :3]
            # rgb image
            if sen_index == 0:
                if pos_index == 0 :
                    # global main_image
                    self.main_image = array
                elif pos_index == 1 :
                    # global cam_image
                    self.cam_image = array


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================

def game_loop(args):
    goal = np.array([30, 40])
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)
        global world
        world = World(client.get_world(), args)
        # while True:
        #     if world.camera_manager.dep_image is not None:
        #         z = world.camera_manager.dep_image*1000
        #         cv2.imshow('dep_image', z)
        #     else:
        #         print("lol")
        #     if world.camera_manager.cam_image is not None:
        #         cv2.imshow('cam_image', world.camera_manager.cam_image)
        #     # np.save('.\\Windows\\CARLA_0.9.11\\PythonAPI\\Data\\LaneNet_images\\dep_image', z)
        #     if cv2.waitKey(0) & 0xFF == ord('q'):
        #         break

        # Path Planning part
        # world.vehicle.set_autopilot(True)
        controller = VehiclePIDController(
            world.vehicle,
            args_lateral={'K_P':1,'K_D':0.0,'K_I':0.0},
            args_longitudinal={'K_P':1,'K_D':0.0,'K_I':0.0},
        )
        dwa = DWA(goal)
        config = Config()
        state = np.array([0.0, 0.0, (initial_state['yaw']) * math.pi / 180.0, 0.0, 0.0])
        objects = np.array([[100, 0]])# np.array([[7, 35], [0, 45], [6, 53], [5, 70]])
        lanes = [[[-2.5, 0], [-2.5, 30]], [[2.5, 0], [2.5, 24]], [[2.5, 24], [33, 24]], [[-2.5, 30], [27, 30]], [[33, 24], [33, 40]], [[27, 30], [27, 40]]]

        # Displaying lanes and objects
        for lane in lanes:
            x1 = lane[0][0]
            x2 = lane[1][0]
            y1 = lane[0][1]
            y2 = lane[1][1]
            plt.plot([y1, y2], [x1, x2])
        y = [state[1], goal[1], 70]
        x = [state[0], goal[0], 70]
        for obj in objects:
            y.append(obj[1])
            x.append(obj[0])
        plt.plot(y, x, 'ro')
        plt.xlabel('Y Direction')
        plt.ylabel('X Direction')

        # Simulation Starts
        while True:
            # self._image_processing()
            u, predicted_trajectory = dwa.dwa_control(state, objects, lanes)
            px = state[0]
            py = state[1]
            state = dwa.motion(state, u, config.dt)  # simulate robot
            plt.plot([state[1], py], [state[0], px])
            destination = carla.Location(
                x=state[0]+initial_state['x'], 
                y=state[1]+initial_state['y'], 
                z=initial_state["z"],
            )

            # move vehicle
            waypoint = world.world.get_map().get_waypoint(destination)
            control_signal = controller.run_step(state[3], waypoint)
            world.vehicle.apply_control(control_signal)
            time.sleep(3*self._config.dt)

            # edge case to reach at goal
            dist_to_goal = math.hypot(state[0] - goal[0], state[1] - goal[1])
            if dist_to_goal <= config.robot_radius:
                print("Goal Reached!!")
                break

            # debug
            # print(self._vehicle.get_location().y - initial_state['y'], self._state[1])
            # break
    finally:
        # plt.show()
        if world is not None:
            print("Destroying World")
            world.destroy()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='Amit',
        help='actor role name (default: "Amit")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    args = argparser.parse_args()
    print('[+] Listening to server {}:{}'.format(args.host, args.port))
    try:
        game_loop(args)
    except KeyboardInterrupt:
        # plt.show()
        world.destroy()
        print('\nCarla is closing...')
        
if __name__ == '__main__':
    main()
