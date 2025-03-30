#!/usr/bin/env python

from __future__ import print_function

import glob
import os
import sys
import argparse
import math
import random
import numpy as np
import carla
import mpc as MPCController
import pygame

def wrap_angle(angle_in_degree):
    angle_in_rad = angle_in_degree / 180.0 * np.pi
    while (angle_in_rad > np.pi):
        angle_in_rad -= 2 * np.pi
    while (angle_in_rad <= -np.pi):
        angle_in_rad += 2 * np.pi
    return angle_in_rad

def get_vehicle_wheelbases(wheels, center_of_mass):
    front_left_wheel = wheels[0]
    front_right_wheel = wheels[1]
    back_left_wheel = wheels[2]
    back_right_wheel = wheels[3]
    front_x = (front_left_wheel.position.x + front_right_wheel.position.x) / 2.0
    front_y = (front_left_wheel.position.y + front_right_wheel.position.y) / 2.0
    front_z = (front_left_wheel.position.z + front_right_wheel.position.z) / 2.0
    back_x = (back_left_wheel.position.x + back_right_wheel.position.x) / 2.0
    back_y = (back_left_wheel.position.y + back_right_wheel.position.y) / 2.0
    back_z = (back_left_wheel.position.z + back_right_wheel.position.z) / 2.0
    l = np.sqrt( (front_x - back_x)**2 + (front_y - back_y)**2 + (front_z - back_z)**2  ) / 100.0
    return l - center_of_mass.x, center_of_mass.x, l

class World(object):
    def __init__(self, carla_world, args):
        self.world = carla_world
        self.actor_role_name = args.rolename
        self.map = self.world.get_map()
        self._actor_filter = args.filter
        self.args = args
        self.waypoint_resolution = args.waypoint_resolution
        self.waypoint_lookahead_distance = args.waypoint_lookahead_distance
        self.desired_speed = args.desired_speed
        self.planning_horizon = args.planning_horizon
        self.time_step = args.time_step
        self.control_mode = args.control_mode
        self.controller = None
        self.control_count = 0.0
        self.player = None
        self.restart()
        
    def restart(self):
        # Get vehicle blueprint
        blueprints = self.world.get_blueprint_library().filter(self._actor_filter)
        blueprint = None
        for blueprint_candidates in blueprints:
            if blueprint_candidates.id == self.args.vehicle_id:
                blueprint = blueprint_candidates
                break
        if blueprint is None:
            blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))

        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
            
        # Spawn the player
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.player.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            
        while self.player is None:
            if self.args.random_spawn == 1:
                spawn_points = self.world.get_map().get_spawn_points()
                spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
                self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            else:
                spawn_location = carla.Location()
                spawn_location.x = float(self.args.spawn_x)
                spawn_location.y = float(self.args.spawn_y)
                spawn_waypoint = self.map.get_waypoint(spawn_location)
                spawn_transform = spawn_waypoint.transform
                spawn_transform.location.z = 1.0
                self.player = self.world.try_spawn_actor(blueprint, spawn_transform)

            self.control_count = 0
            physic_control = self.player.get_physics_control()
            lf, lr, l = get_vehicle_wheelbases(physic_control.wheels, physic_control.center_of_mass)
            self.controller = MPCController.Controller(lf=lf, lr=lr, wheelbase=l, 
                                                    planning_horizon=self.args.planning_horizon, 
                                                    time_step=self.args.time_step)
            velocity_vec = self.player.get_velocity()
            current_transform = self.player.get_transform()
            current_location = current_transform.location
            current_roration = current_transform.rotation
            current_x = current_location.x
            current_y = current_location.y
            current_yaw = wrap_angle(current_roration.yaw)
            current_speed = math.sqrt(velocity_vec.x**2 + velocity_vec.y**2 + velocity_vec.z**2)
            self.controller.update_values(current_x, current_y, current_yaw, current_speed, 0, 0)

    def destroy(self):
        if self.player is not None:
            self.player.destroy()

class VehicleControl(object):
    def __init__(self, world):
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0

    def parse_events(self, world, clock):
        # Control loop
        velocity_vec = world.player.get_velocity()
        current_transform = world.player.get_transform()
        current_location = current_transform.location
        current_rotation = current_transform.rotation
        current_x = current_location.x
        current_y = current_location.y
        current_yaw = wrap_angle(current_rotation.yaw)
        current_speed = math.sqrt(velocity_vec.x**2 + velocity_vec.y**2 + velocity_vec.z**2)
        
        ready_to_go = world.controller.update_values(current_x, current_y, current_yaw, current_speed, 0, 0)
        if ready_to_go:
            if world.control_mode == "MPC":
                dist = world.time_step * current_speed + 0.1
                prev_waypoint = world.map.get_waypoint(current_location)
                current_waypoint = prev_waypoint.next(dist)[0]
                waypoints = []
                road_desired_speed = world.desired_speed
                
                for i in range(world.planning_horizon):
                    if world.control_count + i <= 100:
                        desired_speed = (world.control_count + 1 + i)/100.0 * road_desired_speed
                    else:
                        desired_speed = road_desired_speed
                    dist = world.time_step * road_desired_speed
                    current_waypoint = prev_waypoint.next(dist)[0]
                    waypoints.append([current_waypoint.transform.location.x, 
                                     current_waypoint.transform.location.y, 
                                     road_desired_speed, 
                                     wrap_angle(current_waypoint.transform.rotation.yaw)])
                    prev_waypoint = current_waypoint
                    
            world.controller.update_waypoints(waypoints)     
            world.controller.update_controls()
            self._control.throttle, self._control.steer, self._control.brake = world.controller.get_commands()
            world.player.apply_control(self._control)
            world.control_count += 1

def game_loop(args):
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        carla_world = client.load_world(args.map)
        world = World(carla_world, args)
        controller = VehicleControl(world)

        clock = pygame.time.Clock()
        while True:
            clock.tick_busy_loop(args.FPS)
            controller.parse_events(world, clock)

    finally:
        if world is not None:
            world.destroy()

def main():
    argparser = argparse.ArgumentParser(description='CARLA MPC Control Client')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to')
    argparser.add_argument('--filter', metavar='PATTERN', default='vehicle.*', help='actor filter')
    argparser.add_argument('--rolename', metavar='NAME', default='hero', help='actor role name')
    argparser.add_argument('--map', metavar='NAME', default='Town04', help='simulation map')
    argparser.add_argument('--spawn_x', metavar='x', default='-510', help='x position to spawn the agent')
    argparser.add_argument('--spawn_y', metavar='y', default='200', help='y position to spawn the agent')
    argparser.add_argument('--random_spawn', metavar='RS', default='0', type=int, help='Random spawn agent')
    argparser.add_argument('--vehicle_id', metavar='NAME', default='vehicle.ford.mustang', help='vehicle to spawn')
    argparser.add_argument('--vehicle_wheelbase', metavar='NAME', type=float, default='2.89', help='vehicle wheelbase')
    argparser.add_argument('--waypoint_resolution', metavar='WR', default='0.5', type=float, help='waypoint resolution')
    argparser.add_argument('--waypoint_lookahead_distance', metavar='WLD', default='5.0', type=float, help='waypoint look ahead distance')
    argparser.add_argument('--desired_speed', metavar='SPEED', default='25', type=float, help='desired speed')
    argparser.add_argument('--control_mode', metavar='CONT', default='MPC', help='Controller')
    argparser.add_argument('--planning_horizon', metavar='HORIZON', type=int, default='4', help='Planning horizon for MPC')
    argparser.add_argument('--time_step', metavar='DT', default='0.4', type=float, help='Planning time step for MPC')
    argparser.add_argument('--FPS', metavar='FPS', default='10', type=int, help='Frame per second for simulation')

    args = argparser.parse_args()

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':
    main()