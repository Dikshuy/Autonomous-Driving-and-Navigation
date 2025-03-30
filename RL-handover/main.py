#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import re
import math
import random
import weakref
import numpy as np
import pygame
from pygame.locals import KMOD_CTRL, K_ESCAPE, K_q
import argparse
import logging

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

import mpc as MPCController

def wrap_angle(angle_in_degree):
    """Convert degrees to radians and normalize to [-pi, pi]"""
    angle_in_rad = angle_in_degree / 180.0 * np.pi
    while angle_in_rad > np.pi:
        angle_in_rad -= 2 * np.pi
    while angle_in_rad <= -np.pi:
        angle_in_rad += 2 * np.pi
    return angle_in_rad

def get_vehicle_wheelbases(wheels, center_of_mass):
    """Calculate vehicle wheelbase parameters from physics data"""
    front_wheels = wheels[:2]
    rear_wheels = wheels[2:]
    
    front_pos = np.mean([np.array([w.position.x, w.position.y, w.position.z]) for w in front_wheels], axis=0)
    rear_pos = np.mean([np.array([w.position.x, w.position.y, w.position.z]) for w in rear_wheels], axis=0)
    
    wheelbase = np.sqrt(np.sum((front_pos - rear_pos)**2)) / 100.0  # Convert to meters
    
    return wheelbase - center_of_mass.x, center_of_mass.x, wheelbase

class World:
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.map = self.world.get_map()
        self.hud = hud
        self.player = None
        self.camera_manager = None
        self._actor_filter = args.filter
        self._gamma = args.gamma
        self.args = args
        
        self.waypoint_resolution = args.waypoint_resolution
        self.waypoint_lookahead_distance = args.waypoint_lookahead_distance
        self.desired_speed = args.desired_speed
        self.planning_horizon = args.planning_horizon
        self.time_step = args.time_step
        self.controller = None
        self.control_count = 0.0
        
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        
    def restart(self):
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        
        # Select vehicle blueprint
        blueprint = self._get_vehicle_blueprint()
        
        # Spawn the player
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        
        while self.player is None:
            if self.args.random_spawn:
                spawn_points = self.map.get_spawn_points()
                spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            else:
                spawn_location = carla.Location(x=float(self.args.spawn_x), y=float(self.args.spawn_y))
                spawn_waypoint = self.map.get_waypoint(spawn_location)
                spawn_transform = spawn_waypoint.transform
                spawn_transform.location.z = 1.0
                spawn_point = spawn_transform
                
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            
            if self.player:
                self._init_controller()
        
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)

    def _get_vehicle_blueprint(self):
        blueprints = self.world.get_blueprint_library().filter(self._actor_filter)
        
        for bp in blueprints:
            if bp.id == self.args.vehicle_id:
                blueprint = bp
                break
        else:
            blueprint = random.choice(blueprints)
        
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            blueprint.set_attribute('color', random.choice(blueprint.get_attribute('color').recommended_values))
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
            
        return blueprint
    
    def _init_controller(self):
        self.control_count = 0
        physic_control = self.player.get_physics_control()
        lf, lr, l = get_vehicle_wheelbases(physic_control.wheels, physic_control.center_of_mass)
        
        self.controller = MPCController.Controller(
            lf=lf, 
            lr=lr, 
            wheelbase=l, 
            planning_horizon=self.planning_horizon, 
            time_step=self.time_step
        )
        
        # Initialize controller with current state
        transform = self.player.get_transform()
        velocity = self.player.get_velocity()
        
        current_x = transform.location.x
        current_y = transform.location.y
        current_yaw = wrap_angle(transform.rotation.yaw)
        current_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        frame, timestamp = self.hud.get_simulation_information()
        self.controller.update_values(current_x, current_y, current_yaw, current_speed, timestamp, frame)

    def render(self, display):
        self.camera_manager.render(display)

    def tick(self, clock):
        pass

    def destroy(self):
        if self.camera_manager and self.camera_manager.sensor:
            self.camera_manager.sensor.destroy()
        if self.player:
            self.player.destroy()
        self.player = None
        self.camera_manager = None

class VehicleControl:
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            world.player.set_autopilot(self._autopilot_enabled)
        else:
            raise NotImplementedError("Only vehicle actors supported")

    def parse_events(self, client, world, clock):
        if not self._autopilot_enabled:
            self._handle_manual_control(world)
            
    def _handle_manual_control(self, world):
        """Apply MPC control based on waypoints"""
        # Get current vehicle state
        vehicle = world.player
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        
        current_x = transform.location.x
        current_y = transform.location.y
        current_yaw = wrap_angle(transform.rotation.yaw)
        current_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # Update controller with current state
        frame, timestamp = world.hud.get_simulation_information()
        ready_to_go = world.controller.update_values(current_x, current_y, current_yaw, current_speed, timestamp, frame)
        
        if ready_to_go:
            # Generate waypoints
            waypoints = self._generate_waypoints(world, transform.location, current_speed)
            world.controller.update_waypoints(waypoints)
            
            # Compute and apply control
            world.controller.update_controls()
            self._control.throttle, self._control.steer, self._control.brake = world.controller.get_commands()
            vehicle.apply_control(self._control)
            world.control_count += 1
    
    def _generate_waypoints(self, world, current_location, current_speed):
        """Generate waypoints for planning horizon"""
        waypoints = []
        prev_waypoint = world.map.get_waypoint(current_location)
        
        # Calculate distance to next waypoint based on current speed
        dist = world.time_step * current_speed + 0.1
        
        # Get first waypoint
        current_waypoint = prev_waypoint.next(dist)[0]
        
        # Target speed ramp-up
        road_desired_speed = world.desired_speed
        
        # Generate waypoints for planning horizon
        for i in range(world.planning_horizon):
            # Ramp up speed for first 100 control steps
            if world.control_count + i <= 100:
                desired_speed = (world.control_count + 1 + i)/100.0 * road_desired_speed
            else:
                desired_speed = road_desired_speed
                
            # Get next waypoint
            dist = world.time_step * road_desired_speed
            current_waypoint = prev_waypoint.next(dist)[0]
            
            # Store waypoint data [x, y, speed, yaw]
            waypoints.append([
                current_waypoint.transform.location.x,
                current_waypoint.transform.location.y,
                road_desired_speed,
                wrap_angle(current_waypoint.transform.rotation.yaw)
            ])
            
            prev_waypoint = current_waypoint
            
        return waypoints

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

class HUD:
    def __init__(self, width, height):
        self.dim = (width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds
    
    def get_simulation_information(self):
        return self.frame, self.simulation_time

class CameraManager:
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        
        # Camera setup
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), carla.AttachmentType.SpringArm),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), carla.AttachmentType.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), carla.AttachmentType.SpringArm),
            (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), carla.AttachmentType.SpringArm),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), carla.AttachmentType.Rigid)
        ]
        
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
        ]
        
        # Setup camera blueprint
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
            item.append(bp)
        
        self.index = None

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set up the selected sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
            
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
                
            # Spawn the sensor
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1]
            )
            
            # Setup callback
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
            
        self.index = index

    def render(self, display):
        """Render camera view to display"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        """Process incoming image from sensor"""
        self = weak_self()
        if not self:
            return
            
        # Convert image to pygame surface
        image.convert(self.sensors[self.index][1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

def game_loop(args):
    """Main game loop"""
    pygame.init()
    pygame.font.init()
    world = None

    try:
        # Connect to CARLA server
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        # Setup display
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )

        # Load world and setup simulation
        hud = HUD(args.width, args.height)
        carla_world = client.load_world(args.map)
        world = World(carla_world, hud, args)
        controller = VehicleControl(world, args.autopilot)

        # Main loop
        clock = pygame.time.Clock()
        while True:
            clock.tick_busy_loop(args.fps)
            controller.parse_events(client, world, clock)
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:
        if world is not None:
            world.destroy()
        pygame.quit()

def main():
    """Parse arguments and start simulation"""
    argparser = argparse.ArgumentParser(description='CARLA MPC Control')
    
    # Connection settings
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    
    # Display settings
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='1280x720', help='window resolution (default: 1280x720)')
    argparser.add_argument('--gamma', default=2.2, type=float, help='Gamma correction of the camera (default: 2.2)')
    
    # Simulation settings
    argparser.add_argument('--map', metavar='NAME', default='Town04', help='simulation map (default: "Town04")')
    argparser.add_argument('-a', '--autopilot', action='store_true', help='enable autopilot')
    argparser.add_argument('--fps', default=10, type=int, help='Frames per second for simulation (default: 10)')
    
    # Vehicle settings
    argparser.add_argument('--filter', metavar='PATTERN', default='vehicle.*', help='actor filter (default: "vehicle.*")')
    argparser.add_argument('--vehicle_id', metavar='NAME', default='vehicle.audi.a2', 
                          help='vehicle to spawn (default: vehicle.audi.a2)')
    
    # Spawn settings
    argparser.add_argument('--spawn_x', metavar='X', default='-510', help='x position to spawn the agent (default: 120)')
    argparser.add_argument('--spawn_y', metavar='Y', default='200', help='y position to spawn the agent (default: -8)')
    argparser.add_argument('--random_spawn', metavar='RS', default=0, type=int, help='Random spawn agent (default: 0)')
    
    # Controller settings
    argparser.add_argument('--waypoint_resolution', metavar='WR', default=0.5, type=float, 
                          help='waypoint resolution for control (default: 0.5)')
    argparser.add_argument('--waypoint_lookahead_distance', metavar='WLD', default=5.0, type=float,
                          help='waypoint look ahead distance for control (default: 5.0)')
    argparser.add_argument('--desired_speed', metavar='SPEED', default=30, type=float,
                          help='desired speed for driving (default: 30)')
    argparser.add_argument('--planning_horizon', metavar='HORIZON', type=int, default=5,
                          help='Planning horizon for MPC (default: 5)')
    argparser.add_argument('--time_step', metavar='DT', default=0.3, type=float,
                          help='Planning time step for MPC (default: 0.3)')
    
    # Debug options
    argparser.add_argument('-v', '--verbose', action='store_true', dest='debug', help='print debug information')
    
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':
    main()