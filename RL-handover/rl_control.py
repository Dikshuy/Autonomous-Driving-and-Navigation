import numpy as np
import random
import logging
import matplotlib.pyplot as plt
import mpc as MPCController
import carla
from ppo import RLHandoverController
import pygame
import argparse
from pygame.locals import KMOD_CTRL, K_ESCAPE, K_q
from main import World, HUD

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

# Simulated human MPC controller - this inherits from your existing MPCController
class HumanMPCController(MPCController.Controller):
    def __init__(self, waypoints=None, lf=1.5, lr=1.5, wheelbase=2.89, 
                 planning_horizon=10, time_step=0.1, human_style='normal'):
        super().__init__(waypoints, lf, lr, wheelbase, planning_horizon, time_step)
        # Human driving style: 'normal', 'aggressive', 'cautious'
        self.human_style = human_style
        
        # Add human-like characteristics based on style
        if human_style == 'aggressive':
            # Aggressive driver: higher speeds, sharper steering
            self.controller.a_max *= 1.2  # More aggressive acceleration
            self.controller.v_max *= 1.1  # Higher speed
            # Less weight on steering control cost - willing to make sharper turns
            self.controller.R[1, 1] *= 0.8
        elif human_style == 'cautious':
            # Cautious driver: lower speeds, gentler steering
            self.controller.a_max *= 0.8  # Less aggressive acceleration
            self.controller.v_max *= 0.9  # Lower speed
            # More weight on steering control cost - prefers gentler turns
            self.controller.R[1, 1] *= 1.5
        
        # Add random noise to simulate human inconsistency
        self.noise_level = 0.05 if human_style == 'normal' else \
                          0.1 if human_style == 'aggressive' else 0.02
    
    def get_inputs(self, x, y, yaw, v, waypoints):
        # Add small perturbations to the inputs to simulate human inconsistency
        perturbed_x = x + np.random.normal(0, self.noise_level)
        perturbed_y = y + np.random.normal(0, self.noise_level)
        perturbed_yaw = yaw + np.random.normal(0, self.noise_level * 0.1)
        perturbed_v = v + np.random.normal(0, self.noise_level * v)
        
        # Get control inputs using parent method with perturbed values
        return super().get_inputs(perturbed_x, perturbed_y, perturbed_yaw, perturbed_v, waypoints)
    
    def get_commands(self):
        # Get baseline commands
        throttle, steer, brake = super().get_commands()
        
        # Add some randomness to simulate human imprecision
        throttle += np.random.normal(0, self.noise_level * 0.1)
        throttle = max(0.0, min(1.0, throttle))
        
        steer += np.random.normal(0, self.noise_level * 0.1)
        steer = max(-1.0, min(1.0, steer))
        
        brake += np.random.normal(0, self.noise_level * 0.05)
        brake = max(0.0, min(1.0, brake))
        
        return throttle, steer, brake

# Modified VehicleControl class to integrate RL handover
class VehicleControlWithHandover:
    def __init__(self, world, start_in_autopilot, handover_start_time=10.0, 
                 handover_duration=7.0, human_style='normal', training_mode=True):
        self._autopilot_enabled = start_in_autopilot
        self.handover_start_time = handover_start_time
        self.handover_duration = handover_duration
        self.handover_end_time = handover_start_time + handover_duration
        self.human_style = human_style
        self.current_time = 0.0
        self.training_mode = training_mode
        self.episode_num = 0
        self.episode_rewards = []
        self.episode_alphas = []
        self.current_episode_rewards = []
        self.current_episode_alphas = []
        
        # Initialize control
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            world.player.set_autopilot(self._autopilot_enabled)
            
            # Create the human MPC controller (simulated human driver)
            self.human_controller = HumanMPCController(
                lf=world.controller.controller.lf,
                lr=world.controller.controller.lr,
                wheelbase=world.controller.controller.L,
                planning_horizon=world.planning_horizon,
                time_step=world.time_step,
                human_style=human_style
            )
            
            # Create RL handover controller
            self.rl_handover = RLHandoverController(
                autonomous_controller=world.controller,
                human_controller=self.human_controller,
                handover_start_time=handover_start_time,
                handover_duration=handover_duration
            )
            self.rl_handover.training_mode = training_mode
            
            # Set up plotting
            self.fig, self.axes = plt.subplots(2, 1, figsize=(10, 10))
            self.handover_times = []
            self.alpha_values = []
            
        else:
            raise NotImplementedError("Only vehicle actors supported")

    def parse_events(self, client, world, clock):
        if not self._autopilot_enabled:
            self._handle_control_with_handover(world)
    
    def _handle_control_with_handover(self, world):
        """Apply control with RL-based handover between autonomous and simulated human"""
        # Get current vehicle state
        vehicle = world.player
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        
        current_x = transform.location.x
        current_y = transform.location.y
        current_yaw = wrap_angle(transform.rotation.yaw)
        current_speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # Update current time from simulation
        _, timestamp = world.hud.get_simulation_information()
        self.current_time = timestamp
        
        # Generate waypoints
        waypoints = self._generate_waypoints(world, transform.location, current_speed)
        
        # Use the RL handover controller to blend controls
        throttle, steer, brake = self.rl_handover.blend_controls(
            vehicle, waypoints, self.current_time, timestamp)
        
        # Apply the blended control to the vehicle
        self._control.throttle = throttle
        self._control.steer = steer
        self._control.brake = brake
        vehicle.apply_control(self._control)
        
        # Record data for visualization
        if self.rl_handover.is_handover_phase:
            self.handover_times.append(self.current_time)
            self.alpha_values.append(self.rl_handover.current_action[0])
            self.current_episode_alphas.append(self.rl_handover.current_action[0])
            
            if self.rl_handover.last_reward != 0:
                self.current_episode_rewards.append(self.rl_handover.last_reward)
            
            # Visualize the handover process
            if len(self.handover_times) % 10 == 0:
                self._update_visualization()
        
        # Check if handover just completed
        if self.current_time >= self.handover_end_time and self.current_time - world.time_step < self.handover_end_time:
            self._end_episode(world)
    
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
    
    def _update_visualization(self):
        """Update visualization of handover process"""
        # Clear previous plots
        for ax in self.axes:
            ax.clear()
        
        # Plot alpha value over time
        self.axes[0].plot(self.handover_times, self.alpha_values, 'b-')
        self.axes[0].set_xlabel('Simulation Time (s)')
        self.axes[0].set_ylabel('Blending Parameter (α)')
        self.axes[0].set_title('Control Blending During Handover')
        self.axes[0].grid(True)
        
        # Add regions to show handover phase
        self.axes[0].axvspan(self.handover_start_time, self.handover_end_time, 
                            alpha=0.2, color='green', label='Handover Phase')
        self.axes[0].axhline(y=0.5, color='r', linestyle='--', label='Equal Control')
        self.axes[0].legend()
        
        # Plot episode rewards if in training mode
        if self.training_mode and len(self.episode_rewards) > 0:
            episodes = list(range(1, len(self.episode_rewards) + 1))
            self.axes[1].plot(episodes, self.episode_rewards, 'g-')
            self.axes[1].set_xlabel('Episode')
            self.axes[1].set_ylabel('Total Reward')
            self.axes[1].set_title('Learning Progress')
            self.axes[1].grid(True)
        
        plt.tight_layout()
        plt.pause(0.01)
    
    def _end_episode(self, world):
        """Handle end of an episode (one complete handover)"""
        self.episode_num += 1
        
        # Save episode data
        if len(self.current_episode_rewards) > 0:
            total_reward = sum(self.current_episode_rewards)
            self.episode_rewards.append(total_reward)
            
            # Calculate average alpha
            avg_alpha = sum(self.current_episode_alphas) / len(self.current_episode_alphas)
            self.episode_alphas.append(avg_alpha)
            
            print(f"Episode {self.episode_num} completed - Total reward: {total_reward:.2f}, Avg α: {avg_alpha:.2f}")
            
            # Reset episode data
            self.current_episode_rewards = []
            self.current_episode_alphas = []
            
            # Save the model periodically
            if self.episode_num % 10 == 0 and self.training_mode:
                self.rl_handover.save_model(f"handover_model_ep{self.episode_num}.pth")
        
        # Teleport to start position for next episode
        if self.training_mode:
            spawn_points = world.map.get_spawn_points()
            # Either use fixed spawn or random spawn based on preference
            spawn_point = random.choice(spawn_points)
            world.player.set_transform(spawn_point)
            
            # Reset simulation time (would need to be implemented in the world class)
            # This is a placeholder - in practice you might need to restart the simulation
            # or implement a time reset mechanism
            
    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)
    
def game_loop_with_handover(args):
    """Main game loop with RL-based handover control"""
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
        
        # Create handover controller instead of standard controller
        controller = VehicleControlWithHandover(
            world, 
            args.autopilot,
            handover_start_time=args.handover_start_time,
            handover_duration=args.handover_duration,
            human_style=args.human_style,
            training_mode=args.training_mode
        )

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

def main_with_handover():
    """Parse arguments and start simulation with handover capability"""
    argparser = argparse.ArgumentParser(description='CARLA MPC Control with RL Handover')
    
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
    argparser.add_argument('--spawn_x', metavar='X', default='-14', help='x position to spawn the agent (default: -14)')
    argparser.add_argument('--spawn_y', metavar='Y', default='70', help='y position to spawn the agent (default: 70)')
    argparser.add_argument('--random_spawn', metavar='RS', default=0, type=int, help='Random spawn agent (default: 0)')
    
    # Controller settings
    argparser.add_argument('--waypoint_resolution', metavar='WR', default=0.5, type=float, 
                          help='waypoint resolution for control (default: 0.5)')
    argparser.add_argument('--waypoint_lookahead_distance', metavar='WLD', default=5.0, type=float,
                          help='waypoint look ahead distance for control (default: 5.0)')
    argparser.add_argument('--desired_speed', metavar='SPEED', default=30, type=float,
                          help='desired speed for driving (default: 30)')
    argparser.add_argument('--planning_horizon', metavar='HORIZON', default=10, type=int,
                          help='planning horizon for control (default: 10)')
    argparser.add_argument('--time_step', metavar='TS', default=0.1, type=float,
                          help='time step for control (default: 0.1)')
    argparser.add_argument('--handover_start_time', metavar='HST', default=10.0, type=float,
                          help='start time for handover (default: 10.0)')
    argparser.add_argument('--handover_duration', metavar='HD', default=7.0, type=float,
                          help='duration of handover (default: 7.0)')
    argparser.add_argument('--human_style', metavar='HS', default='normal',
                          help='human driving style (normal, aggressive, cautious)')
    argparser.add_argument('--training_mode', action='store_true', help='enable training mode for RL')
    argparser.add_argument('--save_model', metavar='MODEL', default='handover_model.pth',
                          help='path to save the RL model (default: handover_model.pth)')
    argparser.add_argument('--load_model', metavar='MODEL', default=None,
                          help='path to load the RL model (default: None)')
    argparser.add_argument('--render', action='store_true', help='enable rendering')

    # Debug options
    argparser.add_argument('-v', '--verbose', action='store_true', dest='debug', help='print debug information')
    
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    try:
        game_loop_with_handover(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':
    main_with_handover()