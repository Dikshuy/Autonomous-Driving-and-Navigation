import os
import sys
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict, Any
import math

# Import CasADi
import casadi as ca

# Try to import CARLA
try:
    carla_path = os.environ.get('CARLA_PATH', '/opt/carla-simulator')
    sys.path.append(glob.glob(f'{carla_path}/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


class BicycleMPCCarla:
    """
    Model Predictive Controller for a Bicycle Kinematic Model adapted for CARLA.
    """

    def __init__(
        self, 
        wheelbase: float = 2.5, 
        velocity: float = 7.0, 
        horizon_length: int = 10, 
        time_step: float = 0.1
    ):
        self.L = wheelbase
        self.v = velocity
        self.N = horizon_length
        self.dt = time_step
        
        # State cost matrix (position tracking)
        self.Q = np.diag([10.0, 10.0, 1.0])  # Higher weights for x,y tracking
        
        # Control cost matrix (steering angle)
        self.R = np.diag([1.0])
        
        # Setup the optimization problem
        self.setup_optimization()
    
    def setup_optimization(self):
        """Setup the MPC optimization problem."""
        self.bicycle_model = self.create_bicycle_model()
        
        # Initialize optimizer
        self.opti = ca.Opti()
        self.X = self.opti.variable(3, self.N + 1)  # State trajectory: [x, y, theta]
        self.U = self.opti.variable(1, self.N)      # Control inputs: [delta]
        
        # Parameters for the reference trajectory
        self.target_param = self.opti.parameter(2, self.N)  # Reference points [x, y]
        self.current_state_param = self.opti.parameter(3)   # Current state [x, y, theta]
        
        # Set initial values
        self.opti.set_initial(self.X, 0)
        self.opti.set_initial(self.U, 0)
        
        # Objective function
        objective = 0
        
        # Initial state constraint
        self.opti.subject_to(self.X[:, 0] == self.current_state_param)
        
        for k in range(self.N):
            # Extract current state and control
            current_state = self.X[:, k]
            control_input = self.U[:, k]
            
            # Reference tracking cost
            target = ca.vertcat(self.target_param[0, k], self.target_param[1, k], 0)  # Assuming reference has no theta
            tracking_error = current_state - target
            state_cost = ca.mtimes(tracking_error.T, self.Q @ tracking_error)
            
            # Control cost
            control_cost = ca.mtimes(control_input.T, self.R @ control_input)
            
            # Add to objective
            objective += state_cost + control_cost
            
            # If not the last step, add smoothness cost
            if k < self.N - 1:
                next_control = self.U[:, k+1]
                smoothness_cost = 10.0 * ca.sumsqr(next_control - control_input)  # Weight for smoother control
                objective += smoothness_cost
            
            # System dynamics constraint
            self.opti.subject_to(self.X[:, k+1] == self.bicycle_model(self.X[:, k], self.U[:, k]))
        
        # Steering angle constraints
        self.opti.subject_to(self.opti.bounded(-np.pi/4, self.U[0, :], np.pi/4))
        
        # Set objective and solver
        self.opti.minimize(objective)
        
        # Set solver options
        opts = {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0
        }
        self.opti.solver('ipopt', opts)

    def create_bicycle_model(self) -> ca.Function:
        """Create the bicycle kinematic model."""
        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        theta = ca.MX.sym('theta')
        state = ca.vertcat(x, y, theta)
        
        delta = ca.MX.sym('delta')  # Steering angle
        control = ca.vertcat(delta)
        
        # Bicycle model kinematics
        x_next = x + self.v * ca.cos(theta) * self.dt
        y_next = y + self.v * ca.sin(theta) * self.dt
        theta_next = theta + (self.v / self.L) * ca.tan(delta) * self.dt
        
        next_state = ca.vertcat(x_next, y_next, theta_next)
        
        return ca.Function('bicycle_model', [state, control], [next_state])
    
    def solve(self, current_state, reference_trajectory):
        """
        Solve the MPC optimization problem.
        
        Args:
            current_state: Current vehicle state [x, y, theta]
            reference_trajectory: Reference trajectory points, shape (2, N) for [x, y]
        
        Returns:
            Optimal steering angle
        """
        # Set current state and reference trajectory
        self.opti.set_value(self.current_state_param, current_state)
        self.opti.set_value(self.target_param, reference_trajectory)
        
        try:
            # Solve the optimization problem
            solution = self.opti.solve()
            
            # Extract optimal control input (first steering angle)
            optimal_steering = float(solution.value(self.U[0, 0]))
            
            # Get predicted trajectory for visualization
            predicted_trajectory = solution.value(self.X)
            
            return optimal_steering, predicted_trajectory
            
        except Exception as e:
            print(f"MPC solver error: {e}")
            return 0.0, np.zeros((3, self.N + 1))


class CarlaMPCLaneFollower:
    """
    Controller for lane following in CARLA using MPC.
    """
    
    def __init__(
        self,
        client_host: str = '127.0.0.1',
        client_port: int = 2000,
        fixed_delta_seconds: float = 0.05,
        wheelbase: float = 2.5,
        target_speed: float = 7.0,  # m/s
        horizon_length: int = 10,
        lookahead_distance: float = 20.0,  # meters ahead to generate waypoints
        debug_mode: bool = True
    ):
        # Connect to CARLA
        self.client = carla.Client(client_host, client_port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Set fixed delta seconds for deterministic simulation
        settings = self.world.get_settings()
        self.original_settings = settings
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = fixed_delta_seconds
        self.world.apply_settings(settings)
        
        # Get map and debug helpers
        self.map = self.world.get_map()
        self.debug = self.world.debug if debug_mode else None
        self.debug_mode = debug_mode
        
        # Controller parameters
        self.dt = fixed_delta_seconds
        self.target_speed = target_speed
        self.lookahead_distance = lookahead_distance
        self.wheelbase = wheelbase
        
        # Initialize the MPC controller
        self.mpc = BicycleMPCCarla(
            wheelbase=wheelbase,
            velocity=target_speed,
            horizon_length=horizon_length,
            time_step=fixed_delta_seconds
        )
        
        # Vehicle setup
        self.vehicle = None
        self.spectator = self.world.get_spectator()
        
        # For data logging
        self.vehicle_locations = []
        self.waypoints_history = []
        self.steering_commands = []
        self.predicted_trajectories = []
        
    def spawn_vehicle(self, spawn_point_idx=0):
        """Spawn vehicle at a suitable starting location on the map."""
        # Get available spawn points
        spawn_points = self.map.get_spawn_points()
        
        # Choose a spawn point
        spawn_point = spawn_points[spawn_point_idx]
        
        # Spawn the vehicle
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        
        if self.vehicle is not None:
            self.vehicle.destroy()
            
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Vehicle spawned at {spawn_point.location}")
        
        # Position spectator behind the vehicle
        spectator_transform = carla.Transform(
            spawn_point.location + carla.Location(x=-10, z=5),
            carla.Rotation(pitch=-15)
        )
        self.spectator.set_transform(spectator_transform)
        
        # Wait for the world to be ready
        self.world.tick()
        
        return spawn_point
    
    def get_vehicle_state(self) -> np.ndarray:
        """Get current vehicle state in the form [x, y, theta]."""
        vehicle_transform = self.vehicle.get_transform()
        location = vehicle_transform.location
        rotation = vehicle_transform.rotation
        
        # Convert from CARLA's coordinate system to our MPC coordinate system
        x = location.x
        y = location.y
        theta = np.radians(rotation.yaw)
        
        return np.array([x, y, theta])
    
    def generate_lane_waypoints(self, current_location) -> List[carla.Waypoint]:
        """Generate waypoints along the center of the lane ahead of the vehicle."""
        # Find the waypoint closest to the vehicle
        vehicle_waypoint = self.map.get_waypoint(current_location, project_to_road=True)
        
        # Generate waypoints along the center of the lane
        waypoints = [vehicle_waypoint]
        distance = 0.0
        step_size = 1.0  # Distance between waypoints
        
        current_waypoint = vehicle_waypoint
        while distance < self.lookahead_distance:
            # Get next waypoint along the lane
            next_waypoints = current_waypoint.next(step_size)
            
            if not next_waypoints:
                break
                
            next_waypoint = next_waypoints[0]
            waypoints.append(next_waypoint)
            
            distance += step_size
            current_waypoint = next_waypoint
        
        return waypoints
    
    def convert_waypoints_to_array(self, waypoints) -> np.ndarray:
        """Convert CARLA waypoints to a numpy array for the MPC."""
        x_coords = np.array([wp.transform.location.x for wp in waypoints])
        y_coords = np.array([wp.transform.location.y for wp in waypoints])
        
        return np.vstack((x_coords, y_coords))
    
    def visualize_waypoints(self, waypoints):
        """Visualize waypoints in the CARLA world."""
        if not self.debug_mode:
            return
            
        # Draw waypoints as small spheres
        for wp in waypoints:
            loc = wp.transform.location
            self.debug.draw_point(
                loc + carla.Location(z=0.2),
                size=0.1,
                color=carla.Color(r=0, g=255, b=0),
                life_time=0.1
            )
    
    def visualize_predicted_trajectory(self, predicted_traj):
        """Visualize the predicted vehicle trajectory in the CARLA world."""
        if not self.debug_mode or predicted_traj.size == 0:
            return
            
        # Draw predicted trajectory as red line
        for i in range(1, predicted_traj.shape[1]):
            start = carla.Location(x=predicted_traj[0, i-1], y=predicted_traj[1, i-1], z=0.5)
            end = carla.Location(x=predicted_traj[0, i], y=predicted_traj[1, i], z=0.5)
            self.debug.draw_line(
                start,
                end,
                thickness=0.1,
                color=carla.Color(r=255, g=0, b=0),
                life_time=0.1
            )
    
    def apply_control(self, steering_angle):
        """Apply control to the vehicle based on MPC output."""
        # Map steering angle from radians to CARLA's [-1, 1] range
        # CARLA's steering is opposite to the MPC's definition
        carla_steering = -steering_angle / (np.pi/4)
        
        # Get current velocity
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # Adaptive throttle based on current speed
        throttle = 0.7
        if speed > self.target_speed:
            throttle = 0.0
        
        # Create a carla.VehicleControl object
        control = carla.VehicleControl()
        control.steer = float(np.clip(carla_steering, -1.0, 1.0))
        control.throttle = throttle
        control.brake = 0.0 if speed <= self.target_speed * 1.1 else 0.5
        control.hand_brake = False
        control.manual_gear_shift = False
        
        # Apply the control to the vehicle
        self.vehicle.apply_control(control)
        
        # Log steering command
        self.steering_commands.append(steering_angle)
        
        return control
    
    def run_step(self):
        """Run a single step of the MPC controller."""
        # Get current vehicle state
        current_state = self.get_vehicle_state()
        
        # Generate waypoints for the reference trajectory
        current_location = carla.Location(x=current_state[0], y=current_state[1], z=0)
        waypoints = self.generate_lane_waypoints(current_location)
        
        if not waypoints:
            print("No waypoints found, stopping vehicle")
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
        
        # Visualize waypoints in CARLA
        self.visualize_waypoints(waypoints)
        
        # Convert waypoints to reference trajectory
        reference_trajectory = self.convert_waypoints_to_array(waypoints)
        
        # Ensure we have enough waypoints for the MPC horizon
        if reference_trajectory.shape[1] < self.mpc.N:
            # Duplicate the last waypoint if we don't have enough
            last_point = reference_trajectory[:, -1:]
            missing = self.mpc.N - reference_trajectory.shape[1]
            reference_trajectory = np.hstack([reference_trajectory, np.tile(last_point, (1, missing))])
        else:
            # Trim if we have too many
            reference_trajectory = reference_trajectory[:, :self.mpc.N]
        
        # Solve MPC problem
        steering_angle, predicted_traj = self.mpc.solve(current_state, reference_trajectory)
        
        # Visualize predicted trajectory in CARLA
        self.visualize_predicted_trajectory(predicted_traj)
        
        # Log data for later visualization
        self.vehicle_locations.append((current_state[0], current_state[1], current_state[2]))
        self.waypoints_history.append([(wp.transform.location.x, wp.transform.location.y) for wp in waypoints])
        self.predicted_trajectories.append(predicted_traj[:2, :].T)  # Store only x,y
        
        # Apply control to the vehicle
        return self.apply_control(steering_angle)
    
    def visualize_results(self):
        """Visualize the results of the MPC controller."""
        if not self.vehicle_locations:
            print("No data to visualize")
            return
            
        plt.figure(figsize=(15, 12))
        
        # Plot vehicle trajectory and waypoints
        plt.subplot(3, 1, 1)
        vehicle_x, vehicle_y, vehicle_theta = zip(*self.vehicle_locations)
        plt.plot(vehicle_x, vehicle_y, 'b-', marker='.', label='Vehicle Trajectory')
        
        # Plot waypoints (only every 5th time step to avoid cluttering)
        sample_rate = 5
        for i, waypoints in enumerate(self.waypoints_history[::sample_rate]):
            if waypoints:
                waypoint_x, waypoint_y = zip(*waypoints)
                if i == 0:  # Only add label for the first one
                    plt.plot(waypoint_x, waypoint_y, 'g.', markersize=2, label='Lane Waypoints')
                else:
                    plt.plot(waypoint_x, waypoint_y, 'g.', markersize=2)
        
        # Plot selected predicted trajectories
        sample_indices = np.linspace(0, len(self.predicted_trajectories)-1, 10, dtype=int)
        for i in sample_indices:
            if i < len(self.predicted_trajectories) and len(self.predicted_trajectories[i]) > 0:
                pred_traj = self.predicted_trajectories[i]
                if i == sample_indices[0]:  # Only add label for the first one
                    plt.plot(pred_traj[:, 0], pred_traj[:, 1], 'r-', alpha=0.5, linewidth=1, label='Predicted Trajectories')
                else:
                    plt.plot(pred_traj[:, 0], pred_traj[:, 1], 'r-', alpha=0.5, linewidth=1)
        
        plt.title('Vehicle Trajectory, Lane Waypoints, and Predicted Trajectories')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        # Plot heading angle
        plt.subplot(3, 1, 2)
        time_steps = np.arange(len(vehicle_theta)) * self.dt
        # Convert radians to degrees for better readability
        plt.plot(time_steps, np.degrees(vehicle_theta), label='Vehicle Heading (degrees)')
        plt.title('Vehicle Heading over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Heading Angle (degrees)')
        plt.grid(True)
        plt.legend()
        
        # Plot steering angle commands
        plt.subplot(3, 1, 3)
        time_steps = np.arange(len(self.steering_commands)) * self.dt
        plt.plot(time_steps, self.steering_commands, label='Steering Angle (rad)')
        # Add lines at the steering limits
        plt.axhline(y=np.pi/4, color='r', linestyle='--', alpha=0.5, label='Steering Limits')
        plt.axhline(y=-np.pi/4, color='r', linestyle='--', alpha=0.5)
        plt.title('Steering Commands over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Steering Angle (rad)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('carla_mpc_results.png')
        print("Results saved to carla_mpc_results.png")
        plt.show()
    
    def run_simulation(self, simulation_time=60.0, spawn_point_idx=0):
        """Run the simulation for a specified time in seconds."""
        # Spawn the vehicle
        self.spawn_vehicle(spawn_point_idx)
        
        # Reset tracking variables
        self.vehicle_locations = []
        self.waypoints_history = []
        self.steering_commands = []
        self.predicted_trajectories = []
        
        try:
            start_time = time.time()
            frame_count = 0
            
            while time.time() - start_time < simulation_time:
                # Update spectator to follow the vehicle
                if frame_count % 10 == 0:  # Update spectator every 10 frames for performance
                    vehicle_transform = self.vehicle.get_transform()
                    specX = -10 * np.cos(np.radians(vehicle_transform.rotation.yaw))
                    specY = -10 * np.sin(np.radians(vehicle_transform.rotation.yaw))
                    spectator_transform = carla.Transform(
                        vehicle_transform.location + carla.Location(x=specX, y=specY, z=5),
                        carla.Rotation(pitch=-15, yaw=vehicle_transform.rotation.yaw)
                    )
                    self.spectator.set_transform(spectator_transform)
                
                # Run MPC control step
                self.run_step()
                
                # Tick the world
                self.world.tick()
                frame_count += 1
                
            print(f"Simulation completed after {simulation_time} seconds")
            
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        finally:
            if self.vehicle is not None:
                self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                self.world.tick()
                
            # Visualize results
            self.visualize_results()
    
    def cleanup(self):
        """Clean up resources and restore original world settings."""
        if self.vehicle is not None:
            self.vehicle.destroy()
        
        # Restore original settings
        self.world.apply_settings(self.original_settings)
        print("Restored original world settings")


def main():
    try:
        # Create controller with visualization enabled
        controller = CarlaMPCLaneFollower(
            target_speed=7.0,         # m/s (about 25 km/h)
            horizon_length=10,        # prediction horizon
            lookahead_distance=20.0,  # meters of waypoints to generate ahead
            debug_mode=True           # Enable CARLA debug visualization
        )
        
        # Run simulation (optionally specify a spawn point index)
        controller.run_simulation(simulation_time=60.0, spawn_point_idx=0)
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'controller' in locals():
            controller.cleanup()


if __name__ == "__main__":
    main()