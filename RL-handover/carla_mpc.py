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
        horizon_length: int = 15,  # Increased horizon for better prediction
        time_step: float = 0.1
    ):
        self.L = wheelbase
        self.v = velocity
        self.N = horizon_length
        self.dt = time_step
        
        # State cost matrix (position tracking) - Adjusted weights
        self.Q = np.diag([20.0, 20.0, 5.0])  # Higher weights for x,y tracking, moderate for heading
        
        # Control cost matrix (steering angle) - Increased to penalize aggressive steering
        self.R = np.diag([5.0])
        
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
            
            # Reference tracking cost - consider heading alignment with path
            target_x = self.target_param[0, k]
            target_y = self.target_param[1, k]
            
            # Estimate target heading if not the last point
            if k < self.N - 1:
                next_x = self.target_param[0, k + 1]
                next_y = self.target_param[1, k + 1]
                dx = next_x - target_x
                dy = next_y - target_y
                target_heading = ca.atan2(dy, dx)
            else:
                # For the last point, use the same heading as previous
                target_heading = 0
                
            target = ca.vertcat(target_x, target_y, target_heading)
            tracking_error = current_state - target
            state_cost = ca.mtimes(tracking_error.T, self.Q @ tracking_error)
            
            # Control cost
            control_cost = ca.mtimes(control_input.T, self.R @ control_input)
            
            # Add to objective
            objective += state_cost + control_cost
            
            # Add rate of change penalty for smoother control
            if k < self.N - 1:
                next_control = self.U[:, k+1]
                smoothness_cost = 15.0 * ca.sumsqr(next_control - control_input)  # Increased weight for smoother steering
                objective += smoothness_cost
            
            # System dynamics constraint
            self.opti.subject_to(self.X[:, k+1] == self.bicycle_model(self.X[:, k], self.U[:, k]))
        
        # Steering angle constraints - reduced slightly for more stable control
        self.opti.subject_to(self.opti.bounded(-np.pi/5, self.U[0, :], np.pi/5))
        
        # Set objective and solver
        self.opti.minimize(objective)
        
        # Set solver options - increased max_iter for better convergence
        opts = {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0,
            'ipopt.max_iter': 100,  # Increased max iterations
            'ipopt.tol': 1e-4       # Tighter tolerance
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
            Optimal steering angle and predicted trajectory
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
            # Return previous steering value if available or a small value
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
        target_speed: float = 5.0,  # Reduced speed for more stability
        horizon_length: int = 15,   # Increased horizon
        lookahead_distance: float = 25.0,  # Extended lookahead for better path planning
        debug_mode: bool = True
    ):
        # Connect to CARLA
        self.client = carla.Client(client_host, client_port)
        self.client.set_timeout(10.0)
        
        # Load Town04 specifically
        self.world = self.client.load_world('Town04')
        print("Loaded Town04 map")
        
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
        
        # For error recovery
        self.last_steering_angle = 0.0
        self.error_counter = 0
        self.max_errors = 5
        
        # For speed PID control
        self.speed_error_sum = 0.0
        self.last_speed_error = 0.0
        self.kp_speed = 0.5
        self.ki_speed = 0.1
        self.kd_speed = 0.1
        
    def find_good_spawn_points(self, num_points=10):
        """Find good spawn points in Town04 that are on straight roads."""
        spawn_points = self.map.get_spawn_points()
        good_spawn_points = []
        
        for i, spawn_point in enumerate(spawn_points):
            # Get waypoint at spawn location
            waypoint = self.map.get_waypoint(spawn_point.location)
            
            # Check if the waypoint is on a straight road segment
            if waypoint.lane_type == carla.LaneType.Driving:
                # Get some waypoints ahead to check road straightness
                next_waypoints = []
                current = waypoint
                for _ in range(5):
                    nexts = current.next(5.0)
                    if not nexts:
                        break
                    next_waypoints.append(nexts[0])
                    current = nexts[0]
                
                # Check if road is relatively straight by comparing headings
                if len(next_waypoints) >= 3:
                    headings = [wp.transform.rotation.yaw for wp in next_waypoints]
                    max_heading_diff = max([abs(headings[i] - headings[i+1]) for i in range(len(headings)-1)])
                    
                    # If the heading doesn't change much, it's a good point
                    if max_heading_diff < 5.0:
                        good_spawn_points.append((i, spawn_point))
                
                if len(good_spawn_points) >= num_points:
                    break
        
        # Print good spawn points for reference
        print(f"Found {len(good_spawn_points)} good spawn points in Town04:")
        for i, (idx, _) in enumerate(good_spawn_points):
            print(f"  Good spawn point {i}: index {idx}")
        
        return good_spawn_points
        
    def spawn_vehicle(self, spawn_point_idx=None):
        """Spawn vehicle at a suitable starting location on the map."""
        # Get available spawn points
        spawn_points = self.map.get_spawn_points()
        
        # Find good spawn points if no specific index provided
        if spawn_point_idx is None:
            good_points = self.find_good_spawn_points()
            if good_points:
                spawn_point_idx = good_points[0][0]
                print(f"Using automatically selected good spawn point at index {spawn_point_idx}")
            else:
                spawn_point_idx = 0
        
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
    
    def get_current_speed(self):
        """Get the current speed of the vehicle in m/s."""
        velocity = self.vehicle.get_velocity()
        return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    
    def compute_speed_control(self, current_speed):
        """Compute throttle and brake using PID control."""
        # Calculate speed error
        speed_error = self.target_speed - current_speed
        
        # PID control
        p_term = self.kp_speed * speed_error
        self.speed_error_sum += speed_error * self.dt
        i_term = self.ki_speed * self.speed_error_sum
        d_term = self.kd_speed * (speed_error - self.last_speed_error) / self.dt
        self.last_speed_error = speed_error
        
        # Calculate throttle from PID
        throttle_raw = p_term + i_term + d_term
        
        # Determine throttle and brake based on control signal
        if throttle_raw > 0:
            throttle = min(max(throttle_raw, 0.0), 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(max(-throttle_raw * 0.5, 0.0), 1.0)  # Scale brake to be less aggressive
        
        return throttle, brake
    
    def apply_control(self, steering_angle):
        """Apply control to the vehicle based on MPC output."""
        # Add steering smoothing to prevent jerky movements
        alpha = 0.3  # Smoothing factor (adjust as needed)
        smoothed_steering = alpha * steering_angle + (1 - alpha) * self.last_steering_angle
        self.last_steering_angle = smoothed_steering
        
        # Map steering angle from radians to CARLA's [-1, 1] range
        # CARLA's steering is opposite to the MPC's definition
        carla_steering = -smoothed_steering / (np.pi/4)
        
        # Get current speed and compute speed controls
        current_speed = self.get_current_speed()
        throttle, brake = self.compute_speed_control(current_speed)
        
        # Create a carla.VehicleControl object
        control = carla.VehicleControl()
        control.steer = float(np.clip(carla_steering, -1.0, 1.0))
        control.throttle = throttle
        control.brake = brake
        control.hand_brake = False
        control.manual_gear_shift = False
        
        # Apply the control to the vehicle
        self.vehicle.apply_control(control)
        
        # Log steering command
        self.steering_commands.append(smoothed_steering)
        
        return control
    
    def run_step(self):
        """Run a single step of the MPC controller."""
        try:
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
            
            # If solver failed several times in a row, emergency stop
            if np.all(predicted_traj == 0):
                self.error_counter += 1
                if self.error_counter > self.max_errors:
                    print("Too many solver errors, emergency stopping")
                    return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
                # Use last steering angle as fallback
                steering_angle = self.last_steering_angle
            else:
                self.error_counter = 0  # Reset error counter on success
            
            # Visualize predicted trajectory in CARLA
            self.visualize_predicted_trajectory(predicted_traj)
            
            # Log data for later visualization
            self.vehicle_locations.append((current_state[0], current_state[1], current_state[2]))
            self.waypoints_history.append([(wp.transform.location.x, wp.transform.location.y) for wp in waypoints])
            self.predicted_trajectories.append(predicted_traj[:2, :].T if predicted_traj.size > 0 else [])
            
            # Apply control to the vehicle
            return self.apply_control(steering_angle)
            
        except Exception as e:
            print(f"Error in run_step: {e}")
            # Emergency stop on unexpected errors
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
    
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
    
    def run_simulation(self, simulation_time=60.0, spawn_point_idx=None):
        """Run the simulation for a specified time in seconds."""
        # Spawn the vehicle
        self.spawn_vehicle(spawn_point_idx)
        
        # Reset tracking variables
        self.vehicle_locations = []
        self.waypoints_history = []
        self.steering_commands = []
        self.predicted_trajectories = []
        self.speed_error_sum = 0.0
        self.last_speed_error = 0.0
        self.error_counter = 0
        
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
                
                # Print current speed occasionally
                if frame_count % 20 == 0:
                    current_speed = self.get_current_speed()
                    print(f"Current speed: {current_speed:.2f} m/s, Target: {self.target_speed:.2f} m/s")
                
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
            target_speed=5.0,         # m/s (about 18 km/h) - Reduced for stability
            horizon_length=15,        # Longer prediction horizon
            lookahead_distance=25.0,  # Extended lookahead
            debug_mode=True           # Enable CARLA debug visualization
        )
        
        # Run simulation (spawn point will be automatically selected)
        controller.run_simulation(simulation_time=60.0)
        
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