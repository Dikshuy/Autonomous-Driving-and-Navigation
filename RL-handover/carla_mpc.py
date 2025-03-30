import os
import sys
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict, Any

from typing import Tuple, Optional, List
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt


class BicycleMPC:
    """
    Model Predictive Controller for a Bicycle Kinematic Model.
    """

    def __init__(
        self, 
        wheelbase: float = 2.5, 
        velocity: float = 7.0, 
        horizon_length: int = 5, 
        time_step: float = 0.1
    ):
        self.L = wheelbase
        self.v = velocity
        self.N = horizon_length
        self.dt = time_step
        
        self.Q = np.eye(3)
        
        self.R = np.eye(1)

    def create_bicycle_model(self) -> ca.Function:
        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        theta = ca.MX.sym('theta')
        state = ca.vertcat(x, y, theta)
        
        delta = ca.MX.sym('delta')
        control = ca.vertcat(delta)

        x_next = x + self.v * ca.cos(theta) * self.dt
        y_next = y + self.v * ca.sin(theta) * self.dt
        theta_next = theta + (self.v / self.L) * ca.tan(delta) * self.dt
        
        next_state = ca.vertcat(x_next, y_next, theta_next)
        
        return ca.Function('bicycle_model', [state, control], [next_state])

    @staticmethod
    def generate_reference_trajectory(
        t_values: np.ndarray, 
        t0: float = 10.0, 
        alpha: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_values = 1 / (1 + np.exp(-alpha * (t_values - t0)))
        return t_values, y_values

    def cost_function(
        self, 
        state: ca.MX, 
        control: ca.MX, 
        target: ca.MX, 
        control_prev: Optional[ca.MX] = None
    ) -> ca.MX:
        state_cost = ca.mtimes((state[:2] - target).T, self.Q[:2, :2] @ (state[:2] - target))
        
        control_cost = ca.mtimes(control.T, self.R @ control)
        
        if control_prev is not None:
            jerk_penalty = ca.sumsqr(control - control_prev)
        else:
            jerk_penalty = 0
        
        return ca.sum1(state_cost) + ca.sum1(control_cost) + jerk_penalty

    def run_mpc_simulation(self) -> Tuple[List[float], List[float], List[float]]:
        bicycle_model = self.create_bicycle_model()

        t_values = np.linspace(0, 20, 50)
        target_x, target_y = self.generate_reference_trajectory(t_values)

        opti = ca.Opti()
        X = opti.variable(3, self.N + 1)  # State trajectory
        U = opti.variable(1, self.N)      # Control inputs

        opti.set_initial(X, 0)
        opti.set_initial(U, 0)

        target_param = opti.parameter(2, self.N)

        previous_control = np.zeros(1)

        objective = 0

        for k in range(self.N):
            current_state = X[:, k]
            next_state = X[:, k + 1]
            control_input = U[:, k]

            control_previous = previous_control if k == 0 else U[:, k - 1]
            
            target = target_param[:, k]
            
            objective += self.cost_function(current_state, control_input, target, control_previous)

        for k in range(self.N):
            opti.subject_to(X[:, k+1] == bicycle_model(X[:, k], U[:, k]))
        
        opti.subject_to(opti.bounded(-np.pi/4, U[0, :], np.pi/4))

        opti.minimize(objective)
        opti.solver('ipopt')

        x_trajectory: List[float] = []
        y_trajectory: List[float] = []
        steering_angle_all: List[float] = []
        current_state = np.array([0, 0, 0])

        time_steps = np.arange(len(target_x)) * self.dt
        for t in range(len(target_x) - self.N):
            opti.set_initial(X[:, 0], current_state)

            target_segment = np.vstack((target_x[t:t + self.N], target_y[t:t + self.N]))
            opti.set_value(target_param, target_segment)

            solution = opti.solve()

            optimal_u = solution.value(U[:, 0])
            previous_control = optimal_u

            current_state = solution.value(X[:, 1])
            
            x_trajectory.append(current_state[0])
            y_trajectory.append(current_state[1])
            steering_angle_all.append(optimal_u)

        return x_trajectory, y_trajectory, steering_angle_all

    def visualize_results(
        self, 
        x_trajectory: List[float], 
        y_trajectory: List[float], 
        steering_angles: List[float]
    ):
        t_values = np.linspace(0, 20, 50)
        target_x, target_y = self.generate_reference_trajectory(t_values)
        time_steps = np.arange(len(target_x)) * self.dt

        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(x_trajectory, y_trajectory, marker='o', label='Actual Trajectory')
        plt.plot(target_x, target_y, 'r--', label='Desired Trajectory')
        plt.title('Vehicle Trajectory')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(time_steps[:len(steering_angles)], steering_angles, label='Steering Angle (rad)')
        plt.title('Control Inputs over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Steering Angle (rad)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

try:
    carla_path = os.environ.get('CARLA_PATH', '/opt/carla-simulator')
    sys.path.append(glob.glob(f'{carla_path}/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


class CarlaMPCController:
    def __init__(
        self,
        client_host: str = '127.0.0.1',
        client_port: int = 2000,
        fixed_delta_seconds: float = 0.05,
        wheelbase: float = 2.5,
        target_speed: float = 7.0,  # m/s
        horizon_length: int = 10,
        lookahead_distance: float = 15.0  # meters ahead to generate waypoints
    ):
        # Connect to CARLA
        self.client = carla.Client(client_host, client_port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Set fixed delta seconds for deterministic simulation
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = fixed_delta_seconds
        self.world.apply_settings(settings)
        
        # Get map and create MPC controller
        self.map = self.world.get_map()
        self.dt = fixed_delta_seconds
        self.target_speed = target_speed
        self.lookahead_distance = lookahead_distance
        
        # Initialize the MPC controller
        self.mpc = BicycleMPC(
            wheelbase=wheelbase,
            velocity=target_speed,
            horizon_length=horizon_length,
            time_step=fixed_delta_seconds
        )
        
        # Vehicle setup
        self.vehicle = None
        self.spectator = self.world.get_spectator()
        
        # Initialize MPC solver
        self.bicycle_model = self.mpc.create_bicycle_model()
        self.opti = ca.Opti()
        self.X = self.opti.variable(3, self.mpc.N + 1)  # State trajectory
        self.U = self.opti.variable(1, self.mpc.N)      # Control inputs
        self.target_param = self.opti.parameter(2, self.mpc.N)
        
        self.opti.set_initial(self.X, 0)
        self.opti.set_initial(self.U, 0)
        
        self.previous_control = np.zeros(1)
        self.objective = 0
        
        # Setup MPC optimization problem
        for k in range(self.mpc.N):
            current_state = self.X[:, k]
            control_input = self.U[:, k]
            
            control_previous = self.previous_control if k == 0 else self.U[:, k - 1]
            target = self.target_param[:, k]
            
            self.objective += self.mpc.cost_function(current_state, control_input, target, control_previous)
            
            # System dynamics constraint
            self.opti.subject_to(self.X[:, k+1] == self.bicycle_model(self.X[:, k], self.U[:, k]))
        
        # Steering angle constraints
        self.opti.subject_to(self.opti.bounded(-np.pi/4, self.U[0, :], np.pi/4))
        
        # Set objective and solver
        self.opti.minimize(self.objective)
        self.opti.solver('ipopt', {'print_level': 0})  # Suppress solver output
        
        # For data logging
        self.vehicle_locations = []
        self.waypoints_history = []
        self.steering_commands = []
        
    def spawn_vehicle(self):
        """Spawn vehicle at a suitable starting location on the map."""
        # Get available spawn points
        spawn_points = self.map.get_spawn_points()
        
        # Choose a spawn point
        spawn_point = spawn_points[0]
        
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
    
    def generate_waypoints(self, current_location) -> List[carla.Location]:
        """Generate waypoints along the center of the lane ahead of the vehicle."""
        waypoints = []
        
        # Find the waypoint closest to the vehicle
        vehicle_waypoint = self.map.get_waypoint(current_location, project_to_road=True)
        
        # Generate waypoints along the center of the lane
        current_waypoint = vehicle_waypoint
        distance = 0.0
        step_size = 1.0  # Distance between waypoints
        
        while distance < self.lookahead_distance:
            # Get next waypoint along the lane
            next_waypoints = current_waypoint.next(step_size)
            
            if not next_waypoints:
                break
                
            next_waypoint = next_waypoints[0]
            waypoints.append(next_waypoint.transform.location)
            
            distance += step_size
            current_waypoint = next_waypoint
        
        return waypoints
    
    def waypoints_to_trajectory(self, waypoints, vehicle_state) -> Tuple[np.ndarray, np.ndarray]:
        """Convert waypoints to a trajectory in the MPC's coordinate system."""
        # Extract x and y coordinates from waypoints
        x_coords = np.array([wp.x for wp in waypoints])
        y_coords = np.array([wp.y for wp in waypoints])
        
        return x_coords, y_coords
    
    def apply_control(self, steering_angle):
        """Apply control to the vehicle based on MPC output."""
        # Map steering angle from radians to CARLA's [-1, 1] range
        # CARLA's steering is opposite to the MPC's definition
        carla_steering = -steering_angle / (np.pi/4)
        
        # Create a carla.VehicleControl object
        control = carla.VehicleControl()
        control.steer = float(np.clip(carla_steering, -1.0, 1.0))
        control.throttle = 0.7  # Fixed throttle to maintain target speed
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False
        
        # Apply the control to the vehicle
        self.vehicle.apply_control(control)
        
        # Log steering command
        self.steering_commands.append(steering_angle)
        
        return control
    
    def run_step(self) -> carla.VehicleControl:
        """Run a single step of the MPC controller."""
        # Get current vehicle state
        current_state = self.get_vehicle_state()
        
        # Generate waypoints for the reference trajectory
        current_location = carla.Location(x=current_state[0], y=current_state[1], z=0)
        waypoints = self.generate_waypoints(current_location)
        
        if not waypoints:
            print("No waypoints found, stopping vehicle")
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
        
        # Convert waypoints to trajectory
        target_x, target_y = self.waypoints_to_trajectory(waypoints, current_state)
        
        # Ensure we have enough waypoints for the horizon
        if len(target_x) < self.mpc.N:
            # Duplicate the last waypoint if we don't have enough
            last_x, last_y = target_x[-1], target_y[-1]
            target_x = np.append(target_x, [last_x] * (self.mpc.N - len(target_x)))
            target_y = np.append(target_y, [last_y] * (self.mpc.N - len(target_y)))
        
        # Prepare target for MPC
        target_segment = np.vstack((target_x[:self.mpc.N], target_y[:self.mpc.N]))
        
        # Update MPC with current state and target trajectory
        self.opti.set_initial(self.X[:, 0], current_state)
        self.opti.set_value(self.target_param, target_segment)
        
        try:
            # Solve MPC optimization problem
            solution = self.opti.solve()
            
            # Extract optimal control input
            optimal_u = solution.value(self.U[:, 0])
            self.previous_control = optimal_u
            
            # Log vehicle position and waypoints
            self.vehicle_locations.append((current_state[0], current_state[1]))
            self.waypoints_history.append([(wp.x, wp.y) for wp in waypoints])
            
            # Apply control to the vehicle
            return self.apply_control(float(optimal_u))
            
        except RuntimeError as e:
            print(f"MPC solver error: {e}")
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
    
    def visualize_results(self):
        """Visualize the results of the MPC controller."""
        plt.figure(figsize=(12, 10))
        
        # Plot vehicle trajectory and waypoints
        plt.subplot(2, 1, 1)
        vehicle_x, vehicle_y = zip(*self.vehicle_locations) if self.vehicle_locations else ([], [])
        plt.plot(vehicle_x, vehicle_y, 'b-', marker='o', label='Vehicle Trajectory')
        
        # Plot all waypoints (flattened list)
        all_waypoints = [wp for waypoints in self.waypoints_history for wp in waypoints]
        if all_waypoints:
            waypoint_x, waypoint_y = zip(*all_waypoints)
            plt.scatter(waypoint_x, waypoint_y, c='r', s=5, label='Waypoints')
        
        plt.title('Vehicle Trajectory and Waypoints')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.legend()
        plt.grid(True)
        
        # Plot steering angle commands
        plt.subplot(2, 1, 2)
        time_steps = np.arange(len(self.steering_commands)) * self.dt
        plt.plot(time_steps, self.steering_commands, label='Steering Angle (rad)')
        plt.title('Steering Commands over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Steering Angle (rad)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def run_simulation(self, simulation_time=60.0):
        """Run the simulation for a specified time in seconds."""
        # Spawn the vehicle
        self.spawn_vehicle()
        
        # Reset tracking variables
        self.vehicle_locations = []
        self.waypoints_history = []
        self.steering_commands = []
        
        try:
            start_time = time.time()
            while time.time() - start_time < simulation_time:
                # Update spectator position to follow the vehicle
                vehicle_transform = self.vehicle.get_transform()
                spectator_transform = carla.Transform(
                    vehicle_transform.location + carla.Location(z=5, x=-10),
                    carla.Rotation(pitch=-15)
                )
                self.spectator.set_transform(spectator_transform)
                
                # Run MPC control step
                self.run_step()
                
                # Tick the world
                self.world.tick()
                
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
        """Clean up resources."""
        if self.vehicle is not None:
            self.vehicle.destroy()
            
        # Reset world settings to original
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)


def main():
    try:
        # Create controller
        controller = CarlaMPCController(
            target_speed=7.0,       # m/s (about 25 km/h)
            horizon_length=10,      # prediction horizon
            lookahead_distance=20.0 # meters of waypoints to generate ahead
        )
        
        # Run simulation
        controller.run_simulation(simulation_time=60.0)
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'controller' in locals():
            controller.cleanup()


if __name__ == "__main__":
    main()