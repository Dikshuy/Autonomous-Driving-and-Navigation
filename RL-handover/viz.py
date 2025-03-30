import carla
import numpy as np
import casadi as ca
import random
from typing import Tuple, Optional, List
from collections import deque

class BicycleMPC:
    """
    Model Predictive Controller for a Bicycle Kinematic Model integrated with CARLA.
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
        
        # Tuning parameters
        self.Q = np.diag([10.0, 10.0, 1.0])  # State cost weights (x, y, theta)
        self.R = np.diag([0.1])               # Control cost weight (steering)
        
        # CARLA specific attributes
        self.client = None
        self.world = None
        self.vehicle = None
        self.spectator = None
        self.reference_waypoints = []
        self.current_waypoint_index = 0
        
        # Visualization
        self.waypoint_visuals = []
        self.trajectory_visuals = []
        self.debug_helper = None

    def create_bicycle_model(self) -> ca.Function:
        """Create the bicycle kinematic model function."""
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

    def cost_function(
        self, 
        state: ca.MX, 
        control: ca.MX, 
        target: ca.MX, 
        control_prev: Optional[ca.MX] = None
    ) -> ca.MX:
        """Calculate the cost for MPC optimization."""
        state_cost = ca.mtimes((state - target).T, self.Q @ (state - target))
        control_cost = ca.mtimes(control.T, self.R @ control)
        
        if control_prev is not None:
            jerk_penalty = 0.1 * ca.sumsqr(control - control_prev)
        else:
            jerk_penalty = 0
        
        return state_cost + control_cost + jerk_penalty

    def setup_carla_simulation(self, town: str = 'Town04'):
        """Connect to CARLA and set up the simulation environment."""
        try:
            # Connect to CARLA server
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)
            
            # Load the specified town
            self.world = self.client.load_world(town)
            
            # Set up debug helper
            self.debug_helper = self.world.debug
            
            # Set synchronous mode for deterministic simulation
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.dt
            self.world.apply_settings(settings)
            
            # Spawn a vehicle at any available spawn point
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
            
            # Get all spawn points and select one at random
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise RuntimeError("No spawn points available in the map")
                
            spawn_point = random.choice(spawn_points)
            print(f"Spawning vehicle at: {spawn_point.location}")
            
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            
            # Set spectator view
            self.spectator = self.world.get_spectator()
            transform = carla.Transform(
                self.vehicle.get_transform().location + carla.Location(z=50),
                carla.Rotation(pitch=-90))
            self.spectator.set_transform(transform)
            
            # Generate reference trajectory (straight line from spawn point)
            self.generate_reference_waypoints()
            
            print("CARLA simulation setup complete")
            
        except Exception as e:
            print(f"Error setting up CARLA simulation: {e}")
            raise

    def generate_reference_waypoints(self, distance: float = 5.0, num_points: int = 100):
        """Generate reference waypoints in a straight line from the vehicle's starting position."""
        map = self.world.get_map()
        current_location = self.vehicle.get_location()
        current_waypoint = map.get_waypoint(current_location)
        
        self.reference_waypoints = []
        
        # Get vehicle's forward vector
        transform = self.vehicle.get_transform()
        forward_vector = transform.get_forward_vector()
        
        for i in range(num_points):
            # Calculate next point in a straight line
            next_location = carla.Location(
                x=current_location.x + forward_vector.x * distance * i,
                y=current_location.y + forward_vector.y * distance * i,
                z=current_location.z)
            
            # Get the nearest waypoint on the road
            waypoint = map.get_waypoint(next_location)
            if waypoint is None:
                # If no road waypoint found, just use the straight line point
                waypoint = current_waypoint
                waypoint.transform.location = next_location
            
            self.reference_waypoints.append(waypoint)
            
            # Visualize waypoints
            self.world.debug.draw_string(
                waypoint.transform.location, 
                '•', 
                color=carla.Color(255, 0, 0), 
                life_time=100.0)
    
    def get_reference_trajectory(self, current_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get the reference trajectory segment for MPC."""
        x_ref = []
        y_ref = []
        
        for i in range(self.N):
            idx = min(current_index + i, len(self.reference_waypoints) - 1)
            waypoint = self.reference_waypoints[idx]
            x_ref.append(waypoint.transform.location.x)
            y_ref.append(waypoint.transform.location.y)
            
        return np.array(x_ref), np.array(y_ref)
    
    def get_vehicle_state(self) -> np.ndarray:
        """Get the current vehicle state (x, y, theta)."""
        transform = self.vehicle.get_transform()
        x = transform.location.x
        y = transform.location.y
        theta = np.radians(transform.rotation.yaw)
        return np.array([x, y, theta])
    
    def run_mpc_control(self):
        """Run the MPC controller in the CARLA simulation."""
        if not self.vehicle or not self.reference_waypoints:
            raise RuntimeError("CARLA simulation not properly set up")
            
        bicycle_model = self.create_bicycle_model()
        
        # Initialize MPC optimization problem
        opti = ca.Opti()
        X = opti.variable(3, self.N + 1)  # State trajectory
        U = opti.variable(1, self.N)       # Control inputs (steering)
        
        opti.set_initial(X, 0)
        opti.set_initial(U, 0)
        
        # Target trajectory parameters
        x_target = opti.parameter(self.N)
        y_target = opti.parameter(self.N)
        
        previous_control = np.zeros(1)
        
        # Build the MPC problem
        objective = 0
        
        for k in range(self.N):
            current_state = X[:, k]
            next_state = X[:, k + 1]
            control_input = U[:, k]
            
            # Target state (x, y, theta)
            # For theta, we'll use the direction to the next waypoint
            target_state = ca.vertcat(
                x_target[k], 
                y_target[k],
                ca.atan2(y_target[k] - current_state[1], x_target[k] - current_state[0]))
            
            # Add to objective function
            control_prev = previous_control if k == 0 else U[:, k - 1]
            objective += self.cost_function(current_state, control_input, target_state, control_prev)
            
            # Dynamics constraint
            opti.subject_to(next_state == bicycle_model(current_state, control_input))
        
        # Steering angle constraints
        opti.subject_to(opti.bounded(-np.pi/4, U[0, :], np.pi/4))
        
        opti.minimize(objective)
        opti.solver('ipopt')
        
        # Main simulation loop
        try:
            current_index = 0
            trajectory_history = deque(maxlen=100)
            
            while current_index < len(self.reference_waypoints) - self.N:
                # Get current state
                current_state = self.get_vehicle_state()
                
                # Get reference trajectory segment
                x_ref, y_ref = self.get_reference_trajectory(current_index)
                
                # Set initial state
                opti.set_initial(X[:, 0], current_state)
                
                # Set reference trajectory
                opti.set_value(x_target, x_ref)
                opti.set_value(y_target, y_ref)
                
                # Solve MPC problem
                try:
                    solution = opti.solve()
                    optimal_control = solution.value(U[:, 0])
                    predicted_trajectory = solution.value(X)
                    
                    # Apply control to vehicle
                    steer = float(optimal_control[0])
                    throttle = 0.5  # Fixed throttle for simplicity
                    
                    control = carla.VehicleControl()
                    control.steer = np.clip(steer, -0.5, 0.5)
                    control.throttle = throttle
                    control.brake = 0.0
                    self.vehicle.apply_control(control)
                    
                    # Visualize predicted trajectory
                    self.visualize_trajectory(predicted_trajectory)
                    
                    # Store trajectory for visualization
                    trajectory_history.append((current_state[0], current_state[1]))
                    
                except Exception as e:
                    print(f"MPC solver error: {e}")
                    # If solver fails, continue with previous control
                    steer = previous_control[0]
                
                previous_control = np.array([steer])
                current_index += 1
                
                # Tick the simulation
                self.world.tick()
                
                # Update spectator view
                transform = carla.Transform(
                    self.vehicle.get_transform().location + carla.Location(z=50),
                    carla.Rotation(pitch=-90))
                self.spectator.set_transform(transform)
                
        except KeyboardInterrupt:
            print("Simulation stopped by user")
        finally:
            # Clean up
            if self.vehicle:
                self.vehicle.destroy()
            if self.client:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
    
    def visualize_trajectory(self, trajectory: np.ndarray):
        """Visualize the predicted trajectory in CARLA."""
        # Clear previous trajectory visuals
        for visual in self.trajectory_visuals:
            visual.destroy()
        self.trajectory_visuals.clear()
        
        # Draw new trajectory
        for i in range(trajectory.shape[1] - 1):
            start_loc = carla.Location(x=trajectory[0, i], y=trajectory[1, i], z=0.5)
            end_loc = carla.Location(x=trajectory[0, i+1], y=trajectory[1, i+1], z=0.5)
            
            self.trajectory_visuals.append(
                self.debug_helper.draw_line(
                    start_loc, end_loc, thickness=0.1, 
                    color=carla.Color(0, 255, 0), life_time=self.dt))
            
            self.trajectory_visuals.append(
                self.debug_helper.draw_point(
                    start_loc, size=0.1, 
                    color=carla.Color(0, 255, 0), life_time=self.dt))

def main():
    try:
        # Initialize MPC controller
        mpc = BicycleMPC(
            wheelbase=2.5, 
            velocity=7.0, 
            horizon_length=10, 
            time_step=0.1)
        
        # Set up CARLA simulation (will work with any town)
        mpc.setup_carla_simulation(town='Town04')  # Can change to 'Town01', 'Town02', etc.
        
        # Run MPC control
        mpc.run_mpc_control()
        
    except Exception as e:
        print(f"Error in main execution: {e}")
    finally:
        print("Simulation ended")

if __name__ == "__main__":
    main()