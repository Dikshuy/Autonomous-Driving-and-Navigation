import carla
import numpy as np
import casadi as ca
import random
import time
from typing import Tuple, Optional, List
from collections import deque

class BicycleMPC:
    """
    Enhanced Model Predictive Controller for CARLA with better visualization and solver stability.
    """
    def __init__(
        self, 
        wheelbase: float = 2.5, 
        velocity: float = 7.0, 
        horizon_length: int = 10,  # Increased horizon
        time_step: float = 0.1
    ):
        self.L = wheelbase
        self.v = velocity
        self.N = horizon_length
        self.dt = time_step
        
        # More conservative tuning for better solver convergence
        self.Q = np.diag([5.0, 5.0, 0.5])  # Reduced state cost weights
        self.R = np.diag([0.5])            # Increased control cost weight
        
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
        self.camera = None

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
        """Calculate the cost for MPC optimization with smoother penalties."""
        state_cost = ca.mtimes((state - target).T, self.Q @ (state - target))
        control_cost = ca.mtimes(control.T, self.R @ control)
        
        if control_prev is not None:
            # Smoother control change penalty
            jerk_penalty = 0.05 * ca.sumsqr(control - control_prev)
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
            time.sleep(2)  # Give time for map to load
            
            # Set up debug helper
            self.debug_helper = self.world.debug
            
            # Set synchronous mode for deterministic simulation
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.dt
            self.world.apply_settings(settings)
            
            # Spawn a vehicle at any available spawn point
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]  # More stable vehicle
            
            # Get all spawn points and select one at random
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise RuntimeError("No spawn points available in the map")
                
            spawn_point = random.choice(spawn_points)
            print(f"Spawning vehicle at: {spawn_point.location}")
            
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            self.vehicle.set_autopilot(False)  # Ensure we have full control
            
            # Setup camera for better visualization
            self.setup_camera()
            
            # Generate reference trajectory (straight line from spawn point)
            self.generate_reference_waypoints()
            
            print("CARLA simulation setup complete")
            
        except Exception as e:
            print(f"Error setting up CARLA simulation: {e}")
            raise

    def setup_camera(self):
        """Setup a third-person camera for better visualization."""
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        
        # Attach camera behind the vehicle
        camera_transform = carla.Transform(
            carla.Location(x=-8, z=5),  # 8 meters behind, 5 meters up
            carla.Rotation(pitch=-20))   # Slightly looking down
        
        self.camera = self.world.spawn_actor(
            camera_bp, 
            camera_transform, 
            attach_to=self.vehicle)
        
        # Set spectator to follow the vehicle
        self.spectator = self.world.get_spectator()

    def generate_reference_waypoints(self, distance: float = 5.0, num_points: int = 200):
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
                life_time=100.0,
                persistent_lines=True)
    
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
        """Run the MPC controller in the CARLA simulation with improved stability."""
        if not self.vehicle or not self.reference_waypoints:
            raise RuntimeError("CARLA simulation not properly set up")
            
        bicycle_model = self.create_bicycle_model()
        
        # Initialize MPC optimization problem with solver options
        opti = ca.Opti()
        X = opti.variable(3, self.N + 1)  # State trajectory
        U = opti.variable(1, self.N)       # Control inputs (steering)
        
        # Better initial guesses
        opti.set_initial(X, 0)
        opti.set_initial(U, 0)
        
        # Target trajectory parameters
        x_target = opti.parameter(self.N)
        y_target = opti.parameter(self.N)
        
        previous_control = np.zeros(1)
        
        # Build the MPC problem with relaxed constraints
        objective = 0
        
        for k in range(self.N):
            current_state = X[:, k]
            next_state = X[:, k + 1]
            control_input = U[:, k]
            
            # Target state (x, y, theta)
            target_state = ca.vertcat(
                x_target[k], 
                y_target[k],
                ca.atan2(y_target[k] - current_state[1], x_target[k] - current_state[0]))
            
            # Add to objective function
            control_prev = previous_control if k == 0 else U[:, k - 1]
            objective += self.cost_function(current_state, control_input, target_state, control_prev)
            
            # Dynamics constraint with relaxation
            opti.subject_to(next_state == bicycle_model(current_state, control_input))
        
        # More relaxed steering angle constraints
        opti.subject_to(opti.bounded(-np.pi/3, U[0, :], np.pi/3))
        
        # Set solver options for better convergence
        p_opts = {"expand": True}
        s_opts = {"max_iter": 1000, "print_level": 0, "acceptable_tol": 1e-4}
        opti.solver('ipopt', p_opts, s_opts)
        
        # Main simulation loop
        try:
            current_index = 0
            trajectory_history = deque(maxlen=100)
            last_valid_control = 0.0  # Fallback for solver failures
            
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
                
                # Solve MPC problem with error handling
                try:
                    solution = opti.solve()
                    optimal_control = solution.value(U[:, 0])
                    predicted_trajectory = solution.value(X)
                    last_valid_control = float(optimal_control[0])
                    
                    # Visualize predicted trajectory
                    self.visualize_trajectory(predicted_trajectory)
                    
                except Exception as e:
                    print(f"MPC solver warning: {e}")
                    # Use last valid control with damping
                    optimal_control = last_valid_control * 0.8
                
                # Apply control to vehicle
                steer = np.clip(float(optimal_control), -0.5, 0.5)
                throttle = 0.3  # Reduced throttle for stability
                
                control = carla.VehicleControl()
                control.steer = steer
                control.throttle = throttle
                control.brake = 0.0
                self.vehicle.apply_control(control)
                
                # Update camera and spectator views
                self.update_visualization()
                
                previous_control = np.array([steer])
                current_index += 1
                
                # Tick the simulation
                self.world.tick()
                
        except KeyboardInterrupt:
            print("Simulation stopped by user")
        finally:
            self.cleanup()

    def update_visualization(self):
        """Update camera and spectator views."""
        # Update spectator to follow behind the vehicle
        vehicle_transform = self.vehicle.get_transform()
        camera_transform = carla.Transform(
            vehicle_transform.location + carla.Location(x=-8, z=5),
            carla.Rotation(pitch=-20, yaw=vehicle_transform.rotation.yaw))
        self.spectator.set_transform(camera_transform)

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

    def cleanup(self):
        """Clean up CARLA actors and settings."""
        if self.vehicle:
            self.vehicle.destroy()
        if self.camera:
            self.camera.destroy()
        if self.client:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

def main():
    try:
        # Initialize MPC controller
        mpc = BicycleMPC(
            wheelbase=2.5, 
            velocity=5.0,  # Reduced speed for stability
            horizon_length=10, 
            time_step=0.1)
        
        # Set up CARLA simulation
        mpc.setup_carla_simulation(town='Town04')
        
        # Run MPC control
        mpc.run_mpc_control()
        
    except Exception as e:
        print(f"Error in main execution: {e}")
    finally:
        print("Simulation ended")

if __name__ == "__main__":
    main()