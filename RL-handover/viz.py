import carla
import numpy as np
import casadi as ca
import random
import time
from typing import Tuple, Optional, List
from collections import deque

class BicycleMPC:
    """
    Final improved MPC controller for CARLA with proper mid-lane following.
    """
    def __init__(
        self, 
        wheelbase: float = 2.5, 
        velocity: float = 5.0,  # Conservative speed
        horizon_length: int = 10,
        time_step: float = 0.1
    ):
        self.L = wheelbase
        self.v = velocity
        self.N = horizon_length
        self.dt = time_step
        
        # Well-tuned cost matrices
        self.Q = np.diag([10.0, 10.0, 2.0])  # State cost
        self.R = np.diag([1.0])              # Control cost
        
        # CARLA objects
        self.client = None
        self.world = None
        self.vehicle = None
        self.spectator = None
        self.camera = None
        self.reference_waypoints = []
        
        # Visualization
        self.debug_helper = None
        self.waypoint_visuals = []
        self.trajectory_visuals = []

    def setup_carla_simulation(self, town: str = 'Town04'):
        """Properly initialize CARLA simulation with mid-lane following."""
        try:
            # Connect to CARLA
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(20.0)
            self.world = self.client.load_world(town)
            time.sleep(2)  # Allow map to load
            
            # Setup debug helper
            self.debug_helper = self.world.debug
            
            # Set synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.dt
            self.world.apply_settings(settings)
            
            # Spawn vehicle at a good highway location
            self.spawn_vehicle_on_highway()
            
            # Setup camera for proper visualization
            self.setup_camera()
            
            # Generate proper mid-lane reference trajectory
            self.generate_midlane_waypoints()
            
            print("CARLA simulation setup complete")
            
        except Exception as e:
            print(f"Error setting up CARLA: {e}")
            raise

    def spawn_vehicle_on_highway(self):
        """Spawn vehicle at a known good highway location in Town04."""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        
        # Predefined good spawn point for Town04 highway
        spawn_point = carla.Transform(
            carla.Location(x=-30.0, y=-194.0, z=0.5),
            carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0))
        
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.vehicle.set_autopilot(False)

    def setup_camera(self):
        """Setup a proper third-person chase camera."""
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        
        # Camera position - behind and above the vehicle
        camera_transform = carla.Transform(
            carla.Location(x=-8.0, z=5.0),  # 8m behind, 5m up
            carla.Rotation(pitch=-20.0))    # Slightly looking down
        
        self.camera = self.world.spawn_actor(
            camera_bp, 
            camera_transform, 
            attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.SpringArm)
        
        # Set spectator
        self.spectator = self.world.get_spectator()

    def generate_midlane_waypoints(self, distance: float = 5.0, num_points: int = 200):
        """Generate waypoints strictly following the mid-lane."""
        map = self.world.get_map()
        current_location = self.vehicle.get_location()
        current_waypoint = map.get_waypoint(current_location, project_to_road=True)
        
        if not current_waypoint:
            raise RuntimeError("Vehicle not on a road - cannot generate waypoints")
        
        self.reference_waypoints = [current_waypoint]
        
        # Generate chain of waypoints following the lane
        for _ in range(num_points - 1):
            next_waypoints = current_waypoint.next(distance)
            if not next_waypoints:
                break
            
            # Always take the first option (straightest path)
            current_waypoint = next_waypoints[0]
            self.reference_waypoints.append(current_waypoint)
            
            # Visualize waypoints
            self.debug_helper.draw_string(
                current_waypoint.transform.location + carla.Location(z=1.0),
                '•', 
                color=carla.Color(255, 0, 0), 
                life_time=100.0,
                persistent_lines=True)

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

    def cost_function(self, state: ca.MX, control: ca.MX, target: ca.MX) -> ca.MX:
        """Calculate the cost for MPC optimization."""
        state_cost = ca.mtimes((state - target).T, self.Q @ (state - target))
        control_cost = ca.mtimes(control.T, self.R @ control)
        return state_cost + control_cost

    def run_mpc_control(self):
        """Run the MPC controller with proper error handling."""
        if not self.vehicle or not self.reference_waypoints:
            raise RuntimeError("Simulation not properly initialized")
            
        model = self.create_bicycle_model()
        opti = ca.Opti()
        
        # State and control variables
        X = opti.variable(3, self.N + 1)
        U = opti.variable(1, self.N)
        
        # Target parameters
        x_target = opti.parameter(self.N)
        y_target = opti.parameter(self.N)
        theta_target = opti.parameter(self.N)
        
        # Initial guesses
        opti.set_initial(X, 0)
        opti.set_initial(U, 0)
        
        # Build MPC problem
        objective = 0
        for k in range(self.N):
            target_state = ca.vertcat(x_target[k], y_target[k], theta_target[k])
            objective += self.cost_function(X[:, k], U[:, k], target_state)
            opti.subject_to(X[:, k+1] == model(X[:, k], U[:, k]))
        
        # Steering constraints
        opti.subject_to(opti.bounded(-0.6, U, 0.6))  # ~35 degrees
        
        # Solver options
        p_opts = {"expand": True}
        s_opts = {
            "max_iter": 500,
            "print_level": 0,
            "acceptable_tol": 1e-4,
            "linear_solver": "mumps"
        }
        opti.solver('ipopt', p_opts, s_opts)
        
        # Main control loop
        try:
            current_idx = 0
            last_steer = 0.0
            
            while current_idx < len(self.reference_waypoints) - self.N:
                # Get current state
                vehicle_transform = self.vehicle.get_transform()
                x, y = vehicle_transform.location.x, vehicle_transform.location.y
                theta = np.radians(vehicle_transform.rotation.yaw)
                current_state = np.array([x, y, theta])
                
                # Get reference trajectory segment
                ref_segment = self.reference_waypoints[current_idx:current_idx + self.N]
                x_ref = np.array([w.transform.location.x for w in ref_segment])
                y_ref = np.array([w.transform.location.y for w in ref_segment])
                theta_ref = np.array([np.radians(w.transform.rotation.yaw) for w in ref_segment])
                
                # Set MPC parameters
                opti.set_value(x_target, x_ref)
                opti.set_value(y_target, y_ref)
                opti.set_value(theta_target, theta_ref)
                opti.set_initial(X[:, 0], current_state)
                
                # Solve MPC problem
                try:
                    sol = opti.solve()
                    predicted_trajectory = sol.value(X)
                    optimal_steer = float(sol.value(U[:, 0]))
                    last_steer = optimal_steer
                    
                    # Visualize predicted trajectory
                    self.visualize_trajectory(predicted_trajectory)
                    
                except Exception as e:
                    print(f"MPC warning: {str(e)[:100]}...")
                    optimal_steer = last_steer * 0.8  # Dampen the previous control
                
                # Apply control
                control = carla.VehicleControl()
                control.steer = np.clip(optimal_steer, -0.5, 0.5)
                control.throttle = 0.3
                self.vehicle.apply_control(control)
                
                # Update visualization
                self.update_camera_view()
                
                current_idx += 1
                self.world.tick()
                
        except KeyboardInterrupt:
            print("Simulation stopped by user")
        finally:
            self.cleanup()

    def visualize_trajectory(self, trajectory: np.ndarray):
        """Visualize the predicted trajectory."""
        # Clear previous visuals
        for visual in self.trajectory_visuals:
            visual.destroy()
        self.trajectory_visuals.clear()
        
        # Draw new trajectory
        for i in range(trajectory.shape[1] - 1):
            start = carla.Location(x=trajectory[0, i], y=trajectory[1, i], z=0.5)
            end = carla.Location(x=trajectory[0, i+1], y=trajectory[1, i+1], z=0.5)
            
            self.trajectory_visuals.append(
                self.debug_helper.draw_line(
                    start, end, thickness=0.1,
                    color=carla.Color(0, 255, 0), life_time=self.dt))
            
            self.trajectory_visuals.append(
                self.debug_helper.draw_point(
                    start, size=0.1,
                    color=carla.Color(0, 255, 0), life_time=self.dt))

    def update_camera_view(self):
        """Update the camera to follow the vehicle properly."""
        vehicle_transform = self.vehicle.get_transform()
        camera_transform = carla.Transform(
            vehicle_transform.location + carla.Location(x=-8.0, z=5.0),
            carla.Rotation(pitch=-20.0, yaw=vehicle_transform.rotation.yaw))
        self.spectator.set_transform(camera_transform)

    def cleanup(self):
        """Proper cleanup of CARLA actors."""
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
        mpc = BicycleMPC(
            wheelbase=2.5,
            velocity=5.0,
            horizon_length=10,
            time_step=0.1)
        
        mpc.setup_carla_simulation('Town04')
        mpc.run_mpc_control()
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Simulation ended")

if __name__ == "__main__":
    main()