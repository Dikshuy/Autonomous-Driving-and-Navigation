import numpy as np
import carla
import time
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class MPCController:
    def __init__(self, world, vehicle, dt=0.1, horizon=10):
        self.world = world
        self.vehicle = vehicle
        self.dt = dt  # Time step
        self.horizon = horizon  # Prediction horizon
        
        # MPC parameters
        self.Q = np.diag([1.0, 1.0, 0.5])  # State cost matrix (x, y, yaw)
        self.R = np.diag([0.1, 0.1])  # Control cost matrix (throttle, steering)
        
        # Vehicle model parameters (simplified)
        self.wheelbase = 2.8  # meters
        self.max_steer_angle = 0.7  # radians (approximately 40 degrees)
        self.max_throttle = 1.0
        self.max_brake = 1.0
        
        # For visualization
        self.trajectory_history = []
        self.reference_trajectory = []
        
    def kinematic_bicycle_model(self, state, control, dt):
        """
        Simplified kinematic bicycle model
        state = [x, y, yaw, v]
        control = [throttle, steering]
        """
        x, y, yaw, v = state
        throttle, steering = control
        
        # Simplified model for acceleration based on throttle
        if throttle >= 0:
            # Accelerating
            a = 3.0 * throttle  # max acceleration of 3 m/s^2
        else:
            # Braking
            a = 8.0 * throttle  # max deceleration of 8 m/s^2
        
        # Update velocity
        v_next = v + a * dt
        v_next = max(0.0, min(30.0, v_next))  # Limit velocity between 0 and 30 m/s
        
        # Bicycle model
        beta = math.atan(0.5 * math.tan(steering))
        x_next = x + v * math.cos(yaw + beta) * dt
        y_next = y + v * math.sin(yaw + beta) * dt
        yaw_next = yaw + v * math.tan(steering) / self.wheelbase * dt
        
        return [x_next, y_next, yaw_next, v_next]
    
    def objective(self, u, state, reference):
        """
        MPC cost function
        u: control inputs for the entire horizon, flattened [throttle_0, steering_0, ..., throttle_N-1, steering_N-1]
        state: initial state [x, y, yaw, v]
        reference: reference trajectory, list of [x, y, yaw]
        """
        u = u.reshape(self.horizon, 2)
        cost = 0.0
        x_t = state
        
        for i in range(self.horizon):
            # Predict next state
            x_next = self.kinematic_bicycle_model(x_t, u[i], self.dt)
            
            # State error
            state_error = np.array([
                x_next[0] - reference[min(i, len(reference)-1)][0],  # x error
                x_next[1] - reference[min(i, len(reference)-1)][1],  # y error
                x_next[2] - reference[min(i, len(reference)-1)][2]   # yaw error
            ])
            
            # Compute cost
            cost += state_error.T @ self.Q @ state_error  # State cost
            cost += u[i].T @ self.R @ u[i]  # Control cost
            
            # Set up for next prediction step
            x_t = x_next
        
        return cost
    
    def generate_reference_trajectory(self, waypoints, current_state):
        """
        Generate reference trajectory from waypoints
        """
        ref_traj = []
        for wp in waypoints:
            # Extract world location
            wp_loc = wp.transform.location
            # Calculate yaw from waypoint rotation
            wp_yaw = math.radians(wp.transform.rotation.yaw)
            ref_traj.append([wp_loc.x, wp_loc.y, wp_yaw])
        
        self.reference_trajectory = ref_traj
        return ref_traj
    
    def get_control(self, current_state, reference_trajectory):
        """
        Compute MPC control inputs
        current_state: [x, y, yaw, v]
        reference_trajectory: list of [x, y, yaw]
        """
        # Initial control guess (zero control inputs)
        u0 = np.zeros(self.horizon * 2)
        
        # Bounds for control inputs
        bounds = []
        for _ in range(self.horizon):
            bounds.append((-self.max_brake, self.max_throttle))  # Throttle/brake bounds
            bounds.append((-self.max_steer_angle, self.max_steer_angle))  # Steering bounds
        
        # Solve the optimization problem
        result = minimize(
            self.objective,
            u0,
            args=(current_state, reference_trajectory),
            method='SLSQP',
            bounds=bounds
        )
        
        # Extract the first control input
        u_optimal = result.x.reshape(self.horizon, 2)
        throttle_brake, steering = u_optimal[0]
        
        # Convert to CARLA control
        control = carla.VehicleControl()
        
        if throttle_brake >= 0:
            control.throttle = float(throttle_brake)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = float(-throttle_brake)
        
        control.steer = float(steering)
        control.hand_brake = False
        control.manual_gear_shift = False
        
        return control
    
    def update_state(self):
        """
        Get current vehicle state from CARLA
        """
        vehicle_transform = self.vehicle.get_transform()
        vehicle_velocity = self.vehicle.get_velocity()
        
        x = vehicle_transform.location.x
        y = vehicle_transform.location.y
        yaw = math.radians(vehicle_transform.rotation.yaw)
        v = math.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2)
        
        # Record trajectory point for visualization
        self.trajectory_history.append([x, y])
        
        return [x, y, yaw, v]
    
    def visualize_trajectories(self):
        """
        Visualize actual and reference trajectories
        """
        if not self.trajectory_history or not self.reference_trajectory:
            return
        
        # Extract x, y coordinates
        actual_x, actual_y = zip(*self.trajectory_history)
        ref_x, ref_y, _ = zip(*self.reference_trajectory)
        
        plt.figure(figsize=(10, 8))
        plt.plot(actual_x, actual_y, 'b-', label='Actual')
        plt.plot(ref_x, ref_y, 'r--', label='Reference')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.title('MPC Trajectory Tracking')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.savefig('mpc_trajectory.png')
        plt.close()


def main():
    """
    Main function to run the MPC controller in CARLA
    """
    try:
        # Connect to CARLA
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        
        # Get world and blueprint library
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        
        # Set synchronous mode
        settings = world.get_settings()
        sync_mode = False
        
        if not settings.synchronous_mode:
            sync_mode = True
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.1
            world.apply_settings(settings)
        
        # Spawn points
        spawn_points = world.get_map().get_spawn_points()
        start_point = spawn_points[0]
        
        # Create ego vehicle
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        vehicle = world.spawn_actor(vehicle_bp, start_point)
        print(f"Created vehicle: {vehicle.id}")
        
        # Create a camera for visualization
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        
        # Create spectator to follow the vehicle
        spectator = world.get_spectator()
        
        # Create waypoints for the reference trajectory
        waypoints = world.get_map().get_waypoints(vehicle.get_location(), distance=2.0)
        future_waypoints = []
        
        current_waypoint = waypoints[0]
        for _ in range(30):  # Get 30 waypoints
            waypoint_options = current_waypoint.next(2.0)  # Get waypoints 2 meters ahead
            if waypoint_options:
                current_waypoint = waypoint_options[0]
                future_waypoints.append(current_waypoint)
        
        # Create MPC controller
        mpc = MPCController(world, vehicle)
        
        # Main control loop
        try:
            for i in range(300):  # Run for 300 steps
                if sync_mode:
                    world.tick()
                else:
                    time.sleep(0.1)
                
                # Update spectator to follow vehicle
                vehicle_transform = vehicle.get_transform()
                spectator_transform = carla.Transform(
                    vehicle_transform.location + carla.Location(z=50, x=-30),
                    carla.Rotation(pitch=-90)
                )
                spectator.set_transform(spectator_transform)
                
                # Get current state
                current_state = mpc.update_state()
                
                # Generate/update reference trajectory
                if i % 10 == 0:  # Update reference trajectory every 10 steps
                    vehicle_location = vehicle.get_location()
                    waypoints = world.get_map().get_waypoints(vehicle_location, distance=2.0)
                    
                    future_waypoints = []
                    current_waypoint = waypoints[0]
                    for _ in range(30):
                        waypoint_options = current_waypoint.next(2.0)
                        if waypoint_options:
                            current_waypoint = waypoint_options[0]
                            future_waypoints.append(current_waypoint)
                
                reference_trajectory = mpc.generate_reference_trajectory(future_waypoints, current_state)
                
                # Compute control input
                control = mpc.get_control(current_state, reference_trajectory)
                
                # Apply control to vehicle
                vehicle.apply_control(control)
                
                # Print status
                if i % 10 == 0:
                    print(f"Step {i}: Pos=[{current_state[0]:.2f}, {current_state[1]:.2f}], "
                          f"V={current_state[3]:.2f}, Throttle={control.throttle:.2f}, "
                          f"Brake={control.brake:.2f}, Steer={control.steer:.2f}")
            
            # Visualize trajectories
            mpc.visualize_trajectories()
            
        finally:
            # Reset synchronous mode settings
            if sync_mode:
                settings.synchronous_mode = False
                world.apply_settings(settings)
            
            # Destroy actors
            if camera:
                camera.destroy()
            if vehicle:
                vehicle.destroy()
                
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        print("Simulation completed.")


if __name__ == "__main__":
    main()