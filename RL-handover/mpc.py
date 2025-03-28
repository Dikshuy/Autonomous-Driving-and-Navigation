import carla
import numpy as np
import cvxpy as cp
import math

class VehicleMPC:
    def __init__(self, prediction_horizon=10, control_horizon=5):
        """
        Initialize Model Predictive Control for Vehicle
        
        State vector: [β, ψ_dot, ψ, y]
        Control input: [δ] (steering angle)
        
        Parameters:
        - prediction_horizon: Number of future timesteps to predict
        - control_horizon: Number of control inputs to optimize
        """
        # Vehicle parameters (example values, may need tuning)
        self.L = 2.7  # Wheelbase length (m)
        self.Cf = 60000.0  # Front cornering stiffness
        self.Cr = 60000.0  # Rear cornering stiffness
        self.m = 1500.0  # Vehicle mass (kg)
        self.Iz = 2875.0  # Moment of inertia (kg*m^2)
        
        # MPC configuration
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        
        # Cost function weights (ensure float values)
        self.Q_state = np.diag([10.0, 5.0, 10.0, 20.0])  # State cost (β, ψ_dot, ψ, y)
        self.R_control = np.diag([1.0])  # Control input cost
        
        # Constraints
        self.max_steering = float(np.deg2rad(30))  # Maximum steering angle
        self.max_steering_rate = float(np.deg2rad(10))  # Maximum steering rate
        
        # Simulation timestep
        self.dt = 0.1  # 100ms timestep
    
    def vehicle_dynamics(self, state, steering_input):
        """
        Bicycle kinematic model with lateral dynamics
        
        Args:
        - state: [β, ψ_dot, ψ, y]
        - steering_input: δ (steering angle)
        
        Returns:
        - Next state prediction
        """
        beta, psi_dot, psi, y = state
        
        # Lateral slip angle calculations
        alpha_f = steering_input - math.atan2(psi_dot * self.L / 2.0, 1.0)
        alpha_r = -math.atan2(psi_dot * self.L / 2.0, 1.0)
        
        # Lateral force calculations
        Fy_f = self.Cf * alpha_f
        Fy_r = self.Cr * alpha_r
        
        # State derivatives
        beta_dot = (Fy_f + Fy_r) / (self.m * 9.81)
        psi_dot_dot = (Fy_f * self.L / 2.0 - Fy_r * self.L / 2.0) / self.Iz
        psi_dot_new = psi_dot + psi_dot_dot * self.dt
        psi_new = psi + psi_dot * self.dt
        y_dot = math.sin(psi) * 9.81  # Assuming constant velocity
        y_new = y + y_dot * self.dt
        
        return [beta_dot, psi_dot_new, psi_new, y_new]
    
    def solve_mpc(self, current_state, reference_trajectory):
        """
        Solve Model Predictive Control optimization problem
        
        Args:
        - current_state: Current vehicle state [β, ψ_dot, ψ, y]
        - reference_trajectory: Target trajectory to follow
        
        Returns:
        - Optimal control input (steering angle)
        """
        # Ensure inputs are numpy arrays with float type
        current_state = np.array(current_state, dtype=float)
        reference_trajectory = np.array(reference_trajectory, dtype=float)
        
        # Define optimization variables
        u = cp.Variable((self.control_horizon, 1))  # Control inputs
        x = cp.Variable((self.prediction_horizon + 1, 4))  # State trajectory
        
        # Initial state constraint
        constraints = [x[0, :] == current_state]
        
        # Build cost function and dynamics constraints
        cost = 0.0
        for t in range(self.prediction_horizon):
            # Ensure reference trajectory is within bounds
            ref_state = reference_trajectory[min(t, reference_trajectory.shape[0]-1), :]
            
            # State cost
            state_diff = x[t, :] - ref_state
            cost += cp.quad_form(state_diff, np.eye(4))  # Use identity matrix for simplicity
            
            # Control input cost
            if t < self.control_horizon:
                cost += cp.quad_form(u[t], np.eye(1))
            
            # Input constraints
            if t < self.control_horizon:
                constraints += [
                    cp.abs(u[t]) <= self.max_steering
                ]
                
                # Steering rate constraint (only after first timestep)
                if t > 0:
                    constraints += [
                        cp.abs(u[t] - u[t-1]) <= self.max_steering_rate
                    ]
            
            # Dynamics constraints
            control_input = u[min(t, self.control_horizon-1)]
            dynamics_pred = self.vehicle_dynamics(x[t, :], control_input)
            constraints += [x[t+1, :] == dynamics_pred]
        
        # Terminal cost
        terminal_ref = reference_trajectory[-1, :]
        cost += cp.quad_form(x[-1, :] - terminal_ref, 10.0 * np.eye(4))
        
        # Solve optimization problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.ECOS)  # Use ECOS solver for better compatibility
        
        # Return first control input or 0 if solve failed
        return float(u.value[0, 0]) if u.value is not None else 0.0
    
    def get_vehicle_state(self, vehicle):
        """
        Extract vehicle state from Carla vehicle
        
        Returns:
        - State vector [β, ψ_dot, ψ, y]
        """
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        
        # Compute state vector components
        beta = 0.0  # Lateral slip angle (simplified)
        psi_dot = float(velocity.z)  # Yaw rate
        psi = float(transform.rotation.yaw)  # Heading
        y = float(transform.location.y)  # Lateral position
        
        return [beta, psi_dot, psi, y]
    
    def generate_reference_trajectory(self):
        """
        Generate a reference trajectory for the vehicle to track
        
        Returns:
        - Trajectory as numpy array
        """
        # Example: circular trajectory
        trajectory = np.zeros((self.prediction_horizon + 1, 4))
        for t in range(self.prediction_horizon + 1):
            trajectory[t, 0] = 0.0  # β
            trajectory[t, 1] = 0.0  # ψ_dot
            trajectory[t, 2] = float(t * self.dt)  # ψ
            trajectory[t, 3] = float(math.sin(t * self.dt))  # y
        
        return trajectory

def spawn_vehicle(world, blueprint_library, spawn_attempts=10):
    """
    Robustly spawn a vehicle in the Carla world
    
    Args:
    - world: Carla world instance
    - blueprint_library: Blueprint library to select vehicle from
    - spawn_attempts: Number of attempts to spawn vehicle
    
    Returns:
    - Spawned vehicle actor or None if unsuccessful
    """
    # Filter for vehicle blueprints (excluding bikes, etc.)
    vehicle_blueprints = blueprint_library.filter('vehicle.*')
    
    # Get all possible spawn points
    spawn_points = world.get_map().get_spawn_points()
    
    for attempt in range(spawn_attempts):
        try:
            # Randomly select blueprint and spawn point
            blueprint = random.choice(vehicle_blueprints)
            spawn_point = random.choice(spawn_points)
            
            # Attempt to spawn vehicle
            vehicle = world.spawn_actor(blueprint, spawn_point)
            
            # Optional: Additional checks
            if vehicle:
                print(f"Successfully spawned {blueprint.id} at {spawn_point}")
                return vehicle
        
        except Exception as e:
            print(f"Spawn attempt {attempt + 1} failed: {e}")
    
    print("Failed to spawn vehicle after multiple attempts")
    return None

def main():
    try:
        # Connect to Carla
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)  # Set a timeout
        
        # Get world and blueprint library
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        
        # Spawn vehicle with robust method
        vehicle = spawn_vehicle(world, blueprint_library)
        
        if vehicle is None:
            print("Could not spawn vehicle")
            return
        
        # Initialize MPC controller
        mpc_controller = VehicleMPC()
        
        # Main control loop
        for _ in range(100):  # Run for 100 iterations as example
            # Get current vehicle state
            current_state = mpc_controller.get_vehicle_state(vehicle)
            
            # Generate reference trajectory
            reference_trajectory = mpc_controller.generate_reference_trajectory()
            
            # Solve MPC and get steering command
            steering_command = mpc_controller.solve_mpc(current_state, reference_trajectory)
            
            # Apply control to vehicle
            vehicle.apply_control(carla.VehicleControl(
                steer=steering_command,
                throttle=0.5  # Constant speed example
            ))
            
            # Tick the world
            world.tick()
    
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if 'vehicle' in locals() and vehicle is not None:
            vehicle.destroy()

if __name__ == '__main__':
    import random
    main()