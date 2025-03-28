import carla
import numpy as np
import cvxpy as cp

class VehicleMPC:
    def __init__(self, prediction_horizon=10, control_horizon=5):
        """
        Initialize Model Predictive Control for Vehicle
        
        State vector: [β, ψ_dot, ψ, y]
        Control input: [δ] (steering angle)
        """
        # Vehicle parameters
        self.L = 2.7  # Wheelbase length (m)
        self.v = 10.0  # Constant velocity
        
        # MPC configuration
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        
        # Simulation parameters
        self.dt = 0.1  # Timestep
        
        # Cost function weights
        self.Q_state = np.diag([10.0, 5.0, 10.0, 20.0])
        self.R_control = np.diag([1.0])
        
        # Constraints
        self.max_steering = float(np.deg2rad(30))
        self.max_steering_rate = float(np.deg2rad(10))
    
    def solve_mpc(self, current_state, reference_trajectory):
        """
        Solve Model Predictive Control optimization problem
        """
        # Ensure inputs are numpy arrays
        current_state = np.array(current_state, dtype=float)
        reference_trajectory = np.array(reference_trajectory, dtype=float)
        
        # Define optimization variables
        u = cp.Variable((self.control_horizon, 1))  # Control inputs
        x = cp.Variable((self.prediction_horizon + 1, 4))  # State trajectory
        
        # Initial state constraint
        constraints = [x[0, :] == current_state]
        
        # Cost function
        cost = 0.0
        
        for t in range(self.prediction_horizon):
            # Reference state (use last reference if horizon exceeds trajectory)
            ref_state = reference_trajectory[min(t, reference_trajectory.shape[0]-1), :]
            
            # State cost
            state_diff = x[t, :] - ref_state
            cost += cp.quad_form(state_diff, self.Q_state)
            
            # Control input cost
            if t < self.control_horizon:
                cost += cp.quad_form(u[t], self.R_control)
            
            # Input constraints
            if t < self.control_horizon:
                constraints += [
                    cp.abs(u[t]) <= self.max_steering
                ]
                # Steering rate constraint
                if t > 0:
                    constraints += [
                        cp.abs(u[t] - u[t-1]) <= self.max_steering_rate
                    ]
            
            # Dynamics constraints using CVXPY-compatible formulation
            control_input = u[min(t, self.control_horizon-1)]
            
            # Kinematic bicycle model constraints
            beta_dot = 0.0
            psi_dot_new = cp.multiply(self.v, cp.tan(control_input) / self.L)
            psi_new = x[t, 2] + psi_dot_new * self.dt
            y_new = x[t, 3] + self.v * cp.sin(psi_new) * self.dt
            
            constraints += [
                x[t+1, 0] == beta_dot,
                x[t+1, 1] == psi_dot_new,
                x[t+1, 2] == psi_new,
                x[t+1, 3] == y_new
            ]
        
        # Terminal cost
        terminal_ref = reference_trajectory[-1, :]
        cost += cp.quad_form(x[-1, :] - terminal_ref, 10.0 * self.Q_state)
        
        # Solve optimization problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.ECOS)
        
        # Return first control input or 0 if solve failed
        return float(u.value[0, 0]) if u.value is not None else 0.0
    
    def generate_reference_trajectory(self, initial_pose):
        """
        Generate a straight-line reference trajectory
        
        Args:
        - initial_pose: Initial vehicle pose
        
        Returns:
        - Trajectory as numpy array
        """
        trajectory = np.zeros((self.prediction_horizon + 1, 4))
        
        # Extract initial conditions
        _, _, initial_psi, initial_y = initial_pose
        
        for t in range(self.prediction_horizon + 1):
            trajectory[t, 0] = 0.0  # β (slip angle)
            trajectory[t, 1] = 0.0  # ψ_dot (yaw rate)
            trajectory[t, 2] = initial_psi  # ψ (heading) - keep constant for straight line
            trajectory[t, 3] = initial_y  # y (lateral position) - keep constant
        
        return trajectory
    
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

def spawn_vehicle(world, blueprint_library, spawn_attempts=10):
    """
    Robustly spawn a vehicle in the Carla world
    """
    vehicle_blueprints = blueprint_library.filter('vehicle.*')
    spawn_points = world.get_map().get_spawn_points()
    
    for attempt in range(spawn_attempts):
        try:
            blueprint = random.choice(vehicle_blueprints)
            spawn_point = random.choice(spawn_points)
            
            vehicle = world.spawn_actor(blueprint, spawn_point)
            
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
        client.set_timeout(10.0)
        
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
            
            # Generate reference trajectory based on current state
            reference_trajectory = mpc_controller.generate_reference_trajectory(current_state)
            
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