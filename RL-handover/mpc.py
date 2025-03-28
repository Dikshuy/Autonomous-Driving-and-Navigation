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
        self.Cf = 60000  # Front cornering stiffness
        self.Cr = 60000  # Rear cornering stiffness
        self.m = 1500  # Vehicle mass (kg)
        self.Iz = 2875  # Moment of inertia (kg*m^2)
        
        # MPC configuration
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        
        # Cost function weights
        self.Q_state = np.diag([10, 5, 10, 20])  # State cost (β, ψ_dot, ψ, y)
        self.R_control = np.diag([1])  # Control input cost
        
        # Constraints
        self.max_steering = np.deg2rad(30)  # Maximum steering angle
        self.max_steering_rate = np.deg2rad(10)  # Maximum steering rate
    
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
        alpha_f = steering_input - math.atan2(psi_dot * self.L / 2, 1)
        alpha_r = -math.atan2(psi_dot * self.L / 2, 1)
        
        # Lateral force calculations
        Fy_f = self.Cf * alpha_f
        Fy_r = self.Cr * alpha_r
        
        # State derivatives
        beta_dot = (Fy_f + Fy_r) / (self.m * 9.81)
        psi_dot_dot = (Fy_f * self.L / 2 - Fy_r * self.L / 2) / self.Iz
        psi_dot_new = psi_dot + psi_dot_dot * 0.1  # Assuming 0.1s timestep
        psi_new = psi + psi_dot * 0.1
        y_dot = math.sin(psi) * 9.81  # Assuming constant velocity
        y_new = y + y_dot * 0.1
        
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
        # Define optimization variables
        u = cp.Variable((self.control_horizon, 1))  # Control inputs
        x = cp.Variable((self.prediction_horizon + 1, 4))  # State trajectory
        
        # Initial state constraint
        constraints = [x[0, :] == current_state]
        
        # Build cost function and dynamics constraints
        cost = 0
        for t in range(self.prediction_horizon):
            # State cost
            cost += cp.quad_form(x[t, :] - reference_trajectory[t, :], self.Q_state)
            
            # Control input cost
            if t < self.control_horizon:
                cost += cp.quad_form(u[t], self.R_control)
            
            # Input constraints
            if t < self.control_horizon:
                constraints += [
                    cp.abs(u[t]) <= self.max_steering,
                    cp.abs(u[t] - u[max(0, t-1)]) <= self.max_steering_rate
                ]
            
            # Dynamics constraints
            dynamics_pred = self.vehicle_dynamics(x[t, :], u[min(t, self.control_horizon-1)])
            constraints += [x[t+1, :] == dynamics_pred]
        
        # Terminal cost
        cost += cp.quad_form(x[self.prediction_horizon, :] - reference_trajectory[-1, :], 
                              10 * self.Q_state)
        
        # Solve optimization problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()
        
        # Return first control input
        return u.value[0, 0]
    
    def run_in_carla(self, world, vehicle):
        """
        Run MPC controller in Carla simulation
        
        Args:
        - world: Carla world instance
        - vehicle: Carla vehicle actor
        """
        # Generate reference trajectory (example)
        reference_trajectory = self.generate_reference_trajectory()
        
        while True:
            # Get current vehicle state
            current_state = self.get_vehicle_state(vehicle)
            
            # Solve MPC and get steering command
            steering_command = self.solve_mpc(current_state, reference_trajectory)
            
            # Apply control to vehicle
            vehicle.apply_control(carla.VehicleControl(
                steer=steering_command,
                throttle=0.5  # Constant speed example
            ))
            
            # Simulation update logic
            world.tick()
    
    def get_vehicle_state(self, vehicle):
        """
        Extract vehicle state from Carla vehicle
        
        Returns:
        - State vector [β, ψ_dot, ψ, y]
        """
        # Placeholder - implement actual state extraction
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        
        # Compute state vector components
        beta = 0  # Lateral slip angle
        psi_dot = velocity.z  # Yaw rate
        psi = transform.rotation.yaw  # Heading
        y = transform.location.y  # Lateral position
        
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
            trajectory[t, 0] = 0  # β
            trajectory[t, 1] = 0  # ψ_dot
            trajectory[t, 2] = t * 0.1  # ψ
            trajectory[t, 3] = math.sin(t * 0.1)  # y
        
        return trajectory

# Example usage
def main():
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    
    # Spawn vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('model3')[0]
    spawn_point = carla.Transform(carla.Location(x=0, y=0, z=1))
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    
    # Initialize MPC
    mpc_controller = VehicleMPC()
    mpc_controller.run_in_carla(world, vehicle)

if __name__ == '__main__':
    main()