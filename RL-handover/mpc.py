import carla
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import pygame
import time

class ModelPredictiveController:
    def __init__(self, prediction_horizon=10, control_horizon=5):
        """
        Initialize Model Predictive Controller with specified vehicle state
        
        State vector: [β (sideslip), ψ_dot (yaw rate), ψ (heading), y (lateral position)]
        Input: steering angle
        
        :param prediction_horizon: Number of steps to predict ahead
        :param control_horizon: Number of control steps to optimize
        """
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        
        # Vehicle parameters (these need to be tuned to your specific vehicle)
        self.L = 2.7    # wheelbase length
        self.Cf = 1000  # front cornering stiffness
        self.Cr = 1000  # rear cornering stiffness
        self.m = 1500   # vehicle mass
        self.Iz = 2000  # moment of inertia
        self.g = 9.81   # gravity
        
        # Constraints
        self.max_steering = np.radians(30)  # maximum steering angle
        self.max_acceleration = 5.0  # maximum acceleration/deceleration
    
    def vehicle_model(self, state, u_steering, v=10, dt=0.1):
        """
        Vehicle dynamic model
        
        :param state: Current state [β, ψ_dot, ψ, y]
        :param u_steering: Steering input
        :param v: Velocity (assumed constant for simplicity)
        :param dt: Time step
        :return: Next state
        """
        beta, psi_dot, psi, y = state
        
        # Simplified dynamic model
        # You would replace these with more accurate dynamic equations
        # based on your specific vehicle characteristics
        
        # Lateral acceleration
        ay = psi_dot * v
        
        # Sideslip angle derivative
        beta_dot = (self.Cf * u_steering - (self.Cf + self.Cr) * beta / (m * v)) 
        
        # Yaw rate derivative
        psi_dot_dot = (self.Cf * u_steering * self.L - (self.Cf * self.L - self.Cr * self.L) * beta / (self.Iz * v))
        
        # Update state
        next_beta = beta + beta_dot * dt
        next_psi_dot = psi_dot + psi_dot_dot * dt
        next_psi = psi + psi_dot * dt
        next_y = y + v * np.sin(beta) * dt
        
        return [next_beta, next_psi_dot, next_psi, next_y]
    
    def solve_mpc(self, current_state, target_trajectory, v=10):
        """
        Solve Model Predictive Control optimization problem
        
        :param current_state: Current vehicle state [β, ψ_dot, ψ, y]
        :param target_trajectory: Target trajectory to follow
        :param v: Velocity
        :return: Optimal steering input
        """
        # Define optimization variables
        u_steering = cp.Variable((self.control_horizon, 1))
        
        # Predicted states
        states = np.zeros((self.prediction_horizon + 1, 4))
        states[0] = current_state
        
        # Cost function
        cost = 0.0
        constraints = []
        
        dt = 0.1  # time step
        
        # Simulate prediction and construct optimization problem
        for t in range(self.prediction_horizon):
            # Simulate next state
            states[t+1] = self.vehicle_model(states[t], 
                                             u_steering[min(t, self.control_horizon-1)], 
                                             v, dt)
            
            # Track trajectory cost
            track_cost = cp.norm2(states[t+1][3] - target_trajectory[t+1][1])  # Track y-position
            
            # Control input cost (smoothness)
            control_cost = cp.norm2(u_steering[min(t, self.control_horizon-1)])
            
            cost += track_cost + 0.1 * control_cost
            
            # Constraints
            constraints += [
                cp.abs(u_steering[min(t, self.control_horizon-1)]) <= self.max_steering
            ]
        
        # Solve optimization problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()
        
        return u_steering.value[0][0]

def find_safe_spawn_point(world, blueprint):
    """
    Find a safe spawn point in the Carla world
    
    :param world: Carla world object
    :param blueprint: Vehicle blueprint
    :return: Safe spawn transform
    """
    spawn_points = world.get_map().get_spawn_points()
    
    # If no predefined spawn points, create a custom one
    if not spawn_points:
        spawn_points = [carla.Transform(carla.Location(x=0, y=0, z=1))]
    
    # Try spawning at different points
    for spawn_point in spawn_points:
        try:
            # Check for collisions
            nearby_vehicles = world.get_actors().filter('vehicle.*')
            collision = False
            
            for vehicle in nearby_vehicles:
                distance = spawn_point.location.distance(vehicle.get_location())
                if distance < 5.0:  # Minimum safe distance
                    collision = True
                    break
            
            if not collision:
                # Attempt to spawn vehicle
                vehicle = world.try_spawn_actor(blueprint, spawn_point)
                if vehicle is not None:
                    return vehicle, spawn_point
        
        except Exception as e:
            print(f"Spawn attempt failed: {e}")
    
    # If all spawn attempts fail
    raise RuntimeError("Could not find a safe spawn point")

def run_carla_mpc():
    """
    Main function to run MPC in Carla with visualization
    """
    # Connect to Carla
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # Spawn vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('model3')[0]

    try:
        # Find safe spawn point
        vehicle, spawn_point = find_safe_spawn_point(world, vehicle_bp)
        
        print(f"Vehicle spawned at: {spawn_point.location}")
            
        # spawn_point = carla.Transform(carla.Location(x=0, y=0, z=1))
        # vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        
        # Spectator for tracking
        spectator = world.get_spectator()
        
        # Create MPC controller
        mpc_controller = ModelPredictiveController()
        
        # Generate sample target trajectory (sinusoidal path)
        target_trajectory = []
        for x in np.linspace(0, 200, mpc_controller.prediction_horizon + 1):
            y = 3 * np.sin(x * 0.05)  # Sinusoidal path with amplitude 3
            target_trajectory.append((x, y))
        
        # Visualization setup
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Tracking plot
        tracking_line, = ax1.plot([], [], 'r-', label='Vehicle Path')
        target_line, = ax1.plot([], [], 'b--', label='Target Path')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.legend()
        
        # State plot
        state_lines = [ax2.plot([], [], label=f'State {i}')[0] for i in range(4)]
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('State Values')
        ax2.legend()
        
        # Data storage for plotting
        vehicle_path = {'x': [], 'y': []}
        states_history = {'beta': [], 'psi_dot': [], 'psi': [], 'y': []}
        
        try:
            # Main control loop
            velocity = 10  # Constant velocity
            current_state = [0, 0, 0, 0]  # Initial state [β, ψ_dot, ψ, y]
            
            for step in range(200):  # Limit steps for demonstration
                # Get current vehicle state from Carla
                transform = vehicle.get_transform()
                
                # Compute MPC control
                steering = mpc_controller.solve_mpc(current_state, target_trajectory, velocity)
                
                # Apply control to vehicle
                control = carla.VehicleControl()
                control.steer = steering / mpc_controller.max_steering  # Normalize steering
                control.throttle = 0.5  # Constant throttle
                
                vehicle.apply_control(control)
                
                # Update spectator to follow vehicle
                spectator_transform = carla.Transform(
                    transform.location + carla.Location(z=20),
                    carla.Rotation(pitch=-90)
                )
                spectator.set_transform(spectator_transform)
                
                # Simulate next state using our model
                current_state = mpc_controller.vehicle_model(
                    current_state, steering, velocity
                )
                
                # Store path and states for visualization
                vehicle_path['x'].append(transform.location.x)
                vehicle_path['y'].append(transform.location.y)
                
                for i, key in enumerate(['beta', 'psi_dot', 'psi', 'y']):
                    states_history[key].append(current_state[i])
                
                # Update plots
                tracking_line.set_xdata(vehicle_path['x'])
                tracking_line.set_ydata(vehicle_path['y'])
                
                target_x = [p[0] for p in target_trajectory]
                target_y = [p[1] for p in target_trajectory]
                target_line.set_xdata(target_x)
                target_line.set_ydata(target_y)
                
                for i, line in enumerate(state_lines):
                    state_keys = ['beta', 'psi_dot', 'psi', 'y']
                    line.set_xdata(range(len(states_history[state_keys[i]])))
                    line.set_ydata(states_history[state_keys[i]])
                
                ax1.relim()
                ax1.autoscale_view()
                ax2.relim()
                ax2.autoscale_view()
                
                plt.pause(0.1)
                
                # Small delay to control simulation speed
                time.sleep(0.1)
            
        except Exception as e:
            print(f"Error occurred: {e}")
    finally:
        # Clean up
        vehicle.destroy()
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    run_carla_mpc()