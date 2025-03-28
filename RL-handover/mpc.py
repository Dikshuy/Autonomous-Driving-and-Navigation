import carla
import numpy as np
from casadi import *
import matplotlib.pyplot as plt
import time
import pygame

# Vehicle parameters (example values - adjust according to your vehicle)
class VehicleParams:
    def __init__(self):
        self.m = 1500.0       # mass [kg]
        self.lf = 1.2         # distance from CG to front axle [m]
        self.lr = 1.8         # distance from CG to rear axle [m]
        self.Iz = 3000.0      # yaw moment of inertia [kg*m^2]
        self.Cf = 80000.0    # front cornering stiffness [N/rad]
        self.Cr = 80000.0    # rear cornering stiffness [N/rad]
        self.vx = 20.0        # longitudinal velocity [m/s] (assumed constant)

# MPC parameters
class MPCParams:
    def __init__(self):
        self.N = 10           # prediction horizon
        self.dt = 0.1         # time step [s]
        self.max_iter = 100   # maximum iterations for MPC solver
        self.max_steer = np.deg2rad(30)  # maximum steering angle [rad]

# Vehicle dynamics model
def vehicle_model(params):
    # States: [β (sideslip angle), ψ_dot (yaw rate), ψ (yaw angle), y (lateral displacement)]
    # Input: δ (steering angle)
    
    # Define symbolic variables
    x = SX.sym('x', 4)  # states
    u = SX.sym('u', 1)  # control input (steering)
    
    # Extract parameters
    m = params.m
    lf = params.lf
    lr = params.lr
    Iz = params.Iz
    Cf = params.Cf
    Cr = params.Cr
    vx = params.vx
    
    # Extract states
    beta = x[0]
    psi_dot = x[1]
    psi = x[2]
    y = x[3]
    
    # Extract control input
    delta = u[0]
    
    # Vehicle dynamics equations (bicycle model)
    # Sideslip angle derivative
    beta_dot = (2/(m*vx)) * (Cf*(delta - beta - (lf*psi_dot)/vx) + 
                             Cr*(-beta + (lr*psi_dot)/vx)) - psi_dot
    
    # Yaw acceleration
    psi_ddot = (2/Iz) * (lf*Cf*(delta - beta - (lf*psi_dot)/vx) - 
                          lr*Cr*(-beta + (lr*psi_dot)/vx))
    
    # Yaw angle derivative
    psi_dot = psi_dot  # trivial
    
    # Lateral velocity
    y_dot = vx * np.sin(psi + beta)
    
    # State derivatives
    xdot = vertcat(beta_dot, psi_ddot, psi_dot, y_dot)
    
    # Create CasADI function
    f = Function('f', [x, u], [xdot], ['x', 'u'], ['xdot'])
    return f

# MPC controller class
class MPCController:
    def __init__(self, vehicle_params, mpc_params):
        self.vehicle_params = vehicle_params
        self.mpc_params = mpc_params
        self.model = vehicle_model(vehicle_params)
        
        # Initialize MPC
        self.setup_mpc()
        
    def setup_mpc(self):
        N = self.mpc_params.N
        dt = self.mpc_params.dt
        
        # Define symbolic variables for MPC
        X = SX.sym('X', 4, N+1)  # states over horizon
        U = SX.sym('U', 1, N)    # controls over horizon
        P = SX.sym('P', 4 + 4)   # parameters (initial state + reference)
        
        # Cost function and constraints
        obj = 0  # objective
        g = []   # constraints
        
        # Initial condition constraint
        g.append(X[:, 0] - P[0:4])
        
        # Weights for cost function
        Q = np.diag([1.0, 0.1, 1.0, 10.0])  # state weights
        R = np.diag([0.1])                  # control weights
        
        # Reference state
        x_ref = P[4:8]
        
        # Build MPC problem
        for k in range(N):
            # Current state and control
            x = X[:, k]
            u = U[:, k]
            
            # Add running cost
            obj += mtimes([(x - x_ref).T, Q, (x - x_ref)]) + mtimes([u.T, R, u])
            
            # Dynamics constraint
            x_next = X[:, k+1]
            xdot = self.model(x, u)
            x_next_pred = x + xdot * dt
            g.append(x_next - x_next_pred)
            
        # Terminal cost
        obj += 10 * mtimes([(X[:, N] - x_ref).T, Q, (X[:, N] - x_ref)])
        
        # Create NLP problem
        nlp = {'x': vertcat(reshape(X, -1, 1), reshape(U, -1, 1)),
               'f': obj,
               'g': vertcat(*g),
               'p': P}
        
        # Solver options
        opts = {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0,
            'ipopt.max_iter': self.mpc_params.max_iter
        }
        
        # Create solver
        self.solver = nlpsol('solver', 'ipopt', nlp, opts)
        
        # Variable bounds
        self.lbx = [-inf] * (4*(N+1)) + [-self.mpc_params.max_steer] * N
        self.ubx = [inf] * (4*(N+1)) + [self.mpc_params.max_steer] * N
        
        # Constraint bounds
        # 1 initial condition constraint (4 eq) + N dynamics constraints (4 eq each)
        num_constraints = 4 + 4*N
        self.lbg = [0] * num_constraints
        self.ubg = [0] * num_constraints
        
    def solve_mpc(self, x0, x_ref):
        N = self.mpc_params.N
        
        # Initial guess (straight line, zero steering)
        x_init = np.tile(x0, N+1)
        u_init = np.zeros(N)
        vars_init = np.concatenate([x_init, u_init])
        
        # Parameters (initial state + reference)
        p = np.concatenate([x0, x_ref])
        
        # Solve NLP
        sol = self.solver(
            x0=vars_init,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=self.lbg,
            ubg=self.ubg,
            p=p
        )
        
        # Extract solution
        vars_opt = sol['x'].full().flatten()
        x_opt = vars_opt[:4*(N+1)].reshape(N+1, 4).T
        u_opt = vars_opt[4*(N+1):]
        
        return u_opt[0], x_opt

# Carla interface class
class CarlaInterface:
    def __init__(self, mpc_params):
        self.mpc_params = mpc_params
        
        # Connect to CARLA server
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        
        # Get world and set synchronous mode
        self.world = self.client.get_world()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.mpc_params.dt
        self.world.apply_settings(settings)
        
        # Get blueprint library
        self.blueprint_library = self.world.get_blueprint_library()
        
        # Initialize visualization structures
        self.display = None
        self.spectator = None
        self.collision_sensor = None
        self.lane_sensor = None
        
        # First spawn the vehicle
        self.spawn_vehicle()
        
        # Then setup visualization that depends on the vehicle
        self.setup_visualization()
        
        # For plotting
        self.time_history = []
        self.state_history = []
        self.control_history = []
    
    def spawn_vehicle(self):
        """Spawn the vehicle first"""
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = spawn_points[0]
        
        vehicle_bp = self.blueprint_library.find('vehicle.tesla.model3')
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.vehicle.set_autopilot(False)
        
        # Initialize spectator
        self.spectator = self.world.get_spectator()
        
        # Add RGB camera
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.vehicle
        )
    
    def setup_visualization(self):
        """Setup visualization after vehicle exists"""
        try:
            # Initialize pygame
            pygame.init()
            self.display = pygame.display.set_mode(
                (800, 600),
                pygame.HWSURFACE | pygame.DOUBLEBUF
            )
            pygame.display.set_caption("CARLA MPC Control Visualization")
            
            # Add collision sensor
            collision_bp = self.blueprint_library.find('sensor.other.collision')
            self.collision_sensor = self.world.spawn_actor(
                collision_bp,
                carla.Transform(),
                attach_to=self.vehicle
            )
            self.collision_sensor.listen(lambda event: print(f"Collision with {event.other_actor.type_id}"))
            
            # Add lane invasion sensor
            lane_bp = self.blueprint_library.find('sensor.other.lane_invasion')
            self.lane_sensor = self.world.spawn_actor(
                lane_bp,
                carla.Transform(),
                attach_to=self.vehicle
            )
            self.lane_sensor.listen(lambda event: print("Lane invasion detected"))
            
        except Exception as e:
            print(f"Visualization setup error: {e}")
            self.cleanup()
            raise
    
    def update_visualization(self):
        """Update the visualization"""
        if not self.vehicle:
            return False
            
        try:
            # Update spectator view
            transform = self.vehicle.get_transform()
            self.spectator.set_transform(carla.Transform(
                transform.location + carla.Location(z=50),
                carla.Rotation(pitch=-90)
            ))
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
            
            return True
            
        except Exception as e:
            print(f"Visualization update error: {e}")
            return False
        
    def add_camera(self):
        # Add a RGB camera
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        
    def get_vehicle_state(self):
        # Get current vehicle state
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        angular_velocity = self.vehicle.get_angular_velocity()
        
        # Convert to our state representation [β, ψ_dot, ψ, y]
        vx = np.sqrt(velocity.x**2 + velocity.y**2)
        vy = velocity.y
        beta = np.arctan2(vy, vx) if vx > 0.1 else 0
        psi_dot = angular_velocity.z
        psi = np.deg2rad(transform.rotation.yaw)
        y = transform.location.y  # assuming global y is lateral direction
        
        return np.array([beta, psi_dot, psi, y])
    
    def apply_control(self, steer):
        # Create and apply control command
        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = 0.3  # maintain some speed
        self.vehicle.apply_control(control)
        
    def run_simulation(self, controller, duration=30.0):
        start_time = time.time()
        current_time = 0.0
        
        # Reference state (straight line driving)
        x_ref = np.array([0.0, 0.0, 0.0, 0.0])
        
        try:
            while current_time < duration:
                # Update CARLA world
                self.world.tick()
                
                # Get current state
                x0 = self.get_vehicle_state()
                
                # Solve MPC
                steer, _ = controller.solve_mpc(x0, x_ref)
                
                # Apply control
                self.apply_control(steer)
                
                # Update visualization
                if not self.update_visualization():
                    break
                
                # Record data
                self.time_history.append(current_time)
                self.state_history.append(x0)
                self.control_history.append(steer)
                
                # Wait for next time step (handled by CARLA's synchronous mode)
                current_time = time.time() - start_time
                
        finally:
            # Ensure we disable synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            
    def plot_results(self):
        # Convert to numpy arrays
        time_array = np.array(self.time_history)
        state_array = np.array(self.state_history)
        control_array = np.array(self.control_history)
        
        # Create plots
        plt.figure(figsize=(12, 8))
        
        # Plot states
        plt.subplot(3, 1, 1)
        plt.plot(time_array, state_array[:, 0], label='β (sideslip) [rad]')
        plt.plot(time_array, state_array[:, 1], label='ψ_dot (yaw rate) [rad/s]')
        plt.ylabel('States')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(time_array, state_array[:, 2], label='ψ (yaw angle) [rad]')
        plt.plot(time_array, state_array[:, 3], label='y (lateral disp) [m]')
        plt.ylabel('States')
        plt.legend()
        plt.grid(True)
        
        # Plot control
        plt.subplot(3, 1, 3)
        plt.plot(time_array, np.rad2deg(control_array), label='δ (steering) [deg]')
        plt.xlabel('Time [s]')
        plt.ylabel('Control Input')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def cleanup(self):
        """Proper cleanup of all actors"""
        if hasattr(self, 'lane_sensor') and self.lane_sensor:
            self.lane_sensor.destroy()
        if hasattr(self, 'collision_sensor') and self.collision_sensor:
            self.collision_sensor.destroy()
        if hasattr(self, 'camera') and self.camera:
            self.camera.destroy()
        if hasattr(self, 'vehicle') and self.vehicle:
            self.vehicle.destroy()
        if pygame.get_init():
            pygame.quit()

# Main function
def main():
    try:
        # Initialize parameters
        vehicle_params = VehicleParams()
        mpc_params = MPCParams()
        
        # Create controller
        controller = MPCController(vehicle_params, mpc_params)
        
        # Connect to Carla and run simulation
        carla_interface = CarlaInterface(mpc_params)
        carla_interface.run_simulation(controller, duration=20.0)
        
        # Plot results
        carla_interface.plot_results()
        
    except KeyboardInterrupt:
        print("Simulation stopped by user")
    finally:
        # Clean up
        if 'carla_interface' in locals():
            carla_interface.vehicle.destroy()
            carla_interface.camera.destroy()
            carla_interface.collision_sensor.destroy()
            carla_interface.lane_sensor.destroy()
            pygame.quit()
        print("Simulation ended.")

if __name__ == '__main__':
    main()