import carla
import numpy as np
from casadi import *
import matplotlib.pyplot as plt
import time
import random
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
        Q = np.diag([10.0, 5.0, 20.0, 50.0])  # state weights
        R = np.diag([5.0])                  # control weights
        
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

class CarlaInterface:
    def __init__(self, mpc_params):
        self.mpc_params = mpc_params
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Set synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.mpc_params.dt
        self.world.apply_settings(settings)
        
        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_vehicle_on_straight_road()  # New method for better spawning
        self.setup_visualization()
        
        # For collision recovery
        self.collision_count = 0
        self.max_collisions = 3

    def spawn_vehicle_on_straight_road(self):
        """Find a straight road segment to spawn the vehicle"""
        map = self.world.get_map()
        spawn_points = map.get_spawn_points()
        
        # Filter spawn points on straight roads
        straight_spawns = []
        for sp in spawn_points:
            # Check if the waypoint is on a straight segment
            wp = map.get_waypoint(sp.location)
            if abs(wp.transform.rotation.yaw - sp.rotation.yaw) < 5:  # Nearly straight
                straight_spawns.append(sp)
        
        if not straight_spawns:
            print("Warning: No straight spawn points found, using default")
            straight_spawns = spawn_points
        
        # Choose a random straight spawn point
        spawn_point = random.choice(straight_spawns)
        
        # Spawn vehicle
        vehicle_bp = self.blueprint_library.find('vehicle.tesla.model3')
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.vehicle.set_autopilot(False)
        
        # Set spectator
        self.spectator = self.world.get_spectator()
        self.update_spectator_view()

    def update_spectator_view(self):
        """Update spectator to follow vehicle from a good angle"""
        if not hasattr(self, 'vehicle') or not self.vehicle:
            return
            
        transform = self.vehicle.get_transform()
        # Third-person view from behind
        self.spectator.set_transform(carla.Transform(
            transform.location + carla.Location(x=-10, z=3),
            carla.Rotation(yaw=180, pitch=-15)
        ))

    def handle_collision(self, event):
        """Handle collision events and attempt recovery"""
        self.collision_count += 1
        print(f"Collision #{self.collision_count} with {event.other_actor.type_id}")
        
        if self.collision_count >= self.max_collisions:
            print("Max collisions reached, stopping simulation")
            self.should_stop = True
            return
            
        # Attempt recovery by resetting position
        current_loc = self.vehicle.get_location()
        self.vehicle.set_location(carla.Location(
            x=current_loc.x,
            y=current_loc.y,
            z=current_loc.z + 0.5  # Lift slightly to avoid getting stuck
        ))
        self.vehicle.set_velocity(carla.Vector3D(0, 0, 0))
        self.vehicle.set_angular_velocity(carla.Vector3D(0, 0, 0))

    def run_simulation(self, controller, duration=30.0):
        self.should_stop = False
        start_time = time.time()
        
        # Reference state (straight line driving)
        x_ref = np.array([0.0, 0.0, 0.0, 0.0])
        
        # Add collision sensor
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.collision_sensor.listen(self.handle_collision)
        
        try:
            while (time.time() - start_time) < duration and not self.should_stop:
                self.world.tick()
                
                # Get state
                x0 = self.get_vehicle_state()
                
                # Solve MPC with adjusted reference if needed
                steer, _ = controller.solve_mpc(x0, x_ref)
                
                # Apply control with smoothing
                self.apply_control(steer * 0.8)  # Reduced gain for smoother steering
                
                # Update visualization
                self.update_spectator_view()
                
                # Small delay to prevent overloading
                time.sleep(0.01)
                
        finally:
            self.cleanup()

    def apply_control(self, steer):
        """Apply control with additional smoothing"""
        control = carla.VehicleControl()
        control.steer = np.clip(steer, 
                               -self.mpc_params.max_steer, 
                               self.mpc_params.max_steer)
        control.throttle = 0.3
        control.brake = 0.0
        self.vehicle.apply_control(control)

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
    

    def cleanup(self):
        """Clean up all actors"""
        actors = [
            getattr(self, attr) for attr in ['vehicle', 'camera', 
                                           'collision_sensor', 'lane_sensor']
            if hasattr(self, attr)
        ]
        for actor in actors:
            if actor and actor.is_alive:
                actor.destroy()
        
        # Disable synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)

def main():
    try:
        vehicle_params = VehicleParams()
        mpc_params = MPCParams()
        
        controller = MPCController(vehicle_params, mpc_params)
        carla_interface = CarlaInterface(mpc_params)
        
        print("Starting straight-line driving simulation...")
        carla_interface.run_simulation(controller, duration=30.0)
        
    except KeyboardInterrupt:
        print("Simulation stopped by user")
    finally:
        if 'carla_interface' in locals():
            carla_interface.cleanup()
        print("Simulation ended")

if __name__ == '__main__':
    main()