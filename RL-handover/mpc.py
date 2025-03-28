import carla
import numpy as np
from casadi import *
import random
import time

class VehicleParams:
    def __init__(self):
        self.m = 1500.0       # mass [kg]
        self.lf = 1.2         # distance from CG to front axle [m]
        self.lr = 1.8         # distance from CG to rear axle [m]
        self.Iz = 3000.0      # yaw moment of inertia [kg*m^2]
        self.Cf = 100000.0    # increased cornering stiffness [N/rad]
        self.Cr = 100000.0    # increased cornering stiffness [N/rad]
        self.vx = 10.0        # reduced speed for better control [m/s]

class MPCParams:
    def __init__(self):
        self.N = 10           # prediction horizon
        self.dt = 0.1         # time step [s]
        self.max_iter = 100   # maximum iterations
        self.max_steer = np.deg2rad(3)  # very conservative steering limit

class MPCController:
    def __init__(self, vehicle_params, mpc_params):
        self.vehicle_params = vehicle_params
        self.mpc_params = mpc_params
        self.model = self.create_vehicle_model()
        self.setup_mpc()

    def create_vehicle_model(self):
        x = SX.sym('x', 4)  # [β, ψ_dot, ψ, y]
        u = SX.sym('u', 1)  # δ
        
        params = self.vehicle_params
        beta = x[0]
        psi_dot = x[1]
        psi = x[2]
        y = x[3]
        delta = u[0]
        
        # Vehicle dynamics
        beta_dot = (2/(params.m*params.vx)) * (
            params.Cf*(delta - beta - (params.lf*psi_dot)/params.vx) + 
            params.Cr*(-beta + (params.lr*psi_dot)/params.vx)
        ) - psi_dot
        
        psi_ddot = (2/params.Iz) * (
            params.lf*params.Cf*(delta - beta - (params.lf*psi_dot)/params.vx) - 
            params.lr*params.Cr*(-beta + (params.lr*psi_dot)/params.vx)
        )
        
        y_dot = params.vx * np.sin(psi + beta)
        
        xdot = vertcat(beta_dot, psi_ddot, psi_dot, y_dot)
        return Function('f', [x, u], [xdot], ['x', 'u'], ['xdot'])

    def setup_mpc(self):
        N = self.mpc_params.N
        dt = self.mpc_params.dt
        
        X = SX.sym('X', 4, N+1)
        U = SX.sym('U', 1, N)
        P = SX.sym('P', 8)  # x0 + x_ref
        
        obj = 0
        g = []
        g.append(X[:, 0] - P[0:4])
        
        Q = np.diag([10.0, 5.0, 20.0, 50.0])
        R = np.diag([10.0])
        x_ref = P[4:8]
        
        for k in range(N):
            x = X[:, k]
            u = U[:, k]
            obj += mtimes([(x - x_ref).T, Q, (x - x_ref)]) + mtimes([u.T, R, u])
            
            x_next = X[:, k+1]
            xdot = self.model(x, u)
            x_next_pred = x + xdot * dt
            g.append(x_next - x_next_pred)
            
        obj += 10 * mtimes([(X[:, N] - x_ref).T, Q, (X[:, N] - x_ref)])
        
        nlp = {'x': vertcat(reshape(X, -1, 1), reshape(U, -1, 1)),
               'f': obj,
               'g': vertcat(*g),
               'p': P}
        
        opts = {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0,
            'ipopt.max_iter': self.mpc_params.max_iter
        }
        
        self.solver = nlpsol('solver', 'ipopt', nlp, opts)
        self.lbx = [-inf]*(4*(N+1)) + [-self.mpc_params.max_steer]*N
        self.ubx = [inf]*(4*(N+1)) + [self.mpc_params.max_steer]*N
        self.lbg = [0]*(4 + 4*N)
        self.ubg = [0]*(4 + 4*N)

    def solve_mpc(self, x0, x_ref):
        N = self.mpc_params.N
        x_init = np.tile(x0, N+1)
        u_init = np.zeros(N)
        vars_init = np.concatenate([x_init, u_init])
        p = np.concatenate([x0, x_ref])
        
        sol = self.solver(
            x0=vars_init,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=self.lbg,
            ubg=self.ubg,
            p=p
        )
        
        vars_opt = sol['x'].full().flatten()
        u_opt = vars_opt[4*(N+1):]
        return u_opt[0]

class CarlaInterface:
    def __init__(self, mpc_params):
        self.mpc_params = mpc_params
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(20.0)
        
        try:
            self.world = self.client.get_world()
            self.map = self.world.get_map()
            
            # Set synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.mpc_params.dt
            self.world.apply_settings(settings)
            
            self.blueprint_library = self.world.get_blueprint_library()
            self.spawn_vehicle_on_straight_road()
            self.setup_visualization()
            
            self.collision_count = 0
            self.max_collisions = 2
            self.last_steer = 0.0
            self.waypoints = []
            
        except Exception as e:
            print(f"Initialization failed: {e}")
            self.cleanup()
            raise

    def spawn_vehicle_on_straight_road(self):
        """Find a long straight road segment"""
        spawn_points = self.map.get_spawn_points()
        
        # Prefer highway spawn points
        highway_spawns = [sp for sp in spawn_points 
                         if self.map.get_waypoint(sp.location).is_junction == False]
        
        if not highway_spawns:
            highway_spawns = spawn_points
        
        # Find spawn point with straight path ahead
        best_spawn = None
        max_straight_distance = 0
        
        for sp in highway_spawns:
            wp = self.map.get_waypoint(sp.location)
            straight_dist = self.check_straight_path(wp, distance=100.0)
            if straight_dist > max_straight_distance:
                max_straight_distance = straight_dist
                best_spawn = sp
        
        if not best_spawn:
            best_spawn = random.choice(highway_spawns)
        
        # Spawn vehicle without physics control attribute
        vehicle_bp = self.blueprint_library.find('vehicle.tesla.model3')
        self.vehicle = self.world.spawn_actor(vehicle_bp, best_spawn)
        self.vehicle.set_autopilot(False)
        
        # Generate waypoints
        self.generate_waypoints_ahead(150.0)
        
        # Set spectator
        self.spectator = self.world.get_spectator()
        self.update_spectator_view()

    def check_straight_path(self, waypoint, distance=50.0):
        """Check how straight the road is ahead"""
        current = waypoint
        total_distance = 0
        while total_distance < distance:
            next_wps = current.next(5.0)
            if not next_wps:
                break
            current = next_wps[0]
            total_distance += 5.0
        return total_distance

    def generate_waypoints_ahead(self, distance):
        """Create reference path using waypoints"""
        current_loc = self.vehicle.get_location()
        current_wp = self.map.get_waypoint(current_loc)
        
        self.waypoints = []
        wp = current_wp
        dist = 0
        
        while dist < distance:
            self.waypoints.append(wp)
            next_wps = wp.next(5.0)
            if not next_wps:
                break
            wp = next_wps[0]
            dist += 5.0

    def get_reference_state(self):
        """Get reference state from waypoints"""
        if not self.waypoints:
            return np.array([0.0, 0.0, 0.0, 0.0])
        
        # Find closest waypoint
        vehicle_loc = self.vehicle.get_location()
        closest_wp = min(self.waypoints, 
                        key=lambda wp: wp.transform.location.distance(vehicle_loc))
        
        # Calculate desired yaw and lateral position
        desired_yaw = math.radians(closest_wp.transform.rotation.yaw)
        desired_y = closest_wp.transform.location.y
        
        return np.array([0.0, 0.0, desired_yaw, desired_y])

    def update_spectator_view(self):
        """Third-person chase view"""
        if not hasattr(self, 'vehicle') or not self.vehicle:
            return
            
        transform = self.vehicle.get_transform()
        self.spectator.set_transform(carla.Transform(
            transform.location + carla.Location(x=-15, z=5),
            carla.Rotation(yaw=180, pitch=-20)
        ))

    def setup_visualization(self):
        # Add camera
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.vehicle
        )

    def get_vehicle_state(self):
        """Get current vehicle state: [β, ψ_dot, ψ, y]"""
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        angular_velocity = self.vehicle.get_angular_velocity()
        
        vx = np.sqrt(velocity.x**2 + velocity.y**2)
        vy = velocity.y
        beta = np.arctan2(vy, vx) if vx > 0.1 else 0
        psi_dot = angular_velocity.z
        psi = math.radians(transform.rotation.yaw)
        y = transform.location.y
        
        return np.array([beta, psi_dot, psi, y])

    def handle_collision(self, event):
        """Improved collision recovery"""
        self.collision_count += 1
        print(f"Collision #{self.collision_count} with {event.other_actor.type_id}")
        
        if self.collision_count >= self.max_collisions:
            print("Max collisions reached, stopping simulation")
            self.should_stop = True
            return
            
        # Reset vehicle state
        current_transform = self.vehicle.get_transform()
        reset_location = carla.Location(
            x=current_transform.location.x,
            y=current_transform.location.y,
            z=current_transform.location.z + 0.5
        )
        
        reset_transform = carla.Transform(
            reset_location,
            current_transform.rotation
        )
        
        self.vehicle.set_transform(reset_transform)
        
        # Stop the vehicle
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 1.0
        control.steer = 0.0
        self.vehicle.apply_control(control)
        
        # Regenerate waypoints
        self.generate_waypoints_ahead(100.0)
        time.sleep(1.0)  # Pause after collision

    def apply_control(self, steer):
        """Smooth, limited control application"""
        # Low-pass filter for steering
        smoothed_steer = 0.7 * self.last_steer + 0.3 * steer
        self.last_steer = smoothed_steer
        
        # Create and apply control
        control = carla.VehicleControl()
        control.steer = np.clip(smoothed_steer, 
                               -self.mpc_params.max_steer, 
                               self.mpc_params.max_steer)
        control.throttle = 0.3
        control.brake = 0.0
        self.vehicle.apply_control(control)

    def run_simulation(self, controller, duration=30.0):
        self.should_stop = False
        start_time = time.time()
        
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
                
                # Get state and reference
                x0 = self.get_vehicle_state()
                x_ref = self.get_reference_state()
                
                # Solve MPC
                steer = controller.solve_mpc(x0, x_ref)
                self.apply_control(steer)
                
                # Visual updates
                self.update_spectator_view()
                self.debug_draw_waypoints()
                
                time.sleep(0.01)
                
        except Exception as e:
            print(f"Simulation error: {e}")
        finally:
            self.cleanup()

    def debug_draw_waypoints(self):
        """Visualize the reference path"""
        debug = self.world.debug
        for i in range(len(self.waypoints)-1):
            debug.draw_line(
                self.waypoints[i].transform.location + carla.Location(z=0.5),
                self.waypoints[i+1].transform.location + carla.Location(z=0.5),
                thickness=0.1,
                color=carla.Color(0, 255, 0),
                life_time=1.0/self.mpc_params.dt
            )

    def cleanup(self):
        """Proper cleanup of all actors"""
        actor_names = ['vehicle', 'camera', 'collision_sensor']
        for name in actor_names:
            if hasattr(self, name):
                actor = getattr(self, name)
                if actor and actor.is_alive:
                    actor.destroy()
        
        # Reset synchronous mode
        if hasattr(self, 'world'):
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

def main():
    try:
        vehicle_params = VehicleParams()
        mpc_params = MPCParams()
        
        controller = MPCController(vehicle_params, mpc_params)
        carla_interface = CarlaInterface(mpc_params)
        
        print("Starting stable MPC simulation...")
        carla_interface.run_simulation(controller, duration=30.0)
        
    except KeyboardInterrupt:
        print("Simulation stopped by user")
    except Exception as e:
        print(f"Main error: {e}")
    finally:
        if 'carla_interface' in locals():
            carla_interface.cleanup()
        print("Simulation ended")

if __name__ == '__main__':
    main()