import carla
import numpy as np
from casadi import *
import random
import time
import math

class VehicleParams:
    def __init__(self):
        self.m = 1700.0       # mass [kg]
        self.lf = 1.3         # distance from CG to front axle [m]
        self.lr = 1.7         # distance from CG to rear axle [m]
        self.Iz = 3500.0      # yaw moment of inertia [kg*m^2]
        self.Cf = 80000.0     # reduced cornering stiffness for stability [N/rad]
        self.Cr = 80000.0     # reduced cornering stiffness for stability [N/rad]
        self.vx = 5.0         # reduced speed for better control [m/s]

class MPCParams:
    def __init__(self):
        self.N = 10           # prediction horizon
        self.dt = 0.1         # time step [s]
        self.max_iter = 100   # maximum iterations
        self.max_steer = np.deg2rad(3)  # conservative steering limit
        self.steer_smoothing = 0.7      # stronger smoothing

class MPCController:
    def __init__(self, vehicle_params, mpc_params):
        self.vehicle_params = vehicle_params
        self.mpc_params = mpc_params
        self.model = self.create_vehicle_model()
        self.setup_mpc()
        self.last_steer = 0.0
        self.last_state = np.zeros(4)

    def create_vehicle_model(self):
        x = SX.sym('x', 4)  # [β, ψ_dot, ψ, y]
        u = SX.sym('u', 1)  # δ
        
        params = self.vehicle_params
        beta = x[0]
        psi_dot = x[1]
        psi = x[2]
        y = x[3]
        delta = u[0]
        
        # Safe vehicle dynamics with numerical stability
        vx_safe = max(0.1, params.vx)  # Prevent division by zero
        
        # Bicycle model with limited inputs
        a11 = -(params.Cf + params.Cr)/(params.m * vx_safe)
        a12 = -1 + (params.lr*params.Cr - params.lf*params.Cf)/(params.m * vx_safe**2)
        a21 = (params.lr*params.Cr - params.lf*params.Cf)/params.Iz
        a22 = -(params.lf**2 * params.Cf + params.lr**2 * params.Cr)/(params.Iz * vx_safe)
        
        b1 = params.Cf/(params.m * vx_safe)
        b2 = params.lf*params.Cf/params.Iz
        
        beta_dot = a11*beta + a12*psi_dot + b1*delta
        psi_ddot = a21*beta + a22*psi_dot + b2*delta
        
        # Safe lateral dynamics
        y_dot = vx_safe * (sin(psi) + beta * cos(psi))
        
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
        
        # Well-conditioned weights
        Q = np.diag([2.0, 0.5, 5.0, 10.0])  # Reduced weights for stability
        R = np.diag([5.0])                  # Higher control penalty
        Q_terminal = np.diag([5.0, 1.0, 10.0, 20.0])
        
        x_ref = P[4:8]
        
        for k in range(N):
            x = X[:, k]
            u = U[:, k]
            
            # State cost with input protection
            state_diff = x - x_ref
            state_diff[0] = if_else(fabs(x[0]) > 0.5, 0, state_diff[0])  # Limit beta influence
            state_diff[3] = if_else(fabs(x[3]) > 10.0, 0, state_diff[3]) # Limit y influence
            
            obj += mtimes([state_diff.T, Q, state_diff]) + mtimes([u.T, R, u])
            
            # Strong steering rate penalty
            if k > 0:
                obj += 10.0 * (U[:, k] - U[:, k-1])**2
                
            # Dynamics constraint with protection
            x_next = X[:, k+1]
            xdot = self.model(x, u)
            x_next_pred = x + xdot * dt
            g.append(x_next - x_next_pred)
            
        obj += mtimes([(X[:, N] - x_ref).T, Q_terminal, (X[:, N] - x_ref)])
        
        nlp = {'x': vertcat(reshape(X, -1, 1), reshape(U, -1, 1)),
               'f': obj,
               'g': vertcat(*g),
               'p': P}
        
        opts = {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0,
            'ipopt.max_iter': self.mpc_params.max_iter,
            'ipopt.tol': 1e-3  # Looser tolerance for stability
        }
        
        self.solver = nlpsol('solver', 'ipopt', nlp, opts)
        
        # Conservative bounds
        self.lbx = [-0.5, -1.0, -np.pi, -10.0]*(N+1) + [-self.mpc_params.max_steer]*N
        self.ubx = [0.5, 1.0, np.pi, 10.0]*(N+1) + [self.mpc_params.max_steer]*N
        
        # Constraint bounds
        self.lbg = [-0.1, -0.1, -0.1, -0.1]*(N+1)  # Looser constraints
        self.ubg = [0.1, 0.1, 0.1, 0.1]*(N+1)

    def solve_mpc(self, x0, x_ref):
        N = self.mpc_params.N
        
        # State filtering and protection
        x0_safe = np.nan_to_num(x0, nan=0.0, posinf=0.5, neginf=-0.5)
        x_ref_safe = np.nan_to_num(x_ref, nan=0.0, posinf=0.5, neginf=-0.5)
        
        # Apply low-pass filter to state
        x0_filtered = 0.3 * self.last_state + 0.7 * x0_safe
        self.last_state = x0_filtered
        
        # Initial guess
        x_init = np.tile(x0_filtered, N+1)
        u_init = np.ones(N) * self.last_steer
        vars_init = np.concatenate([x_init, u_init])
        p = np.concatenate([x0_filtered, x_ref_safe])
        
        # Solve NLP
        try:
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
            
            # Strong smoothing
            smoothed_steer = self.mpc_params.steer_smoothing * self.last_steer + \
                            (1 - self.mpc_params.steer_smoothing) * u_opt[0]
            self.last_steer = smoothed_steer
            
            return smoothed_steer
            
        except Exception as e:
            print(f"MPC solve error: {e}, using last steering")
            return self.last_steer

class CarlaInterface:
    def __init__(self, mpc_params, vehicle_params):
        self.mpc_params = mpc_params
        self.vehicle_params = vehicle_params
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
            self.waypoints = []
            self.last_update_time = 0
            self.should_stop = False
            
        except Exception as e:
            print(f"Initialization failed: {e}")
            self.cleanup()
            raise

    def spawn_vehicle_on_straight_road(self):
        """Find a long straight highway segment"""
        spawn_points = self.map.get_spawn_points()
        
        # Filter points on straight roads
        best_spawn = None
        max_dist = 0
        
        for sp in spawn_points:
            wp = self.map.get_waypoint(sp.location)
            dist = self.check_straight_path(wp, distance=100.0)
            if dist > max_dist:
                max_dist = dist
                best_spawn = sp
        
        if not best_spawn:
            best_spawn = random.choice(spawn_points)
        
        # Spawn with collision avoidance
        vehicle_bp = self.blueprint_library.find('vehicle.tesla.model3')
        spawn_transform = carla.Transform(
            best_spawn.location + carla.Location(z=0.5),
            best_spawn.rotation
        )
        
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
        if not self.vehicle:
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_transform)
        
        self.vehicle.set_autopilot(False)
        self.generate_waypoints_ahead(100.0)
        
        # Set spectator
        self.spectator = self.world.get_spectator()
        self.update_spectator_view()

    def check_straight_path(self, waypoint, distance=100.0):
        """Check path straightness ahead"""
        current = waypoint
        total_distance = 0
        while total_distance < distance:
            next_wps = current.next(10.0)
            if not next_wps:
                break
                
            angle_diff = abs(current.transform.rotation.yaw - next_wps[0].transform.rotation.yaw)
            if angle_diff > 2.0:  # Very strict straightness check
                break
                
            current = next_wps[0]
            total_distance += 10.0
            
            if not self.is_location_clear(current.transform.location):
                break
                
        return total_distance

    def is_location_clear(self, location, radius=5.0):
        """Check for nearby static objects"""
        for actor in self.world.get_actors():
            if 'static' in actor.type_id and location.distance(actor.get_location()) < radius:
                return False
        return True

    def generate_waypoints_ahead(self, distance):
        """Generate smooth reference path"""
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

    def update_waypoints(self):
        """Maintain sufficient lookahead"""
        if time.time() - self.last_update_time < 0.5:  # Update every 0.5s
            return
            
        self.last_update_time = time.time()
        vehicle_loc = self.vehicle.get_location()
        
        # Remove passed waypoints
        self.waypoints = [wp for wp in self.waypoints 
                         if wp.transform.location.distance(vehicle_loc) > 5.0]
        
        # Extend path if needed
        if len(self.waypoints) < 15:
            last_wp = self.waypoints[-1] if self.waypoints else self.map.get_waypoint(vehicle_loc)
            new_wps = last_wp.next(50.0)
            if new_wps:
                self.waypoints.extend(new_wps[:10])

    def get_reference_state(self):
        """Get safe reference state"""
        if not self.waypoints:
            return np.array([0.0, 0.0, 0.0, 0.0])
        
        # Find lookahead point (adaptive based on speed)
        speed = self.vehicle.get_velocity().length()
        lookahead_dist = min(10.0, max(3.0, speed * 0.8))
        lookahead_idx = min(int(lookahead_dist / 5.0), len(self.waypoints)-1)
        target_wp = self.waypoints[lookahead_idx]
        
        desired_yaw = math.radians(target_wp.transform.rotation.yaw)
        desired_y = target_wp.transform.location.y
        
        return np.array([0.0, 0.0, desired_yaw, desired_y])

    def update_spectator_view(self):
        """Smooth following camera"""
        if not hasattr(self, 'vehicle') or not self.vehicle:
            return
            
        transform = self.vehicle.get_transform()
        speed = self.vehicle.get_velocity().length()
        
        # Camera parameters
        distance = 8.0 + speed * 0.3
        height = 3.0 + speed * 0.1
        pitch = -15 - speed * 0.5
        
        camera_pos = transform.location + carla.Location(
            x=-distance,
            z=height
        )
        
        rotation = carla.Rotation(
            pitch=pitch,
            yaw=transform.rotation.yaw
        )
        
        self.spectator.set_transform(carla.Transform(camera_pos, rotation))

    def setup_visualization(self):
        """Setup sensors and debug tools"""
        # Collision sensor
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.collision_sensor.listen(self.handle_collision)
        
        # Debug drawing
        self.debug = self.world.debug

    def get_vehicle_state(self):
        """Robust state estimation"""
        try:
            transform = self.vehicle.get_transform()
            velocity = self.vehicle.get_velocity()
            angular_velocity = self.vehicle.get_angular_velocity()
            
            vx = math.sqrt(velocity.x**2 + velocity.y**2)
            vy = velocity.y
            
            # Filter noisy measurements
            if vx < 0.1:
                beta = 0.0
            else:
                beta = math.atan2(vy, vx)
                beta = np.clip(beta, -0.5, 0.5)  # Limit sideslip
            
            psi_dot = angular_velocity.z
            psi_dot = np.clip(psi_dot, -1.0, 1.0)  # Limit yaw rate
            
            psi = math.radians(transform.rotation.yaw % 360)
            y = transform.location.y
            y = np.clip(y, -10.0, 10.0)  # Limit lateral deviation
            
            return np.array([beta, psi_dot, psi, y])
            
        except Exception as e:
            print(f"State estimation error: {e}")
            return np.array([0.0, 0.0, 0.0, 0.0])

    def handle_collision(self, event):
        """Collision recovery system"""
        self.collision_count += 1
        print(f"Collision #{self.collision_count} with {event.other_actor.type_id}")
        
        if self.collision_count >= self.max_collisions:
            print("Max collisions reached, stopping simulation")
            self.should_stop = True
            return
            
        # Find safe recovery location
        current_loc = self.vehicle.get_location()
        recovery_loc = self.find_recovery_location(current_loc)
        
        # Reset vehicle
        self.vehicle.set_transform(carla.Transform(
            recovery_loc,
            self.vehicle.get_transform().rotation
        ))
        
        # Full stop
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 1.0
        control.steer = 0.0
        self.vehicle.apply_control(control)
        
        # Regenerate path
        self.generate_waypoints_ahead(100.0)
        time.sleep(1.0)  # Recovery pause

    def find_recovery_location(self, location):
        """Find nearest safe location"""
        waypoints = self.map.generate_waypoints(5.0)
        nearest = sorted(waypoints, key=lambda wp: wp.transform.location.distance(location))
        
        for wp in nearest[:20]:
            if self.is_location_clear(wp.transform.location, radius=10.0):
                return wp.transform.location + carla.Location(z=0.5)
        
        # Fallback - lift vehicle
        return location + carla.Location(z=1.0)

    def apply_control(self, steer):
        """Safe control application"""
        if not hasattr(self, 'vehicle') or not self.vehicle:
            return
            
        # Clip and filter steering
        steer = np.clip(steer, -self.mpc_params.max_steer, self.mpc_params.max_steer)
        
        # Adaptive throttle based on speed error
        current_speed = self.vehicle.get_velocity().length()
        target_speed = self.vehicle_params.vx
        speed_error = target_speed - current_speed
        
        throttle = 0.15 + 0.005 * speed_error
        throttle = np.clip(throttle, 0.1, 0.2)  # Conservative throttle
        
        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = 0.0
        
        self.vehicle.apply_control(control)

    def run_simulation(self, controller, duration=60.0):
        self.should_stop = False
        start_time = time.time()
        
        try:
            while (time.time() - start_time) < duration and not self.should_stop:
                self.world.tick()
                
                # Update path
                self.update_waypoints()
                
                # Get states
                x0 = self.get_vehicle_state()
                x_ref = self.get_reference_state()
                
                # MPC control
                steer = controller.solve_mpc(x0, x_ref)
                self.apply_control(steer)
                
                # Visualization
                self.update_spectator_view()
                self.draw_debug_info()
                
                time.sleep(0.01)
                
        except Exception as e:
            print(f"Simulation error: {e}")
        finally:
            self.cleanup()

    def draw_debug_info(self):
        """Visualize important information"""
        if not hasattr(self, 'vehicle') or not hasattr(self, 'waypoints'):
            return
            
        # Draw path
        for i in range(len(self.waypoints)-1):
            self.debug.draw_line(
                self.waypoints[i].transform.location + carla.Location(z=0.5),
                self.waypoints[i+1].transform.location + carla.Location(z=0.5),
                thickness=0.1,
                color=carla.Color(0, 255, 0),
                life_time=0.2
            )
        
        # Vehicle info
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        speed = 3.6 * velocity.length()  # km/h
        
        self.debug.draw_string(
            transform.location + carla.Location(z=5),
            f"Speed: {speed:.1f} km/h\nSteering: {math.degrees(self.vehicle.get_control().steer):.1f}°\nCollisions: {self.collision_count}",
            False,
            color=carla.Color(255, 255, 255),
            life_time=0.1
        )

    def cleanup(self):
        """Proper cleanup"""
        actors = ['vehicle', 'collision_sensor']
        for name in actors:
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
        carla_interface = CarlaInterface(mpc_params, vehicle_params)
        
        print("Starting robust MPC simulation...")
        print("Vehicle parameters:", vars(vehicle_params))
        print("MPC parameters:", vars(mpc_params))
        
        carla_interface.run_simulation(controller, duration=60.0)
        
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