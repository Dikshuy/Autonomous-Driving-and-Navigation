import carla
import numpy as np
from casadi import *
import random
import time
import math

class VehicleParams:
    def __init__(self):
        self.m = 1500.0       # mass [kg]
        self.lf = 1.2         # distance from CG to front axle [m]
        self.lr = 1.8         # distance from CG to rear axle [m]
        self.Iz = 3000.0      # yaw moment of inertia [kg*m^2]
        self.Cf = 70000.0     # front cornering stiffness [N/rad] - adjusted for better handling
        self.Cr = 80000.0     # rear cornering stiffness [N/rad] - adjusted for better handling
        self.vx = 8.0         # longitudinal velocity [m/s] - reduced for better control

class MPCParams:
    def __init__(self):
        self.N = 15           # increased prediction horizon for better anticipation
        self.dt = 0.1         # time step [s]
        self.max_iter = 100   # maximum iterations
        self.max_steer = np.deg2rad(25)  # more realistic steering limit [rad]
        self.steer_rate_limit = np.deg2rad(30) * 0.1  # steering rate limit per time step

class MPCController:
    def __init__(self, vehicle_params, mpc_params):
        self.vehicle_params = vehicle_params
        self.mpc_params = mpc_params
        self.model = self.create_vehicle_model()
        self.setup_mpc()
        self.prev_steer = 0.0  # Store previous steering command for rate limiting

    def create_vehicle_model(self):
        # State: [β (sideslip), ψ_dot (yaw rate), ψ (yaw), y (lateral position), e_y_dot (lateral error derivative)]
        x = SX.sym('x', 5)
        # Control input: steering angle δ
        u = SX.sym('u', 1)
        
        params = self.vehicle_params
        beta = x[0]
        psi_dot = x[1]
        psi = x[2]
        y = x[3]
        e_y_dot = x[4]  # Added lateral error derivative for better dynamics
        delta = u[0]
        
        # Improved vehicle dynamics
        beta_dot = (params.Cf*(delta - beta - (params.lf*psi_dot)/params.vx) + 
                    params.Cr*(-beta + (params.lr*psi_dot)/params.vx)) / (params.m*params.vx) - psi_dot
        
        psi_ddot = (params.lf*params.Cf*(delta - beta - (params.lf*psi_dot)/params.vx) - 
                    params.lr*params.Cr*(-beta + (params.lr*psi_dot)/params.vx)) / params.Iz
        
        y_dot = params.vx * (psi + beta)  # Small angle approximation for better numerical stability
        
        # Lateral error derivative dynamics
        e_y_ddot = params.vx * (psi_dot + beta_dot)
        
        xdot = vertcat(beta_dot, psi_ddot, psi_dot, y_dot, e_y_ddot)
        return Function('f', [x, u], [xdot], ['x', 'u'], ['xdot'])

    def setup_mpc(self):
        N = self.mpc_params.N
        dt = self.mpc_params.dt
        
        # Decision variables for states and controls
        X = SX.sym('X', 5, N+1)  # Updated for 5 states
        U = SX.sym('U', 1, N)
        
        # Parameters (current state and reference trajectory)
        P = SX.sym('P', 5 + 5*N)  # initial state (5) + reference trajectory (5×N)
        
        # Define objective and constraints
        obj = 0
        g = []
        
        # Initial state constraint
        g.append(X[:, 0] - P[0:5])
        
        # Weights for state error
        Q = diag(SX([10.0, 5.0, 20.0, 50.0, 10.0]))  # Added weight for lateral error derivative
        
        # Weights for control input and steering rate
        R_steer = 5.0
        R_steer_rate = 50.0
        
        # Define the objective function with reference trajectory
        for k in range(N):
            # Current state and control
            x_k = X[:, k]
            u_k = U[:, k]
            
            # Reference values for this step (extracted from parameter vector)
            x_ref_k = P[5+5*k:5+5*(k+1)]
            
            # State tracking error cost
            state_cost = mtimes([(x_k - x_ref_k).T, Q, (x_k - x_ref_k)])
            
            # Control input cost
            control_cost = R_steer * u_k**2
            
            # Steering rate cost (if not the first step)
            rate_cost = 0
            if k > 0:
                rate_cost = R_steer_rate * (u_k - U[:, k-1])**2
            
            # Add costs to objective
            obj += state_cost + control_cost + rate_cost
            
            # System dynamics constraint
            x_next = X[:, k+1]
            xdot = self.model(x_k, u_k)
            x_next_pred = x_k + xdot * dt
            g.append(x_next - x_next_pred)
        
        # Terminal cost
        x_ref_N = P[5+5*(N-1):5+5*N]  # Last reference point
        obj += 100 * mtimes([(X[:, N] - x_ref_N).T, Q, (X[:, N] - x_ref_N)])
        
        # Define optimization variables
        opt_vars = vertcat(reshape(X, -1, 1), reshape(U, -1, 1))
        
        # Define NLP problem
        nlp = {'x': opt_vars, 'f': obj, 'g': vertcat(*g), 'p': P}
        
        # Solver options
        opts = {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0,
            'ipopt.max_iter': self.mpc_params.max_iter,
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 1e-3
        }
        
        # Create solver
        self.solver = nlpsol('solver', 'ipopt', nlp, opts)
        
        # Define bounds
        self.lbx = [-np.pi/4] * 5 * (N+1)  # Lower bounds for states
        self.lbx += [-self.mpc_params.max_steer] * N  # Lower bounds for controls
        
        self.ubx = [np.pi/4] * 5 * (N+1)  # Upper bounds for states
        self.ubx += [self.mpc_params.max_steer] * N  # Upper bounds for controls
        
        # Equality constraints bounds (all zeros)
        self.lbg = [0] * (5 + 5*N)
        self.ubg = [0] * (5 + 5*N)

    def solve_mpc(self, x0, reference_trajectory):
        """
        Solve the MPC problem
        
        Args:
            x0: Current state [beta, psi_dot, psi, y, e_y_dot]
            reference_trajectory: Array of reference states for the horizon
        """
        N = self.mpc_params.N
        
        # Ensure reference trajectory has enough points
        if len(reference_trajectory) < N:
            last_ref = reference_trajectory[-1]
            reference_trajectory = np.vstack([reference_trajectory, 
                                             np.tile(last_ref, (N - len(reference_trajectory), 1))])
        
        # Initial guess
        x_init = np.tile(x0, (N+1, 1)).T.flatten()
        u_init = np.zeros(N)
        
        if hasattr(self, 'prev_sol'):
            # Warm start from previous solution
            x_init = np.concatenate([self.prev_sol[5:5*(N+1)], self.prev_sol[-5:], np.zeros(5)])
            u_init = np.concatenate([self.prev_sol[5*(N+1)+1:], [0.0]])
        
        vars_init = np.concatenate([x_init, u_init])
        
        # Create parameter vector
        p = np.concatenate([x0, reference_trajectory[:N].flatten()])
        
        # Add steering rate constraints
        if hasattr(self, 'prev_steer'):
            steer_rate_limit = self.mpc_params.steer_rate_limit
            
            # Adjust bounds for first control to respect rate limit
            self.lbx[5*(N+1)] = max(-self.mpc_params.max_steer, self.prev_steer - steer_rate_limit)
            self.ubx[5*(N+1)] = min(self.mpc_params.max_steer, self.prev_steer + steer_rate_limit)
        
        # Solve optimization problem
        sol = self.solver(
            x0=vars_init,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=self.lbg,
            ubg=self.ubg,
            p=p
        )
        
        # Extract solution
        self.prev_sol = sol['x'].full().flatten()
        u_opt = self.prev_sol[5*(N+1):]
        
        # Store steering command for rate limiting
        steer_command = float(u_opt[0])
        self.prev_steer = steer_command
        
        return steer_command

class CarlaInterface:
    def __init__(self, mpc_params):
        self.mpc_params = mpc_params
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(20.0)
        
        try:
            self.world = self.client.get_world()
            self.map = self.world.get_map()
            
            # Set synchronous mode with fixed time step
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.mpc_params.dt
            self.world.apply_settings(settings)
            
            self.blueprint_library = self.world.get_blueprint_library()
            self.spawn_vehicle_on_straight_road()
            self.setup_sensors()
            
            self.collision_count = 0
            self.max_collisions = 3
            self.last_steer = 0.0
            self.waypoints = []
            self.should_stop = False
            self.prev_steer_command = 0.0
            
        except Exception as e:
            print(f"Initialization failed: {e}")
            self.cleanup()
            raise

    def spawn_vehicle_on_straight_road(self):
        """Find a suitable road segment for driving"""
        spawn_points = self.map.get_spawn_points()
        
        # Prefer highways or major roads
        highway_spawns = [sp for sp in spawn_points 
                         if self.map.get_waypoint(sp.location).is_junction == False and
                         self.map.get_waypoint(sp.location).lane_type == carla.LaneType.Driving]
        
        if not highway_spawns:
            highway_spawns = spawn_points
        
        # Find spawn point with good straight/curved path ahead
        best_spawn = None
        max_quality_score = 0
        
        for sp in highway_spawns:
            wp = self.map.get_waypoint(sp.location)
            path_score = self.evaluate_path_quality(wp, distance=150.0)
            if path_score > max_quality_score:
                max_quality_score = path_score
                best_spawn = sp
        
        if not best_spawn:
            best_spawn = random.choice(highway_spawns)
        
        # Spawn vehicle
        vehicle_bp = self.blueprint_library.find('vehicle.tesla.model3')
        
        # Try to set minimum physics control for better stability
        if vehicle_bp.has_attribute('physics_control'):
            physics_control = vehicle_bp.get_attribute('physics_control').recommended_values
            if physics_control:
                vehicle_bp.set_attribute('physics_control', physics_control[0])
        
        self.vehicle = self.world.spawn_actor(vehicle_bp, best_spawn)
        self.vehicle.set_autopilot(False)
        
        # Generate waypoints for the reference path
        self.generate_reference_path(200.0)
        
        # Set up spectator camera
        self.spectator = self.world.get_spectator()
        self.update_spectator_view()

    def evaluate_path_quality(self, waypoint, distance=100.0):
        """
        Evaluate the quality of a potential path.
        Returns a score combining straightness and curvature.
        """
        current = waypoint
        total_distance = 0
        curvature_changes = 0
        last_heading = current.transform.rotation.yaw
        
        while total_distance < distance:
            next_wps = current.next(5.0)
            if not next_wps:
                break
            
            current = next_wps[0]
            current_heading = current.transform.rotation.yaw
            
            # Detect significant heading changes (curvature)
            if abs((current_heading - last_heading + 180) % 360 - 180) > 5:
                curvature_changes += 1
                
            last_heading = current_heading
            total_distance += 5.0
        
        # Score that prefers some curves (not too many, not too few)
        straightness_score = total_distance / distance
        curve_score = min(5, curvature_changes) / 5.0  # Optimal is having some curves
        
        # Combined score (0-1)
        return 0.7 * straightness_score + 0.3 * curve_score

    def generate_reference_path(self, distance):
        """Create a more detailed reference path with more waypoints"""
        current_loc = self.vehicle.get_location()
        current_wp = self.map.get_waypoint(current_loc)
        
        self.waypoints = []
        wp = current_wp
        dist = 0
        step_size = 2.0  # Smaller steps for more detailed path
        
        while dist < distance:
            self.waypoints.append(wp)
            next_wps = wp.next(step_size)
            if not next_wps:
                break
            wp = next_wps[0]
            dist += step_size
        
        print(f"Generated {len(self.waypoints)} waypoints covering {dist:.1f} meters")

    def get_reference_trajectory(self, look_ahead_points=15):
        """
        Create reference trajectory for MPC horizon
        Returns array of reference states [β, ψ_dot, ψ, y, e_y_dot]
        """
        if not self.waypoints:
            # Default reference if no waypoints available
            return np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.0]), (look_ahead_points, 1))
        
        # Get vehicle state
        vehicle_loc = self.vehicle.get_location()
        vehicle_rot = self.vehicle.get_transform().rotation
        vehicle_yaw = math.radians(vehicle_rot.yaw)
        
        # Find closest waypoint index
        closest_idx = min(range(len(self.waypoints)), 
                          key=lambda i: self.waypoints[i].transform.location.distance(vehicle_loc))
        
        # Create reference trajectory
        reference = []
        
        # If we're near the end of the path, regenerate
        if closest_idx + look_ahead_points >= len(self.waypoints) - 5:
            self.generate_reference_path(200.0)
            return self.get_reference_trajectory(look_ahead_points)
        
        v_lon = self.vehicle_params.vx
        ref_points = min(look_ahead_points, len(self.waypoints) - closest_idx)
        
        for i in range(ref_points):
            idx = closest_idx + i
            if idx >= len(self.waypoints):
                break
                
            wp = self.waypoints[idx]
            
            # Calculate desired yaw
            if idx < len(self.waypoints) - 1:
                next_wp = self.waypoints[idx + 1]
                dx = next_wp.transform.location.x - wp.transform.location.x
                dy = next_wp.transform.location.y - wp.transform.location.y
                desired_yaw = np.arctan2(dy, dx)
            else:
                desired_yaw = math.radians(wp.transform.rotation.yaw)
            
            # Desired lateral position (simplified to waypoint y coordinate)
            desired_y = wp.transform.location.y
            
            # Calculate desired yaw rate from path curvature
            if i < ref_points - 1:
                next_yaw = math.radians(self.waypoints[idx + 1].transform.rotation.yaw)
                yaw_diff = (next_yaw - desired_yaw + np.pi) % (2 * np.pi) - np.pi
                desired_yaw_rate = yaw_diff / (self.mpc_params.dt * v_lon)
            else:
                desired_yaw_rate = 0.0
            
            # Calculate desired lateral speed
            if i < ref_points - 1:
                next_y = self.waypoints[idx + 1].transform.location.y
                desired_y_dot = (next_y - desired_y) / self.mpc_params.dt
            else:
                desired_y_dot = 0.0
            
            # Desired sideslip is assumed to be 0 for most cases
            desired_beta = 0.0
            
            reference.append([desired_beta, desired_yaw_rate, desired_yaw, desired_y, desired_y_dot])
        
        # Fill remaining slots with the last reference point if needed
        if len(reference) < look_ahead_points:
            last_ref = reference[-1]
            reference.extend([last_ref] * (look_ahead_points - len(reference)))
        
        return np.array(reference)

    def setup_sensors(self):
        """Set up sensors for monitoring and visualization"""
        # RGB camera
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '90')
        
        camera_transform = carla.Transform(carla.Location(x=-8, z=6), carla.Rotation(pitch=-20))
        self.camera = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.vehicle
        )
        
        # Collision sensor
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.collision_sensor.listen(self.handle_collision)

    def update_spectator_view(self):
        """Update the spectator camera for better visualization"""
        if not hasattr(self, 'vehicle') or not self.vehicle:
            return
            
        transform = self.vehicle.get_transform()
        vehicle_location = transform.location
        
        # Calculate camera location for better visibility
        camera_distance = 10
        camera_height = 5
        yaw_rad = math.radians(transform.rotation.yaw)
        
        camera_x = vehicle_location.x - camera_distance * math.cos(yaw_rad)
        camera_y = vehicle_location.y - camera_distance * math.sin(yaw_rad)
        
        camera_location = carla.Location(x=camera_x, y=camera_y, z=vehicle_location.z + camera_height)
        camera_rotation = carla.Rotation(pitch=-20, yaw=transform.rotation.yaw)
        
        self.spectator.set_transform(carla.Transform(camera_location, camera_rotation))

    def get_vehicle_state(self):
        """Get enhanced vehicle state: [β, ψ_dot, ψ, y, e_y_dot]"""
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        angular_velocity = self.vehicle.get_angular_velocity()
        
        # Calculate longitudinal and lateral velocities
        vel_magnitude = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        yaw = math.radians(transform.rotation.yaw)
        
        # Convert velocities to vehicle frame
        vx = velocity.x * math.cos(yaw) + velocity.y * math.sin(yaw)
        vy = -velocity.x * math.sin(yaw) + velocity.y * math.cos(yaw)
        
        # Calculate sideslip angle
        beta = math.atan2(vy, vx) if abs(vx) > 0.1 else 0.0
        
        # Yaw rate (angular velocity around z-axis)
        psi_dot = angular_velocity.z
        
        # Vehicle position and orientation
        psi = yaw
        y = transform.location.y
        
        # Lateral error derivative (approximately lateral velocity)
        e_y_dot = vy
        
        return np.array([beta, psi_dot, psi, y, e_y_dot])

    def handle_collision(self, event):
        """Improved collision recovery with diagnostics"""
        if not self.vehicle or not self.vehicle.is_alive:
            return
            
        self.collision_count += 1
        collision_type = event.other_actor.type_id
        
        print(f"Collision #{self.collision_count} detected with {collision_type}")
        
        # Get vehicle state for diagnostics
        state = self.get_vehicle_state()
        print(f"Vehicle state at collision: sideslip={math.degrees(state[0]):.1f}°, "
              f"yaw_rate={math.degrees(state[1]):.1f}°/s, lateral_pos={state[3]:.2f}m")
        
        if self.collision_count >= self.max_collisions:
            print("Max collisions reached, stopping simulation")
            self.should_stop = True
            return
            
        # Reset to safe position
        current_transform = self.vehicle.get_transform()
        
        # Try to find nearby safe position
        current_waypoint = self.map.get_waypoint(current_transform.location)
        
        # Move vehicle to the center of the lane, slightly ahead
        safe_location = current_waypoint.transform.location
        safe_location.z += 0.5  # Lift slightly to avoid ground collisions
        
        # Use waypoint orientation for safer reset
        safe_rotation = current_waypoint.transform.rotation
        
        reset_transform = carla.Transform(safe_location, safe_rotation)
        
        # Reset vehicle
        self.vehicle.set_transform(reset_transform)
        
        # Stop the vehicle
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 1.0
        control.steer = 0.0
        self.vehicle.apply_control(control)
        
        # Regenerate waypoints
        self.generate_reference_path(200.0)
        time.sleep(0.5)  # Give time for physics to settle

    def apply_control(self, steer_command, vehicle_params):
        """Apply control with smooth acceleration and proper steering"""
        # Smooth steering transitions
        max_steer_change = self.mpc_params.steer_rate_limit
        steer_diff = steer_command - self.last_steer
        
        if abs(steer_diff) > max_steer_change:
            steer_command = self.last_steer + np.sign(steer_diff) * max_steer_change
            
        self.last_steer = steer_command
        
        # Create control command
        control = carla.VehicleControl()
        
        # Apply steering - convert to Carla's [-1, 1] range
        normalized_steer = steer_command / self.mpc_params.max_steer
        control.steer = np.clip(normalized_steer, -1.0, 1.0)
        
        # Apply throttle based on desired speed
        current_speed = math.sqrt(
            self.vehicle.get_velocity().x**2 + 
            self.vehicle.get_velocity().y**2
        )
        
        if current_speed < vehicle_params.vx - 0.5:
            control.throttle = min(0.6, (vehicle_params.vx - current_speed) * 0.1)
            control.brake = 0.0
        elif current_speed > vehicle_params.vx + 0.5:
            control.throttle = 0.0
            control.brake = min(0.3, (current_speed - vehicle_params.vx) * 0.1)
        else:
            control.throttle = 0.2
            control.brake = 0.0
        
        self.vehicle.apply_control(control)

    def debug_draw_waypoints(self):
        """Visualize the reference path with better color coding"""
        if not self.waypoints:
            return
            
        debug = self.world.debug
        vehicle_loc = self.vehicle.get_location()
        
        # Find closest waypoint for color coding
        closest_idx = min(range(len(self.waypoints)), 
                         key=lambda i: self.waypoints[i].transform.location.distance(vehicle_loc))
        
        # Draw waypoints with different colors
        for i in range(len(self.waypoints)-1):
            color = carla.Color(0, 255, 0)  # Default: green
            
            # Closest waypoints in red
            if i >= closest_idx and i < closest_idx + 5:
                color = carla.Color(255, 0, 0)
            # Near future waypoints in yellow
            elif i >= closest_idx + 5 and i < closest_idx + 15:
                color = carla.Color(255, 255, 0)
            
            debug.draw_line(
                self.waypoints[i].transform.location + carla.Location(z=0.5),
                self.waypoints[i+1].transform.location + carla.Location(z=0.5),
                thickness=0.1,
                color=color,
                life_time=0.1
            )
            
            # Draw points at each waypoint
            if i % 3 == 0:  # Only draw every 3rd waypoint for clarity
                debug.draw_point(
                    self.waypoints[i].transform.location + carla.Location(z=0.5),
                    size=0.1,
                    color=color,
                    life_time=0.1
                )

    def debug_draw_vehicle_state(self, state):
        """Draw velocity and trajectory prediction vectors"""
        if not self.vehicle:
            return
            
        debug = self.world.debug
        transform = self.vehicle.get_transform()
        location = transform.location
        
        # Draw velocity vector
        velocity = self.vehicle.get_velocity()
        vel_magnitude = math.sqrt(velocity.x**2 + velocity.y**2)
        
        if vel_magnitude > 0.1:
            vel_dir = carla.Location(
                x=velocity.x/vel_magnitude,
                y=velocity.y/vel_magnitude,
                z=0
            )
            
            debug.draw_arrow(
                location + carla.Location(z=0.5),
                location + carla.Location(z=0.5) + vel_dir * 5,
                thickness=0.1,
                arrow_size=0.2,
                color=carla.Color(0, 0, 255),
                life_time=0.1
            )
        
        # Draw heading vector
        yaw_rad = math.radians(transform.rotation.yaw)
        heading_dir = carla.Location(
            x=math.cos(yaw_rad),
            y=math.sin(yaw_rad),
            z=0
        )
        
        debug.draw_arrow(
            location + carla.Location(z=1.0),
            location + carla.Location(z=1.0) + heading_dir * 3,
            thickness=0.1,
            arrow_size=0.2,
            color=carla.Color(255, 165, 0),  # Orange
            life_time=0.1
        )

    def run_simulation(self, controller, vehicle_params, duration=30.0):
        """Run the simulation with enhanced monitoring and debugging"""
        self.should_stop = False
        start_time = time.time()
        self.vehicle_params = vehicle_params  # Store for reference
        
        # Performance metrics
        total_lateral_error = 0
        total_steps = 0
        max_lateral_error = 0
        
        try:
            while (time.time() - start_time) < duration and not self.should_stop:
                # Tick the CARLA world
                self.world.tick()
                
                # Get vehicle state
                current_state = self.get_vehicle_state()
                
                # Get reference trajectory for the MPC horizon
                reference_trajectory = self.get_reference_trajectory(controller.mpc_params.N)
                
                # Solve MPC
                try:
                    steer_command = controller.solve_mpc(current_state, reference_trajectory)
                    self.apply_control(steer_command, vehicle_params)
                except Exception as e:
                    print(f"MPC solver error: {e}")
                    # Use previous steering as fallback
                    self.apply_control(self.last_steer, vehicle_params)
                
                # Update visualization
                self.update_spectator_view()
                self.debug_draw_waypoints()
                self.debug_draw_vehicle_state(current_state)
                
                # Calculate metrics
                if len(self.waypoints) > 0:
                    vehicle_loc = self.vehicle.get_location()
                    closest_idx = min(range(len(self.waypoints)), 
                                    key=lambda i: self.waypoints[i].transform.location.distance(vehicle_loc))
                    
                    if closest_idx < len(self.waypoints):
                        lateral_error = abs(self.waypoints[closest_idx].transform.location.y - vehicle_loc.y)
                        total_lateral_error += lateral_error
                        max_lateral_error = max(max_lateral_error, lateral_error)
                        total_steps += 1

            # Print performance metrics
            if total_steps > 0:
                avg_lateral_error = total_lateral_error / total_steps
                print(f"Average lateral error: {avg_lateral_error:.2f} m")
                print(f"Maximum lateral error: {max_lateral_error:.2f} m")
        except Exception as e:
            print(f"Simulation error: {e}")
        finally:
            self.cleanup()
            print("Cleaning up CARLA actors...")
            print("Simulation ended.")

    def cleanup(self):
        """Clean up CARLA actors and settings"""
        if hasattr(self, 'vehicle') and self.vehicle.is_alive:
            self.vehicle.destroy()
        if hasattr(self, 'camera') and self.camera.is_alive:
            self.camera.destroy()
        if hasattr(self, 'collision_sensor') and self.collision_sensor.is_alive:
            self.collision_sensor.destroy()
        if hasattr(self, 'world'):
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

if __name__ == "__main__":
    # Initialize vehicle and MPC parameters
    vehicle_params = VehicleParams()
    mpc_params = MPCParams()
    
    # Create Carla interface and MPC controller
    carla_interface = CarlaInterface(mpc_params)
    mpc_controller = MPCController(vehicle_params, mpc_params)
    
    # Run the simulation
    try:
        carla_interface.run_simulation(mpc_controller, vehicle_params, duration=60.0)
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
        carla_interface.cleanup()
    except Exception as e:
        print(f"Unexpected error: {e}")
        carla_interface.cleanup()
    finally:
        print("Final cleanup...")
        carla_interface.cleanup()
        print("All actors cleaned up.")
        print("Simulation finished.")