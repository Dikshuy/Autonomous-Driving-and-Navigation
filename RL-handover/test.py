import numpy as np
import casadi as ca
import carla
import math
import time
import pygame
from collections import deque
import random

class VehicleDynamics:
    """Bicycle model for vehicle dynamics used in MPC"""
    def __init__(self, wheelbase=2.65, dt=0.1):
        self.L = wheelbase  # Vehicle wheelbase (m)
        self.dt = dt       # Time step (s)
        
    def create_model(self):
        # States: [x, y, psi, v]
        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        psi = ca.MX.sym('psi')
        v = ca.MX.sym('v')
        states = ca.vertcat(x, y, psi, v)
        n_states = states.size()[0]
        
        # Controls: [steering, acceleration]
        steering = ca.MX.sym('steering')
        acceleration = ca.MX.sym('acceleration')
        controls = ca.vertcat(steering, acceleration)
        n_controls = controls.size()[0]
        
        # Improved dynamics equations with better steering model
        # Note: tan(steering) * v / L can cause instability at high speeds
        # This model better represents how a vehicle responds to steering input
        beta = ca.atan(ca.tan(steering) * 0.5)  # Slip angle at vehicle center
        
        rhs = ca.vertcat(
            v * ca.cos(psi + beta),  # x dot - accounts for slip angle
            v * ca.sin(psi + beta),  # y dot - accounts for slip angle
            v * ca.sin(beta) / (self.L * 0.5),  # yaw rate - more stable at high speeds
            acceleration  # velocity change
        )
        
        # CasADi function for dynamics
        f = ca.Function('f', [states, controls], [rhs], 
                    ['states', 'controls'], ['rhs'])
        
        return states, controls, rhs, f, n_states, n_controls

class MPCController:
    def __init__(self, N=10, dt=0.1, wheelbase=2.65):
        self.N = N          # Prediction horizon
        self.dt = dt        # Time step
        self.wheelbase = wheelbase
        
        # Vehicle dynamics model
        self.dynamics = VehicleDynamics(wheelbase, dt)
        (self.states, self.controls, self.rhs, self.f, 
         self.n_states, self.n_controls) = self.dynamics.create_model()
        
        # Control constraints - adjusted for better performance
        self.max_steer = np.radians(30)  # Max steering angle (rad)
        self.max_accel = 2.0             # Max acceleration (m/s²)
        self.min_accel = -3.0            # Max deceleration (m/s²)
        
        # State weights (x, y, psi, v)
        # Increased positional tracking weights for better lane centering
        self.Q = np.diag([10.0, 10.0, 0.5, 5.0])  # Increased position weights
        
        # Control weights (steering, acceleration)
        # Reduced to allow more aggressive corrections
        self.R = np.diag([0.005, 0.005])  # Lower weights for more aggressive control
        
        # Terminal state weights - higher for better convergence
        self.Qf = np.diag([10.0, 10.0, 1.0, 1.0])
        
        # For visualization
        self.predicted_trajectory = []

        self.trajectory_valid = False

        # Initialize MPC
        self.init_mpc()
        
    def init_mpc(self):
        """Initialize MPC optimization problem with improved trajectory prediction"""
        # Optimization variables
        U = ca.MX.sym('U', self.n_controls, self.N)
        X = ca.MX.sym('X', self.n_states, self.N+1)
        P = ca.MX.sym('P', self.n_states + self.n_states)
        
        # Initial state and reference
        x0 = P[:self.n_states]
        x_ref = P[self.n_states:]
        
        # Cost function
        cost = 0
        
        # Constraint list
        g = []
        
        # Initial state constraint
        g.append(X[:, 0] - x0)
        
        # State prediction over horizon
        for k in range(self.N):
            st = X[:, k]
            con = U[:, k]
            
            # Add running cost with increased positional weights
            error = st - x_ref
            cost += ca.mtimes([error.T, self.Q, error]) + ca.mtimes([con.T, self.R, con])
            
            # Add extra cost for heading error to ensure vehicle faces road direction
            # This is critical for correct trajectory prediction
            heading_error = ca.if_else(
                ca.fabs(st[2] - x_ref[2]) > ca.pi,
                ca.fabs(st[2] - x_ref[2]) - 2*ca.pi,
                ca.fabs(st[2] - x_ref[2])
            )
            cost += 5.0 * heading_error * heading_error
            
            # Predict next state using RK4 integration for better accuracy
            k1 = self.f(st, con)
            k2 = self.f(st + self.dt/2 * k1, con)
            k3 = self.f(st + self.dt/2 * k2, con)
            k4 = self.f(st + self.dt * k3, con)
            st_pred = st + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            # Add dynamics constraint
            g.append(X[:, k+1] - st_pred)
        
        # Add terminal cost with higher weights
        terminal_error = X[:, -1] - x_ref
        cost += ca.mtimes([terminal_error.T, self.Qf, terminal_error])
        
        # Convert to NLP problem
        OPT_variables = ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))
        nlp_prob = {
            'f': cost,
            'x': OPT_variables,
            'g': ca.vertcat(*g),
            'p': P
        }
        
        # Solver options - adjusted for better convergence
        opts = {
            'ipopt': {
                'max_iter': 200,
                'print_level': 0,
                'acceptable_tol': 1e-6,
                'acceptable_obj_change_tol': 1e-6,
                'hessian_approximation': 'limited-memory',
                'warm_start_init_point': 'yes'  # Enable warm start
            },
            'print_time': 0
        }
        
        # Create solver
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
        
        # Variable bounds
        lbx = ca.DM.zeros((self.n_states*(self.N+1) + self.n_controls*self.N, 1))
        ubx = ca.DM.zeros((self.n_states*(self.N+1) + self.n_controls*self.N, 1))
        
        # State bounds - less restrictive on velocity
        lbx[0:self.n_states*(self.N+1):self.n_states] = -ca.inf  # x
        lbx[1:self.n_states*(self.N+1):self.n_states] = -ca.inf  # y
        lbx[2:self.n_states*(self.N+1):self.n_states] = -ca.inf  # psi
        lbx[3:self.n_states*(self.N+1):self.n_states] = 0.0      # v >= 0
        
        ubx[0:self.n_states*(self.N+1):self.n_states] = ca.inf   # x
        ubx[1:self.n_states*(self.N+1):self.n_states] = ca.inf   # y
        ubx[2:self.n_states*(self.N+1):self.n_states] = ca.inf   # psi
        ubx[3:self.n_states*(self.N+1):self.n_states] = 30.0     # v <= 30 m/s
        
        # Control bounds
        lbx[self.n_states*(self.N+1)::self.n_controls] = -self.max_steer  # steering
        lbx[self.n_states*(self.N+1)+1::self.n_controls] = self.min_accel # acceleration
        
        ubx[self.n_states*(self.N+1)::self.n_controls] = self.max_steer   # steering
        ubx[self.n_states*(self.N+1)+1::self.n_controls] = self.max_accel # acceleration
        
        self.lbx = lbx
        self.ubx = ubx
        
    def solve(self, x0, x_ref):
        """Solve MPC problem with better trajectory initialization"""
        # Initial guess - improved for better trajectory prediction
        u0 = ca.DM.zeros((self.n_controls, self.N))
        X0 = ca.DM.zeros((self.n_states, self.N+1))
        
        # Set initial state
        X0[:, 0] = x0
        
        # DEBUG: Print x0 to verify starting state is correct
        print(f"MPC initial state: x={x0[0]:.2f}, y={x0[1]:.2f}")
        
        # IMPORTANT: Better initialization - dynamic forward projection
        # Starting from ACTUAL vehicle position (crucial fix)
        curr_state = x0.copy()
        for i in range(1, self.N+1):
            # Initialize control (small steering, constant velocity)
            if i-1 < self.N:
                u0[0, i-1] = 0.0  # Neutral steering
                u0[1, i-1] = 0.0  # Maintain speed
            
            # Propagate state using simple dynamics
            next_x = curr_state[0] + curr_state[3] * np.cos(curr_state[2]) * self.dt
            next_y = curr_state[1] + curr_state[3] * np.sin(curr_state[2]) * self.dt
            next_psi = curr_state[2]  # Maintain heading initially
            next_v = curr_state[3]    # Maintain speed initially
            
            curr_state = np.array([next_x, next_y, next_psi, next_v])
            X0[:, i] = curr_state
        
        # Parameters (initial state + reference)
        p = ca.vertcat(x0, x_ref)
        
        # Optimization variable initial guess
        init_var = ca.vertcat(
            X0.reshape((-1, 1)),
            u0.reshape((-1, 1)))
        
        try:
            # Solve NLP
            sol = self.solver(
                x0=init_var,
                lbx=self.lbx,
                ubx=self.ubx,
                p=p,
                lbg=0,
                ubg=0)
            
            # Extract solution
            opt_var = sol['x'].full()
            
            # IMPORTANT: Clear the previous trajectory first
            self.predicted_trajectory = []
            
            # Get predicted states for visualization
            pred_states = opt_var[:self.n_states*(self.N+1)].reshape((self.n_states, self.N+1))
            
            # CRITICAL FIX: Check first point distance
            first_x = float(pred_states[0, 0])
            first_y = float(pred_states[1, 0])
            vehicle_pos = np.array([x0[0], x0[1]])
            traj_start = np.array([first_x, first_y])
            distance = np.linalg.norm(traj_start - vehicle_pos)
            
            if distance > 5.0:  # If first point is far from vehicle
                print(f"Warning: Trajectory first point far from vehicle ({distance:.2f}m), rejecting solution")
                self.trajectory_valid = False
                self.predicted_trajectory = []
                return np.array([0.0, 0.0])  # Neutral control
            
            # Extract the trajectory points with validation
            for i in range(self.N+1):
                # Get position from the solution
                x_pos = float(pred_states[0, i])
                y_pos = float(pred_states[1, i])
                
                # Validate the points
                if not (math.isnan(x_pos) or math.isnan(y_pos) or 
                        math.isinf(x_pos) or math.isinf(y_pos)):
                    self.predicted_trajectory.append((x_pos, y_pos))
            
            # Set flag to indicate valid trajectory
            self.trajectory_valid = len(self.predicted_trajectory) > 1
            
            # Get controls
            u_opt = opt_var[self.n_states*(self.N+1):]
            u_opt = u_opt.reshape((self.n_controls, self.N))
            
            # Return first control input
            return u_opt[:, 0]
            
        except Exception as e:
            print(f"MPC solver error: {e}")
            self.trajectory_valid = False
            self.predicted_trajectory = []
            return np.array([0.0, 0.0])  # Neutral control  # Neutral control

class CarlaVisualizer:
    """Handles all visualization in CARLA"""
    def __init__(self, world):
        self.world = world
        self.debug = world.debug
        self.vehicle = None
        self.camera = None
        self.display = None
        self.image_queue = deque(maxlen=1)
        
    def set_vehicle(self, vehicle):
        self.vehicle = vehicle
        
    def draw_reference_path(self, path, color=carla.Color(0, 255, 0), thickness=0.2):
        """Draw the reference path in the CARLA world"""
        for i in range(len(path)-1):
            start = carla.Location(x=path[i][0], y=path[i][1], z=0.5)
            end = carla.Location(x=path[i+1][0], y=path[i+1][1], z=0.5)
            self.debug.draw_line(start, end, thickness=thickness, color=color, life_time=0.2)
            
    def draw_waypoints(self, waypoints, color=carla.Color(0, 0, 255), size=0.1):
        """Draw waypoints as spheres"""
        for wp in waypoints:
            loc = wp.transform.location
            loc.z += 0.5  # Slightly above ground
            self.debug.draw_point(loc, size=size, color=color, life_time=0.2)
            
    def draw_predicted_trajectory(self, trajectory, color=carla.Color(255, 0, 0), thickness=0.3):
        """Draw the MPC predicted trajectory with proper validation"""
        # Clear any existing trajectory visualization first
        self.debug.draw_string(
            carla.Location(0, 0, 0),  # Dummy location
            "ClearTrajectory",       # Dummy text
            color=carla.Color(0, 0, 0),
            life_time=0.01)          # Very short life to clear
            
        if not trajectory or len(trajectory) < 2:
            return
            
        # Ensure trajectory points are near the vehicle (sanity check)
        if self.vehicle:
            vehicle_loc = self.vehicle.get_location()
            vehicle_pos = np.array([vehicle_loc.x, vehicle_loc.y])
            
            first_point = np.array([trajectory[0][0], trajectory[0][1]])
            distance_to_vehicle = np.linalg.norm(first_point - vehicle_pos)
            
            # If first point is too far from vehicle, don't draw (likely invalid)
            if distance_to_vehicle > 10.0:  # Reduced threshold to 10 meters
                print(f"Warning: Trajectory start point too far from vehicle ({distance_to_vehicle:.2f}m)")
                return
        
        # Draw trajectory with valid points only
        valid_points = []
        for point in trajectory:
            if (isinstance(point, tuple) and len(point) == 2 and
                not math.isnan(point[0]) and not math.isnan(point[1]) and
                not math.isinf(point[0]) and not math.isinf(point[1])):
                valid_points.append(point)
        
        if len(valid_points) < 2:
            return
        
        # Draw trajectory line
        for i in range(len(valid_points)-1):
            start = carla.Location(x=valid_points[i][0], y=valid_points[i][1], z=0.5)
            end = carla.Location(x=valid_points[i+1][0], y=valid_points[i+1][1], z=0.5)
            self.debug.draw_line(start, end, thickness=thickness, color=color, life_time=0.2)
        
        # Draw markers at each point
        for i, point in enumerate(valid_points):
            loc = carla.Location(x=point[0], y=point[1], z=0.5)
            size = 0.1 if i > 0 else 0.2
            self.debug.draw_point(loc, size=size, color=color, life_time=0.2)

            
    def draw_vehicle_info(self, steering, throttle, brake, speed, lateral_error=None):
        """Draw text information near the vehicle"""
        if not self.vehicle:
            return
            
        location = self.vehicle.get_location()
        text_location = carla.Location(x=location.x, y=location.y, z=location.z + 2.5)
        
        info_text = f"Steer: {steering:.2f}\nThrottle: {throttle:.2f}\nBrake: {brake:.2f}\nSpeed: {speed:.2f} km/h"
        if lateral_error is not None:
            info_text += f"\nLateral Error: {lateral_error:.2f} m"
            
        self.debug.draw_string(
            text_location,
            info_text,
            draw_shadow=True,
            color=carla.Color(255, 255, 255),
            life_time=0.2,
            persistent_lines=False)
    
    def setup_camera(self, width=800, height=600, fov=90):
        """Setup a camera sensor for third-person view"""
        if not self.vehicle:
            return
            
        try:
            # Setup pygame display
            pygame.init()
            self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame.display.set_caption("CARLA MPC Lane Following")
            
            # Get camera blueprint
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(width))
            camera_bp.set_attribute('image_size_y', str(height))
            camera_bp.set_attribute('fov', str(fov))
            
            # Spawn camera - better positioning for seeing lane
            camera_transform = carla.Transform(carla.Location(x=-8, z=6), carla.Rotation(pitch=-20))
            self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
            
            # Set callback
            self.camera.listen(self._process_image)
        except Exception as e:
            print(f"Camera setup failed: {e}")
        
    def _process_image(self, image):
        """Process camera image for pygame display"""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        
        self.image_queue.append(array)
        
    def render(self):
        """Render camera view if available"""
        if not self.display or not self.image_queue:
            return
            
        image = self.image_queue[-1]
        surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
        self.display.blit(surface, (0, 0))
        pygame.display.flip()
        
    def destroy(self):
        """Clean up visualizer"""
        if self.camera:
            self.camera.destroy()
        if self.display:
            pygame.quit()

class CarlaMPCClient:
    def __init__(self, host='127.0.0.1', port=2000):
        # Connect to CARLA server
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        
        # Load Town04 map
        print("Loading Town04 map...")
        self.world = self.client.load_world('Town04')
        
        # Modify simulation settings for better performance
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)
        
        time.sleep(2)  # Give time for map to load
        
        # Get blueprint library and map
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        
        # Vehicle and controller
        self.vehicle = None
        self.mpc = MPCController(N=10, dt=0.1)
        self.visualizer = CarlaVisualizer(self.world)
        
        # For trajectory tracking
        self.reference_path = []
        self.carla_waypoints = []
        self.current_target_idx = 0
        
        # For FPS calculation
        self.last_time = time.time()
        self.fps = 0
        
        # For lane tracking statistics
        self.lateral_error = 0.0

    def validate_coordinates(self):
        """Debug function to validate coordinate frames"""
        if not self.vehicle:
            return
            
        # Get vehicle state
        vehicle_state = self.get_vehicle_state()
        if vehicle_state is None:
            return
            
        # Get vehicle position from CARLA
        vehicle_loc = self.vehicle.get_location()
        
        # Compare positions
        print(f"Vehicle position (CARLA): ({vehicle_loc.x:.2f}, {vehicle_loc.y:.2f})")
        print(f"Vehicle position (State): ({vehicle_state[0]:.2f}, {vehicle_state[1]:.2f})")
        
        # Check if first waypoint is reasonably close
        if self.reference_path:
            first_wp = self.reference_path[0]
            dist = np.sqrt((vehicle_state[0] - first_wp[0])**2 + (vehicle_state[1] - first_wp[1])**2)
            print(f"Distance to first waypoint: {dist:.2f}m")
        
    def find_highway_spawn_point(self):
        """Find a suitable spawn point on a straight highway section in Town04"""
        # Try to find any highway waypoint first (more flexible approach)
        highway_waypoints = []
        for wp in self.map.generate_waypoints(2.0):
            if wp.road_id in [46, 43, 42] and abs(wp.lane_id) in [1, 2, 3, 4]:  # Main highways
                highway_waypoints.append(wp)
        
        if highway_waypoints:
            # Pick one at random
            waypoint = random.choice(highway_waypoints)
        else:
            # Fallback to spawn points
            print("No highway waypoints found, using default spawn")
            spawn_points = self.map.get_spawn_points()
            return spawn_points[0]
        
        # Create spawn point at this location
        spawn_transform = waypoint.transform
        spawn_transform.location.z += 0.5  # Slightly raise to prevent collision
        
        print(f"Selected highway spawn point at road_id={waypoint.road_id}, lane_id={waypoint.lane_id}")
        return spawn_transform
        
    def spawn_vehicle(self):
        """Spawn a vehicle at a suitable location in Town04"""
        # Get vehicle blueprint (Tesla Model 3)
        vehicle_bp = self.blueprint_library.filter('model3')[0]
        
        # Find a good spawn point on the highway
        spawn_point = self.find_highway_spawn_point()
        print(f"Selected spawn point at: {spawn_point.location}")
        
        # Try spawning with retries
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                
                # Tick the world to ensure the vehicle is properly placed
                self.world.tick()
                
                self.visualizer.set_vehicle(self.vehicle)
                print(f"Successfully spawned vehicle at {spawn_point.location}")
                
                # Enable physics for the vehicle
                self.vehicle.set_simulate_physics(True)
                return
            except RuntimeError as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    # Try a slightly different position
                    spawn_point.location.x += 2.0
                    time.sleep(0.5)
                    self.world.tick()
        
        raise RuntimeError("Failed to spawn vehicle after multiple attempts")
        
    def generate_lane_center_path(self, path_length=150):
        """Generate a reference path following the center of the current lane"""
        if not self.vehicle:
            return
            
        # Get starting waypoint on the lane
        vehicle_location = self.vehicle.get_location()
        start_waypoint = self.map.get_waypoint(vehicle_location, project_to_road=True)
        
        if not start_waypoint:
            print("Could not find a waypoint for the current vehicle position!")
            return
            
        print(f"Starting waypoint: road_id={start_waypoint.road_id}, lane_id={start_waypoint.lane_id}")
        
        # Generate waypoints along the lane
        self.carla_waypoints = [start_waypoint]
        current_wp = start_waypoint
        
        # Step size for waypoint generation (meters)
        step_size = 2.0  # Smaller step size for better precision
        
        # Generate waypoints for the specified path length
        total_distance = 0.0
        while total_distance < path_length:
            # Get next waypoint
            next_waypoints = current_wp.next(step_size)
            if not next_waypoints:
                print(f"End of road reached after {len(self.carla_waypoints)} waypoints")
                break
                
            current_wp = next_waypoints[0]
            self.carla_waypoints.append(current_wp)
            
            # Update total distance
            total_distance += step_size
        
        # Convert CARLA waypoints to reference path format
        self.reference_path = []
        for wp in self.carla_waypoints:
            x = wp.transform.location.x
            y = wp.transform.location.y
            yaw = np.radians(wp.transform.rotation.yaw)
            
            # Target speed (adjusted for better control)
            speed = 10.0  # 10 m/s (~36 km/h) - increased for better movement
            
            self.reference_path.append((x, y, yaw, speed))
        
        print(f"Generated lane center path with {len(self.reference_path)} waypoints")
    
    def get_vehicle_state(self):
        """Get current vehicle state for MPC"""
        if not self.vehicle:
            return None
            
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        
        # Convert to numpy array [x, y, psi, v]
        x = transform.location.x
        y = transform.location.y
        psi = np.radians(transform.rotation.yaw)
        v = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        return np.array([x, y, psi, v])
    
    def find_closest_waypoint(self, vehicle_state):
        """Find the closest waypoint in the reference path"""
        if not self.reference_path:
            return 0
            
        min_dist = float('inf')
        closest_idx = 0
        
        for i, wp in enumerate(self.reference_path):
            dist = np.sqrt((vehicle_state[0] - wp[0])**2 + (vehicle_state[1] - wp[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                
        return closest_idx
    
    def calculate_lateral_error(self, vehicle_state):
        """Calculate lateral error from lane center"""
        if not self.reference_path or not self.carla_waypoints:
            return 0.0
            
        # Get closest waypoint
        min_dist = float('inf')
        closest_wp = None
        
        for wp in self.carla_waypoints:
            dist = np.sqrt(
                (vehicle_state[0] - wp.transform.location.x)**2 + 
                (vehicle_state[1] - wp.transform.location.y)**2)
            
            if dist < min_dist:
                min_dist = dist
                closest_wp = wp
        
        if not closest_wp:
            return 0.0
            
        # Calculate accurate lateral distance (perpendicular to road direction)
        vehicle_loc = carla.Location(x=vehicle_state[0], y=vehicle_state[1])
        wp_dir = closest_wp.transform.get_forward_vector()
        wp_loc = closest_wp.transform.location
        
        v_vec = np.array([vehicle_loc.x - wp_loc.x, vehicle_loc.y - wp_loc.y, 0])
        w_vec = np.array([wp_dir.x, wp_dir.y, 0])
        
        cross_prod = np.cross(v_vec, w_vec)
        return np.linalg.norm(cross_prod)
    
    def find_closest_waypoint(self, vehicle_state):
        """Find the closest waypoint in the reference path with debug info"""
        if not self.reference_path:
            return 0
            
        min_dist = float('inf')
        closest_idx = 0
        
        for i, wp in enumerate(self.reference_path):
            dist = np.sqrt((vehicle_state[0] - wp[0])**2 + (vehicle_state[1] - wp[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Debug: Print distance to closest point
        if closest_idx < len(self.reference_path):
            print(f"Distance to closest waypoint: {min_dist:.2f}m")
            
        return closest_idx
    
    # In the main control loop:
    # Modify how we use the reference target
    def get_reference_state(self, target_idx, vehicle_state):
        # Get target waypoint
        x_ref = np.array(self.reference_path[target_idx])
        
        # Current position
        x_cur = vehicle_state[0]
        y_cur = vehicle_state[1]
        
        # Vector from current position to target
        dx = x_ref[0] - x_cur
        dy = x_ref[1] - y_cur
        
        # Calculate distance to target
        distance = np.sqrt(dx*dx + dy*dy)
        
        # If target is too far, adjust to a closer point
        max_distance = 15.0  # Maximum acceptable distance
        if distance > max_distance:
            # Scale back to max distance
            scale = max_distance / distance
            adjusted_x = x_cur + dx * scale
            adjusted_y = y_cur + dy * scale
            
            # Desired heading still points toward original target
            psi_desired = np.arctan2(dy, dx)
            
            # Keep the target velocity from the reference
            v_desired = x_ref[3]
            
            # Create modified reference state with adjusted position
            return np.array([adjusted_x, adjusted_y, psi_desired, v_desired])
        
        # Original target is close enough, use as is
        psi_desired = np.arctan2(dy, dx)
        v_desired = x_ref[3]
        return np.array([x_ref[0], x_ref[1], psi_desired, v_desired])
    
    def run(self):
        """Main control loop"""
        if not self.vehicle:
            print("No vehicle spawned!")
            return
            
        # Setup camera first
        self.visualizer.setup_camera()
        
        # Generate lane center path
        print("Generating lane center path...")
        self.generate_lane_center_path()
        self.validate_coordinates()
            
        if not self.reference_path:
            print("Failed to generate a valid reference path!")
            return
        
        # Stronger initial warm-up to get vehicle moving
        print("Starting vehicle with initial thrust...")
        warmup_control = carla.VehicleControl()
        warmup_control.throttle = 0.7  # Stronger initial thrust
        warmup_control.steer = 0.0
        warmup_control.brake = 0.0
        self.vehicle.apply_control(warmup_control)
        
        # Let the vehicle start moving (with simulation ticks)
        print("Warming up vehicle...")
        for i in range(40):  # More warm-up time
            self.world.tick()
            if i % 10 == 0:
                # Re-apply control periodically during warmup
                self.vehicle.apply_control(warmup_control)
            time.sleep(0.05)
        
        # Main loop
        try:
            running = True
            while running:
                # Tick the world
                self.world.tick()
                
                # Process events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                
                # Calculate FPS
                current_time = time.time()
                self.fps = 1.0 / max(0.001, current_time - self.last_time)
                self.last_time = current_time
                
                # Get current state
                x0 = self.get_vehicle_state()
                if x0 is None:
                    print("Vehicle state not available!")
                    time.sleep(0.05)
                    continue
                
                # Find closest waypoint
                closest_idx = self.find_closest_waypoint(x0)
                
                # Calculate lateral error
                self.lateral_error = self.calculate_lateral_error(x0)
                
                # Get target waypoint (with lookahead)
                look_ahead = min(5, len(self.reference_path) - closest_idx - 1)
                target_idx = min(closest_idx + look_ahead, len(self.reference_path) - 1)
                x_ref = self.get_reference_state(target_idx, x0)
                
                # Update current target for tracking
                self.current_target_idx = target_idx
                
                # Solve MPC
                try:
                    u_opt = self.mpc.solve(x0, x_ref)
                    steering, acceleration = u_opt[0], u_opt[1]
                except Exception as e:
                    print(f"MPC solve error: {e}")
                    # Apply emergency control to keep moving
                    control = carla.VehicleControl(throttle=0.3, steer=0.0, brake=0.0)
                    self.vehicle.apply_control(control)
                    time.sleep(0.05)
                    continue
                
                # Convert acceleration to throttle/brake with better mapping
                if acceleration >= 0:
                    throttle = min(1.0, acceleration / self.mpc.max_accel)  # Scale to [0, 1]
                    brake = 0.0
                else:
                    throttle = 0.0
                    brake = min(1.0, -acceleration / abs(self.mpc.min_accel))  # Scale to [0, 1]
                    
                # Apply control with direct mapping
                control = carla.VehicleControl()
                control.steer = float(np.clip(steering / self.mpc.max_steer, -1.0, 1.0))
                control.throttle = float(np.clip(throttle, 0.0, 1.0))
                control.brake = float(np.clip(brake, 0.0, 1.0))
                control.hand_brake = False
                control.reverse = False

                min_throttle = 0.2
                
                # Force non-zero throttle if vehicle is stationary
                if x0[3] < 2.0:  # If almost stationary
                    control.throttle = max(min_throttle, control.throttle)  # Ensure at least 0.5 throttle
                    control.brake = 0.0  # No braking when stationary
                
                # Apply the control to the vehicle
                self.vehicle.apply_control(control)
                
                # Visualization
                self.visualizer.draw_reference_path(
                [(wp[0], wp[1]) for wp in self.reference_path])

                if self.mpc.trajectory_valid and self.mpc.predicted_trajectory:
                    self.visualizer.draw_predicted_trajectory(self.mpc.predicted_trajectory)
                else:
                    print("Invalid trajectory, not drawing")
            
                
                # Draw CARLA waypoints for debugging
                if self.carla_waypoints:
                    self.visualizer.draw_waypoints(self.carla_waypoints)
                    
                self.visualizer.draw_predicted_trajectory(self.mpc.predicted_trajectory)
                self.visualizer.draw_vehicle_info(
                    control.steer, control.throttle, control.brake, 
                    3.6 * x0[3],  # Convert m/s to km/h
                    self.lateral_error
                )
                
                # Render camera view
                self.visualizer.render()
                
                # Print status with more debug info
                print(f"\rMPC running | FPS: {self.fps:.1f} | Speed: {3.6*x0[3]:.1f} km/h | " +
                      f"Controls: Steer={control.steer:.2f}, Throttle={control.throttle:.2f}, Brake={control.brake:.2f} | " +
                      f"Target: {self.current_target_idx}/{len(self.reference_path)} | " +
                      f"Lateral Error: {self.lateral_error:.2f}m", end="")
                
                # Check if we've reached the end of the path or need a new path
                if self.current_target_idx >= len(self.reference_path) - 5 or self.lateral_error > 5.0:
                    print("\nRegenerating path...")
                    self.generate_lane_center_path()
                
                # Sleep for control frequency if needed
                control_dt = max(0, self.mpc.dt - (time.time() - current_time))
                if control_dt > 0:
                    time.sleep(control_dt)
                
        except KeyboardInterrupt:
            print("\nStopping controller...")
        finally:
            # Reset world settings
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            
            if self.vehicle:
                self.vehicle.destroy()
            self.visualizer.destroy()

if __name__ == "__main__":
    print("Starting CARLA MPC client for lane-following in Town04...")
    client = CarlaMPCClient()
    
    # Spawn vehicle
    try:
        client.spawn_vehicle()
    except Exception as e:
        print(f"Failed to spawn vehicle: {e}")
        exit(1)
    
    # Run controller
    client.run()