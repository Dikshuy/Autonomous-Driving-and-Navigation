import numpy as np
import casadi as ca
import carla
import math
import time
import pygame
from collections import deque
import random

class VehicleDynamics:
    def __init__(self, wheelbase=2.65, dt=0.1):
        self.L = wheelbase
        self.dt = dt
        
    def create_model(self):
        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        psi = ca.MX.sym('psi')
        v = ca.MX.sym('v')
        states = ca.vertcat(x, y, psi, v)
        n_states = states.size()[0]
        
        steering = ca.MX.sym('steering')
        acceleration = ca.MX.sym('acceleration')
        controls = ca.vertcat(steering, acceleration)
        n_controls = controls.size()[0]
        
        beta = ca.atan(ca.tan(steering) * 0.5)
        
        rhs = ca.vertcat(
            v * ca.cos(psi + beta),
            v * ca.sin(psi + beta),
            v * ca.sin(beta) / (self.L * 0.5),
            acceleration)
        
        f = ca.Function('f', [states, controls], [rhs], 
                    ['states', 'controls'], ['rhs'])
        
        return states, controls, rhs, f, n_states, n_controls

class MPCController:
    def __init__(self, N=10, dt=0.1, wheelbase=2.65):
        self.N = N 
        self.dt = dt
        self.wheelbase = wheelbase

        self.dynamics = VehicleDynamics(wheelbase, dt)
        (self.states, self.controls, self.rhs, self.f, 
         self.n_states, self.n_controls) = self.dynamics.create_model()

        self.max_steer = np.radians(30)
        self.max_accel = 2.0
        self.min_accel = -3.0
        
        self.Q = np.diag([10.0, 10.0, 0.5, 5.0])
        self.R = np.diag([0.005, 0.005])        
        self.Qf = np.diag([10.0, 10.0, 1.0, 1.0])

        self.predicted_trajectory = []
        self.trajectory_valid = False
        
        # Add previous solution storage for warm start
        self.prev_u = None
        self.prev_x = None

        self.init_mpc()
        
    def init_mpc(self):
        U = ca.MX.sym('U', self.n_controls, self.N)
        X = ca.MX.sym('X', self.n_states, self.N+1)
        P = ca.MX.sym('P', self.n_states + self.n_states)
        
        x0 = P[:self.n_states]
        x_ref = P[self.n_states:]
        
        cost = 0
        
        g = []
        
        g.append(X[:, 0] - x0)
        
        for k in range(self.N):
            st = X[:, k]
            con = U[:, k]
            
            error = st - x_ref
            cost += ca.mtimes([error.T, self.Q, error]) + ca.mtimes([con.T, self.R, con])
            
            # Handle heading error properly with circular difference
            heading_error = ca.if_else(
                ca.fabs(st[2] - x_ref[2]) > ca.pi,
                ca.fabs(st[2] - x_ref[2]) - 2*ca.pi,
                ca.fabs(st[2] - x_ref[2])
            )
            cost += 5.0 * heading_error * heading_error
            
            # RK4 integration for more accurate dynamics
            k1 = self.f(st, con)
            k2 = self.f(st + self.dt/2 * k1, con)
            k3 = self.f(st + self.dt/2 * k2, con)
            k4 = self.f(st + self.dt * k3, con)
            st_pred = st + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            g.append(X[:, k+1] - st_pred)
        
        terminal_error = X[:, -1] - x_ref
        cost += ca.mtimes([terminal_error.T, self.Qf, terminal_error])
        
        OPT_variables = ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))
        nlp_prob = {
            'f': cost,
            'x': OPT_variables,
            'g': ca.vertcat(*g),
            'p': P
        }
        
        opts = {
            'ipopt': {
                'max_iter': 200,
                'print_level': 0,
                'acceptable_tol': 1e-6,
                'acceptable_obj_change_tol': 1e-6,
                'hessian_approximation': 'limited-memory',
                'warm_start_init_point': 'yes'
            },
            'print_time': 0
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
        
        lbx = ca.DM.zeros((self.n_states*(self.N+1) + self.n_controls*self.N, 1))
        ubx = ca.DM.zeros((self.n_states*(self.N+1) + self.n_controls*self.N, 1))
        
        lbx[0:self.n_states*(self.N+1):self.n_states] = -ca.inf  # x
        lbx[1:self.n_states*(self.N+1):self.n_states] = -ca.inf  # y
        lbx[2:self.n_states*(self.N+1):self.n_states] = -ca.inf  # psi
        lbx[3:self.n_states*(self.N+1):self.n_states] = 0.0      # v >= 0
        
        ubx[0:self.n_states*(self.N+1):self.n_states] = ca.inf   # x
        ubx[1:self.n_states*(self.N+1):self.n_states] = ca.inf   # y
        ubx[2:self.n_states*(self.N+1):self.n_states] = ca.inf   # psi
        ubx[3:self.n_states*(self.N+1):self.n_states] = 30.0     # v <= 30 m/s
        
        lbx[self.n_states*(self.N+1)::self.n_controls] = -self.max_steer  # steering
        lbx[self.n_states*(self.N+1)+1::self.n_controls] = self.min_accel # acceleration
        
        ubx[self.n_states*(self.N+1)::self.n_controls] = self.max_steer   # steering
        ubx[self.n_states*(self.N+1)+1::self.n_controls] = self.max_accel # acceleration
        
        self.lbx = lbx
        self.ubx = ubx
        
    def solve(self, x0, x_ref):
        # Print initial state info for debugging
        print(f"MPC solve input - x0: {x0}, x_ref: {x_ref}")

        x0[2] = (x0[2] + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
        x_ref[2] = (x_ref[2] + np.pi) % (2 * np.pi) - np.pi     
        
        # IMPORTANT: Convert x0 and x_ref to CasADi DM type to avoid numerical issues
        x0_ca = ca.DM(x0)
        x_ref_ca = ca.DM(x_ref)
        
        # Initialize with zeros or previous solution for warm start
        # In MPCController.solve(), modify the initial guess generation:
        if self.prev_u is not None and self.prev_x is not None:
            x0[:, 0] = x0_ca
            for k in range(1, self.N+1):
                # Use previous solution shifted by one step
                x0[:, k] = self.prev_x[:, k-1]
            u0 = self.prev_u
        else:
            # Create a better initial guess that follows the reference path
            x0[:, 0] = x0_ca
            for k in range(1, self.N+1):
                # Interpolate between current state and reference
                alpha = k / self.N
                x0[0, k] = (1-alpha)*x0[0] + alpha*x_ref[0]
                x0[1, k] = (1-alpha)*x0[1] + alpha*x_ref[1]
                x0[2, k] = x_ref[2]  # Use reference heading
                x0[3, k] = x_ref[3]  # Use reference speed
            u0 = ca.DM.zeros((self.n_controls, self.N))
                    
        # Forward simulate with simple model to get better initial guess
        curr_state = x0.copy()
        for i in range(1, self.N+1):
            # Simple forward Euler integration
            next_x = curr_state[0] + curr_state[3] * np.cos(curr_state[2]) * self.dt
            next_y = curr_state[1] + curr_state[3] * np.sin(curr_state[2]) * self.dt
            next_psi = curr_state[2]
            next_v = curr_state[3]
            
            curr_state = np.array([next_x, next_y, next_psi, next_v])
            x0[:, i] = curr_state

        # Debug prints
        print(f"MPC initial state: x={x0[0]:.2f}, y={x0[1]:.2f}, psi={x0[2]:.2f}, v={x0[3]:.2f}")
        print(f"Target state: x={x_ref[0]:.2f}, y={x_ref[1]:.2f}, psi={x_ref[2]:.2f}, v={x_ref[3]:.2f}")
        print(f"Initial guess first point: x={float(x0[0,0]):.2f}, y={float(x0[1,0]):.2f}")
        print(f"Initial guess last point: x={float(x0[0,self.N]):.2f}, y={float(x0[1,self.N]):.2f}")
        
        # Set parameters
        p = ca.vertcat(x0_ca, x_ref_ca)
        
        # Set initial values for the NLP solver
        init_var = ca.vertcat(
            x0.reshape((-1, 1)),
            u0.reshape((-1, 1)))
        
        # Solve the optimization problem
        try:
            sol = self.solver(
                x0=init_var,
                lbx=self.lbx,
                ubx=self.ubx,
                p=p,
                lbg=0,
                ubg=0)
            
            opt_var = sol['x'].full()
            
            # Extract solution
            pred_states = opt_var[:self.n_states*(self.N+1)].reshape((self.n_states, self.N+1))
            u_opt = opt_var[self.n_states*(self.N+1):].reshape((self.n_controls, self.N))
            
            # Store solution for next warm start (make deep copies)
            self.prev_x = pred_states.copy()
            self.prev_u = u_opt.copy()
            
            # Check if MPC solution's first state matches vehicle state
            first_x = float(pred_states[0, 0])
            first_y = float(pred_states[1, 0])
            vehicle_pos = np.array([x0[0], x0[1]])
            traj_start = np.array([first_x, first_y])
            distance = np.linalg.norm(traj_start - vehicle_pos)
            
            # Debug info for trajectory starting point
            print(f"Trajectory start: ({first_x:.2f}, {first_y:.2f}), Vehicle: ({x0[0]:.2f}, {x0[1]:.2f}), Distance: {distance:.2f}m")
            
            # Force the first point to be the current vehicle position
            pred_states[0, 0] = x0[0]
            pred_states[1, 0] = x0[1]
            
            # Generate trajectory points from the solution
            self.predicted_trajectory = []
            for i in range(self.N+1):
                x_pos = float(pred_states[0, i])
                y_pos = float(pred_states[1, i])
                
                if not (math.isnan(x_pos) or math.isnan(y_pos) or 
                        math.isinf(x_pos) or math.isinf(y_pos)):
                    self.predicted_trajectory.append((x_pos, y_pos))
            
            self.trajectory_valid = len(self.predicted_trajectory) > 1
            
            # Final debug check on the trajectory
            if self.trajectory_valid and len(self.predicted_trajectory) > 1:
                first_point = self.predicted_trajectory[0]
                distance = np.sqrt((first_point[0] - x0[0])**2 + (first_point[1] - x0[1])**2)
                print(f"Final trajectory first point: ({first_point[0]:.2f}, {first_point[1]:.2f}), distance: {distance:.2f}m")
            
            return u_opt[:, 0]
            
        except Exception as e:
            print(f"MPC solver error: {e}")
            self.trajectory_valid = False
            self.predicted_trajectory = []
            return np.array([0.0, 0.0])

class CarlaVisualizer:
    def __init__(self, world):
        self.world = world
        self.debug = world.debug
        self.vehicle = None
        self.camera = None
        self.display = None
        self.image_queue = deque(maxlen=1)
        
    def set_vehicle(self, vehicle):
        self.vehicle = vehicle
        
    def draw_reference_path(self, path, color=carla.Color(0, 255, 0), thickness=0.14):
        for i in range(len(path)-1):
            start = carla.Location(x=path[i][0], y=path[i][1], z=0.5)
            end = carla.Location(x=path[i+1][0], y=path[i+1][1], z=0.5)
            self.debug.draw_line(start, end, thickness=thickness, color=color, life_time=0.2)
            
    def draw_waypoints(self, waypoints, color=carla.Color(0, 0, 255), size=0.07):
        for wp in waypoints:
            loc = wp.transform.location
            loc.z += 0.5
            self.debug.draw_point(loc, size=size, color=color, life_time=0.2)
            
    def draw_predicted_trajectory(self, trajectory, color=carla.Color(255, 0, 0), thickness=0.2):
        self.debug.draw_string(
            carla.Location(0, 0, 0),
            "ClearTrajectory",
            color=carla.Color(0, 0, 0),
            life_time=0.01)
            
        if not trajectory or len(trajectory) < 2:
            print("Invalid trajectory, not drawing")
            return

        if self.vehicle:
            vehicle_loc = self.vehicle.get_location()
            vehicle_pos = np.array([vehicle_loc.x, vehicle_loc.y])
            
            first_point = np.array([trajectory[0][0], trajectory[0][1]])
            distance_to_vehicle = np.linalg.norm(first_point - vehicle_pos)
            
            if distance_to_vehicle > 10.0:
                print(f"Warning: Trajectory start point too far from vehicle ({distance_to_vehicle:.2f}m)")
                return
            
        valid_points = []
        for point in trajectory:
            if (isinstance(point, tuple) and len(point) == 2 and
                not math.isnan(point[0]) and not math.isnan(point[1]) and
                not math.isinf(point[0]) and not math.isinf(point[1])):
                valid_points.append(point)
        
        if len(valid_points) < 2:
            return

        for i in range(len(valid_points)-1):
            start = carla.Location(x=valid_points[i][0], y=valid_points[i][1], z=0.5)
            end = carla.Location(x=valid_points[i+1][0], y=valid_points[i+1][1], z=0.5)
            self.debug.draw_line(start, end, thickness=thickness, color=color, life_time=0.2)

        for i, point in enumerate(valid_points):
            loc = carla.Location(x=point[0], y=point[1], z=0.5)
            size = 0.1 if i > 0 else 0.2
            self.debug.draw_point(loc, size=size, color=color, life_time=0.2)

            
    def draw_vehicle_info(self, steering, throttle, brake, speed, lateral_error=None):
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
        pygame.init()
        self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("CARLA MPC Lane Following")
        
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(width))
        camera_bp.set_attribute('image_size_y', str(height))
        camera_bp.set_attribute('fov', str(fov))
        
        camera_transform = carla.Transform(carla.Location(x=-8, z=6), carla.Rotation(pitch=-20))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        
        self.camera.listen(self._process_image)
        
    def _process_image(self, image):
        if not image:
            return
            
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        
        self.image_queue.append(array)
        
    def render(self):
        if not self.image_queue:
            return
            
        image = self.image_queue[-1]
        surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
        self.display.blit(surface, (0, 0))
        pygame.display.flip()
        
    def destroy(self):
        if self.camera:
            self.camera.destroy()
        if self.display:
            pygame.quit()

class CarlaMPCClient:
    def __init__(self, host='127.0.0.1', port=2000):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        
        print("Loading Town04 map...")
        self.world = self.client.load_world('Town04')
        
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05 
        self.world.apply_settings(settings)
        
        time.sleep(2)
        
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        
        self.vehicle = None
        self.mpc = MPCController(N=10, dt=0.1)
        self.visualizer = CarlaVisualizer(self.world)
        
        self.reference_path = []
        self.carla_waypoints = []
        self.current_target_idx = 0
        
        self.last_time = time.time()
        self.fps = 0
        
        self.lateral_error = 0.0
        
        # Add safety controls
        self.last_valid_control = carla.VehicleControl()
        self.last_valid_control.steer = 0.0
        self.last_valid_control.throttle = 0.0
        self.last_valid_control.brake = 0.0

    def validate_coordinates(self):
        vehicle_state = self.get_vehicle_state()
        if vehicle_state is None:
            return

        vehicle_loc = self.vehicle.get_location()

        print(f"Vehicle position (CARLA): ({vehicle_loc.x:.2f}, {vehicle_loc.y:.2f})")
        print(f"Vehicle position (State): ({vehicle_state[0]:.2f}, {vehicle_state[1]:.2f})")
        print(f"Vehicle heading: {np.degrees(vehicle_state[2]):.2f} degrees")

        if self.reference_path:
            first_wp = self.reference_path[0]
            dist = np.sqrt((vehicle_state[0] - first_wp[0])**2 + (vehicle_state[1] - first_wp[1])**2)
            print(f"Distance to first waypoint: {dist:.2f}m")
        
    def find_highway_spawn_point(self):
        highway_waypoints = []
        for wp in self.map.generate_waypoints(2.0):
            if wp.road_id in [46, 43, 42] and abs(wp.lane_id) in [1, 2, 3, 4]:
                highway_waypoints.append(wp)
        
        if highway_waypoints:
            waypoint = random.choice(highway_waypoints)
        else:
            print("No highway waypoints found, using default spawn")
            spawn_points = self.map.get_spawn_points()
            return spawn_points[0]
        
        spawn_transform = waypoint.transform
        spawn_transform.location.z += 0.5
        
        print(f"Selected highway spawn point at road_id={waypoint.road_id}, lane_id={waypoint.lane_id}")
        return spawn_transform
        
    def spawn_vehicle(self):
        vehicle_bp = self.blueprint_library.filter('model3')[0]
        
        spawn_point = self.find_highway_spawn_point()
        print(f"Selected spawn point at: {spawn_point.location}")
        
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                
                self.world.tick()
                
                self.visualizer.set_vehicle(self.vehicle)
                print(f"Successfully spawned vehicle at {spawn_point.location}")

                self.vehicle.set_simulate_physics(True)
                return
            except RuntimeError as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    spawn_point.location.x += 2.0
                    time.sleep(0.5)
                    self.world.tick()
        
        raise RuntimeError("Failed to spawn vehicle after multiple attempts")
        
    def generate_lane_center_path(self, path_length=150):
        vehicle_location = self.vehicle.get_location()
        start_waypoint = self.map.get_waypoint(vehicle_location, project_to_road=True)
        
        print(f"Starting waypoint: road_id={start_waypoint.road_id}, lane_id={start_waypoint.lane_id}")
        
        self.carla_waypoints = [start_waypoint]
        current_wp = start_waypoint
        
        step_size = 2.0
        
        total_distance = 0.0
        while total_distance < path_length:
            next_waypoints = current_wp.next(step_size)
            if not next_waypoints:
                print(f"End of road reached after {len(self.carla_waypoints)} waypoints")
                break
                
            current_wp = next_waypoints[0]
            self.carla_waypoints.append(current_wp)

            total_distance += step_size
        
        self.reference_path = []
        for wp in self.carla_waypoints:
            x = wp.transform.location.x
            y = wp.transform.location.y
            yaw = np.radians(wp.transform.rotation.yaw)
            
            # Set target speed based on road type or curvature
            speed = 10.0
            
            self.reference_path.append((x, y, yaw, speed))
        
        print(f"Generated lane center path with {len(self.reference_path)} waypoints")
    
    def get_vehicle_state(self):
        if not self.vehicle:
            return None
            
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        
        x = transform.location.x
        y = transform.location.y
        psi = np.radians(transform.rotation.yaw)
        v = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        return np.array([x, y, psi, v])
    
    def find_closest_waypoint(self, vehicle_state):
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
        if not self.reference_path or not self.carla_waypoints:
            return 0.0
            
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
            
        vehicle_loc = carla.Location(x=vehicle_state[0], y=vehicle_state[1])
        wp_dir = closest_wp.transform.get_forward_vector()
        wp_loc = closest_wp.transform.location
        
        v_vec = np.array([vehicle_loc.x - wp_loc.x, vehicle_loc.y - wp_loc.y, 0])
        w_vec = np.array([wp_dir.x, wp_dir.y, 0])
        
        cross_prod = np.cross(v_vec, w_vec)
        return np.linalg.norm(cross_prod)
    
    # In get_reference_state(), modify to:
    def get_reference_state(self, target_idx, vehicle_state):
        # Get multiple waypoints ahead for smoother reference
        lookahead = min(5, len(self.reference_path) - target_idx - 1)
        target_idx = min(target_idx + lookahead, len(self.reference_path) - 1)
        
        # Use the immediate next waypoint for position, but further one for speed
        x_ref = np.array(self.reference_path[target_idx])
        
        # For heading, use the average of next few waypoints
        avg_heading = 0
        count = min(3, len(self.reference_path) - target_idx)
        for i in range(count):
            avg_heading += self.reference_path[target_idx + i][2]
        avg_heading /= count
        
        return np.array([x_ref[0], x_ref[1], avg_heading, x_ref[3]])
        
    def run(self):
        try:
            self.visualizer.setup_camera()

            # Generate initial path
            self.generate_lane_center_path()
            self.validate_coordinates()

            # Start with a simple control to get the vehicle moving
            warmup_control = carla.VehicleControl()
            warmup_control.throttle = 0.7
            warmup_control.steer = 0.0
            warmup_control.brake = 0.0
            self.vehicle.apply_control(warmup_control)
            
            # Give the vehicle time to start moving
            for i in range(40):
                self.world.tick()
                if i % 10 == 0:
                    self.vehicle.apply_control(warmup_control)
                time.sleep(0.05)
            
            # Main control loop
            running = True
            control_counter = 0  # For safety fallback

            while running:
                self.world.tick()
                # Check for quit events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                # Calculate FPS
                current_time = time.time()
                self.fps = 1.0 / max(0.001, current_time - self.last_time)
                self.last_time = current_time
                
                # Get current vehicle state
                x0 = self.get_vehicle_state()
                if x0 is None:
                    print("Vehicle state is None, skipping iteration")
                    continue
                    
                # Find closest waypoint and set target
                closest_idx = self.find_closest_waypoint(x0)
                
                # Calculate lateral error for visualization
                self.lateral_error = self.calculate_lateral_error(x0)
                
                # Look ahead for smoother tracking
                look_ahead = min(5, len(self.reference_path) - closest_idx - 1)
                target_idx = min(closest_idx + look_ahead, len(self.reference_path) - 1)
                x_ref = self.get_reference_state(target_idx, x0)

                self.current_target_idx = target_idx
                
                # Solve MPC optimization
                u_opt = self.mpc.solve(x0, x_ref)
                steering, acceleration = u_opt[0], u_opt[1]

                if acceleration >= 0:
                    throttle = min(1.0, acceleration / self.mpc.max_accel)
                    brake = 0.0
                else:
                    throttle = 0.0
                    brake = min(1.0, -acceleration / abs(self.mpc.min_accel))
                    
                control = carla.VehicleControl()
                control.steer = float(np.clip(steering / self.mpc.max_steer, -1.0, 1.0))
                control.throttle = float(np.clip(throttle, 0.0, 1.0))
                control.brake = float(np.clip(brake, 0.0, 1.0))
                control.hand_brake = False
                control.reverse = False

                min_throttle = 0.2

                if x0[3] < 2.0:
                    control.throttle = max(min_throttle, control.throttle)
                    control.brake = 0.0
                
                self.vehicle.apply_control(control)
                
                # Clear the previous debug drawings
                self.world.debug.draw_string(carla.Location(0, 0, 0), "ClearAll", color=carla.Color(0, 0, 0), life_time=0.01)

                # Draw reference path
                self.visualizer.draw_reference_path([(wp[0], wp[1]) for wp in self.reference_path])

                # Draw waypoints
                if self.carla_waypoints:
                    self.visualizer.draw_waypoints(self.carla_waypoints)

                # Draw predicted trajectory - CRITICAL PART
                if self.mpc.trajectory_valid and self.mpc.predicted_trajectory:
                    # Check if trajectory first point is close to vehicle
                    vehicle_pos = np.array([x0[0], x0[1]])
                    traj_first = np.array(self.mpc.predicted_trajectory[0])
                    dist = np.linalg.norm(traj_first - vehicle_pos)
                    
                    if dist < 10.0:  # Only draw if reasonably close
                        self.visualizer.draw_predicted_trajectory(self.mpc.predicted_trajectory)
                    else:
                        print(f"Not drawing trajectory: first point too far ({dist:.2f}m)")
                        # Force the first point to match vehicle position
                        if len(self.mpc.predicted_trajectory) > 1:
                            fixed_trajectory = [(x0[0], x0[1])] + self.mpc.predicted_trajectory[1:]
                            self.visualizer.draw_predicted_trajectory(fixed_trajectory)
                else:
                    print("No valid trajectory to draw")

                # if self.carla_waypoints:
                #     self.visualizer.draw_waypoints(self.carla_waypoints)
                    
                # self.visualizer.draw_predicted_trajectory(self.mpc.predicted_trajectory)
                # self.visualizer.draw_vehicle_info(
                #     control.steer, control.throttle, control.brake, 
                #     3.6 * x0[3],
                #     self.lateral_error
                # )
                
                self.visualizer.render()
                
                print(f"\rMPC running | FPS: {self.fps:.1f} | Speed: {3.6*x0[3]:.1f} km/h | " +
                      f"Controls: Steer={control.steer:.2f}, Throttle={control.throttle:.2f}, Brake={control.brake:.2f} | " +
                      f"Target: {self.current_target_idx}/{len(self.reference_path)} | " +
                      f"Lateral Error: {self.lateral_error:.2f}m", end="")
                
                if self.current_target_idx >= len(self.reference_path) - 5 or self.lateral_error > 5.0:
                    print("\nRegenerating path...")
                    self.generate_lane_center_path()
                
                control_dt = max(0, self.mpc.dt - (time.time() - current_time))
                if control_dt > 0:
                    time.sleep(control_dt)
                
        except KeyboardInterrupt:
            print("\nStopping controller...")
        finally:
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
    client.spawn_vehicle()
    
    client.run()