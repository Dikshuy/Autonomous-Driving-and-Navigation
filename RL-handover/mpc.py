import carla
import numpy as np
from casadi import *
import random
import time

class VehicleParams:
    def __init__(self):
        self.m = 1500.0
        self.lf = 1.2
        self.lr = 1.8  
        self.Iz = 3000.0
        self.Cf = 80000.0
        self.Cr = 80000.0
        self.vx = 20.0

class MPCParams:
    def __init__(self):
        self.N = 10
        self.dt = 0.1
        self.max_iter = 100
        self.max_steer = np.deg2rad(5)  # Reduced max steering

class MPCController:
    def __init__(self, vehicle_params, mpc_params):
        self.vehicle_params = vehicle_params
        self.mpc_params = mpc_params
        self.model = self.create_vehicle_model()
        self.setup_mpc()

    def create_vehicle_model(self):
        x = SX.sym('x', 4)  # [β, ψ_dot, ψ, y]
        u = SX.sym('u', 1)  # δ
        
        m = self.vehicle_params.m
        lf = self.vehicle_params.lf
        lr = self.vehicle_params.lr
        Iz = self.vehicle_params.Iz
        Cf = self.vehicle_params.Cf
        Cr = self.vehicle_params.Cr
        vx = self.vehicle_params.vx
        
        beta = x[0]
        psi_dot = x[1]
        psi = x[2]
        y = x[3]
        delta = u[0]
        
        # Vehicle dynamics
        beta_dot = (2/(m*vx))*(Cf*(delta - beta - (lf*psi_dot)/vx) + 
                              Cr*(-beta + (lr*psi_dot)/vx)) - psi_dot
        psi_ddot = (2/Iz)*(lf*Cf*(delta - beta - (lf*psi_dot)/vx) - 
                          lr*Cr*(-beta + (lr*psi_dot)/vx))
        y_dot = vx * np.sin(psi + beta)
        
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
        R = np.diag([5.0])
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
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.mpc_params.dt
        self.world.apply_settings(settings)
        
        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_vehicle_on_straight_road()
        self.setup_visualization()
        
        self.collision_count = 0
        self.max_collisions = 3

    def spawn_vehicle_on_straight_road(self):
        map = self.world.get_map()
        spawn_points = map.get_spawn_points()
        
        # Find spawn points on straight roads
        straight_spawns = []
        for sp in spawn_points:
            wp = map.get_waypoint(sp.location)
            if abs(wp.transform.rotation.yaw - sp.rotation.yaw) < 5:
                straight_spawns.append(sp)
        
        spawn_point = random.choice(straight_spawns if straight_spawns else spawn_points)
        
        vehicle_bp = self.blueprint_library.find('vehicle.tesla.model3')
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.vehicle.set_autopilot(False)
        
        self.spectator = self.world.get_spectator()
        self.update_spectator_view()

    def setup_visualization(self):
        # Add camera
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.vehicle
        )

    def update_spectator_view(self):
        if not hasattr(self, 'vehicle') or not self.vehicle:
            return
            
        transform = self.vehicle.get_transform()
        self.spectator.set_transform(carla.Transform(
            transform.location + carla.Location(x=-10, z=3),
            carla.Rotation(yaw=180, pitch=-15)
        ))

    def get_vehicle_state(self):
        """Get current vehicle state: [β, ψ_dot, ψ, y]"""
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        angular_velocity = self.vehicle.get_angular_velocity()
        
        vx = np.sqrt(velocity.x**2 + velocity.y**2)
        vy = velocity.y
        beta = np.arctan2(vy, vx) if vx > 0.1 else 0
        psi_dot = angular_velocity.z
        psi = np.deg2rad(transform.rotation.yaw)
        y = transform.location.y
        
        return np.array([beta, psi_dot, psi, y])

    def handle_collision(self, event):
        self.collision_count += 1
        print(f"Collision #{self.collision_count} with {event.other_actor.type_id}")
        
        if self.collision_count >= self.max_collisions:
            print("Max collisions reached, stopping simulation")
            self.should_stop = True
            return
            
        # Reset vehicle position
        current_loc = self.vehicle.get_location()
        self.vehicle.set_location(carla.Location(
            x=current_loc.x,
            y=current_loc.y,
            z=current_loc.z + 0.5
        ))
        self.vehicle.set_velocity(carla.Vector3D(0, 0, 0))
        self.vehicle.set_angular_velocity(carla.Vector3D(0, 0, 0))

    def apply_control(self, steer):
        control = carla.VehicleControl()
        control.steer = np.clip(steer, -self.mpc_params.max_steer, self.mpc_params.max_steer)
        control.throttle = 0.3
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
        
        # Reference state (straight line)
        x_ref = np.array([0.0, 0.0, 0.0, 0.0])
        
        try:
            while (time.time() - start_time) < duration and not self.should_stop:
                self.world.tick()
                
                # Get and control vehicle state
                x0 = self.get_vehicle_state()
                steer = controller.solve_mpc(x0, x_ref)
                self.apply_control(steer * 0.8)  # Smoother steering
                
                self.update_spectator_view()
                time.sleep(0.01)
                
        finally:
            self.cleanup()

    def cleanup(self):
        actors = [a for a in [self.vehicle, self.camera, self.collision_sensor] 
                 if hasattr(self, a) and getattr(self, a)]
        for actor in actors:
            if actor.is_alive:
                actor.destroy()
        
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)

def main():
    try:
        vehicle_params = VehicleParams()
        mpc_params = MPCParams()
        
        controller = MPCController(vehicle_params, mpc_params)
        carla_interface = CarlaInterface(mpc_params)
        
        print("Starting simulation...")
        carla_interface.run_simulation(controller, duration=30.0)
        
    except KeyboardInterrupt:
        print("Simulation stopped by user")
    finally:
        if 'carla_interface' in locals():
            carla_interface.cleanup()
        print("Simulation ended")

if __name__ == '__main__':
    main()