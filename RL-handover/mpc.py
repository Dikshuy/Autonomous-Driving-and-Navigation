import carla
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# =============================================
# 1. CARLA Initialization
# =============================================

# Connect to CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Spawn ego vehicle
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
spawn_point = carla.Transform(carla.Location(x=100, y=0, z=0.5), carla.Rotation(yaw=0))
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Enable traffic (optional)
world.set_weather(carla.WeatherParameters.ClearNoon)
traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)

# =============================================
# 2. Vehicle Dynamics Model (Bicycle Model)
# =============================================

class BicycleModel:
    def __init__(self):
        self.m = 1500.0    # mass (kg)
        self.Iz = 3000.0   # yaw inertia (kg*m^2)
        self.lf = 1.2      # CG to front axle (m)
        self.lr = 1.6      # CG to rear axle (m)
        self.Cf = 80000.0  # Front cornering stiffness (N/rad)
        self.Cr = 80000.0  # Rear cornering stiffness (N/rad)
        self.Vx = 10.0     # Longitudinal speed (m/s)
    
    def predict(self, x, delta):
        beta, psi_dot, psi, y = x
        
        # Slip angles
        alpha_f = delta - (beta + self.lf * psi_dot / self.Vx)
        alpha_r = -(beta - self.lr * psi_dot / self.Vx)
        
        # Lateral forces
        Fyf = self.Cf * alpha_f
        Fyr = self.Cr * alpha_r
        
        # Dynamics
        d_beta = (Fyf + Fyr) / (self.m * self.Vx) - psi_dot
        d_psi_dot = (self.lf * Fyf - self.lr * Fyr) / self.Iz
        d_psi = psi_dot
        d_y = self.Vx * (beta + psi)
        
        return np.array([d_beta, d_psi_dot, d_psi, d_y])

# =============================================
# 3. MPC Controller
# =============================================

class MPC:
    def __init__(self):
        self.model = BicycleModel()
        self.dt = 0.1     # Control timestep (s)
        self.horizon = 10  # Prediction horizon
        
    def cost_function(self, u, *args):
        x, y_ref = args
        cost = 0.0
        x_pred = x.copy()
        
        for i in range(self.horizon):
            delta = u[i]
            dx = self.model.predict(x_pred, delta)
            x_pred += dx * self.dt
            cost += (x_pred[3] - y_ref)**2 + 0.1 * delta**2  # Track ref + minimize steering
        
        return cost
    
    def optimize(self, x, y_ref):
        # Initial guess (zero steering)
        u0 = np.zeros(self.horizon)
        
        # Bounds (-0.5 to 0.5 radian steering)
        bounds = [(-0.5, 0.5) for _ in range(self.horizon)]
        
        # Solve optimization
        res = minimize(
            self.cost_function,
            u0,
            args=(x, y_ref),
            bounds=bounds,
            method='SLSQP'
        )
        
        return res.x[0]  # Return first control input

# =============================================
# 4. Simulation Loop
# =============================================

def main():
    mpc = MPC()
    vehicle.set_autopilot(False)
    
    # Logging
    time_log = []
    y_log = []
    y_ref_log = []
    
    try:
        for t in np.arange(0, 30, 0.1):  # Run for 30 seconds
            # Get current state (simplified - in reality use sensors)
            transform = vehicle.get_transform()
            y = transform.location.y
            psi = np.radians(transform.rotation.yaw)
            
            # Simple state estimation (for demo)
            x = np.array([0.0, 0.0, psi, y])  # [beta, psi_dot, psi, y]
            
            # Reference path (sine wave)
            y_ref = 2.0 * np.sin(0.1 * t)
            
            # MPC control
            delta = mpc.optimize(x, y_ref)
            
            # Apply control
            vehicle.apply_control(
                carla.VehicleControl(
                    steer=delta,
                    throttle=0.3  # Constant speed
                )
            )
            
            # Log
            time_log.append(t)
            y_log.append(y)
            y_ref_log.append(y_ref)
            
            # CARLA tick
            world.tick()
    
    finally:
        # Cleanup
        vehicle.destroy()
        
        # Plot results
        plt.figure()
        plt.plot(time_log, y_log, label='Actual Path')
        plt.plot(time_log, y_ref_log, '--', label='Reference Path')
        plt.xlabel('Time (s)')
        plt.ylabel('Lateral Position (m)')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    main()