import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

class VehicleModel:
    def __init__(self, mass, a, b, I_z, C_af, C_ar, delta, dt=0.01, sim_time=8):
        self.m = mass
        self.a = a
        self.b = b
        self.I_z = I_z
        self.C_af = C_af
        self.C_ar = C_ar
        self.delta = delta
        self.dt = dt
        self.t = np.arange(0, sim_time, dt)
        
    def create_system(self, speed):
        u = speed * (1000 / 3600)
        A = np.array([[-(self.C_af + self.C_ar) / (u * self.m), -((self.a * self.C_af - self.b * self.C_ar) / (u * self.m)) - u],
                      [-(self.a * self.C_af - self.b * self.C_ar) / (u * self.I_z), -((self.a ** 2) * self.C_af + (self.b ** 2) * self.C_ar) / (u * self.I_z)]])
        B = np.array([[self.C_af / self.m], [self.a * (self.C_af / self.I_z)]])
        C = np.eye(2)
        D = np.zeros((2, 1))
        return signal.StateSpace(A, B, C, D), A
    
    def simulate(self, speed):
        sys, A = self.create_system(speed)
        sys_d = signal.cont2discrete((sys.A, sys.B, sys.C, sys.D), self.dt)
        delta_input = np.ones_like(self.t) * self.delta
        _, y, _ = signal.dlsim(sys_d, delta_input)
        return y, A
    
    def compute_trajectory(self, y, speed):
        u = speed * (1000 / 3600)
        theta, X, Y, a_lat, alpha_f, alpha_r = np.zeros((6, len(self.t)))
        y = np.array(y).reshape(len(self.t), -1)
        
        for n in range(1, len(self.t)):
            theta[n] = y[n, 1] * self.dt + theta[n-1]
            X[n] = (u * np.cos(theta[n]) + y[n, 0] * np.sin(theta[n])) * self.dt + X[n-1]
            Y[n] = (u * np.sin(theta[n]) - y[n, 0] * np.cos(theta[n])) * self.dt + Y[n-1]
            a_lat[n] = (y[n, 0] - y[n-1, 0]) / self.dt
            alpha_f[n] = self.delta - (y[n, 0] + self.a * y[n, 1]) / u
            alpha_r[n] = (self.b * y[n, 1] - y[n, 0]) / u
        
        return X, Y, theta, a_lat, alpha_f, alpha_r
    
    def plot_results(self, speeds):
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        labels = ["Lateral Velocity (m/s)", "Yaw Rate (rad/s)", "Lateral Acceleration (m/s²)",
                  "Vehicle Path", "Front Slip Angle (rad)", "Rear Slip Angle (rad)"]
        
        for speed in speeds:
            y, _ = self.simulate(speed)
            X, Y, theta, a_lat, alpha_f, alpha_r = self.compute_trajectory(y, speed)
            data = [y[:, 0], y[:, 1], a_lat, (X, Y), alpha_f, alpha_r]
            
            for j, d in enumerate(data):
                ax = axs[j // 3, j % 3]
                if j == 3:
                    ax.plot(d[0], d[1], label=f"Speed {speed} km/h", linewidth=2)
                else:
                    ax.plot(self.t, d, label=f"Speed {speed} km/h", linewidth=2)
                ax.set_title(labels[j], fontsize=14)
                ax.set_xlabel("Time (s)" if j != 3 else "X (m)", fontsize=12)
                ax.set_ylabel(labels[j], fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend()
                ax.tick_params(axis='both', which='major', labelsize=10)
        
        plt.tight_layout()
        plt.savefig("q2_results_1.png")
        plt.show()
    
    def check_stability(self, speeds):
        for speed in speeds:
            _, A = self.simulate(speed)
            poles = np.linalg.eigvals(A)
            print(f"Poles for speed {speed} km/h: {poles}")
            stability = "Stable" if np.all(np.real(poles) < 0) else "Unstable"
            print(f"System at {speed} km/h is {stability}\n")

if __name__ == "__main__":
    vehicle = VehicleModel(mass=1780, a=1.32, b=1.46, I_z=4400, C_af=70500, C_ar=70500, delta=0.04)
    speeds = [65, 135]
    vehicle.plot_results(speeds)
    vehicle.check_stability(speeds)