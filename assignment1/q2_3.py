import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

class VehicleModel:
    def __init__(self, mass, a, b, I_z, C_af, C_ar, delta, dt=0.01, sim_time=10):
        self.m = mass
        self.a = a
        self.b = b
        self.I_z = I_z
        self.C_af = C_af
        self.C_ar = C_ar
        self.delta = delta
        self.dt = dt
        self.t = np.arange(0, sim_time, dt)
        self.alpha_s = 0.1222 # 7' in rad
        self.F_yf = self.C_af * self.alpha_s
        self.F_yr = self.C_ar * self.alpha_s

    def create_system(self, speed):
        u = speed * (1000 / 3600)
        A = np.array([[-(self.C_af + self.C_ar) / (u * self.m), -((self.a * self.C_af - self.b * self.C_ar) / (u * self.m)) - u],
                      [-(self.a * self.C_af - self.b * self.C_ar) / (u * self.I_z), -((self.a ** 2) * self.C_af + (self.b ** 2) * self.C_ar) / (u * self.I_z)]])
        B = np.array([[self.C_af / self.m], [self.a * (self.C_af / self.I_z)]])
        C = np.eye(2)
        D = np.zeros((2, 1))
        return signal.StateSpace(A, B, C, D)

    def create_additional_systems(self, speed, condition):
        u = speed * (1000 / 3600)
        if condition == "front_saturated":
            A = np.array([[-self.C_ar / (u * self.m), -((self.b * self.C_ar) / (u * self.m)) - u],
                          [-(-self.b * self.C_ar) / (u * self.I_z), -((self.b ** 2 * self.C_ar) / (u * self.I_z))]])
            B = np.array([[0], [0]])
        elif condition == "rear_saturated":
            A = np.array([[-self.C_af / (u * self.m), -((self.a * self.C_af) / (u * self.m)) - u],
                          [-(self.a * self.C_af) / (u * self.I_z), -((self.a ** 2 * self.C_af) / (u * self.I_z))]])
            B = np.array([[self.C_af / self.m], [self.a * (self.C_af / self.I_z)]])
        elif condition == "both_saturated":
            A = np.array([[0, -u],
                          [0, 0]])
            B = np.array([[0], [0]])
        C = np.eye(2)
        D = np.zeros((2, 1))
        return signal.StateSpace(A, B, C, D)

    def simulate(self, speed):
        sys = self.create_system(speed)
        sys_d = sys.to_discrete(self.dt)
        x = np.zeros((2, len(self.t)))
        theta = np.zeros(len(self.t))
        X = np.zeros(len(self.t))
        Y = np.zeros(len(self.t))
        a_lat = np.zeros(len(self.t))
        alpha_f = np.zeros(len(self.t))
        alpha_r = np.zeros(len(self.t))

        alpha_f[0] = self.delta
        alpha_r[0] = 0

        for n in range(1, len(self.t)):
            if alpha_f[n-1] < self.alpha_s and alpha_r[n-1] < self.alpha_s:
                x[:, n] = sys_d.A @ x[:, n-1] + sys_d.B.flatten() * self.delta
            elif alpha_f[n-1] >= self.alpha_s and alpha_r[n-1] < self.alpha_s:
                sys_additional = self.create_additional_systems(speed, "front_saturated")
                sys_d_additional = sys_additional.to_discrete(self.dt)
                x[:, n] = sys_d_additional.A @ x[:, n-1] + np.array([self.F_yf / self.m, self.a * self.F_yf / self.I_z]) * self.dt
            elif alpha_f[n-1] < self.alpha_s and alpha_r[n-1] >= self.alpha_s:
                sys_additional = self.create_additional_systems(speed, "rear_saturated")
                sys_d_additional = sys_additional.to_discrete(self.dt)
                x[:, n] = sys_d_additional.A @ x[:, n-1] + sys_d_additional.B.flatten() * self.delta + np.array([self.F_yr / self.m, -self.b * self.F_yr / self.I_z]) * self.dt
            else:
                sys_additional = self.create_additional_systems(speed, "both_saturated")
                sys_d_additional = sys_additional.to_discrete(self.dt)
                x[:, n] = sys_d_additional.A @ x[:, n-1] + np.array([(self.F_yf + self.F_yr) / self.m, (self.a * self.F_yf - self.b * self.F_yr) / self.I_z]) * self.dt

            theta[n] = x[1, n] * self.dt + theta[n-1]
            X[n] = (speed * (1000 / 3600) * np.cos(theta[n]) + x[0, n] * np.sin(theta[n])) * self.dt + X[n-1]
            Y[n] = (speed * (1000 / 3600) * np.sin(theta[n]) - x[0, n] * np.cos(theta[n])) * self.dt + Y[n-1]
            a_lat[n] = (x[0, n] - x[0, n-1]) / self.dt
            alpha_f[n] = self.delta - (x[0, n] + self.a * x[1, n]) / (speed * (1000 / 3600))
            alpha_r[n] = (self.b * x[1, n] - x[0, n]) / (speed * (1000 / 3600))

        return x, theta, X, Y, a_lat, alpha_f, alpha_r

    def plot_results(self, speeds):
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        labels = ["Lateral Velocity (m/s)", "Yaw Rate (rad/s)", "Lateral Acceleration (m/s²)",
                  "Vehicle Path", "Front Slip Angle (rad)", "Rear Slip Angle (rad)"]

        for speed in speeds:
            x, theta, X, Y, a_lat, alpha_f, alpha_r = self.simulate(speed)
            data = [x[0, :], x[1, :], a_lat, (X, Y), alpha_f, alpha_r]

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
        plt.savefig("q2_results_3.png")
        plt.show()

    def check_stability(self, speeds):
        for speed in speeds:
            sys = self.create_system(speed)
            poles = np.linalg.eigvals(sys.A)
            print(f"Poles for speed {speed} km/h: {poles}")
            stability = "Stable" if np.all(np.real(poles) < 0) else "Unstable"
            print(f"System at {speed} km/h is {stability}\n")

if __name__ == "__main__":
    vehicle = VehicleModel(mass=1780, a=1.32, b=1.46, I_z=4400, C_af=70500, C_ar=70500, delta=0.04)
    speeds = [65, 135]
    vehicle.plot_results(speeds)
    vehicle.check_stability(speeds)