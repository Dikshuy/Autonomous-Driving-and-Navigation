from typing import Tuple, Optional, List
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt


class BicycleMPC:
    """
    Model Predictive Controller for a Bicycle Kinematic Model.
    """

    def __init__(
        self, 
        wheelbase: float = 2.5, 
        velocity: float = 7.0, 
        horizon_length: int = 5, 
        time_step: float = 0.1
    ):
        self.L = wheelbase
        self.v = velocity
        self.N = horizon_length
        self.dt = time_step
        
        self.Q = np.eye(3)
        
        self.R = np.eye(1)

    def create_bicycle_model(self) -> ca.Function:
        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        theta = ca.MX.sym('theta')
        state = ca.vertcat(x, y, theta)
        
        delta = ca.MX.sym('delta')
        control = ca.vertcat(delta)

        x_next = x + self.v * ca.cos(theta) * self.dt
        y_next = y + self.v * ca.sin(theta) * self.dt
        theta_next = theta + (self.v / self.L) * ca.tan(delta) * self.dt
        
        next_state = ca.vertcat(x_next, y_next, theta_next)
        
        return ca.Function('bicycle_model', [state, control], [next_state])

    @staticmethod
    def generate_reference_trajectory(
        t_values: np.ndarray, 
        t0: float = 10.0, 
        alpha: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_values = 1 / (1 + np.exp(-alpha * (t_values - t0)))
        return t_values, y_values

    def cost_function(
        self, 
        state: ca.MX, 
        control: ca.MX, 
        target: ca.MX, 
        control_prev: Optional[ca.MX] = None
    ) -> ca.MX:
        state_cost = ca.mtimes((state[:2] - target).T, self.Q[:2, :2] @ (state[:2] - target))
        
        control_cost = ca.mtimes(control.T, self.R @ control)
        
        if control_prev is not None:
            jerk_penalty = ca.sumsqr(control - control_prev)
        else:
            jerk_penalty = 0
        
        return ca.sum1(state_cost) + ca.sum1(control_cost) + jerk_penalty

    def run_mpc_simulation(self) -> Tuple[List[float], List[float], List[float]]:
        bicycle_model = self.create_bicycle_model()

        t_values = np.linspace(0, 20, 50)
        target_x, target_y = self.generate_reference_trajectory(t_values)

        opti = ca.Opti()
        X = opti.variable(3, self.N + 1)  # State trajectory
        U = opti.variable(1, self.N)      # Control inputs

        opti.set_initial(X, 0)
        opti.set_initial(U, 0)

        target_param = opti.parameter(2, self.N)

        previous_control = np.zeros(1)

        objective = 0

        for k in range(self.N):
            current_state = X[:, k]
            next_state = X[:, k + 1]
            control_input = U[:, k]

            control_previous = previous_control if k == 0 else U[:, k - 1]
            
            target = target_param[:, k]
            
            objective += self.cost_function(current_state, control_input, target, control_previous)

        for k in range(self.N):
            opti.subject_to(X[:, k+1] == bicycle_model(X[:, k], U[:, k]))
        
        opti.subject_to(opti.bounded(-np.pi/4, U[0, :], np.pi/4))

        opti.minimize(objective)
        opti.solver('ipopt')

        x_trajectory: List[float] = []
        y_trajectory: List[float] = []
        steering_angle_all: List[float] = []
        current_state = np.array([0, 0, 0])

        time_steps = np.arange(len(target_x)) * self.dt
        for t in range(len(target_x) - self.N):
            opti.set_initial(X[:, 0], current_state)

            target_segment = np.vstack((target_x[t:t + self.N], target_y[t:t + self.N]))
            opti.set_value(target_param, target_segment)

            solution = opti.solve()

            optimal_u = solution.value(U[:, 0])
            previous_control = optimal_u

            current_state = solution.value(X[:, 1])
            
            x_trajectory.append(current_state[0])
            y_trajectory.append(current_state[1])
            steering_angle_all.append(optimal_u)

        return x_trajectory, y_trajectory, steering_angle_all

    def visualize_results(
        self, 
        x_trajectory: List[float], 
        y_trajectory: List[float], 
        steering_angles: List[float]
    ):
        t_values = np.linspace(0, 20, 50)
        target_x, target_y = self.generate_reference_trajectory(t_values)
        time_steps = np.arange(len(target_x)) * self.dt

        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(x_trajectory, y_trajectory, marker='o', label='Actual Trajectory')
        plt.plot(target_x, target_y, 'r--', label='Desired Trajectory')
        plt.title('Vehicle Trajectory')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(time_steps[:len(steering_angles)], steering_angles, label='Steering Angle (rad)')
        plt.title('Control Inputs over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Steering Angle (rad)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        
def main():
    mpc = BicycleMPC()

    x_traj, y_traj, steering_angles = mpc.run_mpc_simulation()

    mpc.visualize_results(x_traj, y_traj, steering_angles)

if __name__ == "__main__":
    main()
