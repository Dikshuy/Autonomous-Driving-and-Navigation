import carla
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import time
import random
from collections import deque

# CARLA Setup
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# Spawn ego vehicle
vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
spawn_point = carla.Transform(carla.Location(x=100, y=0, z=0.5), carla.Rotation(yaw=0))
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Enable autopilot (for traffic, if needed)
vehicle.set_autopilot(False)

# Sensors (for state estimation)
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=vehicle)

class BicycleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = 1500.0  # kg
        self.Iz = 3000.0  # kg*m^2
        self.lf = 1.2  # m
        self.lr = 1.6  # m
        self.Cf = 80000.0  # N/rad
        self.Cr = 80000.0  # N/rad
        self.Vx = 10.0  # m/s (constant speed)
    
    def forward(self, x, delta):
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
        
        return torch.stack([d_beta, d_psi_dot, d_psi, d_y])

def discrete_update(x, delta, dt=0.1):
    dx = bicycle_model(x, delta)
    return x + dx * dt

def autonomous_mpc(x, y_ref):
    def cost(delta):
        x_next = discrete_update(x, delta)
        return (x_next[3] - y_ref)**2 + 0.1 * delta**2
    
    delta = torch.linspace(-0.5, 0.5, 100)
    costs = torch.stack([cost(d) for d in delta])
    return delta[torch.argmin(costs)]

class HumanMPC:
    def __init__(self):
        self.delay_buffer = deque(maxlen=5)  # 0.5s delay
        self.noise_std = 0.05
    
    def get_steering(self, x, y_ref):
        self.delay_buffer.append((x.clone(), y_ref))
        if len(self.delay_buffer) == self.delay_buffer.maxlen:
            x_delayed, y_ref_delayed = self.delay_buffer.popleft()
            delta = autonomous_mpc(x_delayed, y_ref_delayed)
            delta += torch.randn(1) * self.noise_std
            return torch.clamp(delta, -0.5, 0.5)
        return torch.tensor(0.0)
    
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = torch.sigmoid(self.mean(x))  # alpha ∈ [0,1]
        log_std = self.log_std(x)
        std = torch.exp(log_std)
        return Normal(mean, std)

class SAC:
    def __init__(self, state_dim, action_dim):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.replay_buffer = deque(maxlen=100000)
    
    def act(self, state):
        state = torch.FloatTensor(state)
        dist = self.policy(state)
        action = dist.sample()
        return action.clamp(0, 1).item()
    
    def update(self, batch_size=128):
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # TODO: Implement SAC loss (Q-functions, entropy, etc.)
        # (Omitted for brevity, but follows standard SAC implementation)
        
        self.optimizer.step()

def simulate_handover():
    # Initialize
    sac_agent = SAC(state_dim=5, action_dim=1)
    human_mpc = HumanMPC()
    bicycle_model = BicycleModel()
    
    # State: [beta, psi_dot, psi, y, t_handover]
    x = torch.zeros(4)  # Initial state
    y_ref = 0.0  # Lane center
    alpha_log, y_log, time_log = [], [], []
    
    for t in np.arange(0.0, 20.0, 0.1):
        # Get reference (e.g., slight curve)
        y_ref = 2.0 * np.sin(0.1 * t) if t > 5.0 else 0.0
        
        # State observation
        obs = np.array([x[0], x[1], x[3], max(0, t - 10.0), x[3] - y_ref])
        
        # Control selection
        if t < 10.0:
            delta = autonomous_mpc(x, y_ref)
            alpha = 1.0
        else:
            alpha = sac_agent.act(obs)
            delta_auto = autonomous_mpc(x, y_ref)
            delta_human = human_mpc.get_steering(x, y_ref)
            delta = alpha * delta_auto + (1 - alpha) * delta_human
        
        # Update state
        x = discrete_update(x, delta)
        
        # Apply control in CARLA
        vehicle.apply_control(carla.VehicleControl(steer=delta.item(), throttle=0.3))
        
        # Log
        alpha_log.append(alpha)
        y_log.append(x[3].item())
        time_log.append(t)
        
        # Render (optional)
        world.tick()
    
    # Plot results
    plt.figure()
    plt.plot(time_log, y_log, label="Lateral position")
    plt.plot(time_log, alpha_log, label="RL blending (alpha)")
    plt.axvline(x=10.0, color='r', linestyle='--', label="Handover start")
    plt.legend()
    plt.show()

simulate_handover()