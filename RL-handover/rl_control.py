import os
import random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import glob
import sys
from tqdm import tqdm

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import mpc as MPCController

# Parameters
BUFFER_SIZE = int(1e5)  # Replay buffer size
BATCH_SIZE = 64         # Minibatch size
GAMMA = 0.99            # Discount factor
TAU = 1e-3              # For soft update of target parameters
LR = 5e-4               # Learning rate
UPDATE_EVERY = 4        # How often to update the network

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Fully connected layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        # Check if input is a single sample and add batch dimension if needed
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        x = F.relu(self.fc1(state))
        # Apply batch normalization only during training with batch size > 1
        if x.shape[0] > 1:
            x = self.bn1(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class RLAgent:
    """Agent that interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, num_discrete_actions=10):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            num_discrete_actions (int): number of discrete actions to use
        """
        self.state_size = state_size
        self.action_size = num_discrete_actions  # Number of discrete alpha values
        self.seed = random.seed(seed)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.num_discrete_actions = num_discrete_actions
        
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, num_discrete_actions, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, num_discrete_actions, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        # Initialize target network with same weights as local network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, 1.0)

        # Replay memory
        self.memory = ReplayBuffer(num_discrete_actions, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action_idx, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action_idx, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
                
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            action_idx = torch.argmax(action_values).item()
        else:
            action_idx = random.choice(range(self.action_size))
            
        # Convert discrete action index to continuous alpha value [0,1]
        alpha = action_idx / (self.num_discrete_actions - 1)
        
        return np.array([[alpha]]), action_idx

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get expected Q values from local model
        Q_local = self.qnetwork_local(states)
        Q_expected = Q_local.gather(1, actions)
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
                
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class HandoverController:
    """Controller that blends MPC and human control during handover"""
    
    def __init__(self, mpc_controller, human_controller, handover_trigger_time=5.0, handover_duration=5.0):
        self.mpc_controller = mpc_controller
        self.human_controller = human_controller
        self.handover_trigger_time = handover_trigger_time
        self.handover_duration = handover_duration
        self.handover_complete_time = handover_trigger_time + handover_duration

        self.alpha = 0.0  # 0 = full autonomous, 1 = full human
        self.prev_alpha = 0.0  # Store previous alpha for smoothness calculation
        
    def is_handover_active(self, current_time):
        return (current_time >= self.handover_trigger_time and 
                current_time < self.handover_complete_time)
                
    def get_handover_progress(self, current_time):
        if current_time < self.handover_trigger_time:
            return 0.0
        elif current_time >= self.handover_complete_time:
            return 1.0
        else:
            return (current_time - self.handover_trigger_time) / self.handover_duration
            
    def set_blend_factor(self, alpha):
        self.prev_alpha = self.alpha
        self.alpha = np.clip(alpha, 0.0, 1.0)
        
    def get_alpha_change_rate(self):
        return abs(self.alpha - self.prev_alpha)
        
    def get_control(self, vehicle, timestamp):
        # Get MPC control
        self.mpc_controller.update_controls()
        mpc_throttle, mpc_steer, mpc_brake = self.mpc_controller.get_commands()
        
        # Get human control (simulated)
        self.human_controller.update_controls()
        human_throttle, human_steer, human_brake = self.human_controller.get_commands()
        
        # Blend controls based on alpha
        throttle = (1 - self.alpha) * mpc_throttle + self.alpha * human_throttle
        steer = (1 - self.alpha) * mpc_steer + self.alpha * human_steer
        brake = (1 - self.alpha) * mpc_brake + self.alpha * human_brake
        
        return throttle, steer, brake


class CarlaHandoverEnv:
    """Environment for handover control in CARLA"""
    def __init__(
        self, 
        carla_world, 
        mpc_config, 
        handover_trigger_time=5.0, 
        handover_duration=5.0,
        state_features=['lateral_error', 'heading_error', 'speed_error', 'handover_progress'],
        max_episode_steps=100
    ):
        """Initialize environment for handover control learning"""
        self.world = carla_world
        self.player = None
        self.mpc_config = mpc_config
        self.handover_trigger_time = handover_trigger_time
        self.handover_duration = handover_duration
        self.state_features = state_features
        self.max_episode_steps = max_episode_steps
        
        self.state_dim = len(state_features)
        self.action_dim = 1  # Alpha (blend factor)
        
        self.current_time = 0.0
        self.current_step = 0
        self.episode_rewards = []
        
        self.autonomous_controller = None
        self.human_controller = None
        self.handover_controller = None
        
        self.target_waypoints = None
        self.prev_alpha = 0.0
        
    def reset(self):
        self.current_time = 0.0
        self.current_step = 0
        self.prev_alpha = 0.0
        
        if self.player:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            self.player.destroy()
            
            blueprint = self._get_vehicle_blueprint()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        else:
            blueprint = self._get_vehicle_blueprint()
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points)
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            
        self._init_controllers()
        
        state = self._get_state()
        
        return state
        
    def step(self, action):
        alpha = float(action[0, 0])  # Extract single alpha value from array
        self.handover_controller.set_blend_factor(alpha)
        
        throttle, steer, brake = self.handover_controller.get_control(self.player, self.current_time)
        control = carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
        self.player.apply_control(control)
        
        self.world.tick()
        self.current_time += self.mpc_config.time_step
        self.current_step += 1
        
        next_state = self._get_state()
        
        reward = self._compute_reward(alpha)
        
        done = self.current_step >= self.max_episode_steps or self._is_collision() or self._is_off_road()
        
        info = {
            'time': self.current_time,
            'handover_progress': self.handover_controller.get_handover_progress(self.current_time),
            'alpha': alpha
        }
        
        self.episode_rewards.append(reward)
        self.prev_alpha = alpha
        
        return next_state, reward, done, info
    
    def _get_vehicle_blueprint(self):
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        blueprint = random.choice(blueprints)
        blueprint.set_attribute('role_name', 'hero')
        return blueprint
    
    def _init_controllers(self):
        physic_control = self.player.get_physics_control()
        wheels = physic_control.wheels
        center_of_mass = physic_control.center_of_mass
        
        front_wheels = wheels[:2]
        rear_wheels = wheels[2:]
        front_pos = np.mean([np.array([w.position.x, w.position.y, w.position.z]) for w in front_wheels], axis=0)
        rear_pos = np.mean([np.array([w.position.x, w.position.y, w.position.z]) for w in rear_wheels], axis=0)
        wheelbase = np.sqrt(np.sum((front_pos - rear_pos)**2)) / 100.0  # Convert to meters
        lf = wheelbase - center_of_mass.x
        lr = center_of_mass.x
        
        self.autonomous_controller = MPCController.Controller(
            lf=lf,
            lr=lr,
            wheelbase=wheelbase,
            planning_horizon=self.mpc_config.planning_horizon,
            time_step=self.mpc_config.time_step
        )
        
        # Human controller (simulated with another MPC with different parameters)
        # We use slightly different parameters to simulate human driver style
        self.human_controller = MPCController.Controller(
            lf=lf,
            lr=lr,
            wheelbase=wheelbase,
            planning_horizon=int(self.mpc_config.planning_horizon * 0.8),  # Shorter horizon
            time_step=self.mpc_config.time_step
        )
        
        self.handover_controller = HandoverController(
            self.autonomous_controller, 
            self.human_controller,
            handover_trigger_time=self.handover_trigger_time,
            handover_duration=self.handover_duration
        )
        
        self._update_controllers()
        
    def _update_controllers(self):
        transform = self.player.get_transform()
        velocity = self.player.get_velocity()
        
        current_x = transform.location.x
        current_y = transform.location.y
        current_yaw = self._wrap_angle(transform.rotation.yaw)
        current_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        self.autonomous_controller.update_values(
            current_x, current_y, current_yaw, current_speed, self.current_time, 1  # Changed frame to 1
        )
        self.human_controller.update_values(
            current_x, current_y, current_yaw, current_speed, self.current_time, 1  # Changed frame to 1
        )
        
        waypoints = self._generate_waypoints()
        self.autonomous_controller.update_waypoints(waypoints)
        self.human_controller.update_waypoints(waypoints)
        
    def _generate_waypoints(self):
        waypoints = []
        current_location = self.player.get_transform().location
        prev_waypoint = self.world.get_map().get_waypoint(current_location)
        
        velocity = self.player.get_velocity()
        current_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        dist = max(self.mpc_config.time_step * current_speed + 1.0, 2.0)  # Ensure minimum distance
        
        for i in range(self.mpc_config.planning_horizon + 1):  # +1 to ensure we have enough waypoints
            next_waypoints = prev_waypoint.next(dist)
            if not next_waypoints:
                break
                
            current_waypoint = next_waypoints[0]
            
            waypoints.append([
                current_waypoint.transform.location.x,
                current_waypoint.transform.location.y,
                self.mpc_config.desired_speed,
                self._wrap_angle(current_waypoint.transform.rotation.yaw)
            ])
            
            prev_waypoint = current_waypoint
            
        # Make sure we have enough waypoints for the planning horizon
        while len(waypoints) <= self.mpc_config.planning_horizon:
            # If we run out of waypoints, repeat the last one
            if waypoints:
                last_wp = waypoints[-1].copy()
                # Add a small offset to avoid duplicates
                last_wp[0] += 0.1
                last_wp[1] += 0.1
                waypoints.append(last_wp)
            else:
                # If no waypoints at all, create a default one ahead
                waypoints.append([
                    current_location.x + 1.0,
                    current_location.y,
                    self.mpc_config.desired_speed,
                    0.0
                ])
            
        self.target_waypoints = waypoints
        return waypoints
        
    def _get_state(self):
        self._update_controllers()
        
        transform = self.player.get_transform()
        velocity = self.player.get_velocity()
        
        current_x = transform.location.x
        current_y = transform.location.y
        current_yaw = self._wrap_angle(transform.rotation.yaw)
        current_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        lateral_error = 0.0
        heading_error = 0.0
        speed_error = 0.0
        
        if self.target_waypoints:
            min_dist = float('inf')
            closest_idx = 0
            
            for i, wp in enumerate(self.target_waypoints):
                wp_x, wp_y = wp[0], wp[1]
                dist = math.sqrt((wp_x - current_x)**2 + (wp_y - current_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            
            if closest_idx < len(self.target_waypoints):
                wp = self.target_waypoints[closest_idx]
                wp_x, wp_y, wp_speed, wp_yaw = wp
                
                # Calculate lateral error (cross-track error)
                # Use the closest point and the next point to define the path segment
                next_idx = min(closest_idx + 1, len(self.target_waypoints) - 1)
                if next_idx != closest_idx:
                    next_wp = self.target_waypoints[next_idx]
                    next_x, next_y = next_wp[0], next_wp[1]
                    
                    # Calculate path direction
                    path_dx = next_x - wp_x
                    path_dy = next_y - wp_y
                    path_length = np.sqrt(path_dx**2 + path_dy**2)
                    if path_length > 0:
                        path_dx /= path_length
                        path_dy /= path_length
                        
                        # Calculate perpendicular distance (lateral error)
                        v_dx = current_x - wp_x
                        v_dy = current_y - wp_y
                        lateral_error = abs(v_dx * (-path_dy) + v_dy * path_dx)
                    else:
                        lateral_error = min_dist
                else:
                    lateral_error = min_dist
                
                # Heading error - smallest angle between vehicle yaw and path direction
                heading_error = abs(wp_yaw - current_yaw)
                if heading_error > np.pi:
                    heading_error = 2 * np.pi - heading_error
                
                # Speed error
                speed_error = abs(wp_speed - current_speed)
        
        # Get handover progress
        handover_progress = self.handover_controller.get_handover_progress(self.current_time)
        
        # Build state vector based on requested features
        state = []
        for feature in self.state_features:
            if feature == 'lateral_error':
                state.append(min(lateral_error, 10.0))  # Cap to reasonable range
            elif feature == 'heading_error':
                state.append(heading_error)
            elif feature == 'speed_error':
                state.append(min(speed_error, 10.0))  # Cap to reasonable range
            elif feature == 'handover_progress':
                state.append(handover_progress)
            elif feature == 'current_speed':
                state.append(min(current_speed / 30.0, 1.0))  # Normalize speed
            elif feature == 'current_time':
                state.append(self.current_time / 10.0)  # Normalize time
            elif feature == 'prev_alpha':
                state.append(self.prev_alpha)
        
        return np.array(state, dtype=np.float32)
        
    def _compute_reward(self, alpha):
        transform = self.player.get_transform()
        velocity = self.player.get_velocity()
        
        current_x = transform.location.x
        current_y = transform.location.y
        current_yaw = self._wrap_angle(transform.rotation.yaw)
        current_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        reward = 0.0
        
        # 1. Path following reward (lateral error penalty)
        lateral_error = 0.0
        if self.target_waypoints:
            min_dist = float('inf')
            closest_idx = 0
            
            for i, wp in enumerate(self.target_waypoints):
                wp_x, wp_y = wp[0], wp[1]
                dist = math.sqrt((wp_x - current_x)**2 + (wp_y - current_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            
            lateral_error = min_dist
        
        # Exponential decay for lateral error penalty
        path_reward = -1.0 * np.exp(lateral_error)
        
        # 2. Speed maintenance reward
        speed_error = abs(self.mpc_config.desired_speed - current_speed)
        speed_reward = -0.5 * speed_error
        
        # 3. Handover smoothness reward
        handover_reward = 0.0
        progress = self.handover_controller.get_handover_progress(self.current_time)
        
        # Calculate smoothness penalty based on change in alpha
        smoothness_penalty = -2.0 * abs(alpha - self.prev_alpha)
        
        if self.handover_controller.is_handover_active(self.current_time):
            # During handover, reward smooth transitions that follow a sigmoid curve
            ideal_alpha = 1.0 / (1.0 + np.exp(-10 * (progress - 0.5)))  # Sigmoid function
            handover_reward = -4.0 * abs(alpha - ideal_alpha) + smoothness_penalty
        else:
            # Before handover, alpha should be 0; after handover, alpha should be 1
            if self.current_time < self.handover_trigger_time:
                handover_reward = -4.0 * abs(alpha - 0.0) + smoothness_penalty
            else:
                handover_reward = -4.0 * abs(alpha - 1.0) + smoothness_penalty
        
        # Combine rewards
        reward = path_reward + speed_reward + handover_reward
        
        # Additional penalties
        if self._is_collision():
            reward -= 100.0
            
        if self._is_off_road():
            reward -= 50.0
        
        return reward
    
    def _is_collision(self):
        # Simple collision detection using sensors would be implemented here
        # For now, we'll just return False as a placeholder
        return False
        
    def _is_off_road(self):
        # Check if the vehicle is on a driving lane
        waypoint = self.world.get_map().get_waypoint(self.player.get_transform().location)
        return not waypoint.lane_type == carla.LaneType.Driving
        
    def _wrap_angle(self, angle_in_degree):
        angle_in_rad = angle_in_degree / 180.0 * np.pi
        while angle_in_rad > np.pi:
            angle_in_rad -= 2 * np.pi
        while angle_in_rad <= -np.pi:
            angle_in_rad += 2 * np.pi
        return angle_in_rad


def train_rl_agent(env, agent, num_episodes=1000, max_steps=100):
    """Train the RL agent on the handover control task"""
    rewards = []
    
    for i_episode in tqdm(range(1, num_episodes+1), desc="Training"):
        state = env.reset()
        episode_reward = 0
        
        for t in range(max_steps):
            action, action_idx = agent.act(state, agent.epsilon)
            next_state, reward, done, _ = env.step(action)
            
            agent.step(state, np.array([[action_idx]]), reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        rewards.append(episode_reward)
        
        if i_episode % 100 == 0:
            print(f"Episode {i_episode}/{num_episodes}, Avg Reward: {np.mean(rewards[-100:]):.2f}, Epsilon: {agent.epsilon:.2f}")
    
    return rewards


def evaluate_policy(env, agent, num_episodes=10, render=False):
    """Evaluate the trained policy without exploration"""
    rewards = []
    
    for i in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = agent.act(state, eps=0.0)  # Greedy actions
            next_state, reward, done, info = env.step(action)
            
            if render:
                print(f"Time: {info['time']:.1f}, Progress: {info['handover_progress']:.2f}, "
                      f"Alpha: {info['alpha']:.2f}, Reward: {reward:.2f}")
            
            state = next_state
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    avg_reward = np.mean(rewards)
    print(f"Evaluation over {num_episodes} episodes: {avg_reward:.2f}")
    return avg_reward


def main():
    # Connect to CARLA server
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    world = client.get_world()
    
    # Configure weather and synchronous mode
    world_settings = world.get_settings()
    world_settings.synchronous_mode = True
    world_settings.fixed_delta_seconds = 0.05
    world.apply_settings(world_settings)
    
    # MPC configuration
    class MPCConfig:
        planning_horizon = 10
        time_step = 0.1
        desired_speed = 30
        waypoint_resolution = 0.5
        waypoint_lookahead_distance = 5.0
    
    mpc_config = MPCConfig()
    
    env = CarlaHandoverEnv(
        world,
        mpc_config,
        handover_trigger_time=5.0,
        handover_duration=5.0,
        max_episode_steps=100
    )
    
    state_dim = env.state_dim
    action_dim = env.action_dim
    agent = RLAgent(state_size=state_dim, action_size=action_dim, seed=0)
    
    print("Training RL agent...")
    train_rl_agent(env, agent, num_episodes=500, max_steps=env.max_episode_steps)
    
    torch.save(agent.qnetwork_local.state_dict(), "handover_policy.pth")
    
    print("Evaluating policy...")
    evaluate_policy(env, agent)
    
    print("Training complete!")


if __name__ == "__main__":
    main()