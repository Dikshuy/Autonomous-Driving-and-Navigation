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
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Architecture for continuous action space (alpha)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        return torch.sigmoid(self.fc3(x))


class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class RLAgent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return action_values.cpu().data.numpy()
        else:
            return np.random.uniform(0, 1, size=(1, self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions.long())

        loss = F.mse_loss(Q_expected, Q_targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)  
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def soft_update(self, local_model, target_model, tau):
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
        self.alpha = np.clip(alpha, 0.0, 1.0)
        
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
        
    def reset(self):
        self.current_time = 0.0
        self.current_step = 0
        
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
        alpha = float(action[0])
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
            current_x, current_y, current_yaw, current_speed, self.current_time, 0
        )
        self.human_controller.update_values(
            current_x, current_y, current_yaw, current_speed, self.current_time, 0
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
        
        dist = self.mpc_config.time_step * current_speed + 0.1
        
        for i in range(self.mpc_config.planning_horizon):
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
            
            wp = self.target_waypoints[closest_idx]
            wp_x, wp_y, wp_speed, wp_yaw = wp
            
            lateral_error = min_dist
            
            heading_error = abs(wp_yaw - current_yaw)
            if heading_error > np.pi:
                heading_error = 2 * np.pi - heading_error
            
            speed_error = abs(wp_speed - current_speed)
        
        handover_progress = self.handover_controller.get_handover_progress(self.current_time)
        
        state = []
        
        for feature in self.state_features:
            if feature == 'lateral_error':
                state.append(lateral_error)
            elif feature == 'heading_error':
                state.append(heading_error)
            elif feature == 'speed_error':
                state.append(speed_error)
            elif feature == 'handover_progress':
                state.append(handover_progress)
            elif feature == 'current_speed':
                state.append(current_speed)
            elif feature == 'current_time':
                state.append(self.current_time)
        
        return np.array(state)
        
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
        
        path_reward = -lateral_error * 0.5
        
        # 2. Speed maintenance reward
        speed_reward = -abs(self.mpc_config.desired_speed - current_speed) * 0.1
        
        # 3. Handover smoothness reward
        # During handover, reward smooth transitions in alpha
        handover_reward = 0.0
        if self.handover_controller.is_handover_active(self.current_time):
            # Get handover progress (0 to 1)
            progress = self.handover_controller.get_handover_progress(self.current_time)
            
            # Simple linear blend would be alpha = progress
            # Reward smaller deviation from this baseline during transition
            baseline_alpha = progress
            handover_reward = -abs(alpha - baseline_alpha) * 2.0
        else:
            # Before handover, alpha should be 0; after handover, alpha should be 1
            if self.current_time < self.handover_trigger_time:
                handover_reward = -abs(alpha - 0.0) * 2.0
            else:
                handover_reward = -abs(alpha - 1.0) * 2.0
        
        reward = path_reward + speed_reward + handover_reward
        
        if self._is_off_road():
            reward -= 50.0
        
        return reward
        
    def _is_off_road(self):
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
    rewards = []
    
    for i_episode in tqdm(range(1, num_episodes+1), desc="Training"):
        state = env.reset()
        episode_reward = 0
        
        for t in range(max_steps):
            action = agent.act(state, agent.epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        rewards.append(episode_reward)
        
        if i_episode % 100 == 0:
            print(f"Episode {i_episode}/{num_episodes}, Avg Reward: {np.mean(rewards[-100:]):.2f}, Epsilon: {agent.epsilon:.2f}")
    
    return rewards


def evaluate_policy(env, agent, num_episodes=10):
    rewards = []
    
    for i in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.act(state, eps=0.0)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    avg_reward = np.mean(rewards)
    print(f"Evaluation over {num_episodes} episodes: {avg_reward:.2f}")
    return avg_reward


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    world = client.get_world()
    
    class MPCConfig:
        planning_horizon = 5
        time_step = 0.3
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
    rewards = train_rl_agent(env, agent, num_episodes=500, max_steps=env.max_episode_steps)
    
    torch.save(agent.qnetwork_local.state_dict(), "handover_policy.pth")
    
    print("Evaluating policy...")
    eval_reward = evaluate_policy(env, agent)
    
    print("Training complete!")


if __name__ == "__main__":
    main()