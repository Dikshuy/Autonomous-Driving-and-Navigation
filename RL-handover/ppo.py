import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (torch.FloatTensor(state), torch.FloatTensor(action), 
                torch.FloatTensor(reward), torch.FloatTensor(next_state), 
                torch.FloatTensor(done))
    
    def __len__(self):
        return len(self.buffer)

class RLHandoverController:
    def __init__(self, autonomous_controller, human_controller, state_dim=10, action_dim=1, 
                 handover_start_time=10.0, handover_duration=7.0, lr=0.001, gamma=0.99):
        self.autonomous_controller = autonomous_controller  # MPC controller for autonomous driving
        self.human_controller = human_controller            # MPC controller simulating human driving
        
        self.handover_start_time = handover_start_time
        self.handover_duration = handover_duration
        self.handover_end_time = handover_start_time + handover_duration
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = 64
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.current_state = None
        self.current_action = None
        self.training_mode = True
        self.last_reward = 0
        self.episode_rewards = []
        self.is_handover_phase = False
        
    def get_state(self, vehicle, current_time, waypoints):
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        acceleration = vehicle.get_acceleration()
        angular_velocity = vehicle.get_angular_velocity()
        
        x = transform.location.x
        y = transform.location.y
        yaw = transform.rotation.yaw * np.pi / 180.0
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        target_x, target_y, target_speed, target_yaw = waypoints[0]
        
        handover_progress = 0.0
        if current_time > self.handover_start_time:
            handover_progress = min(1.0, (current_time - self.handover_start_time) / self.handover_duration)

        state = np.array([
            x, y, yaw, speed,
            velocity.x, velocity.y,
            acceleration.x, acceleration.y,
            handover_progress,
            current_time
        ], dtype=np.float32)
        
        return state
    
    def select_action(self, state, explore=True):
        """
        Select blending parameter α using ε-greedy policy
        α = 1: fully autonomous control
        α = 0: fully human control
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if explore and random.random() < self.epsilon:
            handover_progress = state[8]
            # Bias exploration around the natural progression (1 -> 0)
            mean = 1.0 - handover_progress
            # Add some noise around this mean
            action = np.clip(np.random.normal(mean, 0.2), 0.0, 1.0)
            return np.array([action], dtype=np.float32)
        else:
            with torch.no_grad():
                action = self.policy_net(state_tensor).cpu().numpy()[0]
            return action
    
    def compute_reward(self, vehicle, waypoints, blending_param, prev_blending_param, current_time):
        """
        Calculate reward based on:
        1. Safety (distance to obstacles, staying on road)
        2. Comfort (smooth control, acceleration, jerk)
        3. Handover smoothness (smooth transition in control authority)
        """
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # Calculate tracking error to waypoints
        target_x, target_y, target_speed, target_yaw = waypoints[0]
        position_error = np.sqrt((transform.location.x - target_x)**2 + (transform.location.y - target_y)**2)
        speed_error = abs(speed - target_speed)
        
        # Calculate handover smoothness - penalize abrupt changes in blending parameter
        blend_change = abs(blending_param - prev_blending_param)
        
        # Calculate progress of handover
        handover_progress = 0.0
        if current_time > self.handover_start_time:
            handover_progress = min(1.0, (current_time - self.handover_start_time) / self.handover_duration)
        
        # Weighted reward components
        tracking_reward = -0.5 * position_error - 0.3 * speed_error
        smoothness_reward = -5.0 * blend_change  # Strongly penalize abrupt changes
        
        # Goal-oriented reward: parameter should approach 0 (human control) by the end of handover
        target_alpha = max(0, 1.0 - handover_progress)
        goal_reward = -2.0 * abs(blending_param[0] - target_alpha)
        
        # Combine rewards
        reward = tracking_reward + smoothness_reward + goal_reward
        
        # Add bonus for completing handover
        if handover_progress >= 0.99 and blending_param[0] < 0.1:
            reward += 10.0
            
        return reward
    
    def blend_controls(self, vehicle, waypoints, current_time, simulation_time):
        """
        Blend controls from autonomous and human controllers
        based on the learned blending parameter α
        """
        # Get vehicle state
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        current_x = transform.location.x
        current_y = transform.location.y
        current_yaw = transform.rotation.yaw * np.pi / 180.0
        current_speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # Determine if we're in handover phase
        self.is_handover_phase = (current_time >= self.handover_start_time and 
                                 current_time < self.handover_end_time)
        
        # Get state
        state = self.get_state(vehicle, current_time, waypoints)
        
        if self.is_handover_phase:
            # We're in handover phase, use RL to determine blending parameter
            action = self.select_action(state, explore=self.training_mode)
            alpha = action[0]
            
            # Update controllers with current state
            self.autonomous_controller.update_values(
                current_x, current_y, current_yaw, current_speed, simulation_time, True)
            self.human_controller.update_values(
                current_x, current_y, current_yaw, current_speed, simulation_time, True)
            
            # Update waypoints for both controllers
            self.autonomous_controller.update_waypoints(waypoints)
            self.human_controller.update_waypoints(waypoints)
            
            # Get controls from both controllers
            self.autonomous_controller.update_controls()
            auto_throttle, auto_steer, auto_brake = self.autonomous_controller.get_commands()
            
            self.human_controller.update_controls()
            human_throttle, human_steer, human_brake = self.human_controller.get_commands()
            
            # Blend controls using alpha
            throttle = alpha * auto_throttle + (1 - alpha) * human_throttle
            steer = alpha * auto_steer + (1 - alpha) * human_steer
            brake = alpha * auto_brake + (1 - alpha) * human_brake
            
            # Store transition for training
            if self.current_state is not None and self.current_action is not None and self.training_mode:
                reward = self.compute_reward(vehicle, waypoints, action, self.current_action, current_time)
                self.last_reward = reward
                self.episode_rewards.append(reward)
                self.replay_buffer.push(
                    self.current_state, self.current_action, [reward], state, [0])
                
                # Train if we have enough samples
                if len(self.replay_buffer) > self.batch_size:
                    self.train()
            
            # Update current state and action
            self.current_state = state
            self.current_action = action
            
            return throttle, steer, brake
            
        elif current_time < self.handover_start_time:
            # Before handover, use autonomous controller
            self.autonomous_controller.update_values(
                current_x, current_y, current_yaw, current_speed, simulation_time, True)
            self.autonomous_controller.update_waypoints(waypoints)
            self.autonomous_controller.update_controls()
            return self.autonomous_controller.get_commands()
            
        else:
            # After handover, use human controller
            self.human_controller.update_values(
                current_x, current_y, current_yaw, current_speed, simulation_time, True)
            self.human_controller.update_waypoints(waypoints)
            self.human_controller.update_controls()
            return self.human_controller.get_commands()
    
    def train(self):
        """Train the policy network using saved experiences"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample a batch from replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        
        # Compute current Q value
        current_alpha = self.policy_net(state)
        
        # Compute next Q value
        next_alpha = self.policy_net(next_state)
        
        # Compute loss (using mean squared error)
        # This is a simple value-based update; in a full implementation, 
        # consider using actor-critic or policy gradient methods
        target = reward + self.gamma * next_alpha * (1 - done)
        loss = nn.MSELoss()(current_alpha, target.detach())
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_model(self, path="handover_model.pth"):
        """Save the policy network"""
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load_model(self, path="handover_model.pth"):
        """Load the policy network"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']