import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
import random
import math
from pygame.locals import K_ESCAPE, K_q, KMOD_CTRL
import pygame
import carla
import matplotlib.pyplot as plt
from carla import ColorConverter as cc
from main import World
from main import HUD
from collections import deque

# Reuse the PPO implementation from your code
class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim=1, hidden_dim=128):  # Only one action: alpha
        super(PPONetwork, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy) head - outputs mean for alpha
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # Ensure alpha is between 0 and 1
        )
        self.actor_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        shared_features = self.shared(x)
        action_mean = self.actor_mean(shared_features)
        action_std = torch.exp(self.actor_std).expand_as(action_mean)
        value = self.critic(shared_features)
        return torch.distributions.Normal(action_mean, action_std), value

class HandoverPPOAgent:
    def __init__(self, state_dim, lr=3e-4, gamma=0.99, 
                 clip_epsilon=0.2, entropy_coef=0.01, epochs=4, batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = PPONetwork(state_dim, action_dim=1).to(self.device)  # Alpha is a single value
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Memory buffers
        self.clear_memory()
    
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist, value = self.policy(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]
    
    def store_transition(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
    
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def compute_returns_and_advantages(self):
        returns = []
        advantages = []
        R = 0
        advantage = 0
        
        # Compute returns in reverse
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t] * next_non_terminal
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t+1] * next_non_terminal
                
            delta = self.rewards[t] + self.gamma * next_value - self.values[t]
            advantage = delta + self.gamma * 0.95 * next_non_terminal * advantage
            advantages.insert(0, advantage)
            
            R = self.rewards[t] + self.gamma * next_non_terminal * R
            returns.insert(0, R)
        
        # Normalize advantages
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return np.array(returns), advantages
    
    def update(self):
        if len(self.states) < self.batch_size:
            return
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        
        returns, advantages = self.compute_returns_and_advantages()
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Optimize policy for K epochs
        for _ in range(self.epochs):
            # Shuffle indices for mini-batch updates
            indices = np.arange(len(self.states))
            np.random.shuffle(indices)
            
            for start in range(0, len(self.states), self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]
                
                # Get mini-batch
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_returns = returns[idx]
                mb_advantages = advantages[idx]
                
                # Evaluate current policy
                dist, values = self.policy(mb_states)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                
                # Compute ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                # Policy loss
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 
                                  1.0 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = 0.5 * (mb_returns - values.squeeze()).pow(2).mean()
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
                
                # Update policy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
        
        # Clear memory after update
        self.clear_memory()
        return True

class SimulatedHumanController:
    """
    A simulated human controller - implemented as a variant of MPC with some added noise
    to represent human imperfections
    """
    def __init__(self, mpc_controller):
        # Clone the MPC controller and modify it for "human-like" behavior
        self.controller = copy.deepcopy(mpc_controller)
        self.noise_level = 0.1  # Steer noise to simulate human imperfection
        self.reaction_delay = 3  # Simulated reaction delay in frames
        self.control_buffer = deque(maxlen=10)  # Store recent controls
    
    def update_values(self, *args, **kwargs):
        return self.controller.update_values(*args, **kwargs)
    
    def update_waypoints(self, waypoints):
        # Add a bit of target path noise for the human controller
        noisy_waypoints = []
        for wp in waypoints:
            noise_x = random.gauss(0, 0.1)
            noise_y = random.gauss(0, 0.1)
            noisy_wp = wp.copy()
            noisy_wp[0] += noise_x
            noisy_wp[1] += noise_y
            noisy_waypoints.append(noisy_wp)
            
        self.controller.update_waypoints(noisy_waypoints)
    
    def update_controls(self):
        # Add delay to simulate human reaction time
        if len(self.control_buffer) < self.reaction_delay:
            self.controller.update_controls()
            self.control_buffer.append(self.controller.get_commands())
            return self.get_commands()
        else:
            self.controller.update_controls()
            self.control_buffer.append(self.controller.get_commands())
            return self.get_commands()
    
    def get_commands(self):
        if len(self.control_buffer) < self.reaction_delay:
            throttle, steer, brake = 0.0, 0.0, 0.0
        else:
            throttle, steer, brake = self.control_buffer[0]
            # Add noise to steering to simulate human imperfection
            steer += random.gauss(0, self.noise_level)
            steer = max(-1.0, min(1.0, steer))  # Clamp steering
            
        return throttle, steer, brake

class HandoverControlManager:
    """
    Manager for transitioning control between autonomous system and human driver
    """
    def __init__(self, mpc_controller, handover_duration=7.0):
        self.autonomous_controller = mpc_controller
        self.human_controller = SimulatedHumanController(mpc_controller)
        
        # Handover settings
        self.handover_duration = handover_duration  # seconds
        self.handover_trigger_time = None
        self.handover_complete_time = None
        self.handover_in_progress = False
        
        # RL agent settings
        self.state_dim = 10  # Adjust based on your state representation
        self.rl_agent = HandoverPPOAgent(state_dim=self.state_dim)
        
        # Current alpha value (blending parameter)
        self.alpha = 1.0  # Start with full autonomous control
        
        # Performance metrics
        self.trajectory_error = 0.0
        self.control_smoothness = 0.0
        self.last_controls = None
        
        # Episode tracking
        self.episode_timesteps = 0
        self.max_episode_length = 200  # Approximately 20 seconds at 10Hz
        self.total_reward = 0.0
        self.episode_count = 0
        self.training_mode = True
    
    def trigger_handover(self, timestamp):
        """Trigger the handover process"""
        self.handover_trigger_time = timestamp
        self.handover_complete_time = timestamp + self.handover_duration
        self.handover_in_progress = True
        self.alpha = 1.0  # Start with full autonomous control
        print(f"Handover triggered at {timestamp:.2f}s, will complete at {self.handover_complete_time:.2f}s")
        
    def is_handover_complete(self, timestamp):
        """Check if handover is complete"""
        if not self.handover_in_progress:
            return False
        
        if timestamp >= self.handover_complete_time:
            if self.handover_in_progress:
                print(f"Handover completed at {timestamp:.2f}s")
                self.handover_in_progress = False
                self.alpha = 0.0  # Full human control
            return True
        
        return False
    
    def get_state_representation(self, vehicle, waypoints, timestamp):
        """Create a state representation for the RL agent"""
        # Get vehicle state
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        
        current_x = transform.location.x
        current_y = transform.location.y
        current_yaw = wrap_angle(transform.rotation.yaw)
        current_speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # Get handover progress
        if self.handover_in_progress:
            handover_progress = (timestamp - self.handover_trigger_time) / self.handover_duration
        else:
            handover_progress = 0.0
        
        # Get closest waypoint and distance
        if waypoints:
            distances = []
            for wp in waypoints:
                dx = wp[0] - current_x
                dy = wp[1] - current_y
                distances.append(np.sqrt(dx*dx + dy*dy))
            
            min_dist_idx = np.argmin(distances)
            closest_wp = waypoints[min_dist_idx]
            path_error = distances[min_dist_idx]
            
            # Heading error
            target_yaw = closest_wp[3]  # Assuming waypoint format [x, y, speed, yaw]
            yaw_error = np.abs(current_yaw - target_yaw)
            while yaw_error > np.pi:
                yaw_error -= 2 * np.pi
            yaw_error = np.abs(yaw_error)
        else:
            path_error = 0.0
            yaw_error = 0.0
        
        # Construct state vector
        state = [
            current_speed,
            handover_progress,
            path_error,
            yaw_error,
            self.trajectory_error,
            self.control_smoothness,
            self.alpha,  # Current blending value
            self.episode_timesteps / self.max_episode_length,  # Normalized timestep
            int(self.handover_in_progress),
            # Add additional relevant state components
        ]
        
        return np.array(state, dtype=np.float32)
    
    def calculate_reward(self, vehicle, waypoints, control_a, control_h, blended_control):
        """Calculate reward for RL agent"""
        # Extract controls
        throttle_a, steer_a, brake_a = control_a
        throttle_h, steer_h, brake_h = control_h
        throttle, steer, brake = blended_control
        
        # Calculate trajectory error
        transform = vehicle.get_transform()
        current_x = transform.location.x
        current_y = transform.location.y
        
        if waypoints:
            # Path following reward
            min_dist = float('inf')
            for wp in waypoints:
                dx = wp[0] - current_x
                dy = wp[1] - current_y
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = min(min_dist, dist)
            
            # Penalize deviation from path
            path_error = -min_dist
        else:
            path_error = 0.0
        
        # Control smoothness reward
        control_change = 0.0
        if self.last_controls is not None:
            last_throttle, last_steer, last_brake = self.last_controls
            control_change = abs(throttle - last_throttle) + abs(steer - last_steer) + abs(brake - last_brake)
        
        control_smoothness = -control_change * 2.0  # Penalize abrupt control changes
        
        # Handover progress reward
        if self.handover_in_progress:
            # Encourage gradual decrease in alpha during handover
            target_alpha = 1.0 - (self.episode_timesteps / self.max_episode_length)
            alpha_error = -abs(self.alpha - target_alpha) * 0.5
        else:
            alpha_error = 0.0
        
        # Combine rewards
        reward = path_error * 0.5 + control_smoothness * 0.3 + alpha_error * 0.2
        
        # Update metrics for next iteration
        self.trajectory_error = min_dist if waypoints else 0.0
        self.control_smoothness = control_change
        self.last_controls = (throttle, steer, brake)
        
        return reward
    
    def blend_controls(self, control_a, control_h):
        """Blend autonomous and human controls based on alpha"""
        throttle_a, steer_a, brake_a = control_a
        throttle_h, steer_h, brake_h = control_h
        
        # Linear blending
        throttle = self.alpha * throttle_a + (1 - self.alpha) * throttle_h
        steer = self.alpha * steer_a + (1 - self.alpha) * steer_h
        brake = self.alpha * brake_a + (1 - self.alpha) * brake_h
        
        return throttle, steer, brake
    
    def update(self, vehicle, waypoints, timestamp, frame):
        """Update controllers and blend controls using the RL agent"""
        # Update episode tracking
        self.episode_timesteps += 1
        
        # Check if handover should be triggered (at 5 seconds)
        if not self.handover_in_progress and timestamp >= 5.0 and timestamp < 5.1:
            self.trigger_handover(timestamp)
        
        # Update both controllers with current state
        self.autonomous_controller.update_values(
            vehicle.get_transform().location.x,
            vehicle.get_transform().location.y,
            wrap_angle(vehicle.get_transform().rotation.yaw),
            np.sqrt(vehicle.get_velocity().x**2 + vehicle.get_velocity().y**2 + vehicle.get_velocity().z**2),
            timestamp,
            frame
        )
        
        self.human_controller.update_values(
            vehicle.get_transform().location.x,
            vehicle.get_transform().location.y,
            wrap_angle(vehicle.get_transform().rotation.yaw),
            np.sqrt(vehicle.get_velocity().x**2 + vehicle.get_velocity().y**2 + vehicle.get_velocity().z**2),
            timestamp,
            frame
        )
        
        # Update waypoints for both controllers
        self.autonomous_controller.update_waypoints(waypoints)
        self.human_controller.update_waypoints(waypoints)
        
        # Update controls for both controllers
        self.autonomous_controller.update_controls()
        self.human_controller.update_controls()
        
        # Get individual controls
        control_a = self.autonomous_controller.get_commands()
        control_h = self.human_controller.get_commands()
        
        # Get state for RL agent
        state = self.get_state_representation(vehicle, waypoints, timestamp)
        
        # Get action (alpha) from RL agent if handover is in progress
        if self.handover_in_progress and self.training_mode:
            action, log_prob, value = self.rl_agent.act(state)
            self.alpha = float(action[0])  # Extract alpha value
        elif self.handover_in_progress and not self.training_mode:
            # Just use the policy without exploration
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.rl_agent.device)
                dist, _ = self.rl_agent.policy(state_tensor)
                self.alpha = float(dist.mean.cpu().numpy()[0])
        else:
            # Not in handover, use fixed values
            self.alpha = 1.0 if not self.is_handover_complete(timestamp) else 0.0
            action, log_prob, value = [self.alpha], 0.0, 0.0
        
        # Blend controls
        blended_control = self.blend_controls(control_a, control_h)
        
        # Calculate reward if in training mode
        if self.training_mode:
            reward = self.calculate_reward(vehicle, waypoints, control_a, control_h, blended_control)
            self.total_reward += reward
            
            # Check for episode termination
            done = self.is_handover_complete(timestamp) or self.episode_timesteps >= self.max_episode_length
            
            # Store transition
            if self.handover_in_progress:
                self.rl_agent.store_transition(state, action, log_prob, reward, int(done), value)
            
            # Update policy if episode is done
            if done:
                if len(self.rl_agent.states) > 0:
                    self.rl_agent.update()
                
                print(f"Episode {self.episode_count} finished. Total reward: {self.total_reward:.2f}")
                self.episode_count += 1
                self.total_reward = 0.0
                self.episode_timesteps = 0
                self.handover_in_progress = False
                self.handover_trigger_time = None
                self.handover_complete_time = None
        
        return blended_control, self.alpha

def wrap_angle(angle_in_degree):
    """Convert degrees to radians and normalize to [-pi, pi]"""
    angle_in_rad = angle_in_degree / 180.0 * np.pi
    while angle_in_rad > np.pi:
        angle_in_rad -= 2 * np.pi
    while angle_in_rad <= -np.pi:
        angle_in_rad += 2 * np.pi
    return angle_in_rad

class VehicleControl:
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            world.player.set_autopilot(self._autopilot_enabled)
            
            # Initialize the handover control manager
            self.handover_manager = HandoverControlManager(world.controller)
            
            # For visualization purposes
            self.alpha_history = []
            self.reward_history = []
            self.timestamp_history = []
            
        else:
            raise NotImplementedError("Only vehicle actors supported")

    def parse_events(self, client, world, clock):
        if not self._autopilot_enabled:
            self._handle_blended_control(world)
            
    def _handle_blended_control(self, world):
        """Apply blended control from autonomous and human controllers"""
        # Get current vehicle state
        vehicle = world.player
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        
        current_x = transform.location.x
        current_y = transform.location.y
        current_yaw = wrap_angle(transform.rotation.yaw)
        current_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # Get simulation info
        frame, timestamp = world.hud.get_simulation_information()
        
        # Generate waypoints
        waypoints = self._generate_waypoints(world, transform.location, current_speed)
        
        # Update handover manager and get blended control
        blended_control, alpha = self.handover_manager.update(vehicle, waypoints, timestamp, frame)
        
        # Store history for visualization
        self.alpha_history.append(alpha)
        self.timestamp_history.append(timestamp)
        
        # Apply the blended control to the vehicle
        self._control.throttle, self._control.steer, self._control.brake = blended_control
        vehicle.apply_control(self._control)
        
        # Update control count for waypoint generation
        world.control_count += 1
        
        # Handle episode termination and reset
        if timestamp > 17.0:  # End episode after 17 seconds
            # Reset the simulation to start a new episode
            world.restart()
            self.handover_manager = HandoverControlManager(world.controller)
            self.alpha_history = []
            self.timestamp_history = []
    
    def _generate_waypoints(self, world, current_location, current_speed):
        """Generate waypoints for planning horizon"""
        waypoints = []
        prev_waypoint = world.map.get_waypoint(current_location)
        
        # Calculate distance to next waypoint based on current speed
        dist = world.time_step * current_speed + 0.1
        
        # Get first waypoint
        current_waypoint = prev_waypoint.next(dist)[0]
        
        # Target speed ramp-up
        road_desired_speed = world.desired_speed
        
        # Generate waypoints for planning horizon
        for i in range(world.planning_horizon):
            # Ramp up speed for first 100 control steps
            if world.control_count + i <= 100:
                desired_speed = (world.control_count + 1 + i)/100.0 * road_desired_speed
            else:
                desired_speed = road_desired_speed
                
            # Get next waypoint
            dist = world.time_step * road_desired_speed
            current_waypoint = prev_waypoint.next(dist)[0]
            
            # Store waypoint data [x, y, speed, yaw]
            waypoints.append([
                current_waypoint.transform.location.x,
                current_waypoint.transform.location.y,
                road_desired_speed,
                wrap_angle(current_waypoint.transform.rotation.yaw)
            ])
            
            prev_waypoint = current_waypoint
            
        return waypoints

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)
    
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def train_handover_agent(num_episodes=1000, save_interval=100, model_path='./handover_model'):
    """
    Train the handover agent for a specified number of episodes
    """
    # Create directory for saving models if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    
    # Setup CARLA environment and run training
    args = get_default_args()
    args.training_mode = True
    
    # Training stats
    episode_rewards = []
    avg_alpha_values = []
    
    try:
        # Connect to CARLA and setup training environment
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        
        # Setup display (can be headless for faster training)
        if not args.headless:
            pygame.init()
            pygame.font.init()
            display = pygame.display.set_mode(
                (args.width, args.height),
                pygame.HWSURFACE | pygame.DOUBLEBUF
            )
        else:
            display = None
        
        # Load world and setup simulation
        hud = HUD(args.width, args.height)
        carla_world = client.load_world(args.map)
        world = World(carla_world, hud, args)
        controller = VehicleControl(world, args.autopilot)
        
        # Main training loop
        clock = pygame.time.Clock()
        episode = 0
        
        while episode < num_episodes:
            clock.tick_busy_loop(args.fps)
            
            # Update environment
            controller.parse_events(client, world, clock)
            world.tick(clock)
            
            # Render if display is enabled
            if display:
                world.render(display)
                pygame.display.flip()
            
            # Check if episode completed
            if controller.handover_manager.episode_count > episode:
                # New episode completed
                episode = controller.handover_manager.episode_count
                episode_rewards.append(controller.handover_manager.total_reward)
                
                # Calculate average alpha during handover
                if len(controller.alpha_history) > 0:
                    avg_alpha = np.mean(controller.alpha_history)
                    avg_alpha_values.append(avg_alpha)
                    controller.alpha_history = []
                
                print(f"Episode {episode}/{num_episodes} completed. Reward: {episode_rewards[-1]:.2f}")
                
                # Save model periodically
                if episode % save_interval == 0:
                    torch.save(controller.handover_manager.rl_agent.policy.state_dict(), 
                              f"{model_path}/handover_model_{episode}.pt")
                    
                    # Plot and save training progress
                    plot_training_progress(episode_rewards, avg_alpha_values, model_path)
        
        # Save final model
        torch.save(controller.handover_manager.rl_agent.policy.state_dict(), 
                  f"{model_path}/handover_model_final.pt")
        
    finally:
        if world is not None:
            world.destroy()
        if pygame.get_init():
            pygame.quit()

def get_default_args():
    """Return default arguments for training"""
    args = type('', (), {})()
    
    # Connection settings
    args.host = '127.0.0.1'
    args.port = 2000
    
    # Display settings
    args.width = 1280
    args.height = 720
    args.gamma = 2.2
    args.headless = False  # Set to True for faster training without visualization
    
    # Simulation settings
    args.map = 'Town04'
    args.autopilot = False
    args.fps = 10
    
    # Vehicle settings
    args.filter = 'vehicle.*'
    args.vehicle_id = 'vehicle.audi.a2'
    
    # Spawn settings
    args.spawn_x = '-14'
    args.spawn_y = '70'
    args.random_spawn = 0
    
    # Controller settings
    args.waypoint_resolution = 0.5
    args.waypoint_lookahead_distance = 5.0
    args.desired_speed = 30
    args.planning_horizon = 5
    args.time_step = 0.3
    
    # Debug options
    args.debug = False
    
    return args

def plot_training_progress(episode_rewards, avg_alpha_values, save_path):
    """Plot and save training progress"""
    plt.figure(figsize=(12, 10))
    
    # Plot episode rewards
    plt.subplot(2, 1, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot moving average of rewards
    if len(episode_rewards) > 10:
        moving_avg = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
        plt.plot(moving_avg, label='Moving Avg (10 episodes)', color='orange')
        plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{save_path}/training_progress.png")
    plt.close()

if __name__ == "__main__":
    # Train the handover agent
    train_handover_agent(num_episodes=1000, save_interval=100, model_path='./handover_model')
