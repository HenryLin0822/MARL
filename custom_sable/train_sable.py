"""Training script for Sable on Multi-Agent Particle Environments.

This script demonstrates how to train the Sable network on various multi-agent
particle environment scenarios like simple_spread, simple_adversary, etc.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from collections import deque
import time
import argparse
from pathlib import Path

# Add multiagent-particle-envs to path
sys.path.append('multiagent-particle-envs')

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

# Import Sable components
from sable_network_fully_aligned import SableNetwork
from config import SableConfig
from retention_fully_aligned import get_init_hidden_state


class MultiAgentParticleWrapper:
    """Wrapper for Multi-Agent Particle Environments to work with Sable."""
    
    def __init__(self, scenario_name: str, benchmark: bool = False):
        self.scenario_name = scenario_name
        self.benchmark = benchmark
        
        # Load scenario and create environment
        scenario = scenarios.load(scenario_name + ".py").Scenario()
        world = scenario.make_world()
        
        if benchmark:
            self.env = MultiAgentEnv(
                world, scenario.reset_world, scenario.reward, 
                scenario.observation, scenario.benchmark_data
            )
        else:
            self.env = MultiAgentEnv(
                world, scenario.reset_world, scenario.reward, scenario.observation
            )
        
        self.n_agents = self.env.n
        self.agent_ids = [f"agent_{i}" for i in range(self.n_agents)]
        
        # Get observation and action dimensions
        self._get_space_info()
        
    def _get_space_info(self):
        """Extract observation and action space information."""
        # Reset environment to get observation shape
        obs_n = self.env.reset()
        
        # Observation dimension
        self.obs_dim = len(obs_n[0])
        
        # Action dimension (discrete actions)
        self.action_dim = self.env.action_space[0].n
        
        # Action space type
        self.action_space_type = "discrete"
        
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment and return observations as dictionary."""
        obs_n = self.env.reset()
        obs_dict = {}
        for i, obs in enumerate(obs_n):
            obs_dict[self.agent_ids[i]] = np.array(obs, dtype=np.float32)
        return obs_dict
    
    def step(self, action_dict: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        """Step environment with actions from all agents."""
        # Convert action dictionary to list
        action_list = []
        for agent_id in self.agent_ids:
            if agent_id in action_dict:
                action_list.append(action_dict[agent_id])
            else:
                action_list.append(0)  # Default action
        
        # Step environment
        obs_n, reward_n, done_n, info_n = self.env.step(action_list)
        
        # Convert to dictionaries
        obs_dict = {}
        reward_dict = {}
        done_dict = {}
        info_dict = {}
        
        for i, agent_id in enumerate(self.agent_ids):
            obs_dict[agent_id] = np.array(obs_n[i], dtype=np.float32)
            reward_dict[agent_id] = float(reward_n[i])
            done_dict[agent_id] = bool(done_n[i]) if isinstance(done_n, list) else bool(done_n)
            info_dict[agent_id] = info_n[i] if isinstance(info_n, list) else {}
            
        return obs_dict, reward_dict, done_dict, info_dict
    
    def render(self, mode='human'):
        """Render environment."""
        return self.env.render(mode)
    
    def close(self):
        """Close environment."""
        self.env.close()


class PPOTrainer:
    """PPO trainer for Sable on Multi-Agent Particle Environments."""
    
    def __init__(
        self,
        config: SableConfig,
        env: MultiAgentParticleWrapper,
        device: str = "cpu",
        learning_rate: float = 3e-4,
        save_dir: str = "checkpoints"
    ):
        self.config = config
        self.env = env
        self.device = torch.device(device)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize Sable network
        self.network = SableNetwork(config).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Training statistics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.training_stats = {
            'episodes': 0,
            'total_steps': 0,
            'best_reward': float('-inf'),
            'losses': {'policy': [], 'value': [], 'entropy': []}
        }
        
    def collect_rollout(self, rollout_length: int) -> Dict[str, List]:
        """Collect a rollout of experiences."""
        rollout_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'next_values': []
        }
        
        # Initialize hidden states
        batch_size = 1
        hidden_states = self.network.init_hidden_states(batch_size, self.device)
        
        # Reset environment
        obs_dict = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(rollout_length):
            # Convert observations to tensor
            obs_list = []
            for agent_id in self.env.agent_ids:
                if agent_id in obs_dict:
                    obs_list.append(obs_dict[agent_id])
                else:
                    obs_list.append(np.zeros(self.env.obs_dim, dtype=np.float32))
            
            obs_tensor = torch.from_numpy(np.stack(obs_list)).unsqueeze(0).to(self.device)
            
            # Get actions from network
            with torch.no_grad():
                actions, log_probs, values, hidden_states = self.network.get_actions(
                    obs_tensor, hidden_states, deterministic=False
                )
            
            # Convert actions to dictionary
            action_dict = {}
            actions_np = actions.squeeze(0).cpu().numpy()
            for i, agent_id in enumerate(self.env.agent_ids):
                if i < len(actions_np):
                    action_dict[agent_id] = int(actions_np[i])
            
            # Step environment
            next_obs_dict, reward_dict, done_dict, info_dict = self.env.step(action_dict)
            
            # Store experience
            rollout_data['observations'].append(obs_tensor.squeeze(0))
            rollout_data['actions'].append(actions.squeeze(0))
            rollout_data['rewards'].append(sum(reward_dict.values()))
            rollout_data['values'].append(values.squeeze())
            rollout_data['log_probs'].append(log_probs.squeeze(0))
            rollout_data['dones'].append(any(done_dict.values()))
            
            # Update for next step
            obs_dict = next_obs_dict
            episode_reward += sum(reward_dict.values())
            episode_length += 1
            
            # Check if episode ended
            if any(done_dict.values()) or episode_length >= 200:  # Max episode length
                # Get final value for bootstrapping
                obs_list = []
                for agent_id in self.env.agent_ids:
                    if agent_id in obs_dict:
                        obs_list.append(obs_dict[agent_id])
                    else:
                        obs_list.append(np.zeros(self.env.obs_dim, dtype=np.float32))
                
                final_obs_tensor = torch.from_numpy(np.stack(obs_list)).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    _, _, final_values, _ = self.network.get_actions(
                        final_obs_tensor, hidden_states, deterministic=True
                    )
                rollout_data['next_values'].append(final_values.squeeze())
                
                # Reset environment
                obs_dict = self.env.reset()
                hidden_states = self.network.init_hidden_states(batch_size, self.device)
                
                # Store episode statistics
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.training_stats['episodes'] += 1
                
                episode_reward = 0
                episode_length = 0
            else:
                # Bootstrap with next value
                obs_list = []
                for agent_id in self.env.agent_ids:
                    if agent_id in obs_dict:
                        obs_list.append(obs_dict[agent_id])
                    else:
                        obs_list.append(np.zeros(self.env.obs_dim, dtype=np.float32))
                
                next_obs_tensor = torch.from_numpy(np.stack(obs_list)).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    _, _, next_values, _ = self.network.get_actions(
                        next_obs_tensor, hidden_states, deterministic=True
                    )
                rollout_data['next_values'].append(next_values.squeeze())
        
        return rollout_data
    
    def compute_advantages(self, rollout_data: Dict[str, List]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        rewards = torch.tensor(rollout_data['rewards'], device=self.device)
        values = torch.stack(rollout_data['values'])
        next_values = torch.stack(rollout_data['next_values'])
        dones = torch.tensor(rollout_data['dones'], device=self.device, dtype=torch.bool)
        
        # Compute GAE
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.config.gamma * next_values[t] * (~dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (~dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update_policy(self, rollout_data: Dict[str, List], advantages: torch.Tensor, returns: torch.Tensor):
        """Update policy using PPO."""
        observations = torch.stack(rollout_data['observations'])
        actions = torch.stack(rollout_data['actions'])
        old_log_probs = torch.stack(rollout_data['log_probs'])
        values = torch.stack(rollout_data['values'])
        
        # Create dummy hidden states and dones for training
        batch_size, seq_len, n_agents = observations.shape[0], observations.shape[1], observations.shape[2]
        dummy_hidden_states = self.network.init_hidden_states(batch_size, self.device)
        dummy_dones = torch.zeros(batch_size, n_agents, dtype=torch.bool, device=self.device)
        
        for epoch in range(self.config.ppo_epochs):
            # Forward pass
            new_values, new_log_probs, entropy = self.network(
                observations, actions, dummy_hidden_states, dones=dummy_dones
            )
            
            # Policy loss
            ratio = torch.exp(new_log_probs.sum(dim=-1) - old_log_probs.sum(dim=-1))
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(new_values.mean(dim=-1), returns)
            
            # Entropy loss
            entropy_loss = -entropy.mean()
            
            # Total loss
            total_loss = (
                policy_loss + 
                self.config.vf_coef * value_loss + 
                self.config.ent_coef * entropy_loss
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            
            # Store losses
            self.training_stats['losses']['policy'].append(policy_loss.item())
            self.training_stats['losses']['value'].append(value_loss.item())
            self.training_stats['losses']['entropy'].append(entropy_loss.item())
    
    def train(self, total_timesteps: int, log_interval: int = 10, save_interval: int = 100):
        """Main training loop."""
        print(f"Starting training for {total_timesteps} timesteps...")
        print(f"Environment: {self.env.scenario_name}")
        print(f"Agents: {self.env.n_agents}, Obs dim: {self.env.obs_dim}, Action dim: {self.env.action_dim}")
        
        steps_per_rollout = self.config.rollout_length
        total_updates = total_timesteps // steps_per_rollout
        
        start_time = time.time()
        
        for update in range(total_updates):
            # Collect rollout
            rollout_data = self.collect_rollout(steps_per_rollout)
            
            # Compute advantages
            advantages, returns = self.compute_advantages(rollout_data)
            
            # Update policy
            self.update_policy(rollout_data, advantages, returns)
            
            self.training_stats['total_steps'] += steps_per_rollout
            
            # Logging
            if update % log_interval == 0:
                avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
                
                print(f"Update {update}/{total_updates}")
                print(f"  Episodes: {self.training_stats['episodes']}")
                print(f"  Avg Reward: {avg_reward:.2f}")
                print(f"  Avg Length: {avg_length:.1f}")
                print(f"  Time: {time.time() - start_time:.1f}s")
                
                # Update best reward
                if avg_reward > self.training_stats['best_reward']:
                    self.training_stats['best_reward'] = avg_reward
                    self.save_checkpoint(f"best_model.pt")
            
            # Save checkpoint
            if update % save_interval == 0:
                self.save_checkpoint(f"checkpoint_{update}.pt")
        
        print("Training completed!")
        self.save_checkpoint("final_model.pt")
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats,
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths)
        }
        torch.save(checkpoint, self.save_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """Load training checkpoint."""
        checkpoint = torch.load(self.save_dir / filename, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']
        if 'episode_rewards' in checkpoint:
            self.episode_rewards.extend(checkpoint['episode_rewards'])
        if 'episode_lengths' in checkpoint:
            self.episode_lengths.extend(checkpoint['episode_lengths'])
    
    def evaluate(self, num_episodes: int = 10, render: bool = False):
        """Evaluate the trained policy."""
        print(f"Evaluating for {num_episodes} episodes...")
        
        self.network.eval()
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            obs_dict = self.env.reset()
            hidden_states = self.network.init_hidden_states(1, self.device)
            
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done and episode_length < 200:
                # Convert observations to tensor
                obs_list = []
                for agent_id in self.env.agent_ids:
                    if agent_id in obs_dict:
                        obs_list.append(obs_dict[agent_id])
                    else:
                        obs_list.append(np.zeros(self.env.obs_dim, dtype=np.float32))
                
                obs_tensor = torch.from_numpy(np.stack(obs_list)).unsqueeze(0).to(self.device)
                
                # Get actions
                with torch.no_grad():
                    actions, _, _, hidden_states = self.network.get_actions(
                        obs_tensor, hidden_states, deterministic=True
                    )
                
                # Convert to action dictionary
                action_dict = {}
                actions_np = actions.squeeze(0).cpu().numpy()
                for i, agent_id in enumerate(self.env.agent_ids):
                    if i < len(actions_np):
                        action_dict[agent_id] = int(actions_np[i])
                
                # Step environment
                obs_dict, reward_dict, done_dict, _ = self.env.step(action_dict)
                
                episode_reward += sum(reward_dict.values())
                episode_length += 1
                done = any(done_dict.values())
                
                if render:
                    self.env.render()
                    time.sleep(0.05)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
        
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        avg_length = np.mean(episode_lengths)
        
        print(f"\nEvaluation Results:")
        print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Average Length: {avg_length:.1f}")
        
        self.network.train()
        return avg_reward, std_reward, avg_length


def main():
    parser = argparse.ArgumentParser(description="Train Sable on Multi-Agent Particle Environments")
    parser.add_argument("--scenario", default="simple_spread", help="Scenario name")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--device", default="cpu", help="Device to use")
    parser.add_argument("--save_dir", default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--load_checkpoint", help="Path to checkpoint to load")
    parser.add_argument("--evaluate", action="store_true", help="Only evaluate, don't train")
    parser.add_argument("--render", action="store_true", help="Render during evaluation")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--n_head", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_block", type=int, default=2, help="Number of transformer blocks")
    
    args = parser.parse_args()
    
    # Create environment
    env = MultiAgentParticleWrapper(args.scenario)
    
    # Create Sable configuration
    config = SableConfig(
        n_agents=env.n_agents,
        action_dim=env.action_dim,
        obs_dim=env.obs_dim,
        embed_dim=args.embed_dim,
        n_head=args.n_head,
        n_block=args.n_block,
        chunk_size=min(env.n_agents, 4),  # Adaptive chunk size
        action_space_type=env.action_space_type,
        rollout_length=128,
        ppo_epochs=4,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5
    )
    
    # Create trainer
    trainer = PPOTrainer(config, env, device=args.device, save_dir=args.save_dir)
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        trainer.load_checkpoint(args.load_checkpoint)
        print(f"Loaded checkpoint: {args.load_checkpoint}")
    
    if args.evaluate:
        # Evaluate only
        trainer.evaluate(num_episodes=20, render=args.render)
    else:
        # Train
        trainer.train(total_timesteps=args.timesteps)
        
        # Evaluate final policy
        print("\nFinal evaluation:")
        trainer.evaluate(num_episodes=10, render=args.render)
    
    env.close()


if __name__ == "__main__":
    main()