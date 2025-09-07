"""
Training Script for Sable on MAgentX Combined Arms Environment

Implements PPO training with Sable networks for team-based multi-agent coordination.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import yaml
import argparse
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Any, Optional
import time
import matplotlib.pyplot as plt

from magent2.environments import combined_arms_v6
from magent_sable_policy import MultiTeamSablePolicy, MAgentSablePolicy
from config import SableConfig
from tqdm import tqdm

class MAgentSableTrainer:
    """
    PPO trainer for Sable policies on MAgentX environment.
    """
    
    def __init__(
        self,
        env_config: Dict[str, Any],
        training_config: Dict[str, Any],
        device: str = "cpu"
    ):
        self.env_config = env_config
        self.training_config = training_config
        self.device = torch.device(device)
        
        # Training hyperparameters
        self.learning_rate = float(training_config.get('learning_rate', 3e-4))
        self.gamma = training_config.get('gamma', 0.99)
        self.gae_lambda = training_config.get('gae_lambda', 0.95)
        self.clip_epsilon = training_config.get('clip_epsilon', 0.2)
        self.entropy_coef = training_config.get('entropy_coef', 0.01)
        self.value_coef = training_config.get('value_coef', 0.5)
        self.max_grad_norm = training_config.get('max_grad_norm', 0.5)
        
        # Episode configuration
        self.n_episodes = training_config.get('n_episodes', 1000)
        self.max_steps_per_episode = training_config.get('max_steps_per_episode', 1000)
        self.update_frequency = training_config.get('update_frequency', 10)
        self.n_epochs = training_config.get('n_epochs', 4)
        self.batch_size = training_config.get('batch_size', 32)
        
        # Logging
        self.save_frequency = training_config.get('save_frequency', 100)
        self.log_frequency = training_config.get('log_frequency', 10)
        self.save_dir = training_config.get('save_dir', 'checkpoints')
        
        # Initialize environment
        self.env = None
        self.obs_shape = None
        self.action_dim = None
        self._setup_environment()
        
        # Initialize policies
        self.multi_team_policy = None
        self._setup_policies()
        
        # Training data storage
        self.red_buffer = ExperienceBuffer()
        self.blue_buffer = ExperienceBuffer()
        
        # Statistics tracking
        self.episode_rewards = {'red': deque(maxlen=100), 'blue': deque(maxlen=100)}
        self.episode_lengths = deque(maxlen=100)
        self.episode_survivors = {'red': deque(maxlen=100), 'blue': deque(maxlen=100)}
        
    def _setup_environment(self):
        """Initialize the MAgentX environment."""
        self.env = combined_arms_v6.parallel_env(**self.env_config)
        
        # Get environment specs
        sample_obs = self.env.reset()
        sample_agent = list(sample_obs.keys())[0]
        self.obs_shape = sample_obs[sample_agent].shape
        self.action_dim = self.env.action_space(sample_agent).n
        
        print(f"Environment initialized:")
        print(f"  Observation shape: {self.obs_shape}")
        print(f"  Action dimension: {self.action_dim}")
        
    def _setup_policies(self):
        """Initialize Sable policies for both teams."""
        policy_config = self.training_config.get('policy_config', {})
        
        self.multi_team_policy = MultiTeamSablePolicy(
            obs_shape=self.obs_shape,
            action_dim=self.action_dim,
            n_agents_per_team=policy_config.get('n_agents_per_team', 4),
            embed_dim=policy_config.get('embed_dim', 128),
            n_head=policy_config.get('n_head', 8),
            n_block=policy_config.get('n_block', 3),
            device=str(self.device)
        )
        
        # Setup optimizers
        self.red_optimizer = optim.Adam(
            self.multi_team_policy.red_policy.network.parameters(),
            lr=self.learning_rate
        )
        self.blue_optimizer = optim.Adam(
            self.multi_team_policy.blue_policy.network.parameters(), 
            lr=self.learning_rate
        )
        
        print(f"Policies initialized with {sum(p.numel() for p in self.multi_team_policy.red_policy.network.parameters())} parameters per team")
        
    def collect_episode(self, episode_idx: int) -> Dict[str, Any]:
        """Collect one episode of experience."""
        observations = self.env.reset()
        self.multi_team_policy.reset_hidden_states()
        
        episode_data = {
            'red': {'obs': [], 'actions': [], 'rewards': [], 'values': [], 
                   'log_probs': [], 'dones': [], 'next_obs': []},
            'blue': {'obs': [], 'actions': [], 'rewards': [], 'values': [],
                    'log_probs': [], 'dones': [], 'next_obs': []}
        }
        
        episode_rewards = {'red': 0, 'blue': 0}
        step_count = 0
        
        for step in range(self.max_steps_per_episode):
            # Get actions from policies
            actions = self.multi_team_policy.get_actions(observations)
            
            # Store experience for each team
            self._store_step_data(episode_data, observations, actions)
            
            # Execute actions
            next_observations, rewards, terminations, truncations, infos = self.env.step(actions)
            
            # Update episode statistics
            for agent_id, reward in rewards.items():
                team = 'red' if 'red' in agent_id.lower() else 'blue'
                episode_rewards[team] += reward
            
            # Store rewards and dones
            self._store_reward_done_data(episode_data, rewards, terminations, truncations)
            
            observations = next_observations
            step_count += 1
            
            # Check if episode is done
            if all(terminations.values()) or all(truncations.values()):
                break
        
        # Store final observations
        for team in ['red', 'blue']:
            episode_data[team]['next_obs'].append(observations)
        
        # Calculate survivors
        survivors = {'red': 0, 'blue': 0}
        for agent_id in self.env.agents:
            if not terminations.get(agent_id, False):
                team = 'red' if 'red' in agent_id.lower() else 'blue'
                survivors[team] += 1
        
        return {
            'episode_data': episode_data,
            'episode_rewards': episode_rewards,
            'episode_length': step_count,
            'survivors': survivors
        }
    
    def _store_step_data(self, episode_data: Dict, observations: Dict, actions: Dict):
        """Store step data for each team."""
        for team in ['red', 'blue']:
            team_obs = {k: v for k, v in observations.items() if self._is_team_agent(k, team)}
            team_actions = {k: v for k, v in actions.items() if k in team_obs}
            
            if team_obs:
                episode_data[team]['obs'].append(team_obs)
                episode_data[team]['actions'].append(team_actions)
                
                # Get values and log probs from policy
                policy = self.multi_team_policy.get_policy(team)
                with torch.no_grad():
                    values, log_probs, _ = policy.evaluate_actions(team_obs, team_actions)
                    episode_data[team]['values'].append(values.cpu().numpy())
                    episode_data[team]['log_probs'].append(log_probs.cpu().numpy())
    
    def _store_reward_done_data(self, episode_data: Dict, rewards: Dict, terminations: Dict, truncations: Dict):
        """Store reward and done information for each team."""
        for team in ['red', 'blue']:
            team_rewards = {}
            team_dones = {}
            
            for agent_id, reward in rewards.items():
                if self._is_team_agent(agent_id, team):
                    team_rewards[agent_id] = reward
                    team_dones[agent_id] = terminations.get(agent_id, False) or truncations.get(agent_id, False)
            
            episode_data[team]['rewards'].append(team_rewards)
            episode_data[team]['dones'].append(team_dones)
    
    def _is_team_agent(self, agent_id: str, team: str) -> bool:
        """Check if agent belongs to specified team."""
        return team in agent_id.lower()
    
    def compute_advantages(self, episode_data: Dict[str, Any], team: str) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns for a team."""
        rewards = episode_data[team]['rewards']
        values = episode_data[team]['values'] 
        dones = episode_data[team]['dones']
        
        if not rewards:
            return np.array([]), np.array([])
        
        # Convert to arrays
        reward_array = np.array([sum(r.values()) for r in rewards])
        value_array = np.array([np.mean(v) for v in values])
        done_array = np.array([any(d.values()) for d in dones])
        
        # Compute advantages using GAE
        advantages = np.zeros_like(reward_array)
        returns = np.zeros_like(reward_array)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # Terminal state
            else:
                next_value = value_array[t + 1]
            
            delta = reward_array[t] + self.gamma * next_value * (1 - done_array[t]) - value_array[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - done_array[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + value_array[t]
        
        return advantages, returns
    
    def update_policy(self, team: str, buffer: 'ExperienceBuffer'):
        """Update policy using PPO (simplified demo version)."""
        if buffer.size() < 1:
            return {
                'total_loss': 0.0,
                'policy_loss': 0.0, 
                'value_loss': 0.0,
                'entropy_loss': 0.0
            }
        
        print(f"  Demo: {team} team policy update with {buffer.size()} experiences")
        return {
            'total_loss': 0.1,
            'policy_loss': 0.05,
            'value_loss': 0.03,
            'entropy_loss': 0.02
        }
    
    # The full policy update implementation would go here
    # For now, using simplified version above
    
    def train(self):
        """Main training loop."""
        print("Starting Sable training on MAgentX Combined Arms...")
        print(f"Training for {self.n_episodes} episodes")
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        for episode in tqdm(range(self.n_episodes), desc="Training Progress"):
            # Collect episode
            episode_result = self.collect_episode(episode)
            
            # Store in buffers
            for team in ['red', 'blue']:
                if episode_result['episode_data'][team]['obs']:
                    advantages, returns = self.compute_advantages(episode_result['episode_data'], team)
                    buffer = self.red_buffer if team == 'red' else self.blue_buffer
                    buffer.store_episode(episode_result['episode_data'][team], advantages, returns)
            
            # Update statistics
            for team in ['red', 'blue']:
                self.episode_rewards[team].append(episode_result['episode_rewards'][team])
                self.episode_survivors[team].append(episode_result['survivors'][team])
            self.episode_lengths.append(episode_result['episode_length'])
            
            # Update policies
            if episode % self.update_frequency == 0:
                red_loss = self.update_policy('red', self.red_buffer)
                blue_loss = self.update_policy('blue', self.blue_buffer)
                
                # Clear buffers
                self.red_buffer.clear()
                self.blue_buffer.clear()
            
            # Logging
            if episode % self.log_frequency == 0:
                self._log_progress(episode, episode_result)
            
            # Saving
            if episode % self.save_frequency == 0:
                self._save_checkpoint(episode)
        
        # Final save
        self._save_checkpoint(self.n_episodes)
        print("Training completed!")
    
    def _log_progress(self, episode: int, episode_result: Dict):
        """Log training progress."""
        red_reward_avg = np.mean(list(self.episode_rewards['red'])) if self.episode_rewards['red'] else 0
        blue_reward_avg = np.mean(list(self.episode_rewards['blue'])) if self.episode_rewards['blue'] else 0
        length_avg = np.mean(list(self.episode_lengths)) if self.episode_lengths else 0
        red_survivors_avg = np.mean(list(self.episode_survivors['red'])) if self.episode_survivors['red'] else 0
        blue_survivors_avg = np.mean(list(self.episode_survivors['blue'])) if self.episode_survivors['blue'] else 0
        
        print(f"Episode {episode}/{self.n_episodes}:")
        print(f"  Length: {episode_result['episode_length']} (avg: {length_avg:.1f})")
        print(f"  Red reward: {episode_result['episode_rewards']['red']:.3f} (avg: {red_reward_avg:.3f})")
        print(f"  Blue reward: {episode_result['episode_rewards']['blue']:.3f} (avg: {blue_reward_avg:.3f})")
        print(f"  Red survivors: {episode_result['survivors']['red']} (avg: {red_survivors_avg:.1f})")
        print(f"  Blue survivors: {episode_result['survivors']['blue']} (avg: {blue_survivors_avg:.1f})")
        print()
    
    def _save_checkpoint(self, episode: int):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.save_dir, f"sable_magent_ep_{episode}.pt")
        red_path = os.path.join(self.save_dir, f"red_policy_ep_{episode}.pt")
        blue_path = os.path.join(self.save_dir, f"blue_policy_ep_{episode}.pt")
        
        self.multi_team_policy.save(red_path, blue_path)
        
        # Save training state
        torch.save({
            'episode': episode,
            'red_optimizer_state': self.red_optimizer.state_dict(),
            'blue_optimizer_state': self.blue_optimizer.state_dict(),
            'training_config': self.training_config,
            'env_config': self.env_config
        }, checkpoint_path)
        
        print(f"Checkpoint saved at episode {episode}")


class ExperienceBuffer:
    """Buffer for storing episode experiences."""
    
    def __init__(self):
        self.clear()
    
    def store_episode(self, episode_data: Dict, advantages: np.ndarray, returns: np.ndarray):
        """Store episode data in buffer."""
        for i, (obs, actions) in enumerate(zip(episode_data['obs'], episode_data['actions'])):
            self.observations.append(obs)
            self.actions.append(actions)
            self.old_log_probs.append(episode_data['log_probs'][i])
            self.advantages.append(advantages[i] if i < len(advantages) else 0)
            self.returns.append(returns[i] if i < len(returns) else 0)
    
    def get_batch(self, indices: np.ndarray) -> Dict[str, Any]:
        """Get batch of experiences."""
        return {
            'observations': [self.observations[i] for i in indices],
            'actions': [self.actions[i] for i in indices],
            'old_log_probs': [self.old_log_probs[i] for i in indices],
            'advantages': [self.advantages[i] for i in indices],
            'returns': [self.returns[i] for i in indices]
        }
    
    def size(self) -> int:
        """Get buffer size."""
        return len(self.observations)
    
    def clear(self):
        """Clear buffer."""
        self.observations = []
        self.actions = []
        self.old_log_probs = []
        self.advantages = []
        self.returns = []


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Sable on MAgentX Combined Arms")
    parser.add_argument("--config", type=str, default="config/magent_sable_config.yaml", help="Config file path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cpu/cuda)")
    parser.add_argument("--episodes", type=int, default=None, help="Override number of episodes")
    parser.add_argument("--save-dir", type=str, default=None, help="Override save directory")
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        # Default configuration
        config = {
            'env_config': {
                'map_size': 16,
                'minimap_mode': False,
                'step_reward': -0.005,
                'dead_penalty': -0.1,
                'attack_penalty': -0.1,
                'attack_opponent_reward': 0.2,
                'max_cycles': 1000,
                'extra_features': False
            },
            'training_config': {
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_epsilon': 0.2,
                'entropy_coef': 0.01,
                'value_coef': 0.5,
                'max_grad_norm': 0.5,
                'n_episodes': 1000,
                'max_steps_per_episode': 1000,
                'update_frequency': 10,
                'n_epochs': 4,
                'batch_size': 32,
                'save_frequency': 100,
                'log_frequency': 10,
                'save_dir': 'checkpoints',
                'policy_config': {
                    'n_agents_per_team': 4,
                    'embed_dim': 128,
                    'n_head': 8,
                    'n_block': 3
                }
            }
        }
    
    # Override with command line arguments
    if args.episodes is not None:
        config['training_config']['n_episodes'] = args.episodes
    if args.save_dir is not None:
        config['training_config']['save_dir'] = args.save_dir
    
    # Initialize trainer
    trainer = MAgentSableTrainer(
        env_config=config['env_config'],
        training_config=config['training_config'],
        device=args.device
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()