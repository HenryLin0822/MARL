"""Sable training implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import logging
from pathlib import Path
import pickle

from .sable_network import SableNetwork
from .config import SableConfig


class PPOTrajectory:
    """Stores trajectory data for PPO training."""
    
    def __init__(self):
        self.observations: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.dones: List[torch.Tensor] = []
        self.legal_actions: List[torch.Tensor] = []
        
    def add_step(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        log_prob: torch.Tensor,
        done: torch.Tensor,
        legal_action: Optional[torch.Tensor] = None
    ):
        """Add a single step to the trajectory."""
        self.observations.append(obs.clone())
        self.actions.append(action.clone())
        self.values.append(value.clone())
        self.rewards.append(reward.clone())
        self.log_probs.append(log_prob.clone())
        self.dones.append(done.clone())
        if legal_action is not None:
            self.legal_actions.append(legal_action.clone())
    
    def get_batch(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Convert trajectory to training batch."""
        return {
            'observations': torch.stack(self.observations).to(device),
            'actions': torch.stack(self.actions).to(device),
            'values': torch.stack(self.values).to(device),
            'rewards': torch.stack(self.rewards).to(device),
            'log_probs': torch.stack(self.log_probs).to(device),
            'dones': torch.stack(self.dones).to(device),
            'legal_actions': torch.stack(self.legal_actions).to(device) if self.legal_actions else None
        }
    
    def clear(self):
        """Clear all stored data."""
        self.observations.clear()
        self.actions.clear()
        self.values.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.legal_actions.clear()


class SableTrainer:
    """PPO trainer for Sable algorithm."""
    
    def __init__(
        self,
        config: SableConfig,
        env_factory=None,
        device: Optional[torch.device] = None,
        log_dir: Optional[str] = None
    ):
        self.config = config
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize network
        self.network = SableNetwork(config).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.lr)
        
        # Initialize environment factory (user-provided)
        self.env_factory = env_factory
        
        # Training state
        self.step = 0
        self.episode = 0
        
        # Logging
        self.log_dir = log_dir
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            self.setup_logging(log_dir)
        else:
            self.writer = None
            
        self.trajectory = PPOTrajectory()
        
    def setup_logging(self, log_dir: str):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Path(log_dir) / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: [T, B] reward tensor
            values: [T, B] value tensor  
            dones: [T, B] done tensor
            next_value: [B] final value estimate
            
        Returns:
            advantages: [T, B] advantage estimates
            returns: [T, B] discounted returns
        """
        T, B = rewards.shape
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(B, device=rewards.device)
        
        # Bootstrap final value
        next_value = next_value * (1 - dones[-1])
        
        # Compute GAE backwards
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1 - dones[t]
                next_values = next_value
            else:
                next_non_terminal = 1 - dones[t]
                next_values = values[t + 1]
                
            delta = rewards[t] + self.config.gamma * next_values * next_non_terminal - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            
        returns = advantages + values
        return advantages, returns
        
    def ppo_loss(
        self,
        batch: Dict[str, torch.Tensor],
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute PPO loss.
        
        Args:
            batch: Training batch
            advantages: [T, B*N] advantage estimates
            returns: [T, B*N] return targets
            
        Returns:
            total_loss: Combined loss tensor
            loss_info: Dictionary of loss components
        """
        T, B, N, obs_dim = batch['observations'].shape
        
        # Reshape for network processing: [T*B, N, obs_dim] -> [T*B, N*1, obs_dim]
        obs = batch['observations'].view(T * B, N, obs_dim)
        actions = batch['actions'].view(T * B, N, -1)
        old_log_probs = batch['log_probs'].view(T * B, N)
        old_values = batch['values'].view(T * B, N)
        dones = batch['dones'].view(T * B, N)
        legal_actions = batch['legal_actions'].view(T * B, N, -1) if batch['legal_actions'] is not None else None
        
        # Initialize dummy hidden states for training
        dummy_hstates = self.network.init_hidden_states(T * B, self.device)
        
        # Forward pass
        values, log_probs, entropy = self.network(
            obs.view(T * B, N * 1, obs_dim), 
            actions.view(T * B, N * 1, -1),
            dummy_hstates,
            dones.view(T * B, N * 1),
            legal_actions.view(T * B, N * 1, -1) if legal_actions is not None else None
        )
        
        # Reshape outputs back
        values = values.view(T * B, N)
        log_probs = log_probs.view(T * B, N)
        entropy = entropy.view(T * B, N)
        
        # Flatten for loss computation
        advantages_flat = advantages.view(-1)
        returns_flat = returns.view(-1)
        old_log_probs_flat = old_log_probs.view(-1)
        old_values_flat = old_values.view(-1)
        values_flat = values.view(-1)
        log_probs_flat = log_probs.view(-1)
        entropy_flat = entropy.view(-1)
        
        # Normalize advantages
        advantages_norm = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
        
        # PPO actor loss
        ratio = torch.exp(log_probs_flat - old_log_probs_flat)
        surr1 = ratio * advantages_norm
        surr2 = torch.clamp(
            ratio, 
            1 - self.config.clip_eps, 
            1 + self.config.clip_eps
        ) * advantages_norm
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss (clipped)
        value_pred_clipped = old_values_flat + torch.clamp(
            values_flat - old_values_flat,
            -self.config.clip_eps,
            self.config.clip_eps
        )
        value_loss1 = (values_flat - returns_flat).pow(2)
        value_loss2 = (value_pred_clipped - returns_flat).pow(2)
        value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
        
        # Entropy loss
        entropy_loss = entropy_flat.mean()
        
        # Total loss
        total_loss = (
            actor_loss 
            + self.config.vf_coef * value_loss 
            - self.config.ent_coef * entropy_loss
        )
        
        loss_info = {
            'total_loss': total_loss.item(),
            'actor_loss': actor_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'approx_kl': ((log_probs_flat - old_log_probs_flat).pow(2) * 0.5).mean().item()
        }
        
        return total_loss, loss_info
        
    def train_step(self, env_step_data: Dict[str, Any]) -> Dict[str, float]:
        """Perform one training step.
        
        Args:
            env_step_data: Dictionary containing environment interaction data
            Expected keys: 'observations', 'actions', 'rewards', 'values', 
                          'log_probs', 'dones', 'next_obs', 'legal_actions' (optional)
            
        Returns:
            Dictionary of training metrics
        """
        # Get final value estimate for GAE
        with torch.no_grad():
            next_obs = env_step_data['next_obs']
            B, N, _ = next_obs.shape
            dummy_hstates = self.network.init_hidden_states(B, self.device)
            _, _, next_values, _ = self.network.get_actions(
                next_obs, dummy_hstates, deterministic=True
            )
            next_values = next_values.mean(dim=1)  # Average over agents
            
        # Convert step data to training batch
        batch = {
            'observations': env_step_data['observations'],
            'actions': env_step_data['actions'],
            'values': env_step_data['values'],
            'rewards': env_step_data['rewards'],
            'log_probs': env_step_data['log_probs'],
            'dones': env_step_data['dones'],
            'legal_actions': env_step_data.get('legal_actions')
        }
        
        # Compute advantages and returns
        T, B, N = batch['rewards'].shape
        rewards_mean = batch['rewards'].mean(dim=2)  # Average over agents
        values_mean = batch['values'].mean(dim=2)    # Average over agents
        dones_any = batch['dones'].any(dim=2).float()  # Any agent done
        
        advantages, returns = self.compute_gae(
            rewards_mean, values_mean, dones_any, next_values
        )
        
        # Expand advantages and returns back to agent dimension
        advantages = advantages.unsqueeze(2).expand(-1, -1, N)
        returns = returns.unsqueeze(2).expand(-1, -1, N)
        
        # Train for multiple epochs
        train_metrics = {
            'total_loss': 0,
            'actor_loss': 0,
            'value_loss': 0,
            'entropy_loss': 0,
            'approx_kl': 0
        }
        
        for epoch in range(self.config.ppo_epochs):
            # Shuffle data
            batch_size = T * B
            indices = torch.randperm(batch_size)
            minibatch_size = batch_size // self.config.num_minibatches
            
            epoch_metrics = {key: 0 for key in train_metrics.keys()}
            
            for i in range(self.config.num_minibatches):
                start_idx = i * minibatch_size
                end_idx = start_idx + minibatch_size
                mb_indices = indices[start_idx:end_idx]
                
                # Create minibatch
                mb_batch = {}
                for key, value in batch.items():
                    if value is not None:
                        value_flat = value.view(batch_size, *value.shape[2:])
                        mb_batch[key] = value_flat[mb_indices].unsqueeze(1)
                    else:
                        mb_batch[key] = None
                        
                mb_advantages = advantages.view(batch_size, N)[mb_indices]
                mb_returns = returns.view(batch_size, N)[mb_indices]
                
                # Compute loss
                loss, loss_info = self.ppo_loss(mb_batch, mb_advantages, mb_returns)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), 
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                
                # Accumulate metrics
                for key, value in loss_info.items():
                    epoch_metrics[key] += value
                    
            # Average over minibatches
            for key in epoch_metrics:
                epoch_metrics[key] /= self.config.num_minibatches
                train_metrics[key] += epoch_metrics[key]
                
        # Average over epochs
        for key in train_metrics:
            train_metrics[key] /= self.config.ppo_epochs
            
        return train_metrics
        
    def evaluate(self, env, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the current policy.
        
        Args:
            env: Environment instance
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.network.eval()
        episode_returns = []
        episode_lengths = []
        
        with torch.no_grad():
            for _ in range(num_episodes):
                obs = env.reset()
                if not isinstance(obs, torch.Tensor):
                    obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                if obs.dim() == 2:  # Add batch dimension if needed
                    obs = obs.unsqueeze(0)
                    
                hstates = self.network.init_hidden_states(1, self.device)
                episode_return = 0
                episode_length = 0
                done = False
                
                while not done:
                    actions, _, _, hstates = self.network.get_actions(
                        obs, hstates, deterministic=True
                    )
                    
                    # Step environment
                    if hasattr(env, 'step'):
                        next_obs, rewards, dones, info = env.step(actions.squeeze(0))
                    else:
                        # User needs to implement their environment interface
                        raise NotImplementedError("Environment step method not implemented")
                        
                    episode_return += rewards.sum().item() if isinstance(rewards, torch.Tensor) else rewards
                    episode_length += 1
                    
                    if not isinstance(next_obs, torch.Tensor):
                        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
                    if next_obs.dim() == 2:
                        next_obs = next_obs.unsqueeze(0)
                        
                    obs = next_obs
                    done = dones.any() if isinstance(dones, torch.Tensor) else any(dones)
                    
                episode_returns.append(episode_return)
                episode_lengths.append(episode_length)
                
        self.network.train()
        
        return {
            'eval_return_mean': np.mean(episode_returns),
            'eval_return_std': np.std(episode_returns),
            'eval_length_mean': np.mean(episode_lengths),
            'eval_length_std': np.std(episode_lengths)
        }
        
    def save_checkpoint(self, filepath: str, metadata: Optional[Dict] = None):
        """Save training checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            metadata: Optional metadata dictionary
        """
        checkpoint = {
            'step': self.step,
            'episode': self.episode,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'metadata': metadata or {}
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, filepath)
        
    def load_checkpoint(self, filepath: str) -> Dict:
        """Load training checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Metadata dictionary from checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.step = checkpoint['step']
        self.episode = checkpoint['episode']
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['metadata']
        
    def log_metrics(self, metrics: Dict[str, float]):
        """Log training metrics.
        
        Args:
            metrics: Dictionary of metrics to log
        """
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, self.step)
                
        if hasattr(self, 'logger'):
            metric_str = ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
            self.logger.info(f'Step {self.step} | {metric_str}')
            
    def close(self):
        """Clean up resources."""
        if self.writer:
            self.writer.close()