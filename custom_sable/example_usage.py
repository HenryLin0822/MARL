"""
Example usage of the custom Sable implementation.

This script demonstrates how to use the extracted Sable algorithm
with your own custom environment.
"""

import torch
import numpy as np
from typing import Tuple, Dict, Any
import yaml

from custom_sable import SableNetwork, SableTrainer, SableConfig


class DummyEnvironment:
    """
    Example dummy environment for demonstration.
    Replace this with your actual environment implementation.
    """
    
    def __init__(self, n_agents: int = 4, obs_dim: int = 32, action_dim: int = 4):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_steps = 100
        self.current_step = 0
        
    def reset(self) -> torch.Tensor:
        """Reset environment and return initial observations."""
        self.current_step = 0
        # Return random observations [n_agents, obs_dim]
        return torch.randn(self.n_agents, self.obs_dim)
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Take environment step.
        
        Args:
            actions: [n_agents] action tensor for discrete actions
                    or [n_agents, action_dim] for continuous actions
                    
        Returns:
            next_obs: [n_agents, obs_dim] next observations
            rewards: [n_agents] reward tensor
            dones: [n_agents] done flags
            info: Dictionary with additional info
        """
        self.current_step += 1
        
        # Random next observations
        next_obs = torch.randn(self.n_agents, self.obs_dim)
        
        # Random rewards (usually you'd compute based on actions/state)
        rewards = torch.randn(self.n_agents) * 0.1
        
        # Done if max steps reached
        dones = torch.zeros(self.n_agents, dtype=torch.bool)
        if self.current_step >= self.max_steps:
            dones.fill_(True)
            
        info = {'step': self.current_step}
        
        return next_obs, rewards, dones, info
    
    def get_legal_actions(self) -> torch.Tensor:
        """Get legal actions mask.
        
        Returns:
            [n_agents, action_dim] boolean mask of legal actions
        """
        # For this dummy env, all actions are legal
        return torch.ones(self.n_agents, self.action_dim, dtype=torch.bool)


def collect_rollout(
    env: DummyEnvironment,
    network: SableNetwork,
    config: SableConfig,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Collect a rollout from the environment.
    
    Args:
        env: Environment instance
        network: Sable network
        config: Configuration
        device: Device to run on
        
    Returns:
        Dictionary containing rollout data
    """
    network.eval()
    
    # Storage for rollout
    observations = []
    actions = []
    values = []
    rewards = []
    log_probs = []
    dones = []
    legal_actions_list = []
    
    # Reset environment
    obs = env.reset().to(device).unsqueeze(0)  # Add batch dimension
    hstates = network.init_hidden_states(1, device)
    
    with torch.no_grad():
        for step in range(config.rollout_length):
            # Get legal actions if needed
            legal_actions = env.get_legal_actions().to(device).unsqueeze(0)
            
            # Get actions from network
            action, log_prob, value, hstates = network.get_actions(
                obs, hstates, deterministic=False, legal_actions=legal_actions
            )
            
            # Step environment
            next_obs, reward, done, info = env.step(action.squeeze(0))
            
            # Store data
            observations.append(obs.squeeze(0))  # Remove batch dim
            actions.append(action.squeeze(0))
            values.append(value.squeeze(0))
            rewards.append(reward.unsqueeze(0))  # Add time dimension
            log_probs.append(log_prob.squeeze(0))
            dones.append(done.unsqueeze(0).float())
            legal_actions_list.append(legal_actions.squeeze(0))
            
            # Update observation
            obs = next_obs.to(device).unsqueeze(0)
            
            if done.any():
                obs = env.reset().to(device).unsqueeze(0)
                hstates = network.init_hidden_states(1, device)
    
    # Stack all data
    rollout_data = {
        'observations': torch.stack(observations),  # [T, N, obs_dim]
        'actions': torch.stack(actions),            # [T, N] or [T, N, action_dim]
        'values': torch.stack(values),              # [T, N]
        'rewards': torch.stack(rewards),            # [T, N]
        'log_probs': torch.stack(log_probs),        # [T, N]
        'dones': torch.stack(dones),                # [T, N]
        'legal_actions': torch.stack(legal_actions_list),  # [T, N, action_dim]
        'next_obs': obs.squeeze(0)                  # [N, obs_dim]
    }
    
    # Add batch dimension
    for key, value in rollout_data.items():
        if key != 'next_obs':
            rollout_data[key] = value.unsqueeze(1)  # [T, 1, ...]
        else:
            rollout_data[key] = value.unsqueeze(0)  # [1, ...]
            
    return rollout_data


def main():
    """Main training loop example."""
    
    # Configuration
    config = SableConfig(
        n_agents=4,
        action_dim=4,
        obs_dim=32,
        n_block=2,
        n_head=4,
        embed_dim=128,
        chunk_size=4,
        lr=3e-4,
        rollout_length=128,
        ppo_epochs=4,
        num_minibatches=4,
        action_space_type="discrete"
    )
    
    # Save config example
    config.to_yaml("sable_config.yaml")
    print("Config saved to sable_config.yaml")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create environment
    env = DummyEnvironment(
        n_agents=config.n_agents,
        obs_dim=config.obs_dim,
        action_dim=config.action_dim
    )
    
    # Create trainer
    trainer = SableTrainer(
        config=config,
        device=device,
        log_dir="./logs/sable_training"
    )
    
    print("Starting training...")
    
    # Training loop
    num_updates = 100
    eval_freq = 10
    
    for update in range(num_updates):
        # Collect rollout
        rollout_data = collect_rollout(env, trainer.network, config, device)
        
        # Train
        trainer.network.train()
        train_metrics = trainer.train_step(rollout_data)
        trainer.step += 1
        
        # Log metrics
        trainer.log_metrics(train_metrics)
        
        # Evaluate periodically
        if update % eval_freq == 0:
            eval_metrics = trainer.evaluate(env, num_episodes=5)
            trainer.log_metrics(eval_metrics)
            
            print(f"Update {update:3d} | "
                  f"Loss: {train_metrics['total_loss']:.4f} | "
                  f"Return: {eval_metrics['eval_return_mean']:.4f}")
        
        # Save checkpoint
        if update % 50 == 0:
            trainer.save_checkpoint(
                f"./logs/sable_training/checkpoint_{update}.pt",
                metadata={'update': update}
            )
    
    print("Training completed!")
    
    # Final evaluation
    final_metrics = trainer.evaluate(env, num_episodes=20)
    print(f"Final evaluation return: {final_metrics['eval_return_mean']:.4f} ± {final_metrics['eval_return_std']:.4f}")
    
    # Cleanup
    trainer.close()


def test_network():
    """Test the network components individually."""
    print("Testing Sable network components...")
    
    config = SableConfig(n_agents=4, obs_dim=32, action_dim=4, embed_dim=64, n_head=2, n_block=1)
    device = torch.device('cpu')
    
    # Create network
    network = SableNetwork(config).to(device)
    print(f"Network created with {sum(p.numel() for p in network.parameters())} parameters")
    
    # Test forward pass
    batch_size = 2
    seq_len = 8
    
    # Create dummy data
    obs = torch.randn(batch_size, seq_len, config.obs_dim)
    actions = torch.randint(0, config.action_dim, (batch_size, seq_len))
    hstates = network.init_hidden_states(batch_size, device)
    dones = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    
    # Test training forward pass
    values, log_probs, entropy = network(
        obs, actions.unsqueeze(-1).float(), hstates, dones
    )
    
    print(f"Training pass - Values: {values.shape}, Log probs: {log_probs.shape}, Entropy: {entropy.shape}")
    
    # Test action generation
    obs_single = torch.randn(batch_size, config.n_agents, config.obs_dim)
    actions_out, log_probs_out, values_out, hstates_out = network.get_actions(
        obs_single, hstates, deterministic=False
    )
    
    print(f"Action generation - Actions: {actions_out.shape}, Values: {values_out.shape}")
    print("Network test completed successfully!")


if __name__ == "__main__":
    # Test network first
    test_network()
    print("\\n" + "="*50 + "\\n")
    
    # Run main training example
    main()