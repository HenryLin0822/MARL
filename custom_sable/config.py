"""Configuration for Sable algorithm."""

from typing import NamedTuple
import yaml
from pathlib import Path


class SableConfig(NamedTuple):
    """Configuration for Sable network and training."""
    
    # Network architecture
    n_agents: int = 4
    action_dim: int = 4
    obs_dim: int = 32
    
    # Network hyperparameters
    n_block: int = 2
    n_head: int = 4
    embed_dim: int = 128
    
    # Memory/retention config
    chunk_size: int = 4  # Should divide n_agents evenly
    decay_scaling_factor: float = 1.0
    
    # Training hyperparameters
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Training schedule
    rollout_length: int = 128
    ppo_epochs: int = 4
    num_minibatches: int = 4
    
    # Action space
    action_space_type: str = "discrete"  # "discrete" or "continuous"
    
    @classmethod
    def from_yaml(cls, path: str):
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self._asdict(), f, default_flow_style=False)
    
    def validate(self):
        """Validate configuration parameters."""
        assert self.n_agents % self.chunk_size == 0, \
            f"n_agents ({self.n_agents}) must be divisible by chunk_size ({self.chunk_size})"
        assert self.embed_dim % self.n_head == 0, \
            f"embed_dim ({self.embed_dim}) must be divisible by n_head ({self.n_head})"
        assert self.action_space_type in ["discrete", "continuous"], \
            f"action_space_type must be 'discrete' or 'continuous', got {self.action_space_type}"
        assert 0 <= self.decay_scaling_factor <= 1, \
            f"decay_scaling_factor must be between 0 and 1, got {self.decay_scaling_factor}"