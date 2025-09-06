# Custom Sable Implementation

This directory contains a standalone PyTorch implementation of the Sable algorithm, extracted and adapted from the original Mava codebase. The implementation uses standard deep learning libraries (PyTorch, numpy, einops) instead of JAX, making it more accessible and easier to integrate with custom environments.

## Overview

Sable is a multi-agent reinforcement learning algorithm that leverages:
- **Retentive Networks**: For efficient memory mechanism across agents and time
- **Autoregressive Action Selection**: Actions are generated sequentially per agent
- **Advantage Decomposition**: For convergence guarantees in multi-agent settings
- **PPO-style Training**: Proximal Policy Optimization for stable training

## Files

- `__init__.py`: Package initialization and exports
- `config.py`: Configuration dataclass for Sable hyperparameters
- `retention.py`: Multi-scale retention mechanism implementation
- `sable_network.py`: Main Sable network architecture (encoder-decoder)
- `sable_trainer.py`: PPO trainer for Sable algorithm
- `example_usage.py`: Complete example showing how to use the implementation
- `requirements.txt`: Required Python packages
- `README.md`: This documentation file

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. The implementation is self-contained and doesn't require installation. Just import from the directory:
```python
from custom_sable import SableNetwork, SableTrainer, SableConfig
```

## Quick Start

### Basic Usage

```python
import torch
from custom_sable import SableNetwork, SableConfig

# Create configuration
config = SableConfig(
    n_agents=4,
    action_dim=4,
    obs_dim=32,
    n_block=2,
    n_head=4,
    embed_dim=128
)

# Create network
network = SableNetwork(config)

# Initialize hidden states
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hstates = network.init_hidden_states(batch_size=1, device=device)

# Get actions (inference)
obs = torch.randn(1, config.n_agents, config.obs_dim)
actions, log_probs, values, updated_hstates = network.get_actions(
    obs, hstates, deterministic=False
)
```

### Training Example

```python
from custom_sable import SableTrainer

# Create trainer
trainer = SableTrainer(
    config=config,
    device=device,
    log_dir="./logs"
)

# Your training loop would collect environment data and call:
# train_metrics = trainer.train_step(rollout_data)
```

See `example_usage.py` for a complete working example.

## Configuration

The `SableConfig` class contains all hyperparameters:

### Network Architecture
- `n_agents`: Number of agents in the environment
- `action_dim`: Dimension of action space
- `obs_dim`: Dimension of observation space
- `n_block`: Number of encoder/decoder blocks
- `n_head`: Number of attention heads in retention mechanism
- `embed_dim`: Hidden dimension size

### Memory Configuration
- `chunk_size`: Number of agents processed in each chunk (must divide n_agents)
- `decay_scaling_factor`: Scaling factor for retention decay (0-1)

### Training Hyperparameters
- `lr`: Learning rate
- `gamma`: Discount factor
- `gae_lambda`: GAE lambda parameter
- `clip_eps`: PPO clipping epsilon
- `ent_coef`: Entropy coefficient
- `vf_coef`: Value function coefficient
- `max_grad_norm`: Gradient clipping norm

### Training Schedule
- `rollout_length`: Steps per rollout
- `ppo_epochs`: PPO update epochs per rollout
- `num_minibatches`: Number of minibatches per epoch

## Environment Interface

Your environment should implement these methods:

```python
class YourEnvironment:
    def reset(self) -> torch.Tensor:
        \"\"\"Return initial observations [n_agents, obs_dim]\"\"\"
        pass
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        \"\"\"
        Args:
            actions: [n_agents] for discrete or [n_agents, action_dim] for continuous
        Returns:
            next_obs: [n_agents, obs_dim]
            rewards: [n_agents] 
            dones: [n_agents] boolean
            info: Dict with additional info
        \"\"\"
        pass
    
    def get_legal_actions(self) -> torch.Tensor:
        \"\"\"Optional: return [n_agents, action_dim] boolean mask\"\"\"
        pass
```

## Key Features

### Multi-Scale Retention
- Efficient alternative to self-attention
- Multiple heads with different time scales
- Supports both chunk-wise and recurrent processing

### Autoregressive Action Generation
- Actions generated sequentially across agents
- Enables coordination through action dependencies
- Supports both discrete and continuous action spaces

### PPO Training
- Stable policy optimization
- Advantage estimation with GAE
- Gradient clipping and value function clipping

## Differences from Original Mava Implementation

1. **Framework**: PyTorch instead of JAX/Flax
2. **Dependencies**: Minimal dependencies (torch, numpy, einops, etc.)
3. **Environment Interface**: Simplified, user-defined environment interface
4. **Vectorization**: Batch processing instead of JAX vmap
5. **Logging**: TensorBoard integration instead of custom loggers

## Performance Considerations

- The implementation prioritizes readability and modularity over maximum performance
- For production use, consider:
  - Using mixed precision training (`torch.cuda.amp`)
  - Optimizing batch processing for your specific environment
  - Using compiled models (`torch.compile` in PyTorch 2.0+)

## Extending the Implementation

### Adding New Action Spaces
Modify `SableNetwork.forward()` and `SableNetwork.get_actions()` to handle your action space.

### Custom Loss Functions
Extend `SableTrainer.ppo_loss()` to include additional loss terms.

### Environment Wrappers
Create wrappers around your environment to match the expected interface.

## Citation

If you use this implementation, please cite the original Sable paper:

```bibtex
@article{mahjoub2024sable,
    title={Performant, Memory Efficient and Scalable Multi-Agent Reinforcement Learning},
    author={Omayma Mahjoub and Wiem Khlifi and Arnu Pretorius},
    year={2024},
    journal={arXiv preprint arXiv:2410.01706},
}
```

## License

This implementation maintains the Apache 2.0 license from the original Mava codebase.