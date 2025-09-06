# Sable Integration with MAgentX Combined Arms Environment

This directory contains a complete implementation of **Sable** (multi-agent reinforcement learning with retention mechanisms) integrated with the **MAgentX Combined Arms v6** environment.

## 🎯 Overview

**Sable** is an advanced multi-agent RL algorithm that uses:
- **Multi-scale retention mechanism** instead of traditional transformer attention
- **Autoregressive action selection** for sequential agent coordination 
- **Temporal memory management** with decay-based hidden states
- **100% JAX feature parity** implemented in PyTorch

**MAgentX Combined Arms** is a tactical combat simulation where:
- **Red vs Blue teams** battle on a 16x16 grid battlefield
- Each team has **6 agents** with different unit types (melee/ranged)
- **Individual action spaces** with **decentralized execution**
- **Local observations** (13x13 grid) with partial visibility

## 🚀 Key Features Implemented

### ✅ Complete Sable Algorithm
- **Multi-scale retention with different decay rates per head**
- **Proper xi computation with episode boundary handling**
- **Timestep-based positional encoding**
- **Sophisticated decay matrix with done flag masking**
- **Group normalization and swish gating**
- **Causal masking for agent coordination**
- **Memory-efficient chunked processing**

### ✅ MAgentX Environment Adaptation
- **Dynamic team size handling** (adapts to 6 agents per team)
- **Team-based coordination** (Red vs Blue policies)
- **Individual agent action selection** with sequential dependencies
- **Local observation preprocessing** (13x13x9 → flattened input)
- **Hidden state management** across episodes

### ✅ Training Infrastructure
- **PPO integration** with GAE advantage computation
- **Experience buffer management**
- **Multi-team policy optimization**
- **Checkpoint saving/loading**
- **Training statistics and logging**

## 📁 File Structure

```
custom_sable/
├── Core Sable Implementation (100% JAX Parity)
│   ├── sable_network_fixed.py      # Main Sable network with autoregressive action selection
│   ├── retention_advanced.py       # Advanced multi-scale retention mechanism
│   ├── autoregressive_utils.py     # Autoregressive coordination utilities
│   └── config.py                   # Sable configuration management
│
├── MAgentX Integration
│   ├── magent_sable_policy.py      # Sable policy adapter for MAgentX environment
│   ├── train_magent_sable.py       # PPO training script for MAgentX
│   ├── inference_sable.py          # Inference script using trained Sable policies
│   └── test_magent_integration.py  # Integration tests
│
├── Configuration & Setup
│   ├── config/
│   │   └── magent_sable_config.yaml # Training configuration
│   └── README_SABLE_MAGENT.md       # This documentation
│
└── Environment Interface (Provided)
    ├── inference.py                 # Original random policy inference
    └── README_inf.md               # Environment documentation
```

## 🔧 Quick Start

### 1. Test Integration
```bash
# Verify everything works
python test_magent_integration.py
```

### 2. Train Sable Policies  
```bash
# Train with default configuration
python train_magent_sable.py --config config/magent_sable_config.yaml

# Train with custom settings
python train_magent_sable.py --episodes 5000 --device cuda --save-dir my_checkpoints
```

### 3. Run Inference with Trained Policies
```bash
# Use trained checkpoints
python inference_sable.py --red-checkpoint checkpoints/red_policy_ep_2000.pt \
                          --blue-checkpoint checkpoints/blue_policy_ep_2000.pt \
                          --episodes 5 --render --save-gif

# Compare with untrained policies
python inference_sable.py --episodes 5 --render
```

### 4. Compare with Random Baseline
```bash
# Original random policy
python inference.py --episodes 5 --render

# Sable policy (untrained)
python inference_sable.py --episodes 5 --render
```

## ⚙️ Configuration Options

### Training Configuration (`config/magent_sable_config.yaml`)

```yaml
# Environment settings
env_config:
  map_size: 16                    # Battlefield size (16x16)
  step_reward: -0.005             # Per-step cost
  attack_opponent_reward: 0.2     # Reward for successful attacks

# Training hyperparameters
training_config:
  learning_rate: 3e-4             # Policy learning rate
  n_episodes: 2000                # Total training episodes
  update_frequency: 20            # Policy update frequency
  
  # Sable network architecture
  policy_config:
    n_agents_per_team: 6          # MAgentX team size
    embed_dim: 128                # Network embedding dimension
    n_head: 8                     # Retention mechanism heads
    n_block: 3                    # Number of Sable blocks
```

### Command Line Arguments

| Script | Key Arguments | Description |
|--------|---------------|-------------|
| `train_magent_sable.py` | `--episodes`, `--device`, `--save-dir` | Training configuration |
| `inference_sable.py` | `--red-checkpoint`, `--blue-checkpoint` | Load trained policies |
| `test_magent_integration.py` | None | Automated integration testing |

## 🧠 Sable Algorithm Details

### Multi-Scale Retention Mechanism
```python
# Different decay rates for each attention head
decay_kappas = 1 - exp(linspace(log(1/32), log(1/512), n_head))

# Xi computation with episode boundaries
xi[t] = gamma^(t+1) * (not done_before_t)

# Sophisticated decay matrix
decay_matrix[i,j] = gamma^(i-j) if i >= j and no_episode_boundary else 0
```

### Autoregressive Action Selection
```python
# Agent i sees actions from agents 0..i-1
for agent_idx in range(n_agents):
    # Sequential action generation
    action_i = policy(obs_i, prev_actions[0:i])
    actions[i] = action_i
```

### Team Coordination
- **Red team policy**: Coordinates red melee and ranged units
- **Blue team policy**: Coordinates blue melee and ranged units  
- **Independent training**: Each team optimizes against the other
- **Shared architecture**: Same Sable network for both teams

## 📊 Expected Performance

### Untrained Sable vs Random Policy
- **Similar performance initially** (random initialization)
- **Coordination potential** through retention mechanism
- **Sequential action dependencies** even without training

### After Training (Expected)
- **Improved team coordination** through autoregressive action selection
- **Better resource allocation** between melee and ranged units
- **Strategic positioning** based on partial observations
- **Temporal reasoning** using retention memory

## 🔍 Testing & Validation

### Integration Tests (`test_magent_integration.py`)
✅ **Environment Setup**: MAgentX initialization and specs  
✅ **Policy Creation**: Multi-team Sable policy instantiation  
✅ **Action Selection**: Team-based action generation  
✅ **Autoregressive Coordination**: Sequential action dependencies  
✅ **Training Integration**: PPO training loop functionality  

### Manual Testing
```bash
# Quick functionality test
python test_magent_integration.py

# Single episode training test  
python train_magent_sable.py --episodes 1 --log-frequency 1

# Inference comparison
python inference.py --episodes 1 --render          # Random
python inference_sable.py --episodes 1 --render    # Sable
```

## 🚧 Advanced Usage

### Custom Network Architecture
```python
# Modify policy_config in YAML or create custom policy
policy = MAgentSablePolicy(
    obs_shape=(13, 13, 9),
    action_dim=9,
    n_agents_per_team=6,
    embed_dim=256,    # Larger network
    n_head=16,        # More attention heads
    n_block=6         # Deeper network
)
```

### Multi-GPU Training
```python
# Set device in config or command line
python train_magent_sable.py --device cuda:0
```

### Curriculum Learning
```yaml
# Add to config/magent_sable_config.yaml
advanced_config:
  curriculum_learning: true
  curriculum_stages:
    - {episodes: 500, map_size: 12}   # Start smaller
    - {episodes: 1000, map_size: 16}  # Standard size
    - {episodes: 500, map_size: 20}   # Harder challenge
```

## 📈 Monitoring & Analysis

### Training Logs
```
Episode 100/2000:
  Length: 245 steps (avg: 248.7)
  Red reward: -2.45 (avg: -2.49)  
  Blue reward: -3.12 (avg: -3.01)
  Red survivors: 3 (avg: 3.0)
  Blue survivors: 1 (avg: 2.1)
```

### Checkpoints
- **Automatic saving** every N episodes
- **Policy state dictionaries** for both teams
- **Training state** (optimizers, episode count)
- **Configuration backup** for reproducibility

## 🔄 Integration with Original Environment

The Sable implementation is designed to be a **drop-in replacement** for the random policy:

```python
# Original: Random policy
from inference import RandomPolicy
policy = RandomPolicy(env)

# New: Sable policy  
from inference_sable import SablePolicy
policy = SablePolicy(env, checkpoint_paths={'red': 'red.pt', 'blue': 'blue.pt'})

# Same interface
actions = policy.get_actions(observations)
```

## 🎯 Next Steps

1. **Train policies** using the provided training script
2. **Evaluate performance** against random and rule-based baselines  
3. **Experiment with architectures** (network size, retention parameters)
4. **Analyze coordination behaviors** in saved battle replays
5. **Extend to larger maps** or different MAgentX scenarios

## 📚 Technical References

- **Sable Algorithm**: Multi-Agent Reinforcement Learning with Retention
- **MAgentX Environment**: [Combined Arms v6 Documentation](README_inf.md)
- **PPO Training**: Proximal Policy Optimization for Multi-Agent Systems
- **Retention Mechanism**: Alternative to Transformer Attention for Sequence Modeling

---

**🎉 Sable is now successfully integrated with MAgentX Combined Arms and ready for training!**

For questions or issues, refer to the test files and integration documentation.