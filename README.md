# MARL Combat Simulation

A Multi-Agent Reinforcement Learning (MARL) project featuring tactical combat simulations using the MAgentX/magent2 environment. This project implements the Combined Arms v6 scenario where two opposing teams battle on a grid-based battlefield.

## Game Introduction

### Overview
The Combined Arms environment simulates strategic combat between two teams - **Red** and **Blue** - on a 2D grid battlefield. Each team consists of multiple unit types with different capabilities, creating complex tactical scenarios that require coordinated multi-agent strategies.

### Objective
Teams compete to eliminate opposing forces while preserving their own units. The simulation continues until one team is eliminated or the maximum episode length is reached.

### Environment Features
- **Grid-based battlefield** with configurable map size (default: 16x16)
- **Real-time combat** with simultaneous agent actions
- **Partial observability** - agents only see their local surroundings
- **Reward structure** encouraging tactical combat behavior

## Unit Types

Each team deploys 4 distinct unit types, each with specialized roles and characteristics:

### Red Team Units
| Unit Type | Color | Role | Characteristics |
|-----------|-------|------|----------------|
| **Melee 1** | 🔴 Bright Red | Heavy Infantry | High HP, slow movement, close-range combat |
| **Melee 2** | 🔴 Dark Red | Assault Infantry | High HP, slow movement, close-range combat |
| **Ranged 1** | 🟠 Orange | Archer/Sniper | Low HP, fast movement, long-range attacks |
| **Ranged 2** | 🟡 Yellow | Artillery | Low HP, fast movement, long-range attacks |

### Blue Team Units  
| Unit Type | Color | Role | Characteristics |
|-----------|-------|------|----------------|
| **Melee 1** | 🔵 Bright Blue | Heavy Infantry | High HP, slow movement, close-range combat |
| **Melee 2** | 🔵 Dark Blue | Assault Infantry | High HP, slow movement, close-range combat |
| **Ranged 1** | 🔵 Cyan | Archer/Sniper | Low HP, fast movement, long-range attacks |
| **Ranged 2** | 🔵 Sky Blue | Artillery | Low HP, fast movement, long-range attacks |

### Unit Mechanics
- **Melee Units**: Tank-like units that excel in close combat but move slowly
- **Ranged Units**: Glass cannon units that deal damage from distance but are fragile
- **Health System**: Unit brightness indicates health level (brighter = healthier)
- **Death**: Units are removed when health reaches zero

## Action Space and State Representation

### Action Space
Each agent has an **individual discrete action space** and makes decisions independently based on its local observations. Actions are **not joint** - each agent selects its own action simultaneously.

**Action Selection Process**:
```python
# Each agent decides independently
for agent_id, obs in observations.items():
    action = policy.get_action(obs, agent_id)  # Individual decision
    actions[agent_id] = action

# All actions executed simultaneously
next_obs, rewards, terms, truncs, infos = env.step(actions)
```

**Action Space Specification**:
- **Discrete Action Space**: Each agent has `Discrete(9)` action space
- **Total Agents**: 12 agents (6 per team: 3 melee + 3 ranged units each)
- **Action Categories**: 
  - **Movement**: 4 directions (Up, Down, Left, Right) + Stay (1 action)
  - **Attack**: 4 directional attacks (Up, Down, Left, Right)
  - **Total**: 9 discrete actions per agent

**Action Space Properties**:
- **Decentralized execution**: No coordination between agents during action selection
- **Simultaneous processing**: All agent actions are executed at the same timestep
- **Local decision-making**: Actions based only on agent's partial observation
- **Individual control**: Each of the 12 agents selects from 9 possible actions independently

### State Representation

The environment provides both **global state** (for rendering/analysis) and **local observations** (for agent decision-making).

#### Global State
**Shape**: `(height, width, channels)` where `channels = 9`

**Channel Layout**:
```python
Channel 0: Walls/Obstacles (gray in visualization)
Channel 1-4: Red team units by type (red spectrum colors)
  - Channel 1: Red melee unit 1 (bright red)
  - Channel 2: Red melee unit 2 (dark red) 
  - Channel 3: Red ranged unit 1 (orange)
  - Channel 4: Red ranged unit 2 (yellow)
Channel 5-8: Blue team units by type (blue spectrum colors)
  - Channel 5: Blue melee unit 1 (bright blue)
  - Channel 6: Blue melee unit 2 (dark blue)
  - Channel 7: Blue ranged unit 1 (cyan)
  - Channel 8: Blue ranged unit 2 (sky blue)
```

**Value Encoding**:
- **0**: Empty space or dead unit
- **0.1 to 1.0**: Unit present with health level (continuous values, higher = healthier)
- **Walls**: Binary presence (0 or 1)

**Note**: This continuous health representation is the default behavior in the magent2 environment package.

#### Local Agent Observations

Each agent receives a **13x13 grid observation** centered on its position with multiple information channels.

**Observation Dimensions**: `(13, 13, channels)` where `channels = 9-12` depending on configuration

**Channel Structure** (in order):
```python
Channel 0: Obstacles/Walls (off-map boundaries and walls)
Channel 1: My Team Presence (friendly unit locations)
Channel 2: My Team Health (friendly unit health levels)
Channel 3: Enemy Team 1 Presence (opponent locations)
Channel 4: Enemy Team 1 Health (opponent health levels)
Channel 5: Enemy Team 2 Presence (if multiple enemy teams)
Channel 6: Enemy Team 2 Health (if multiple enemy teams)
Channel 7: [Additional channels for extra features if enabled]
Channel 8: [One-hot encoded last action taken by agent]
```

**Physical Meaning**:
- **Grid Position**: Each cell represents one battlefield grid square
- **Agent-Centered**: The 13x13 view is centered on the observing agent
- **Local Visibility**: Agent can only see 6 cells in each direction from its position
- **Multi-layered**: Each channel provides different information about the same spatial area

**Value Ranges**:
- **[0, 2]**: Most observation values fall in this range
- **0**: Empty space or absent feature
- **>0**: Feature present with intensity/health level
- **Health channels**: Continuous values representing unit health

**Key Properties**:
- **Partial observability**: 13x13 window vs full map (default 16x16)
- **Spatial structure**: Preserves relative positions of units
- **Multi-channel**: Rich information beyond just unit presence
- **Agent-centric**: Always centered on the observing agent's position
- **Real-time updates**: Updated each timestep as agents move and act

#### State Access Examples
```python
# Global state (for rendering/analysis)
global_state = env.state()  # Shape: (16, 16, 9)
rgb_image = state_to_rgb(global_state)

# Individual agent observations (for decision-making)
observations = env.reset()  # Dict: {agent_id: local_obs}
for agent_id, local_obs in observations.items():
    action = policy.get_action(local_obs, agent_id)
```

**Observation Space Characteristics**:
- **Numpy arrays**: Typically float32 dtype
- **Continuous values**: Health levels as floating point numbers (not binary)
- **Range**: [0, 1] where 0 = empty/dead, 1 = full health
- **Variable size**: Depends on unit type and observation radius
- **Dynamic**: Observation space may change as agents die

## Rendering and Visualization

### Color Scheme
The environment uses an intuitive color-coding system for easy identification:

- **🔘 Dark Gray**: Walls and obstacles
- **⚫ Black**: Empty battlefield space
- **Red Spectrum** (🔴→🟡): Red team units by type and health
- **Blue Spectrum** (🔵→🔵): Blue team units by type and health

### Rendering Features

#### Real-time Visualization
```python
# Basic rendering during simulation
python inference.py --render
```

#### GIF Generation
```python
# Save battle replay as animated GIF
python inference.py --save-gif --gif-path render/battle.gif
```

#### Customizable Scaling
```python
# Adjust GIF image size for better visibility
python inference.py --save-gif --scale-factor 30
```

### Output Files
- **GIFs**: Saved to `render/` directory
- **Default output**: `render/inference_simulation.gif`
- **Frame rate**: 10 FPS with infinite loop

## Usage Instructions

### Prerequisites
```bash
pip install magent2 numpy pillow imageio
```

### Basic Usage

#### 1. Run Single Episode
```bash
python inference.py
```

#### 2. Multiple Episodes with Rendering  
```bash
python inference.py --episodes 5 --render
```

#### 3. Create Battle Replay
```bash
python inference.py --save-gif --render --steps 2000
```

#### 4. Custom Map Size
```bash
python inference.py --map-size 32 --save-gif
```

### Command Line Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--episodes` | Number of episodes to run | 1 |
| `--steps` | Maximum steps per episode | 1000 |
| `--map-size` | Battlefield dimensions (NxN) | 16 |
| `--render` | Print progress during simulation | False |
| `--save-gif` | Generate animated GIF | False |
| `--gif-path` | Output path for GIF file | `render/inference_simulation.gif` |
| `--scale-factor` | Image scaling for GIF visibility | 20 |

### Advanced Usage Examples

#### Long Battle with High Resolution
```bash
python inference.py --episodes 3 --steps 3000 --map-size 24 \
                   --save-gif --scale-factor 25 --render
```

#### Environment Exploration
```bash
python examine_state.py  # Inspect environment structure
python render_simulation.py  # Quick visualization test
```

## File Structure

```
MARL/
├── inference.py              # Main simulation runner
├── examine_state.py          # Environment exploration tools  
├── render_simulation.py      # Simplified rendering script
└── render/
    └── inference_simulation.gif  # Generated battle replays
```

### Core Components

#### `inference.py`
- **RandomPolicy class**: Baseline random agent behavior
- **state_to_rgb()**: Environment state to RGB conversion
- **run_inference()**: Main simulation loop with statistics

#### `examine_state.py`  
- Environment structure inspection
- Agent observation analysis
- State space exploration

#### `render_simulation.py`
- Lightweight visualization
- Quick GIF generation for testing

## Environment Configuration

### Reward Structure

The Combined Arms v6 environment uses a comprehensive reward system designed to encourage tactical combat behavior while balancing aggression and survival. Rewards are **additive**, meaning multiple reward conditions can apply simultaneously in a single time step.

#### Core Reward Components

| Reward Type | Value | Parameter | Description |
|-------------|-------|-----------|-------------|
| **Step Reward** | -0.005 | `step_reward` | Small penalty each time step to encourage quick mission completion |
| **Death Penalty** | -0.1 | `dead_penalty` | Penalty when an agent dies, discouraging reckless behavior |
| **Attack Penalty** | -0.1 | `attack_penalty` | Cost for making any attack, preventing spam attacking |
| **Opponent Kill Reward** | +0.2 | `attack_opponent_reward` | Reward for successfully killing an enemy agent |
| **Major Kill Bonus** | +5.0 | *(fixed)* | Large bonus for eliminating an opponent (from magent2 core) |

#### Reward Calculation Logic

```python
# Example reward calculation for a single time step:
total_reward = 0

# Every time step
total_reward += step_reward  # -0.005

# If agent attacks (regardless of target)
if agent_attacks:
    total_reward += attack_penalty  # -0.1

# If agent attacks an opponent 
if attacks_opponent:
    total_reward += attack_opponent_reward  # +0.2

# If agent kills an opponent
if kills_opponent:
    total_reward += 5.0  # Major kill bonus

# If agent dies
if agent_dies:
    total_reward += dead_penalty  # -0.1
```

#### Strategic Implications

- **Combat Engagement**: Net positive reward (+5.1) for successfully killing an opponent
- **Attack Decision**: Attacking opponents gives net +0.1 reward, while attacking teammates gives net -0.1
- **Survival Focus**: Death penalty encourages defensive positioning and tactical awareness  
- **Time Pressure**: Step penalty creates urgency to complete objectives efficiently
- **Resource Management**: Attack penalty makes agents selective about when to engage

#### Reward Customization

All reward values can be customized when creating the environment:

```python
env = combined_arms_v6.parallel_env(
    step_reward=-0.01,           # Increase time pressure
    dead_penalty=-0.2,           # Stronger survival incentive  
    attack_penalty=-0.05,        # Lower attack cost
    attack_opponent_reward=0.3,  # Higher engagement reward
    # ... other parameters
)
```

### Technical Settings
- **Parallel environment**: Simultaneous agent actions
- **No minimap mode**: Agents use local observations only
- **Extra features**: Disabled for simplified state space

## Future Extensions

This baseline implementation uses random policies. The codebase is designed to easily integrate:

- **Deep Q-Networks (DQN)** for individual agents
- **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)**
- **Proximal Policy Optimization (PPO)** variants
- **Centralized training, decentralized execution** approaches
- **Communication protocols** between team agents

## Statistics and Analysis

The simulation automatically tracks and reports:
- Episode length (survival time)
- Total team rewards
- Final survivor counts
- Average performance metrics across episodes

Sample output:
```
INFERENCE SUMMARY
==================================================
Episode 1: Length=245, Reward=-2.45, Survivors=3
Episode 2: Length=189, Reward=-1.89, Survivors=1
Episode 3: Length=312, Reward=-3.12, Survivors=5

Averages:
  Length: 248.7 steps
  Reward: -2.49
  Survivors: 3.0
```

This provides baseline metrics for evaluating the performance of future RL implementations.