import numpy as np
from magent2.environments import combined_arms_v6

# Create environment
env = combined_arms_v6.parallel_env(map_size=16, minimap_mode=False, step_reward=-0.005,
                                    dead_penalty=-0.1, attack_penalty=-0.1,
                                    attack_opponent_reward=0.2, max_cycles=1000, extra_features=False)

observations = env.reset()
print("Environment reset!")
print(f"Number of agents: {len(env.agents)}")
print()

# Examine observations for first few agents
for i, (agent_id, obs) in enumerate(observations.items()):
    print(f"Agent {agent_id}:")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation dtype: {obs.dtype}")
    print(f"  Min/Max values: {obs.min():.3f}/{obs.max():.3f}")
    print(f"  Unique values: {np.unique(obs)[:10]}...")  # First 10 unique values
    if i >= 2:  # Just show first few
        break

print()

# Check global state
if hasattr(env, 'state'):
    state = env.state()
    print(f"Global state shape: {state.shape}")
    print(f"Global state dtype: {state.dtype}")
    print(f"Global state range: {state.min():.3f} to {state.max():.3f}")
    print(f"Global state unique values: {np.unique(state)[:15]}...")
else:
    print("No global state method found")

# Check action space
first_agent = env.agents[0]
action_space = env.action_space(first_agent)
print(f"\nAction space for agent {first_agent}: {action_space}")
print(f"Action space size: {action_space.n}")

env.close()