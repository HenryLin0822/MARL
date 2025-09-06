import random
import numpy as np
from PIL import Image
import imageio
import os
from magent2.environments import combined_arms_v6
from DQN import MultiAgentDQNPolicy

def state_to_rgb(state):
    """Convert environment state to RGB image with proper colors
    
    Color scheme:
    - Gray: Walls/obstacles
    - Red/Dark Red: Red team melee units (high HP, slow, close range)
    - Orange/Yellow: Red team ranged units (low HP, fast, long range)  
    - Blue/Dark Blue: Blue team melee units (high HP, slow, close range)
    - Cyan/Sky Blue: Blue team ranged units (low HP, fast, long range)
    - Brightness indicates health level
    """
    height, width, channels = state.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Background (empty space) is black (0,0,0)
    
    # Walls/obstacles (channel 0) - Dark gray
    walls = state[:, :, 0]
    rgb_image[walls > 0] = [64, 64, 64]  # Dark gray
    
    # Red team agents (channels 1-4)
    # Melee units: Solid red colors (higher HP, slower, close combat)
    # Ranged units: Orange/yellow colors (lower HP, faster, long range)
    red_colors = [
        [255, 0, 0],      # Red melee unit 1 - bright red
        [200, 0, 0],      # Red melee unit 2 - dark red  
        [255, 165, 0],    # Red ranged unit 1 - orange
        [255, 255, 0]     # Red ranged unit 2 - yellow
    ]
    
    for i in range(4):
        if i + 1 < channels:
            agents = state[:, :, i + 1]
            mask = agents > 0
            if np.any(mask):
                # Use intensity to show health (brighter = healthier)
                intensity = agents[mask]
                for j in range(3):  # RGB channels
                    rgb_image[mask, j] = (red_colors[i][j] * intensity).astype(np.uint8)
    
    # Blue team agents (channels 5-8)
    # Melee units: Solid blue colors (higher HP, slower, close combat)  
    # Ranged units: Cyan/light blue colors (lower HP, faster, long range)
    blue_colors = [
        [0, 0, 255],      # Blue melee unit 1 - bright blue
        [0, 0, 200],      # Blue melee unit 2 - dark blue
        [0, 255, 255],    # Blue ranged unit 1 - cyan
        [135, 206, 235]   # Blue ranged unit 2 - sky blue
    ]
    
    for i in range(4):
        if i + 5 < channels:
            agents = state[:, :, i + 5]
            mask = agents > 0
            if np.any(mask):
                # Use intensity to show health (brighter = healthier)
                intensity = agents[mask]
                for j in range(3):  # RGB channels
                    rgb_image[mask, j] = (blue_colors[i][j] * intensity).astype(np.uint8)
    
    return rgb_image

class RandomPolicy:
    """Simple random policy for multi-agent RL environment"""
    
    def __init__(self, env):
        self.env = env
    
    def get_action(self, observation, agent_id):
        """
        Get action for a specific agent based on its observation
        
        Args:
            observation: Agent's local observation
            agent_id: ID of the agent
            
        Returns:
            int: Action to take (0 to action_space.n - 1)
        """
        action_space = self.env.action_space(agent_id)
        return random.randint(0, action_space.n - 1)
    
    def get_actions(self, observations):
        """
        Get actions for all agents
        
        Args:
            observations: Dict of {agent_id: observation}
            
        Returns:
            dict: {agent_id: action}
        """
        actions = {}
        for agent_id, obs in observations.items():
            actions[agent_id] = self.get_action(obs, agent_id)
        return actions

def run_inference(num_episodes=1, max_steps=1000, map_size=16, render=False, save_gif=False, gif_path="render/inference_simulation.gif", scale_factor=20, use_dqn=True, training=True, load_models=False):
    """
    Run inference with the policy
    
    Args:
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        map_size: Size of the map
        render: Whether to print progress
        save_gif: Whether to save simulation as GIF
        gif_path: Path to save the GIF file
        scale_factor: Scale factor for GIF images (default 20)
        use_dqn: Whether to use DQN policy (True) or random policy (False)
        training: Whether to train the DQN agents during inference
        load_models: Whether to load existing DQN models
    """
    # Create environment
    env = combined_arms_v6.parallel_env(
        map_size=map_size, 
        minimap_mode=False, 
        step_reward=-0.005,
        dead_penalty=-0.1, 
        attack_penalty=-0.1,
        attack_opponent_reward=0.2, 
        max_cycles=max_steps, 
        extra_features=False
    )
    
    # Initialize policy
    if use_dqn:
        policy = MultiAgentDQNPolicy(env)
        if load_models:
            policy.load_models()
            if render:
                print("Loaded existing DQN models")
        if render:
            print("Using DQN policy with multi-agent learning")
    else:
        policy = RandomPolicy(env)
        if render:
            print("Using random policy")
    
    episode_stats = []
    
    # Initialize GIF recording if requested
    all_frames = []
    if save_gif:
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    
    for episode in range(num_episodes):
        # Reset environment
        observations = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_frames = []
        prev_observations = None
        prev_actions = None
        
        if render:
            print(f"Starting episode {episode + 1}")
            print(f"Initial agents: {len(env.agents)}")
            if use_dqn and training:
                stats = policy.get_training_stats()
                avg_epsilon = np.mean([s['epsilon'] for s in stats.values()])
                print(f"Average epsilon: {avg_epsilon:.3f}")
        
        for step in range(max_steps):
            # Capture frame for GIF if requested
            if save_gif:
                state = env.state()
                rgb_frame = state_to_rgb(state)
                
                # Scale up the image for better visibility
                img = Image.fromarray(rgb_frame)
                img_scaled = img.resize((map_size * scale_factor, map_size * scale_factor), Image.NEAREST)
                episode_frames.append(np.array(img_scaled))
            
            # Get actions from policy
            if use_dqn:
                actions = policy.get_actions(observations, training=training)
            else:
                actions = policy.get_actions(observations)
            
            # Step environment
            next_observations, rewards, terminations, truncations, _ = env.step(actions)
            
            # Train DQN agents if using DQN and training is enabled
            if use_dqn and training and prev_observations is not None:
                policy.step(prev_observations, prev_actions, rewards, observations, terminations)
            
            # Store current state for next iteration
            prev_observations = observations.copy() if observations else None
            prev_actions = actions.copy() if actions else None
            observations = next_observations
            
            # Update episode stats
            episode_reward += sum(rewards.values())
            episode_length = step + 1
            
            # Print progress
            if render and step % 100 == 0:
                alive_agents = len([a for a in env.agents if not terminations.get(a, False)])
                step_reward = sum(rewards.values())
                print(f"  Step {step}: {alive_agents} agents alive, step reward: {step_reward:.3f}")
                if use_dqn and training:
                    stats = policy.get_training_stats()
                    avg_buffer = np.mean([s['buffer_size'] for s in stats.values()])
                    print(f"    Avg buffer size: {avg_buffer:.0f}")
            
            # Check if episode is done
            if all(terminations.values()) or all(truncations.values()):
                if render:
                    print(f"Episode {episode + 1} finished at step {step}")
                break
        
        # Store episode frames for GIF
        if save_gif:
            all_frames.extend(episode_frames)
        
        # Store episode statistics
        final_alive = len([a for a in env.agents if not terminations.get(a, False)])
        episode_stats.append({
            'episode': episode + 1,
            'length': episode_length,
            'total_reward': episode_reward,
            'final_alive_agents': final_alive
        })
        
        if render:
            print(f"Episode {episode + 1} stats:")
            print(f"  Length: {episode_length} steps")
            print(f"  Total reward: {episode_reward:.3f}")
            print(f"  Final alive agents: {final_alive}")
            print()
    
    # Save DQN models if training was enabled
    if use_dqn and training:
        policy.save_models()
        if render:
            print("DQN models saved to 'models/' directory")
    
    env.close()
    
    # Save GIF if requested
    if save_gif and all_frames:
        print(f"Saving GIF with {len(all_frames)} frames to {gif_path}...")
        imageio.mimsave(gif_path, all_frames, fps=10, loop=0)
        print(f"GIF saved successfully!")
    
    return episode_stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MARL inference with DQN or random policy")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--map-size", type=int, default=16, help="Size of the map")
    parser.add_argument("--render", action="store_true", help="Print progress during simulation")
    parser.add_argument("--save-gif", action="store_true", help="Save simulation as GIF")
    parser.add_argument("--gif-path", type=str, default="render/inference_simulation.gif", help="Path to save GIF")
    parser.add_argument("--scale-factor", type=int, default=20, help="Scale factor for GIF images")
    parser.add_argument("--use-dqn", action="store_true", help="Use DQN policy instead of random policy")
    parser.add_argument("--no-training", action="store_true", help="Disable training for DQN (evaluation mode)")
    parser.add_argument("--load-models", action="store_true", help="Load existing DQN models from 'models/' directory")
    
    args = parser.parse_args()
    
    # Determine policy type and training mode
    use_dqn = args.use_dqn
    training = not args.no_training
    
    policy_type = "DQN" if use_dqn else "random"
    mode = "training" if (use_dqn and training) else "evaluation"
    
    print(f"Running MARL inference with {policy_type} policy in {mode} mode...")
    if args.save_gif:
        print(f"Will save GIF to: {args.gif_path}")
    if use_dqn and args.load_models:
        print("Will attempt to load existing DQN models")
    
    # Run inference
    stats = run_inference(
        num_episodes=args.episodes,
        max_steps=args.steps, 
        map_size=args.map_size,
        render=args.render,
        save_gif=args.save_gif,
        gif_path=args.gif_path,
        scale_factor=args.scale_factor,
        use_dqn=use_dqn,
        training=training,
        load_models=args.load_models
    )
    
    # Print summary
    print("=" * 50)
    print("INFERENCE SUMMARY")
    print("=" * 50)
    
    for stat in stats:
        print(f"Episode {stat['episode']}: "
              f"Length={stat['length']}, "
              f"Reward={stat['total_reward']:.2f}, "
              f"Survivors={stat['final_alive_agents']}")
    
    # Calculate averages
    avg_length = np.mean([s['length'] for s in stats])
    avg_reward = np.mean([s['total_reward'] for s in stats])
    avg_survivors = np.mean([s['final_alive_agents'] for s in stats])
    
    print(f"\nAverages:")
    print(f"  Length: {avg_length:.1f} steps")
    print(f"  Reward: {avg_reward:.2f}")
    print(f"  Survivors: {avg_survivors:.1f}")