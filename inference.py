import random
import numpy as np
from PIL import Image
import imageio
import os
from magent2.environments import combined_arms_v6

def policy_1(team_observations):
    """
    Policy for red team
    
    Args:
        team_observations: Dict of {agent_id: observation} for red team agents
    
    Returns:
        dict: {agent_id: action} for red team agents
    """
    team_actions = {}
    for agent_id, obs in team_observations.items():
        # Random policy - select action from 0 to 8 (9 total actions)
        team_actions[agent_id] = random.randint(0, 8)
    return team_actions

def policy_2(team_observations):
    """
    Policy for blue team
    
    Args:
        team_observations: Dict of {agent_id: observation} for blue team agents
    
    Returns:
        dict: {agent_id: action} for blue team agents
    """
    team_actions = {}
    for agent_id, obs in team_observations.items():
        # Random policy - select action from 0 to 8 (9 total actions)
        team_actions[agent_id] = random.randint(0, 8)
    return team_actions

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

def separate_teams(observations):
    """
    Separate observations by team based on agent ID prefix
    
    Args:
        observations: Dict of {agent_id: observation} for all agents
    
    Returns:
        tuple: (red_observations, blue_observations)
    """
    red_observations = {}
    blue_observations = {}
    
    for agent_id, obs in observations.items():
        if agent_id.startswith('red'):
            red_observations[agent_id] = obs
        elif agent_id.startswith('blue'):
            blue_observations[agent_id] = obs
    
    return red_observations, blue_observations

def run_inference(num_episodes=1, max_steps=1000, map_size=16, render=False, save_gif=False, gif_path="render/inference_simulation.gif", scale_factor=20):
    """
    Run inference with team-based policies
    
    Args:
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        map_size: Size of the map
        render: Whether to print progress
        save_gif: Whether to save simulation as GIF
        gif_path: Path to save the GIF file
        scale_factor: Scale factor for GIF images (default 20)
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
        
        if render:
            print(f"Starting episode {episode + 1}")
            print(f"Initial agents: {len(env.agents)}")
        
        for step in range(max_steps):
            # Capture frame for GIF if requested
            if save_gif:
                state = env.state()
                rgb_frame = state_to_rgb(state)
                
                # Scale up the image for better visibility
                img = Image.fromarray(rgb_frame)
                img_scaled = img.resize((map_size * scale_factor, map_size * scale_factor), Image.NEAREST)
                episode_frames.append(np.array(img_scaled))
            
            # Separate observations by team
            red_observations, blue_observations = separate_teams(observations)
            
            # Get actions from team policies
            red_actions = policy_1(red_observations)
            blue_actions = policy_2(blue_observations)
            
            # Combine actions from both teams
            actions = {**red_actions, **blue_actions}
            
            # Step environment
            next_observations, rewards, terminations, truncations, _ = env.step(actions)
            
            # Update observations
            observations = next_observations
            
            # Update episode stats
            episode_reward += sum(rewards.values())
            episode_length = step + 1
            
            # Print progress
            if render and step % 100 == 0:
                alive_agents = len([a for a in env.agents if not terminations.get(a, False)])
                step_reward = sum(rewards.values())
                red_alive = len([a for a in env.agents if a.startswith('red') and not terminations.get(a, False)])
                blue_alive = len([a for a in env.agents if a.startswith('blue') and not terminations.get(a, False)])
                print(f"  Step {step}: {alive_agents} agents alive ({red_alive} red, {blue_alive} blue), step reward: {step_reward:.3f}")
            
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
        final_red_alive = len([a for a in env.agents if a.startswith('red') and not terminations.get(a, False)])
        final_blue_alive = len([a for a in env.agents if a.startswith('blue') and not terminations.get(a, False)])
        
        episode_stats.append({
            'episode': episode + 1,
            'length': episode_length,
            'total_reward': episode_reward,
            'final_alive_agents': final_alive,
            'final_red_alive': final_red_alive,
            'final_blue_alive': final_blue_alive
        })
        
        if render:
            print(f"Episode {episode + 1} stats:")
            print(f"  Length: {episode_length} steps")
            print(f"  Total reward: {episode_reward:.3f}")
            print(f"  Final alive agents: {final_alive} ({final_red_alive} red, {final_blue_alive} blue)")
            print()
    
    env.close()
    
    # Save GIF if requested
    if save_gif and all_frames:
        print(f"Saving GIF with {len(all_frames)} frames to {gif_path}...")
        imageio.mimsave(gif_path, all_frames, fps=10, loop=0)
        print(f"GIF saved successfully!")
    
    return episode_stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MARL inference with team-based policies")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--map-size", type=int, default=16, help="Size of the map")
    parser.add_argument("--render", action="store_true", help="Print progress during simulation")
    parser.add_argument("--save-gif", action="store_true", help="Save simulation as GIF")
    parser.add_argument("--gif-path", type=str, default="render/inference_simulation.gif", help="Path to save GIF")
    parser.add_argument("--scale-factor", type=int, default=20, help="Scale factor for GIF images")
    
    args = parser.parse_args()
    
    print(f"Running MARL inference with team-based policies...")
    if args.save_gif:
        print(f"Will save GIF to: {args.gif_path}")
    
    # Run inference
    stats = run_inference(
        num_episodes=args.episodes,
        max_steps=args.steps, 
        map_size=args.map_size,
        render=args.render,
        save_gif=args.save_gif,
        gif_path=args.gif_path,
        scale_factor=args.scale_factor
    )
    
    # Print summary
    print("=" * 50)
    print("INFERENCE SUMMARY")
    print("=" * 50)
    
    for stat in stats:
        print(f"Episode {stat['episode']}: "
              f"Length={stat['length']}, "
              f"Reward={stat['total_reward']:.2f}, "
              f"Survivors={stat['final_alive_agents']} "
              f"({stat['final_red_alive']} red, {stat['final_blue_alive']} blue)")
    
    # Calculate averages
    avg_length = np.mean([s['length'] for s in stats])
    avg_reward = np.mean([s['total_reward'] for s in stats])
    avg_survivors = np.mean([s['final_alive_agents'] for s in stats])
    avg_red_survivors = np.mean([s['final_red_alive'] for s in stats])
    avg_blue_survivors = np.mean([s['final_blue_alive'] for s in stats])
    
    print(f"\nAverages:")
    print(f"  Length: {avg_length:.1f} steps")
    print(f"  Reward: {avg_reward:.2f}")
    print(f"  Survivors: {avg_survivors:.1f} ({avg_red_survivors:.1f} red, {avg_blue_survivors:.1f} blue)")