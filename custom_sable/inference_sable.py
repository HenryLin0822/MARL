"""
Inference script using trained Sable policies on MAgentX Combined Arms environment.

This script replaces the RandomPolicy in inference.py with trained Sable policies,
demonstrating the learned team coordination behavior.
"""

import argparse
import os
import torch
import numpy as np
from PIL import Image
import imageio
from typing import Dict, Any, Optional

from magent2.environments import combined_arms_v6
from magent_sable_policy_fully_aligned import MultiTeamSablePolicyFullyAligned as MultiTeamSablePolicy


def state_to_rgb(state):
    """Convert environment state to RGB image - same as original inference.py."""
    height, width, channels = state.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Walls/obstacles (channel 0) - Dark gray
    walls = state[:, :, 0]
    rgb_image[walls > 0] = [64, 64, 64]
    
    # Red team agents (channels 1-4)ㄇ
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
                intensity = agents[mask]
                for j in range(3):
                    rgb_image[mask, j] = (red_colors[i][j] * intensity).astype(np.uint8)
    
    # Blue team agents (channels 5-8)
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
                intensity = agents[mask]
                for j in range(3):
                    rgb_image[mask, j] = (blue_colors[i][j] * intensity).astype(np.uint8)
    
    return rgb_image


class SablePolicy:
    """Sable policy wrapper that mimics the RandomPolicy interface."""
    
    def __init__(self, env, checkpoint_paths: Optional[Dict[str, str]] = None, device: str = "cpu"):
        """
        Initialize Sable policy.
        
        Args:
            env: MAgentX environment
            checkpoint_paths: Dict with 'red' and 'blue' checkpoint paths
            device: Device to run on
        """
        self.env = env
        self.device = device
        
        # Get environment specs
        sample_obs = env.reset()
        sample_agent = list(sample_obs.keys())[0]
        obs_shape = sample_obs[sample_agent].shape
        action_dim = env.action_space(sample_agent).n
        
        # Initialize multi-team policy
        self.multi_team_policy = MultiTeamSablePolicy(
            obs_shape=obs_shape,
            action_dim=action_dim,
            n_agents_per_team=6,  # MAgentX has 6 agents per team
            embed_dim=128,
            n_head=8,
            n_block=3,
            device=device
        )
        
        # Load checkpoints if provided
        if checkpoint_paths:
            if 'red' in checkpoint_paths and os.path.exists(checkpoint_paths['red']):
                self.multi_team_policy.red_policy.load(checkpoint_paths['red'])
                print(f"Loaded red policy from {checkpoint_paths['red']}")
            else:
                print("Warning: Using untrained red policy")
                
            if 'blue' in checkpoint_paths and os.path.exists(checkpoint_paths['blue']):
                self.multi_team_policy.blue_policy.load(checkpoint_paths['blue'])
                print(f"Loaded blue policy from {checkpoint_paths['blue']}")
            else:
                print("Warning: Using untrained blue policy")
        else:
            print("Warning: Using untrained policies")
        
        # Set to evaluation mode
        self.multi_team_policy.red_policy.eval()
        self.multi_team_policy.blue_policy.eval()
    
    def get_action(self, observation, agent_id):
        """Get action for a specific agent - compatibility with RandomPolicy interface."""
        observations = {agent_id: observation}
        actions = self.get_actions(observations)
        return actions.get(agent_id, 0)  # Default action if not found
    
    def get_actions(self, observations):
        """Get actions for all agents using Sable's team coordination."""
        return self.multi_team_policy.get_actions(observations, deterministic=True)
    
    def reset_hidden_states(self):
        """Reset hidden states for new episodes."""
        self.multi_team_policy.reset_hidden_states()


def run_sable_inference(
    num_episodes=1, 
    max_steps=1000, 
    map_size=16, 
    render=False, 
    save_gif=False, 
    gif_path="render/sable_simulation.gif", 
    scale_factor=20,
    checkpoint_paths: Optional[Dict[str, str]] = None,
    device="cpu"
):
    """
    Run inference with Sable policies.
    
    Args:
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        map_size: Size of the map
        render: Whether to print progress
        save_gif: Whether to save simulation as GIF
        gif_path: Path to save the GIF file
        scale_factor: Scale factor for GIF images
        checkpoint_paths: Dict with 'red' and 'blue' checkpoint paths
        device: Device to run on
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
    
    # Initialize Sable policy
    policy = SablePolicy(env, checkpoint_paths, device)
    
    episode_stats = []
    
    # Initialize GIF recording if requested
    all_frames = []
    if save_gif:
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    
    for episode in range(num_episodes):
        # Reset environment and policy
        observations = env.reset()
        policy.reset_hidden_states()
        
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
                
                # Scale up the image
                img = Image.fromarray(rgb_frame)
                img_scaled = img.resize((map_size * scale_factor, map_size * scale_factor), Image.NEAREST)
                episode_frames.append(np.array(img_scaled))
            
            # Get actions from Sable policy
            actions = policy.get_actions(observations)
            
            # Step environment
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Update episode stats
            episode_reward += sum(rewards.values())
            episode_length = step + 1
            
            # Print progress
            if render and step % 100 == 0:
                alive_agents = len([a for a in env.agents if not terminations.get(a, False)])
                step_reward = sum(rewards.values())
                red_alive = len([a for a in env.agents if not terminations.get(a, False) and 'red' in a.lower()])
                blue_alive = len([a for a in env.agents if not terminations.get(a, False) and 'blue' in a.lower()])
                print(f"  Step {step}: Red={red_alive}, Blue={blue_alive}, step reward: {step_reward:.3f}")
            
            # Check if episode is done
            if all(terminations.values()) or all(truncations.values()):
                if render:
                    print(f"Episode {episode + 1} finished at step {step}")
                break
        
        # Store episode frames for GIF
        if save_gif:
            all_frames.extend(episode_frames)
        
        # Calculate team survivors
        red_survivors = len([a for a in env.agents if not terminations.get(a, False) and 'red' in a.lower()])
        blue_survivors = len([a for a in env.agents if not terminations.get(a, False) and 'blue' in a.lower()])
        
        # Store episode statistics
        episode_stats.append({
            'episode': episode + 1,
            'length': episode_length,
            'total_reward': episode_reward,
            'red_survivors': red_survivors,
            'blue_survivors': blue_survivors
        })
        
        if render:
            print(f"Episode {episode + 1} stats:")
            print(f"  Length: {episode_length} steps")
            print(f"  Total reward: {episode_reward:.3f}")
            print(f"  Red survivors: {red_survivors}")
            print(f"  Blue survivors: {blue_survivors}")
            print()
    
    env.close()
    
    # Save GIF if requested
    if save_gif and all_frames:
        print(f"Saving GIF with {len(all_frames)} frames to {gif_path}...")
        imageio.mimsave(gif_path, all_frames, fps=10, loop=0)
        print(f"GIF saved successfully!")
    
    return episode_stats


def main():
    parser = argparse.ArgumentParser(description="Run Sable inference on MAgentX Combined Arms")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--map-size", type=int, default=16, help="Size of the map")
    parser.add_argument("--render", action="store_true", help="Print progress during simulation")
    parser.add_argument("--save-gif", action="store_true", help="Save simulation as GIF")
    parser.add_argument("--gif-path", type=str, default="render/sable_simulation.gif", help="Path to save GIF")
    parser.add_argument("--scale-factor", type=int, default=20, help="Scale factor for GIF images")
    parser.add_argument("--red-checkpoint", type=str, default=None, help="Path to red team checkpoint")
    parser.add_argument("--blue-checkpoint", type=str, default=None, help="Path to blue team checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Prepare checkpoint paths
    checkpoint_paths = {}
    if args.red_checkpoint:
        checkpoint_paths['red'] = args.red_checkpoint
    if args.blue_checkpoint:
        checkpoint_paths['blue'] = args.blue_checkpoint
    
    print("Running Sable inference on MAgentX Combined Arms...")
    if checkpoint_paths:
        print(f"Using checkpoints: {checkpoint_paths}")
    else:
        print("Using untrained policies (random initialization)")
    
    if args.save_gif:
        print(f"Will save GIF to: {args.gif_path}")
    
    # Run inference
    stats = run_sable_inference(
        num_episodes=args.episodes,
        max_steps=args.steps,
        map_size=args.map_size,
        render=args.render,
        save_gif=args.save_gif,
        gif_path=args.gif_path,
        scale_factor=args.scale_factor,
        checkpoint_paths=checkpoint_paths if checkpoint_paths else None,
        device=args.device
    )
    
    # Print summary
    print("=" * 60)
    print("SABLE INFERENCE SUMMARY")
    print("=" * 60)
    
    for stat in stats:
        print(f"Episode {stat['episode']}: "
              f"Length={stat['length']}, "
              f"Reward={stat['total_reward']:.2f}, "
              f"Red={stat['red_survivors']}, "
              f"Blue={stat['blue_survivors']}")
    
    # Calculate averages
    avg_length = np.mean([s['length'] for s in stats])
    avg_reward = np.mean([s['total_reward'] for s in stats])
    avg_red = np.mean([s['red_survivors'] for s in stats])
    avg_blue = np.mean([s['blue_survivors'] for s in stats])
    
    print(f"\nAverages:")
    print(f"  Length: {avg_length:.1f} steps")
    print(f"  Reward: {avg_reward:.2f}")
    print(f"  Red survivors: {avg_red:.1f}")
    print(f"  Blue survivors: {avg_blue:.1f}")
    
    # Compare with random baseline if available
    print(f"\n--- Sable Policy Performance ---")
    print(f"Team coordination: {'Enabled (autoregressive)' if checkpoint_paths else 'Random initialization'}")
    print(f"Combat effectiveness: {(avg_red + avg_blue):.1f} average survivors")


if __name__ == "__main__":
    main()