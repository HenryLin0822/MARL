import os
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import argparse
from magent2.environments import combined_arms_v6
from DQN import MultiAgentDQNPolicy
from tqdm import tqdm
class MADQNTrainer:
    def __init__(self, config):
        self.config = config
        
        # Create environment
        self.env = combined_arms_v6.parallel_env(
            map_size=config['map_size'],
            minimap_mode=False,
            step_reward=config['step_reward'],
            dead_penalty=config['dead_penalty'],
            attack_penalty=config['attack_penalty'],
            attack_opponent_reward=config['attack_opponent_reward'],
            max_cycles=config['max_cycles'],
            extra_features=False
        )
        
        # Initialize multi-agent DQN policy
        self.policy = MultiAgentDQNPolicy(
            env=self.env,
            lr=config['learning_rate'],
            buffer_size=config['buffer_size'],
            batch_size=config['batch_size']
        )
        
        # Training statistics
        self.episode_rewards = defaultdict(list)
        self.episode_lengths = []
        self.team_survival_rates = {'red': [], 'blue': []}
        self.win_rates = {'red': 0, 'blue': 0, 'draw': 0}
        
        # Moving averages for tracking progress
        self.reward_window = deque(maxlen=100)
        self.length_window = deque(maxlen=100)
        
        # Create directories for saving
        os.makedirs(config['model_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)
        
    def train(self):
        print("Starting Multi-Agent DQN Training...")
        print(f"Training for {self.config['num_episodes']} episodes")
        print(f"Environment: {self.config['map_size']}x{self.config['map_size']} battlefield")
        print("-" * 60)
        
        best_avg_reward = float('-inf')
        
        for episode in tqdm(range(self.config['num_episodes'])):
            episode_start_time = time.time()
            
            # Reset environment
            observations = self.env.reset()
            episode_rewards = defaultdict(float)
            episode_length = 0
            
            # Track initial agent counts
            initial_agents = set(observations.keys())
            red_agents = {a for a in initial_agents if 'red' in str(a).lower()}
            blue_agents = {a for a in initial_agents if 'blue' in str(a).lower()}
            initial_red_count = len(red_agents)
            initial_blue_count = len(blue_agents)
            
            prev_observations = observations.copy()
            
            # Episode loop
            while self.env.agents:
                # Get actions from policy
                actions = self.policy.get_actions(observations, training=True)
                
                # Step environment
                next_observations, rewards, terminations, truncations, infos = self.env.step(actions)
                
                # Store experiences
                self.policy.step(prev_observations, actions, rewards, next_observations, terminations)
                
                # Update episode statistics
                for agent_id, reward in rewards.items():
                    episode_rewards[agent_id] += reward
                
                # Update for next iteration
                prev_observations = observations.copy()
                observations = next_observations
                episode_length += 1
                
                # Check if episode is done
                if not observations or episode_length >= self.config['max_cycles']:
                    break
            
            # Calculate episode statistics
            total_episode_reward = sum(episode_rewards.values())
            
            # Calculate survival rates
            final_agents = set(observations.keys()) if observations else set()
            final_red_count = len({a for a in final_agents if 'red' in str(a).lower()})
            final_blue_count = len({a for a in final_agents if 'blue' in str(a).lower()})
            
            red_survival_rate = final_red_count / initial_red_count if initial_red_count > 0 else 0
            blue_survival_rate = final_blue_count / initial_blue_count if initial_blue_count > 0 else 0
            
            # Determine winner
            if final_red_count > final_blue_count:
                winner = 'red'
                self.win_rates['red'] += 1
            elif final_blue_count > final_red_count:
                winner = 'blue'
                self.win_rates['blue'] += 1
            else:
                winner = 'draw'
                self.win_rates['draw'] += 1
            
            # Store episode statistics
            for agent_id, reward in episode_rewards.items():
                self.episode_rewards[agent_id].append(reward)
            
            self.episode_lengths.append(episode_length)
            self.team_survival_rates['red'].append(red_survival_rate)
            self.team_survival_rates['blue'].append(blue_survival_rate)
            
            self.reward_window.append(total_episode_reward)
            self.length_window.append(episode_length)
            
            episode_time = time.time() - episode_start_time
            
            # Print progress
            if (episode + 1) % self.config['print_freq'] == 0:
                avg_reward = np.mean(self.reward_window)
                avg_length = np.mean(self.length_window)
                
                # Get training statistics
                training_stats = self.policy.get_training_stats()
                avg_epsilon = np.mean([stats['epsilon'] for stats in training_stats.values()])
                
                print(f"Episode {episode + 1:4d} | "
                      f"Reward: {total_episode_reward:8.2f} | "
                      f"Length: {episode_length:3d} | "
                      f"Winner: {winner:4s} | "
                      f"Avg Reward: {avg_reward:8.2f} | "
                      f"Avg Length: {avg_length:6.1f} | "
                      f"Epsilon: {avg_epsilon:.3f} | "
                      f"Time: {episode_time:.2f}s")
            
            # Save best model
            if len(self.reward_window) >= 50:
                current_avg_reward = np.mean(self.reward_window)
                if current_avg_reward > best_avg_reward:
                    best_avg_reward = current_avg_reward
                    self.policy.save_models(os.path.join(self.config['model_dir'], 'best'))
                    print(f"New best average reward: {best_avg_reward:.2f} - Model saved!")
            
            # Save checkpoint
            if (episode + 1) % self.config['save_freq'] == 0:
                checkpoint_dir = os.path.join(self.config['model_dir'], f'checkpoint_{episode + 1}')
                self.policy.save_models(checkpoint_dir)
                self.save_training_stats(os.path.join(self.config['log_dir'], f'stats_{episode + 1}.npz'))
                print(f"Checkpoint saved at episode {episode + 1}")
        
        # Final save
        self.policy.save_models(os.path.join(self.config['model_dir'], 'final'))
        self.save_training_stats(os.path.join(self.config['log_dir'], 'final_stats.npz'))
        
        # Print final statistics
        self.print_final_statistics()
        
        # Generate plots
        self.plot_training_curves()
        
        print("Training completed!")
    
    def save_training_stats(self, filepath):
        np.savez(filepath,
                episode_lengths=self.episode_lengths,
                red_survival_rates=self.team_survival_rates['red'],
                blue_survival_rates=self.team_survival_rates['blue'],
                win_rates=self.win_rates,
                episode_rewards=dict(self.episode_rewards))
    
    def print_final_statistics(self):
        total_episodes = len(self.episode_lengths)
        
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Total Episodes: {total_episodes}")
        print(f"Average Episode Length: {np.mean(self.episode_lengths):.1f} steps")
        print(f"Average Total Reward: {np.mean([sum(rewards) for rewards in zip(*self.episode_rewards.values())]):.2f}")
        
        print(f"\nWin Rates:")
        print(f"  Red Team:  {self.win_rates['red']:3d} ({self.win_rates['red']/total_episodes*100:.1f}%)")
        print(f"  Blue Team: {self.win_rates['blue']:3d} ({self.win_rates['blue']/total_episodes*100:.1f}%)")
        print(f"  Draws:     {self.win_rates['draw']:3d} ({self.win_rates['draw']/total_episodes*100:.1f}%)")
        
        print(f"\nAverage Survival Rates:")
        print(f"  Red Team:  {np.mean(self.team_survival_rates['red']):.2f}")
        print(f"  Blue Team: {np.mean(self.team_survival_rates['blue']):.2f}")
        
        # Agent-specific statistics
        print(f"\nAgent Performance:")
        for agent_type in ['melee', 'ranged']:
            agent_rewards = []
            for agent_id, rewards in self.episode_rewards.items():
                if agent_type in str(agent_id).lower() or \
                   (agent_type == 'melee' and any(x in str(agent_id).lower() for x in ['_0_', '_1_'])) or \
                   (agent_type == 'ranged' and any(x in str(agent_id).lower() for x in ['_2_', '_3_'])):
                    agent_rewards.extend(rewards)
            
            if agent_rewards:
                print(f"  {agent_type.capitalize()} units: {np.mean(agent_rewards):.3f} avg reward")
    
    def plot_training_curves(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode lengths
        axes[0, 0].plot(self.episode_lengths)
        axes[0, 0].set_title('Episode Lengths')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Length (steps)')
        
        # Total rewards per episode
        total_rewards = [sum(rewards[i] for rewards in self.episode_rewards.values() if i < len(rewards)) 
                        for i in range(len(self.episode_lengths))]
        axes[0, 1].plot(total_rewards)
        axes[0, 1].set_title('Total Episode Rewards')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Total Reward')
        
        # Survival rates
        axes[1, 0].plot(self.team_survival_rates['red'], label='Red Team', color='red', alpha=0.7)
        axes[1, 0].plot(self.team_survival_rates['blue'], label='Blue Team', color='blue', alpha=0.7)
        axes[1, 0].set_title('Team Survival Rates')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Survival Rate')
        axes[1, 0].legend()
        
        # Moving average of rewards
        window_size = min(50, len(total_rewards))
        if window_size > 1:
            moving_avg = np.convolve(total_rewards, np.ones(window_size)/window_size, mode='valid')
            axes[1, 1].plot(range(window_size-1, len(total_rewards)), moving_avg)
            axes[1, 1].set_title(f'Moving Average Rewards (window={window_size})')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Average Reward')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['log_dir'], 'training_curves.png'))
        plt.show()

def get_default_config():
    return {
        # Environment parameters
        'map_size': 16,
        'step_reward': -0.005,
        'dead_penalty': -0.1,
        'attack_penalty': -0.1,
        'attack_opponent_reward': 0.2,
        'max_cycles': 1000,
        
        # Training parameters
        'num_episodes': 2000,
        'learning_rate': 1e-3,
        'buffer_size': 50000,
        'batch_size': 64,
        
        # Logging and saving
        'print_freq': 50,
        'save_freq': 500,
        'model_dir': 'models',
        'log_dir': 'logs'
    }

def main():
    parser = argparse.ArgumentParser(description='Train Multi-Agent DQN on Combat Environment')
    parser.add_argument('--episodes', type=int, default=20, help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--buffer-size', type=int, default=50000, help='Replay buffer size')
    parser.add_argument('--map-size', type=int, default=16, help='Map size')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--print-freq', type=int, default=50, help='Print frequency')
    parser.add_argument('--save-freq', type=int, default=500, help='Save frequency')
    
    args = parser.parse_args()
    
    # Create configuration
    config = get_default_config()
    config.update({
        'num_episodes': args.episodes,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'buffer_size': args.buffer_size,
        'map_size': args.map_size,
        'model_dir': args.model_dir,
        'log_dir': args.log_dir,
        'print_freq': args.print_freq,
        'save_freq': args.save_freq
    })
    
    # Initialize trainer and start training
    trainer = MADQNTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()