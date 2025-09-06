import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple

class AutoRegressiveDQNNetwork(nn.Module):
    """
    Central policy network that takes observations, strategy embeddings, and unit types
    to output actions and new strategy embeddings in an autoregressive manner.
    """
    def __init__(self, input_shape, action_size, strategy_dim=64, unit_types=4):
        super(AutoRegressiveDQNNetwork, self).__init__()
        
        self.input_shape = input_shape  # (13, 13, channels)
        self.action_size = action_size
        self.strategy_dim = strategy_dim
        self.unit_types = unit_types
        
        # Convolutional layers for spatial feature extraction (observation only)
        self.conv1 = nn.Conv2d(input_shape[2], 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Calculate conv output size for spatial features
        conv_out_size = 13 * 13 * 128
        
        # Fully connected layers that combine spatial features, strategy, and unit type
        # Input: conv_features + strategy_embedding + unit_type_onehot
        fc_input_size = conv_out_size + strategy_dim + unit_types
        
        self.fc1 = nn.Linear(fc_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        
        # Output heads
        self.action_head = nn.Linear(128, action_size)  # Action logits
        self.strategy_head = nn.Linear(128, strategy_dim)  # New strategy embedding
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, obs, strategy_embedding, unit_type_onehot):
        """
        Forward pass of the autoregressive network.
        
        Args:
            obs: (batch_size, height, width, channels) - agent observations
            strategy_embedding: (batch_size, strategy_dim) - current team strategy
            unit_type_onehot: (batch_size, unit_types) - one-hot encoded unit type
            
        Returns:
            action_logits: (batch_size, action_size) - Q-values for actions
            new_strategy: (batch_size, strategy_dim) - updated strategy embedding
        """
        # Process spatial observations through conv layers
        # Convert from (batch, height, width, channels) to (batch, channels, height, width)
        x = obs.permute(0, 3, 1, 2)
        
        # Convolutional feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten spatial features
        spatial_features = x.reshape(x.size(0), -1)
        
        # Combine spatial features, strategy embedding, and unit type
        combined_input = torch.cat([spatial_features, strategy_embedding, unit_type_onehot], dim=1)
        
        # Fully connected processing
        x = F.relu(self.fc1(combined_input))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        # Output heads
        action_logits = self.action_head(x)
        new_strategy = self.strategy_head(x)
        
        return action_logits, new_strategy

# Extended experience tuple for autoregressive model
ARExperience = namedtuple('ARExperience', 
                         ('state', 'prev_strategy', 'unit_type', 'action', 'reward', 
                          'next_state', 'next_strategy', 'done'))

class AutoRegressiveReplayBuffer:
    """Replay buffer for autoregressive experiences"""
    def __init__(self, buffer_size, batch_size, device):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
    
    def add(self, state, prev_strategy, unit_type, action, reward, next_state, next_strategy, done):
        experience = ARExperience(state, prev_strategy, unit_type, action, reward, 
                                next_state, next_strategy, done)
        self.memory.append(experience)
    
    def sample(self):
        experiences = random.sample(self.memory, self.batch_size)
        
        states = torch.from_numpy(np.array([e.state for e in experiences])).float().to(self.device)
        prev_strategies = torch.from_numpy(np.array([e.prev_strategy for e in experiences])).float().to(self.device)
        unit_types = torch.from_numpy(np.array([e.unit_type for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.array([e.action for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences])).float().to(self.device)
        next_strategies = torch.from_numpy(np.array([e.next_strategy for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.array([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)
        
        return states, prev_strategies, unit_types, actions, rewards, next_states, next_strategies, dones
    
    def __len__(self):
        return len(self.memory)

class AutoRegressiveDQNAgent:
    """
    DQN Agent that uses autoregressive policy with strategy embeddings
    """
    def __init__(self, input_shape, action_size, strategy_dim=64, unit_types=4, 
                 lr=1e-3, buffer_size=10000, batch_size=64, gamma=0.99, tau=1e-3,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        
        self.input_shape = input_shape
        self.action_size = action_size
        self.strategy_dim = strategy_dim
        self.unit_types = unit_types
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks
        self.q_network = AutoRegressiveDQNNetwork(input_shape, action_size, strategy_dim, unit_types).to(self.device)
        self.target_network = AutoRegressiveDQNNetwork(input_shape, action_size, strategy_dim, unit_types).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = AutoRegressiveReplayBuffer(buffer_size, batch_size, self.device)
        
        # Training step counter
        self.t_step = 0
        
    def get_unit_type_onehot(self, agent_id):
        """Convert agent ID to one-hot encoded unit type"""
        agent_str = str(agent_id).lower()
        
        # Determine unit type based on agent ID patterns
        # Assuming: melee1 (_0_), melee2 (_1_), ranged1 (_2_), ranged2 (_3_)
        if "_0_" in agent_str or "melee" in agent_str:
            unit_type_idx = 0  # melee1
        elif "_1_" in agent_str:
            unit_type_idx = 1  # melee2  
        elif "_2_" in agent_str or "ranged" in agent_str:
            unit_type_idx = 2  # ranged1
        else:
            unit_type_idx = 3  # ranged2
            
        # Create one-hot encoding
        onehot = np.zeros(self.unit_types)
        onehot[unit_type_idx] = 1.0
        return onehot
    
    def act(self, observation, strategy_embedding, agent_id, training=True):
        """
        Select action using epsilon-greedy policy with autoregressive network
        
        Args:
            observation: Agent's local observation
            strategy_embedding: Current team strategy embedding
            agent_id: Agent identifier for unit type determination
            training: Whether in training mode
            
        Returns:
            action: Selected action
            new_strategy: Updated strategy embedding
        """
        # Convert inputs to tensors
        obs_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        strategy_tensor = torch.from_numpy(strategy_embedding).float().unsqueeze(0).to(self.device)
        unit_type_onehot = self.get_unit_type_onehot(agent_id)
        unit_type_tensor = torch.from_numpy(unit_type_onehot).float().unsqueeze(0).to(self.device)
        
        # Epsilon-greedy action selection
        if training and random.random() <= self.epsilon:
            # Random action
            action = random.choice(np.arange(self.action_size))
            # Still need to get new strategy from network
            self.q_network.eval()
            with torch.no_grad():
                _, new_strategy_tensor = self.q_network(obs_tensor, strategy_tensor, unit_type_tensor)
            self.q_network.train()
            new_strategy = new_strategy_tensor.cpu().data.numpy()[0]
        else:
            # Greedy action
            self.q_network.eval()
            with torch.no_grad():
                action_values, new_strategy_tensor = self.q_network(obs_tensor, strategy_tensor, unit_type_tensor)
            self.q_network.train()
            
            action = np.argmax(action_values.cpu().data.numpy())
            new_strategy = new_strategy_tensor.cpu().data.numpy()[0]
        
        return action, new_strategy
    
    def step(self, state, prev_strategy, agent_id, action, reward, next_state, next_strategy, done):
        """Store experience and train if enough samples available"""
        unit_type_onehot = self.get_unit_type_onehot(agent_id)
        
        # Store experience in replay buffer
        self.memory.add(state, prev_strategy, unit_type_onehot, action, reward, 
                       next_state, next_strategy, done)
        
        # Learn every few steps
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
    
    def learn(self, experiences):
        """Update network parameters using batch of experiences"""
        states, prev_strategies, unit_types, actions, rewards, next_states, next_strategies, dones = experiences
        
        # Get Q targets for next states from target model
        with torch.no_grad():
            next_q_values, _ = self.target_network(next_states, next_strategies, unit_types)
            q_targets_next = next_q_values.detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        q_targets = rewards.unsqueeze(1) + (self.gamma * q_targets_next * (1 - dones.unsqueeze(1)))
        
        # Get expected Q values from local model
        q_expected, _ = self.q_network(states, prev_strategies, unit_types)
        q_expected = q_expected.gather(1, actions.unsqueeze(1))
        
        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.soft_update(self.q_network, self.target_network)
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def soft_update(self, local_model, target_model):
        """Soft update model parameters"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def hard_update(self):
        """Hard update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filepath)
    
    def load_model(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

class AutoRegressiveMultiAgentPolicy:
    """
    Multi-agent policy using autoregressive decision making with team strategy embeddings
    """
    def __init__(self, env, lr=1e-3, buffer_size=10000, batch_size=64, strategy_dim=64):
        self.env = env
        self.lr = lr
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.strategy_dim = strategy_dim
        
        # Initialize single shared agent (central policy)
        self.agent = None
        self.team_strategies = {}  # Store current strategy for each team
        
        # Initialize the agent and team strategies
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the central autoregressive agent"""
        # Get observation shape from environment
        obs = self.env.reset()
        if obs:
            sample_obs = list(obs.values())[0]
            input_shape = sample_obs.shape
            
            # Get action space size
            sample_agent = list(obs.keys())[0]
            action_size = self.env.action_space(sample_agent).n
            
            # Create single autoregressive agent
            self.agent = AutoRegressiveDQNAgent(
                input_shape=input_shape,
                action_size=action_size,
                strategy_dim=self.strategy_dim,
                lr=self.lr,
                buffer_size=self.buffer_size,
                batch_size=self.batch_size
            )
        
        # Initialize team strategies with random embeddings
        self.team_strategies = {
            'red': np.random.randn(self.strategy_dim) * 0.1,
            'blue': np.random.randn(self.strategy_dim) * 0.1
        }
    
    def _get_team(self, agent_id):
        """Determine which team an agent belongs to"""
        return 'red' if 'red' in str(agent_id).lower() else 'blue'
    
    def _get_team_agents(self, observations, team):
        """Get sorted list of agents for a specific team"""
        team_agents = [agent_id for agent_id in observations.keys() 
                      if self._get_team(agent_id) == team]
        return sorted(team_agents)  # Sort by agent ID for consistent ordering
    
    def get_actions(self, observations, training=True):
        """
        Get actions for all agents using autoregressive policy
        
        Args:
            observations: Dict of {agent_id: observation}
            training: Whether in training mode
            
        Returns:
            dict: {agent_id: action}
            dict: {team: final_strategy} - updated team strategies
        """
        actions = {}
        new_team_strategies = {}
        
        # Process each team separately and sequentially
        for team in ['red', 'blue']:
            team_agents = self._get_team_agents(observations, team)
            
            if not team_agents:
                continue
                
            # Start with current team strategy
            current_strategy = self.team_strategies[team].copy()
            
            # Process agents sequentially within the team
            for agent_id in team_agents:
                if agent_id in observations:
                    # Get action and updated strategy
                    action, new_strategy = self.agent.act(
                        observation=observations[agent_id],
                        strategy_embedding=current_strategy,
                        agent_id=agent_id,
                        training=training
                    )
                    
                    actions[agent_id] = action
                    current_strategy = new_strategy  # Update strategy for next agent
            
            # Store final team strategy
            new_team_strategies[team] = current_strategy
        
        # Update team strategies
        self.team_strategies.update(new_team_strategies)
        
        return actions
    
    def step(self, prev_observations, actions, rewards, observations, terminations):
        """Store experiences for training"""
        if not prev_observations:
            return
            
        # Process experiences for each team
        for team in ['red', 'blue']:
            team_agents = self._get_team_agents(prev_observations, team)
            
            if not team_agents:
                continue
            
            # Get previous and current team strategies
            prev_strategy = self.team_strategies[team].copy()
            
            # Process each agent's experience
            current_strategy = prev_strategy.copy()
            for agent_id in team_agents:
                if agent_id in prev_observations and agent_id in observations:
                    # Calculate what the next strategy would have been
                    _, next_strategy = self.agent.act(
                        observation=prev_observations[agent_id],
                        strategy_embedding=current_strategy,
                        agent_id=agent_id,
                        training=False  # Just for getting next strategy, no exploration
                    )
                    
                    # Store experience
                    self.agent.step(
                        state=prev_observations[agent_id],
                        prev_strategy=current_strategy,
                        agent_id=agent_id,
                        action=actions.get(agent_id, 0),
                        reward=rewards.get(agent_id, 0),
                        next_state=observations[agent_id],
                        next_strategy=next_strategy,
                        done=terminations.get(agent_id, False)
                    )
                    
                    current_strategy = next_strategy  # Update for next agent
    
    def save_models(self, directory="models"):
        """Save the autoregressive model"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Save central policy
        filepath = f"{directory}/autoregressive_central_policy.pth"
        self.agent.save_model(filepath)
        
        # Save team strategies
        np.save(f"{directory}/team_strategies.npy", self.team_strategies)
    
    def load_models(self, directory="models"):
        """Load the autoregressive model"""
        import os
        
        # Load central policy
        filepath = f"{directory}/autoregressive_central_policy.pth"
        if os.path.exists(filepath):
            self.agent.load_model(filepath)
        
        # Load team strategies
        strategy_file = f"{directory}/team_strategies.npy"
        if os.path.exists(strategy_file):
            loaded_strategies = np.load(strategy_file, allow_pickle=True).item()
            self.team_strategies.update(loaded_strategies)
    
    def get_training_stats(self):
        """Get training statistics"""
        return {
            'central_policy': {
                'epsilon': self.agent.epsilon,
                'buffer_size': len(self.agent.memory),
                'strategy_dim': self.strategy_dim
            }
        }