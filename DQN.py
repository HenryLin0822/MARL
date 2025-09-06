import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple

class DQNNetwork(nn.Module):
    def __init__(self, input_shape, action_size, unit_type="melee"):
        super(DQNNetwork, self).__init__()
        
        self.input_shape = input_shape  # (13, 13, channels)
        self.action_size = action_size
        self.unit_type = unit_type
        
        # Convolutional layers for spatial feature extraction
        self.conv1 = nn.Conv2d(input_shape[2], 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Calculate conv output size
        conv_out_size = 13 * 13 * 128
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, 256)
        
        # Unit type specific processing
        if unit_type == "melee":
            # Melee units focus more on local tactical decisions
            self.fc3 = nn.Linear(256, 128)
        else:  # ranged
            # Ranged units need broader spatial awareness
            self.fc3 = nn.Linear(256, 256)
            
        self.output = nn.Linear(self.fc3.out_features, action_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Input shape: (batch_size, height, width, channels)
        # PyTorch expects: (batch_size, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        
        # Convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten for fully connected layers
        x = x.reshape(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        # Output layer
        return self.output(x)

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)
    
    def sample(self):
        experiences = random.sample(self.memory, self.batch_size)
        
        states = torch.from_numpy(np.array([e.state for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.array([e.action for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.array([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, input_shape, action_size, unit_type="melee", lr=1e-3, 
                 buffer_size=10000, batch_size=64, gamma=0.99, tau=1e-3,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        
        self.input_shape = input_shape
        self.action_size = action_size
        self.unit_type = unit_type
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
        self.q_network = DQNNetwork(input_shape, action_size, unit_type).to(self.device)
        self.target_network = DQNNetwork(input_shape, action_size, unit_type).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size, batch_size, self.device)
        
        # Training step counter
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        # Store experience in replay buffer
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every few steps
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
    
    def act(self, state, training=True):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # Epsilon-greedy action selection
        if training and random.random() <= self.epsilon:
            return random.choice(np.arange(self.action_size))
        else:
            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(state)
            self.q_network.train()
            return np.argmax(action_values.cpu().data.numpy())
    
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        
        # Get Q targets for next states from target model
        q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        q_targets = rewards.unsqueeze(1) + (self.gamma * q_targets_next * (1 - dones.unsqueeze(1)))
        
        # Get expected Q values from local model
        q_expected = self.q_network(states).gather(1, actions.unsqueeze(1))
        
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
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def hard_update(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'unit_type': self.unit_type
        }, filepath)
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

class MultiAgentDQNPolicy:
    def __init__(self, env, lr=1e-3, buffer_size=10000, batch_size=64):
        self.env = env
        self.lr = lr
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        # Agent configurations for different unit types
        self.agent_configs = {}
        self.agents = {}
        
        # Initialize agents based on team and unit type
        self._initialize_agents()
    
    def _initialize_agents(self):
        # Get a sample observation to determine input shape
        obs = self.env.reset()
        if obs:
            sample_obs = list(obs.values())[0]
            input_shape = sample_obs.shape
            
            # Get action space size for any agent
            sample_agent = list(obs.keys())[0]
            action_size = self.env.action_space(sample_agent).n
            
            # Create agents for different unit types
            # We'll determine unit type based on agent ID patterns
            for agent_id in self.env.agents:
                unit_type = self._get_unit_type(agent_id)
                
                # Create DQN agent with unit-specific hyperparameters
                if unit_type == "melee":
                    # Melee units: more conservative, focus on survival
                    agent = DQNAgent(
                        input_shape=input_shape,
                        action_size=action_size,
                        unit_type="melee",
                        lr=self.lr * 0.8,  # Slightly lower learning rate
                        buffer_size=self.buffer_size,
                        batch_size=self.batch_size,
                        gamma=0.99,  # Long-term focus
                        epsilon=0.9,  # More exploration initially
                        epsilon_decay=0.996
                    )
                else:  # ranged
                    # Ranged units: more aggressive, tactical positioning
                    agent = DQNAgent(
                        input_shape=input_shape,
                        action_size=action_size,
                        unit_type="ranged",
                        lr=self.lr,
                        buffer_size=self.buffer_size,
                        batch_size=self.batch_size,
                        gamma=0.95,  # More immediate rewards
                        epsilon=1.0,  # High initial exploration
                        epsilon_decay=0.993
                    )
                
                self.agents[agent_id] = agent
    
    def _get_unit_type(self, agent_id):
        # Determine unit type based on agent ID
        # This is a heuristic - you may need to adjust based on actual agent naming
        agent_str = str(agent_id).lower()
        
        # Common patterns in magent2 agent IDs
        if "melee" in agent_str or "_0_" in agent_str or "_1_" in agent_str:
            return "melee"
        else:
            return "ranged"
    
    def get_action(self, observation, agent_id, training=True):
        if agent_id in self.agents:
            return self.agents[agent_id].act(observation, training=training)
        else:
            # Fallback to random action if agent not found
            action_space = self.env.action_space(agent_id)
            return random.randint(0, action_space.n - 1)
    
    def get_actions(self, observations, training=True):
        actions = {}
        for agent_id, obs in observations.items():
            actions[agent_id] = self.get_action(obs, agent_id, training=training)
        return actions
    
    def step(self, prev_observations, actions, rewards, observations, terminations):
        # Store experiences for all agents
        for agent_id in prev_observations.keys():
            if agent_id in self.agents and agent_id in observations:
                self.agents[agent_id].step(
                    state=prev_observations[agent_id],
                    action=actions[agent_id],
                    reward=rewards.get(agent_id, 0),
                    next_state=observations[agent_id],
                    done=terminations.get(agent_id, False)
                )
    
    def save_models(self, directory="models"):
        import os
        os.makedirs(directory, exist_ok=True)
        
        for agent_id, agent in self.agents.items():
            unit_type = agent.unit_type
            team = "red" if "red" in str(agent_id).lower() else "blue"
            filepath = f"{directory}/{team}_{unit_type}_dqn.pth"
            agent.save_model(filepath)
    
    def load_models(self, directory="models"):
        import os
        for agent_id, agent in self.agents.items():
            unit_type = agent.unit_type
            team = "red" if "red" in str(agent_id).lower() else "blue"
            filepath = f"{directory}/{team}_{unit_type}_dqn.pth"
            
            if os.path.exists(filepath):
                agent.load_model(filepath)
    
    def set_training_mode(self, training=True):
        for agent in self.agents.values():
            if training:
                agent.q_network.train()
            else:
                agent.q_network.eval()
    
    def get_training_stats(self):
        stats = {}
        for agent_id, agent in self.agents.items():
            stats[agent_id] = {
                'epsilon': agent.epsilon,
                'buffer_size': len(agent.memory),
                'unit_type': agent.unit_type
            }
        return stats