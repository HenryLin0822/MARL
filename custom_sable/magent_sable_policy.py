"""
Sable Policy Implementation for MAgentX Combined Arms Environment

Adapts the Sable network for the MAgentX environment with:
- Individual discrete action spaces per agent
- Local observations (13x13 grid with multiple channels)
- Team-based coordination through autoregressive action selection
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sable_network_fixed import SableNetwork
from config import SableConfig


class MAgentSablePolicy:
    """
    Sable policy for MAgentX Combined Arms environment.
    
    Handles team-based coordination while working with individual agent observations.
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, int, int],  # (13, 13, channels)
        action_dim: int,
        n_agents_per_team: int = 4,
        team_id: str = "red",  # "red" or "blue"
        embed_dim: int = 128,
        n_head: int = 8,
        n_block: int = 3,
        device: str = "cpu"
    ):
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.n_agents_per_team = n_agents_per_team
        self.team_id = team_id
        self.device = torch.device(device)
        
        # Flatten observation for network input
        self.obs_dim = np.prod(obs_shape)
        
        # Create Sable configuration with max team size
        self.max_team_size = n_agents_per_team * 2  # Allow for larger teams
        self.config = SableConfig(
            n_agents=self.max_team_size,
            action_dim=action_dim,
            obs_dim=self.obs_dim,
            embed_dim=embed_dim,
            n_head=n_head,
            n_block=n_block,
            chunk_size=self.max_team_size,
            action_space_type="discrete",
            decay_scaling_factor=1.0
        )
        
        # Initialize Sable network
        self.network = SableNetwork(self.config).to(self.device)
        
        # Hidden states for recurrent processing
        self.hidden_states = None
        self.reset_hidden_states()
        
        # Agent ID mapping for consistent ordering
        self.agent_id_mapping = {}
        self.active_agents = []
        
    def reset_hidden_states(self, batch_size: int = 1):
        """Reset hidden states for new episodes."""
        self.hidden_states = self.network.init_hidden_states(batch_size, self.device)
        
    def update_active_agents(self, agent_ids: List[str]):
        """Update the list of active agents and maintain consistent ordering."""
        # Filter for team agents only
        team_agents = [aid for aid in agent_ids if self._is_team_agent(aid)]
        
        # Sort for consistent ordering
        team_agents.sort()
        
        # Update mapping if needed
        if team_agents != self.active_agents:
            self.active_agents = team_agents
            self.agent_id_mapping = {aid: idx for idx, aid in enumerate(team_agents)}
            
            # Adjust hidden states if number of agents changed
            current_agents = len(team_agents)
            if current_agents != self.n_agents_per_team and current_agents > 0:
                # For simplicity, reset hidden states when team size changes
                self.reset_hidden_states()
    
    def _is_team_agent(self, agent_id: str) -> bool:
        """Check if agent belongs to this team."""
        if self.team_id == "red":
            return "red" in agent_id.lower()
        else:
            return "blue" in agent_id.lower()
    
    def _preprocess_observations(self, observations: Dict[str, np.ndarray]) -> torch.Tensor:
        """Convert dict of observations to tensor format for Sable network."""
        # Filter and order observations by team agents
        team_observations = []
        for agent_id in self.active_agents:
            if agent_id in observations:
                obs = observations[agent_id]
                # Flatten spatial observation
                obs_flat = obs.flatten()
                team_observations.append(obs_flat)
            else:
                # Agent might be dead, use zero observation
                zero_obs = np.zeros(self.obs_dim)
                team_observations.append(zero_obs)
        
        # Dynamically adjust to actual team size
        actual_team_size = len(team_observations)
        if actual_team_size == 0:
            # No agents in team, return empty tensor
            return torch.zeros(1, 1, self.obs_dim).to(self.device)
        
        # Store actual team size and pad to max size for network
        self.current_team_size = actual_team_size
        
        # Pad to max team size for consistent network input
        while len(team_observations) < self.max_team_size:
            zero_obs = np.zeros(self.obs_dim)
            team_observations.append(zero_obs)
        
        # Convert to tensor: [1, current_team_size, obs_dim]
        team_observations_np = np.array(team_observations)  # More efficient conversion
        obs_tensor = torch.FloatTensor(team_observations_np).unsqueeze(0).to(self.device)
        return obs_tensor
    
    def _create_legal_actions(self, agent_ids: List[str]) -> torch.Tensor:
        """Create legal actions mask for all team agents."""
        # Create legal actions for max team size  
        batch_size = 1
        legal_actions = torch.ones(
            batch_size, self.max_team_size, self.action_dim, 
            dtype=torch.bool, device=self.device
        )
        return legal_actions
    
    def get_actions(
        self, 
        observations: Dict[str, np.ndarray],
        deterministic: bool = False
    ) -> Dict[str, int]:
        """
        Get actions for all team agents using Sable's autoregressive coordination.
        
        Args:
            observations: Dict of {agent_id: observation}
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Dict of {agent_id: action} for team agents only
        """
        # Update active agents
        self.update_active_agents(list(observations.keys()))
        
        if not self.active_agents:
            return {}
        
        # Preprocess observations
        obs_tensor = self._preprocess_observations(observations)
        legal_actions = self._create_legal_actions(self.active_agents)
        
        # Get actions from Sable network
        with torch.no_grad():
            actions, action_log_probs, values, updated_hstates = self.network.get_actions(
                obs_tensor, self.hidden_states, legal_actions=legal_actions
            )
        
        # Update hidden states
        self.hidden_states = updated_hstates
        
        # Convert actions to dict format
        action_dict = {}
        actions_np = actions.squeeze(0).cpu().numpy()  # [n_agents]
        
        for i, agent_id in enumerate(self.active_agents):
            if i < len(actions_np) and agent_id in observations:
                action_dict[agent_id] = int(actions_np[i])
        
        return action_dict
    
    def evaluate_actions(
        self,
        observations: Dict[str, np.ndarray],
        actions: Dict[str, int],
        dones: Optional[Dict[str, bool]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training (used during policy optimization).
        
        Args:
            observations: Dict of {agent_id: observation}
            actions: Dict of {agent_id: action}
            dones: Dict of {agent_id: done_flag}
            
        Returns:
            values: Value estimates for each agent
            log_probs: Action log probabilities
            entropy: Policy entropy
        """
        # Update active agents
        self.update_active_agents(list(observations.keys()))
        
        if not self.active_agents:
            # Return dummy values if no active agents
            dummy_tensor = torch.zeros(1, 1, device=self.device)
            return dummy_tensor, dummy_tensor, dummy_tensor
        
        # Preprocess inputs
        obs_tensor = self._preprocess_observations(observations)
        legal_actions = self._create_legal_actions(self.active_agents)
        
        # Convert actions to tensor
        action_list = []
        for agent_id in self.active_agents:
            if agent_id in actions:
                action_list.append(actions[agent_id])
            else:
                action_list.append(0)  # Default action for dead agents
        
        # Pad to max team size for consistent network input
        while len(action_list) < self.max_team_size:
            action_list.append(0)  # Padding action
            
        action_tensor = torch.LongTensor(action_list).unsqueeze(0).to(self.device)
        
        # Convert dones to tensor if provided
        dones_tensor = None
        if dones is not None:
            done_list = []
            for agent_id in self.active_agents:
                done_list.append(dones.get(agent_id, True))
            while len(done_list) < self.max_team_size:
                done_list.append(True)  # Padding agents are "done"
            dones_tensor = torch.BoolTensor(done_list).unsqueeze(0).to(self.device)
        
        # Forward pass through network
        values, log_probs, entropy = self.network(
            obs_tensor, action_tensor, self.hidden_states,
            dones=dones_tensor, legal_actions=legal_actions
        )
        
        return values, log_probs, entropy
    
    def get_value(self, observations: Dict[str, np.ndarray]) -> torch.Tensor:
        """Get value estimates for current observations."""
        # Preprocess observations
        obs_tensor = self._preprocess_observations(observations)
        
        # Get values from encoder
        with torch.no_grad():
            values, _, _ = self.network.encoder(obs_tensor, self.hidden_states['encoder'])
        
        return values
    
    def save(self, filepath: str):
        """Save policy parameters."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'config': self.config,
            'team_id': self.team_id,
            'obs_shape': self.obs_shape,
            'action_dim': self.action_dim,
            'n_agents_per_team': self.n_agents_per_team
        }, filepath)
    
    def load(self, filepath: str):
        """Load policy parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        
    def train(self):
        """Set network to training mode."""
        self.network.train()
        
    def eval(self):
        """Set network to evaluation mode."""
        self.network.eval()


class MultiTeamSablePolicy:
    """
    Multi-team wrapper that manages both Red and Blue Sable policies.
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        action_dim: int,
        n_agents_per_team: int = 4,
        embed_dim: int = 128,
        n_head: int = 8,
        n_block: int = 3,
        device: str = "cpu"
    ):
        self.red_policy = MAgentSablePolicy(
            obs_shape, action_dim, n_agents_per_team, "red",
            embed_dim, n_head, n_block, device
        )
        
        self.blue_policy = MAgentSablePolicy(
            obs_shape, action_dim, n_agents_per_team, "blue", 
            embed_dim, n_head, n_block, device
        )
        
    def get_actions(
        self, 
        observations: Dict[str, np.ndarray],
        deterministic: bool = False
    ) -> Dict[str, int]:
        """Get actions for all agents from both teams."""
        red_actions = self.red_policy.get_actions(observations, deterministic)
        blue_actions = self.blue_policy.get_actions(observations, deterministic)
        
        # Combine actions
        all_actions = {**red_actions, **blue_actions}
        return all_actions
    
    def reset_hidden_states(self):
        """Reset hidden states for both teams."""
        self.red_policy.reset_hidden_states()
        self.blue_policy.reset_hidden_states()
    
    def get_policy(self, team: str) -> MAgentSablePolicy:
        """Get policy for specific team."""
        if team.lower() == "red":
            return self.red_policy
        else:
            return self.blue_policy
    
    def save(self, red_path: str, blue_path: str):
        """Save both team policies."""
        self.red_policy.save(red_path)
        self.blue_policy.save(blue_path)
    
    def load(self, red_path: str, blue_path: str):
        """Load both team policies."""
        self.red_policy.load(red_path)
        self.blue_policy.load(blue_path)