"""Autoregressive action selection utilities for Sable."""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from einops import rearrange


def get_shifted_discrete_actions(
    action: torch.Tensor, 
    legal_actions: torch.Tensor, 
    n_agents: int
) -> torch.Tensor:
    """Get shifted discrete action sequence for training.
    
    Args:
        action: [B, S] action indices (S = T * N)
        legal_actions: [B, S, A] legal action mask
        n_agents: Number of agents
        
    Returns:
        shifted_actions: [B, S, A+1] shifted one-hot actions with start tokens
    """
    B, S = action.shape
    A = legal_actions.shape[-1]
    
    # Create shifted action array with extra dimension for start token
    shifted_actions = torch.zeros(B, S, A + 1, device=action.device)
    
    # Start-of-timestep token (index 0 = start token)
    start_timestep_token = torch.zeros(A + 1, device=action.device)
    start_timestep_token[0] = 1.0
    
    # Convert actions to one-hot (indices 1 to A)
    one_hot_action = F.one_hot(action, A).float()
    
    # Insert one-hot actions into shifted array (indices 1 to A)
    shifted_actions[:, :, 1:] = one_hot_action
    
    # Shift by 1 position to create teacher forcing sequence
    shifted_actions = torch.roll(shifted_actions, shifts=1, dims=1)
    
    # Set start token for first agent of each timestep
    shifted_actions[:, ::n_agents, :] = start_timestep_token
    
    return shifted_actions


def get_shifted_continuous_actions(
    action: torch.Tensor, 
    action_dim: int, 
    n_agents: int
) -> torch.Tensor:
    """Get shifted continuous action sequence for training.
    
    Args:
        action: [B, S, action_dim] continuous actions (S = T * N)
        action_dim: Action dimension
        n_agents: Number of agents
        
    Returns:
        shifted_actions: [B, S, action_dim] shifted actions with zero start tokens
    """
    B, S, _ = action.shape
    
    # Create shifted actions with zero padding
    shifted_actions = torch.zeros_like(action)
    start_timestep_token = torch.zeros(action_dim, device=action.device)
    
    # Shift actions: action[t] -> shifted_actions[t+1] 
    shifted_actions[:, 1:, :] = action[:, :-1, :]
    
    # Set zero start token for first agent of each timestep
    shifted_actions[:, ::n_agents, :] = start_timestep_token
    
    return shifted_actions


def discrete_train_decoder_fn(
    decoder,
    obs_rep: torch.Tensor,
    action: torch.Tensor,
    legal_actions: torch.Tensor,
    hstates: Tuple[torch.Tensor, torch.Tensor],
    dones: Optional[torch.Tensor],
    step_count: Optional[torch.Tensor],
    n_agents: int,
    chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parallel training forward pass for discrete actions with proper shifting.
    
    Args:
        decoder: Decoder module
        obs_rep: [B, S, embed_dim] observation representations
        action: [B, S] action indices
        legal_actions: [B, S, A] legal action mask  
        hstates: Tuple of decoder hidden states
        dones: [B, S] done flags (optional)
        step_count: [B, S] step counts (optional)
        n_agents: Number of agents
        chunk_size: Chunk size for processing
        
    Returns:
        action_log_prob: [B, S] action log probabilities
        entropy: [B, S] policy entropy
    """
    # Get shifted actions for teacher forcing
    shifted_actions = get_shifted_discrete_actions(action, legal_actions, n_agents)
    
    B, S, A = legal_actions.shape
    logits = torch.zeros_like(legal_actions, dtype=torch.float32)
    
    # Process in chunks for memory efficiency
    num_chunks = S // chunk_size
    updated_hstates = (torch.zeros_like(hstates[0]), torch.zeros_like(hstates[1]))
    
    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = (chunk_id + 1) * chunk_size
        
        # Extract chunk data
        chunk_obs_rep = obs_rep[:, start_idx:end_idx]
        chunk_shifted_actions = shifted_actions[:, start_idx:end_idx]
        chunk_dones = dones[:, start_idx:end_idx] if dones is not None else None
        chunk_step_count = step_count[:, start_idx:end_idx] if step_count is not None else None
        
        # Forward through decoder
        chunk_logits, chunk_hstates = decoder(
            chunk_shifted_actions,
            chunk_obs_rep, 
            hstates,
            chunk_dones,
            chunk_step_count
        )
        
        # Store results
        logits[:, start_idx:end_idx] = chunk_logits
        updated_hstates = chunk_hstates
        
    # Apply legal action masking
    masked_logits = torch.where(
        legal_actions,
        logits,
        torch.finfo(logits.dtype).min
    )
    
    # Compute log probabilities and entropy
    action_dist = torch.distributions.Categorical(logits=masked_logits)
    action_log_prob = action_dist.log_prob(action)
    entropy = action_dist.entropy()
    
    return action_log_prob, entropy


def continuous_train_decoder_fn(
    decoder,
    obs_rep: torch.Tensor,
    action: torch.Tensor,
    hstates: Tuple[torch.Tensor, torch.Tensor],
    dones: Optional[torch.Tensor],
    step_count: Optional[torch.Tensor],
    n_agents: int,
    chunk_size: int,
    action_dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parallel training forward pass for continuous actions with proper shifting.
    
    Args:
        decoder: Decoder module
        obs_rep: [B, S, embed_dim] observation representations
        action: [B, S, action_dim] continuous actions
        hstates: Tuple of decoder hidden states
        dones: [B, S] done flags (optional)
        step_count: [B, S] step counts (optional)
        n_agents: Number of agents
        chunk_size: Chunk size for processing
        action_dim: Action dimension
        
    Returns:
        action_log_prob: [B, S] action log probabilities
        entropy: [B, S] policy entropy
    """
    # Get shifted actions for teacher forcing
    shifted_actions = get_shifted_continuous_actions(action, action_dim, n_agents)
    
    B, S, _ = action.shape
    means = torch.zeros_like(action)
    
    # Process in chunks for memory efficiency
    num_chunks = S // chunk_size
    
    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = (chunk_id + 1) * chunk_size
        
        # Extract chunk data
        chunk_obs_rep = obs_rep[:, start_idx:end_idx]
        chunk_shifted_actions = shifted_actions[:, start_idx:end_idx]
        chunk_dones = dones[:, start_idx:end_idx] if dones is not None else None
        chunk_step_count = step_count[:, start_idx:end_idx] if step_count is not None else None
        
        # Forward through decoder
        chunk_means, hstates = decoder(
            chunk_shifted_actions,
            chunk_obs_rep,
            hstates, 
            chunk_dones,
            chunk_step_count
        )
        
        # Store results
        means[:, start_idx:end_idx] = chunk_means
    
    # Get standard deviation from decoder
    log_std = decoder.log_std
    std = torch.exp(log_std).expand_as(means)
    
    # Create distribution and compute log prob and entropy
    action_dist = torch.distributions.Normal(means, std)
    
    # For multi-dimensional actions, sum log probs across action dims
    action_log_prob = action_dist.log_prob(action).sum(dim=-1)
    entropy = action_dist.entropy().sum(dim=-1)
    
    return action_log_prob, entropy


def discrete_autoregressive_act(
    decoder,
    obs_rep: torch.Tensor,
    hstates: Tuple[torch.Tensor, torch.Tensor],
    legal_actions: torch.Tensor,
    step_count: Optional[torch.Tensor],
    key: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Autoregressive action generation for discrete actions.
    
    This is the core of Sable - actions are generated sequentially where
    each agent sees the actions of previous agents.
    
    Args:
        decoder: Decoder module  
        obs_rep: [B, N, embed_dim] observation representations
        hstates: Tuple of decoder hidden states
        legal_actions: [B, N, A] legal action mask
        step_count: [B, N] step counts (optional)
        key: Random generator (optional)
        
    Returns:
        actions: [B, N] sampled actions
        action_log_probs: [B, N] action log probabilities  
        updated_hstates: Updated hidden states
    """
    B, N, A = legal_actions.shape
    device = obs_rep.device
    
    # Initialize shifted actions with start token
    shifted_actions = torch.zeros(B, N, A + 1, device=device)
    shifted_actions[:, 0, 0] = 1.0  # Start token for first agent
    
    # Storage for outputs
    output_actions = torch.zeros(B, N, dtype=torch.long, device=device)
    output_action_log_probs = torch.zeros(B, N, device=device)
    
    # Sequential action generation - this is the key Sable feature!
    for agent_idx in range(N):
        # Get current agent's input
        agent_shifted_action = shifted_actions[:, agent_idx:agent_idx+1, :]  # [B, 1, A+1]
        agent_obs_rep = obs_rep[:, agent_idx:agent_idx+1, :]  # [B, 1, embed_dim]
        agent_step_count = step_count[:, agent_idx:agent_idx+1] if step_count is not None else None
        
        # Forward through decoder in recurrent mode
        logits, hstates = decoder.recurrent(
            agent_shifted_action,
            agent_obs_rep,
            hstates,
            agent_step_count
        )
        
        # Apply legal action masking
        agent_legal_actions = legal_actions[:, agent_idx:agent_idx+1, :]
        masked_logits = torch.where(
            agent_legal_actions,
            logits,
            torch.finfo(logits.dtype).min
        )
        
        # Sample action
        action_dist = torch.distributions.Categorical(logits=masked_logits.squeeze(1))
        if key is not None:
            # Use provided generator for reproducible sampling
            action = action_dist.sample(generator=key)
        else:
            action = action_dist.sample()
            
        action_log_prob = action_dist.log_prob(action)
        
        # Store results
        output_actions[:, agent_idx] = action
        output_action_log_probs[:, agent_idx] = action_log_prob
        
        # Update shifted actions for next agent (if not last agent)
        if agent_idx < N - 1:
            # Convert action to one-hot and add to shifted_actions
            action_one_hot = F.one_hot(action, A).float()
            shifted_actions[:, agent_idx + 1, 1:] = action_one_hot
    
    return output_actions, output_action_log_probs, hstates


def continuous_autoregressive_act(
    decoder,
    obs_rep: torch.Tensor,
    hstates: Tuple[torch.Tensor, torch.Tensor],
    step_count: Optional[torch.Tensor],
    action_dim: int,
    key: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Autoregressive action generation for continuous actions.
    
    Args:
        decoder: Decoder module
        obs_rep: [B, N, embed_dim] observation representations
        hstates: Tuple of decoder hidden states  
        step_count: [B, N] step counts (optional)
        action_dim: Action dimension
        key: Random generator (optional)
        
    Returns:
        actions: [B, N, action_dim] sampled actions
        action_log_probs: [B, N] action log probabilities
        updated_hstates: Updated hidden states
    """
    B, N, _ = obs_rep.shape
    device = obs_rep.device
    
    # Initialize shifted actions with zero start token
    shifted_actions = torch.zeros(B, N, action_dim, device=device)
    
    # Storage for outputs
    output_actions = torch.zeros(B, N, action_dim, device=device)
    output_action_log_probs = torch.zeros(B, N, device=device)
    
    # Sequential action generation
    for agent_idx in range(N):
        # Get current agent's input
        agent_shifted_action = shifted_actions[:, agent_idx:agent_idx+1, :]  # [B, 1, action_dim]
        agent_obs_rep = obs_rep[:, agent_idx:agent_idx+1, :]  # [B, 1, embed_dim]
        agent_step_count = step_count[:, agent_idx:agent_idx+1] if step_count is not None else None
        
        # Forward through decoder in recurrent mode
        mean, hstates = decoder.recurrent(
            agent_shifted_action,
            agent_obs_rep,
            hstates,
            agent_step_count
        )
        
        # Get standard deviation
        log_std = decoder.log_std
        std = torch.exp(log_std)  # [action_dim]
        
        # Sample action
        action_dist = torch.distributions.Normal(mean.squeeze(1), std)  # mean: [B, action_dim], std: [action_dim]
        if key is not None:
            # Use provided generator for reproducible sampling
            action = action_dist.sample(generator=key)
        else:
            action = action_dist.sample()
            
        action_log_prob = action_dist.log_prob(action).sum(dim=-1)
        
        # Store results
        output_actions[:, agent_idx, :] = action
        output_action_log_probs[:, agent_idx] = action_log_prob
        
        # Update shifted actions for next agent (if not last agent)
        if agent_idx < N - 1:
            shifted_actions[:, agent_idx + 1, :] = action
    
    return output_actions, output_action_log_probs, hstates