"""Fully aligned autoregressive action generation matching Mava Sable exactly."""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from einops import rearrange
import numpy as np


def get_shifted_discrete_actions(
    action: torch.Tensor,
    legal_actions: torch.Tensor,
    n_agents: int,
    chunk_size: int
) -> torch.Tensor:
    """Get shifted discrete action sequence for training - matches JAX exactly.
    
    Args:
        action: [B, S] action indices (S = T * N where T is timesteps, N is agents)
        legal_actions: [B, S, A] legal action mask
        n_agents: Number of agents per team
        chunk_size: Chunk size for processing
        
    Returns:
        shifted_actions: [B, S, A+1] shifted one-hot actions with start tokens
    """
    B, S = action.shape
    A = legal_actions.shape[-1]
    
    # Create shifted action array with extra dimension for start token
    shifted_actions = torch.zeros(B, S, A + 1, device=action.device, dtype=torch.float32)
    
    # Start-of-timestep token (index 0 = start token)
    start_timestep_token = torch.zeros(A + 1, device=action.device)
    start_timestep_token[0] = 1.0
    
    # Convert actions to one-hot (indices 1 to A)
    # Clamp actions to valid range to avoid index errors
    action_clamped = torch.clamp(action, 0, A - 1)
    one_hot_action = F.one_hot(action_clamped, A).float()
    
    # Insert one-hot actions into shifted array (indices 1 to A)
    shifted_actions[:, :, 1:] = one_hot_action
    
    # Shift by 1 position to create teacher forcing sequence
    shifted_actions = torch.roll(shifted_actions, shifts=1, dims=1)
    
    # Set start token for first agent of each timestep (every n_agents)
    # This creates the autoregressive structure: first agent sees start token,
    # subsequent agents see previous agents' actions
    shifted_actions[:, ::n_agents, :] = start_timestep_token
    
    return shifted_actions


def get_shifted_continuous_actions(
    action: torch.Tensor,
    action_dim: int,
    n_agents: int,
    chunk_size: int
) -> torch.Tensor:
    """Get shifted continuous action sequence for training - matches JAX exactly.
    
    Args:
        action: [B, S, action_dim] continuous actions (S = T * N)
        action_dim: Action dimension
        n_agents: Number of agents per team
        chunk_size: Chunk size for processing
        
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
    """Parallel training forward pass for discrete actions with proper teacher forcing.
    
    This matches the JAX implementation exactly with proper shifted action sequences.
    """
    B, S, embed_dim = obs_rep.shape
    A = legal_actions.shape[-1]
    
    # Get properly shifted actions for teacher forcing
    shifted_actions = get_shifted_discrete_actions(action, legal_actions, n_agents, chunk_size)
    
    # Create step count if not provided - for chunked processing
    if step_count is None:
        step_count = torch.arange(S, device=obs_rep.device, dtype=torch.float32)
        step_count = step_count[None, :, None].expand(B, S, 1)
    
    # Forward through decoder with shifted actions
    logits, _ = decoder(shifted_actions, obs_rep, hstates, dones, step_count)
    
    # Apply legal action masking
    if legal_actions is not None:
        # Create large negative value for illegal actions
        large_neg = torch.finfo(logits.dtype).min
        masked_logits = torch.where(legal_actions, logits, large_neg)
    else:
        masked_logits = logits
    
    # Compute log probabilities and entropy
    log_probs = F.log_softmax(masked_logits, dim=-1)
    probs = F.softmax(masked_logits, dim=-1)
    
    # Get log probabilities for actual actions
    action_clamped = torch.clamp(action, 0, A - 1)
    action_log_probs = log_probs.gather(-1, action_clamped.unsqueeze(-1)).squeeze(-1)
    
    # Compute entropy
    entropy = -(probs * log_probs).sum(dim=-1)
    
    return action_log_probs, entropy


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
    """Parallel training forward pass for continuous actions."""
    B, S, embed_dim = obs_rep.shape
    
    # Get properly shifted actions for teacher forcing
    shifted_actions = get_shifted_continuous_actions(action, action_dim, n_agents, chunk_size)
    
    # Create step count if not provided
    if step_count is None:
        step_count = torch.arange(S, device=obs_rep.device, dtype=torch.float32)
        step_count = step_count[None, :, None].expand(B, S, 1)
    
    # Forward through decoder
    mean, _ = decoder(shifted_actions, obs_rep, hstates, dones, step_count)
    
    # Get log std from decoder
    log_std = decoder.log_std
    std = torch.exp(log_std)
    
    # Compute log probabilities (Gaussian)
    var = std ** 2
    log_prob = -0.5 * (((action - mean) ** 2) / var + 2 * log_std + np.log(2 * np.pi))
    action_log_probs = log_prob.sum(dim=-1)  # Sum across action dimensions
    
    # Compute entropy
    entropy = 0.5 * (action_dim * (1 + np.log(2 * np.pi)) + 2 * log_std.sum())
    entropy = entropy.expand_as(action_log_probs)
    
    return action_log_probs, entropy


def discrete_autoregressive_act(
    decoder,
    obs_rep: torch.Tensor,
    hstates: Tuple[torch.Tensor, torch.Tensor],
    legal_actions: Optional[torch.Tensor],
    step_count: Optional[torch.Tensor],
    key: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Autoregressive action generation for discrete actions - THE CORE OF SABLE.
    
    This matches the JAX implementation exactly with proper sequential generation.
    Each agent's action conditions on all previous agents' actions in the same timestep.
    """
    B, N, embed_dim = obs_rep.shape
    device = obs_rep.device
    
    if legal_actions is not None:
        A = legal_actions.shape[-1]
    else:
        # Infer action dimension from decoder
        A = decoder.head[-1].out_features
        legal_actions = torch.ones(B, N, A, device=device, dtype=torch.bool)
    
    # Initialize shifted actions - matches JAX exactly
    shifted_actions = torch.zeros(B, N, A + 1, device=device)
    shifted_actions[:, 0, 0] = 1.0  # Start token for first agent
    
    # Output containers
    output_action = torch.zeros(B, N, 1, device=device, dtype=torch.long)
    output_action_log = torch.zeros(B, N, 1, device=device)
    
    # Current hidden states
    current_hstates = hstates
    
    # Generate step count matching JAX
    if step_count is None:
        step_count = torch.arange(N, device=device, dtype=torch.float32)
        step_count = step_count[None, :, None].expand(B, N, 1)
    
    # Sequential autoregressive generation - matches JAX loop exactly
    for i in range(N):
        # Get current inputs
        current_action_input = shifted_actions[:, i:i+1]  # [B, 1, A+1]
        obs_i = obs_rep[:, i:i+1]  # [B, 1, embed_dim] 
        step_i = step_count[:, i:i+1]  # [B, 1, 1]
        
        # Forward pass through decoder
        logits_i, current_hstates = decoder.recurrent(
            current_action_input, obs_i, current_hstates, step_i
        )
        
        # Apply legal action masking - matches JAX exactly
        legal_i = legal_actions[:, i:i+1]  # [B, 1, A]
        large_neg = torch.finfo(logits_i.dtype).min
        masked_logits_i = torch.where(legal_i, logits_i, large_neg)
        
        # Sample action using categorical distribution - matches JAX
        if key is not None:
            # Create categorical distribution and sample
            probs_i = F.softmax(masked_logits_i, dim=-1)  # [B, 1, A]
            action_i = torch.multinomial(probs_i.squeeze(1), 1, generator=key)  # [B, 1]
        else:
            probs_i = F.softmax(masked_logits_i, dim=-1)
            action_i = torch.multinomial(probs_i.squeeze(1), 1)  # [B, 1]
            
        # Compute log probability
        log_prob_i = F.log_softmax(masked_logits_i, dim=-1)
        action_log_i = log_prob_i.gather(-1, action_i.unsqueeze(-1))  # [B, 1, 1]
        
        # Store results
        output_action[:, i:i+1] = action_i.unsqueeze(-1)
        output_action_log[:, i:i+1] = action_log_i
        
        # Update shifted actions for next agent (matches JAX mode="drop")
        if i < N - 1:
            # Convert to one-hot and place in next position
            action_one_hot = F.one_hot(action_i.squeeze(-1), A).float()  # [B, A]
            # Set next agent's input (indices 1: are action dimensions)
            shifted_actions[:, i+1, 1:] = action_one_hot
    
    # Convert outputs to match JAX format
    output_actions = output_action.squeeze(-1).long()  # [B, N] 
    output_action_log = output_action_log.squeeze(-1)  # [B, N]
    
    return output_actions, output_action_log, current_hstates


def continuous_autoregressive_act(
    decoder,
    obs_rep: torch.Tensor,
    hstates: Tuple[torch.Tensor, torch.Tensor],
    step_count: Optional[torch.Tensor],
    action_dim: int,
    key: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Autoregressive action generation for continuous actions - matches JAX exactly."""
    B, N, embed_dim = obs_rep.shape
    device = obs_rep.device
    
    # Initialize shifted actions - matches JAX exactly
    shifted_actions = torch.zeros(B, N, action_dim, device=device)
    output_action = torch.zeros(B, N, action_dim, device=device)
    output_action_log = torch.zeros(B, N, device=device)
    
    # Current hidden states
    current_hstates = hstates
    
    # Generate step count matching JAX
    if step_count is None:
        step_count = torch.arange(N, device=device, dtype=torch.float32)
        step_count = step_count[None, :, None].expand(B, N, 1)
    
    # Sequential autoregressive generation - matches JAX loop exactly
    for i in range(N):
        # Get current inputs
        current_action_input = shifted_actions[:, i:i+1]  # [B, 1, action_dim]
        obs_i = obs_rep[:, i:i+1]  # [B, 1, embed_dim]
        step_i = step_count[:, i:i+1]  # [B, 1, 1]
        
        # Forward pass through decoder
        act_mean, current_hstates = decoder.recurrent(
            current_action_input, obs_i, current_hstates, step_i
        )
        
        # Get action std from decoder log_std parameter
        action_std = torch.exp(decoder.log_std).clamp(min=1e-3)  # Min scale for stability
        
        # Sample action using normal distribution with tanh transform
        if key is not None:
            noise = torch.randn_like(act_mean, generator=key)
        else:
            noise = torch.randn_like(act_mean)
            
        # Apply tanh transformation for bounded actions (matches JAX TanhTransformedDistribution)
        raw_action = act_mean + action_std * noise
        action = torch.tanh(raw_action)
        
        # Compute log probability with tanh correction
        # Base normal log prob
        normal_log_prob = -0.5 * (noise ** 2 + np.log(2 * np.pi)) - torch.log(action_std)
        # Tanh correction term
        tanh_correction = torch.log(1 - action ** 2 + 1e-6)
        action_log_prob = (normal_log_prob + tanh_correction).sum(dim=-1)  # Sum over action dims
        
        # Store results
        output_action[:, i] = action.squeeze(1)
        output_action_log[:, i] = action_log_prob.squeeze(1)
        
        # Update shifted actions for next agent (matches JAX mode="drop")
        if i < N - 1:
            shifted_actions[:, i+1] = action.squeeze(1)
    
    return output_action, output_action_log, current_hstates