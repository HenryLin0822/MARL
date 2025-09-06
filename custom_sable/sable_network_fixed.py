"""Sable Network implementation in PyTorch with proper autoregressive action selection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
from einops import rearrange, einsum

from retention_advanced import MultiScaleRetention
from config import SableConfig
from autoregressive_utils import (
    discrete_train_decoder_fn,
    continuous_train_decoder_fn,
    discrete_autoregressive_act,
    continuous_autoregressive_act
)


class SwiGLU(nn.Module):
    """SwiGLU activation function."""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(input_dim, hidden_dim, bias=False)
        self.proj = nn.Linear(input_dim, hidden_dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.gate(x)) * self.proj(x)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class EncodeBlock(nn.Module):
    """Sable encoder block."""
    
    def __init__(self, config: SableConfig, n_agents_per_chunk: int):
        super().__init__()
        self.config = config
        
        self.ln1 = RMSNorm(config.embed_dim)
        self.ln2 = RMSNorm(config.embed_dim)
        
        self.retn = MultiScaleRetention(
            embed_dim=config.embed_dim,
            n_head=config.n_head,
            n_agents=n_agents_per_chunk,
            masked=False,  # Full retention for encoder
            decay_scaling_factor=config.decay_scaling_factor,
            memory_type="standard",
            timestep_positional_encoding=True
        )
        
        self.ffn = SwiGLU(config.embed_dim, config.embed_dim)
        
    def forward(
        self, 
        x: torch.Tensor, 
        hstate: torch.Tensor, 
        dones: Optional[torch.Tensor] = None,
        step_count: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder block."""
        ret, updated_hstate = self.retn(
            key=x, query=x, value=x, hstate=hstate, dones=dones, step_count=step_count
        )
        x = self.ln1(x + ret)
        output = self.ln2(x + self.ffn(x))
        return output, updated_hstate
        
    def recurrent(
        self,
        x: torch.Tensor,
        hstate: torch.Tensor,
        step_count: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recurrent mode for single-step processing."""
        ret, updated_hstate = self.retn.recurrent(
            key_n=x, query_n=x, value_n=x, hstate=hstate, step_count=step_count
        )
        x = self.ln1(x + ret)
        output = self.ln2(x + self.ffn(x))
        return output, updated_hstate


class Encoder(nn.Module):
    """Multi-block encoder."""
    
    def __init__(self, config: SableConfig, n_agents_per_chunk: int):
        super().__init__()
        self.config = config
        
        self.ln = RMSNorm(config.embed_dim)
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            RMSNorm(config.obs_dim),
            nn.Linear(config.obs_dim, config.embed_dim, bias=False),
            nn.GELU()
        )
        
        # Value head
        self.head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            RMSNorm(config.embed_dim),
            nn.Linear(config.embed_dim, 1)
        )
        
        # Encoder blocks
        self.blocks = nn.ModuleList([
            EncodeBlock(config, n_agents_per_chunk)
            for _ in range(config.n_block)
        ])
        
    def forward(
        self,
        obs: torch.Tensor,
        hstate: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
        step_count: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encoder forward pass."""
        B, C, _ = obs.shape
        updated_hstate = torch.zeros_like(hstate)
        obs_rep = self.obs_encoder(obs)
        
        # Generate step count if not provided - chunked processing
        if step_count is None:
            # For chunked processing: create step count based on agent ordering
            step_count = torch.arange(C, device=obs.device, dtype=torch.float32)
            step_count = step_count[None, :, None].expand(B, C, 1)  # [B, C, 1]
        
        # Apply encoder blocks
        for i, block in enumerate(self.blocks):
            block_hstate = hstate[:, i]  # [B, n_head, head_size, head_size]
            obs_rep, block_hstate_new = block(
                self.ln(obs_rep), block_hstate, dones, step_count
            )
            updated_hstate[:, i] = block_hstate_new
            
        value = self.head(obs_rep)
        return value, obs_rep, updated_hstate
        
    def recurrent(
        self,
        obs: torch.Tensor,
        hstate: torch.Tensor,
        step_count: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recurrent mode for single-step processing."""
        B, N, _ = obs.shape
        updated_hstate = torch.zeros_like(hstate)
        obs_rep = self.obs_encoder(obs)
        
        # Generate step count for recurrent mode
        if step_count is None:
            # For action selection: each agent gets sequential step count
            step_count = torch.arange(N, device=obs.device, dtype=torch.float32)
            step_count = step_count[None, :, None].expand(B, N, 1)  # [B, N, 1]
        
        # Apply encoder blocks in recurrent mode
        for i, block in enumerate(self.blocks):
            block_hstate = hstate[:, i]
            obs_rep, block_hstate_new = block.recurrent(
                self.ln(obs_rep), block_hstate, step_count
            )
            updated_hstate[:, i] = block_hstate_new
            
        value = self.head(obs_rep)
        return value, obs_rep, updated_hstate


class DecodeBlock(nn.Module):
    """Sable decoder block."""
    
    def __init__(self, config: SableConfig, n_agents_per_chunk: int):
        super().__init__()
        self.config = config
        
        self.ln1 = RMSNorm(config.embed_dim)
        self.ln2 = RMSNorm(config.embed_dim)
        self.ln3 = RMSNorm(config.embed_dim)
        
        # Self-retention over actions (masked)
        self.retn1 = MultiScaleRetention(
            embed_dim=config.embed_dim,
            n_head=config.n_head,
            n_agents=n_agents_per_chunk,
            masked=True,
            decay_scaling_factor=config.decay_scaling_factor,
            memory_type="standard",
            timestep_positional_encoding=True
        )
        
        # Cross-retention over observations and actions
        self.retn2 = MultiScaleRetention(
            embed_dim=config.embed_dim,
            n_head=config.n_head,
            n_agents=n_agents_per_chunk,
            masked=True,
            decay_scaling_factor=config.decay_scaling_factor,
            memory_type="standard",
            timestep_positional_encoding=True
        )
        
        self.ffn = SwiGLU(config.embed_dim, config.embed_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        obs_rep: torch.Tensor,
        hstates: Tuple[torch.Tensor, torch.Tensor],
        dones: Optional[torch.Tensor] = None,
        step_count: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Decoder block forward pass."""
        hs1, hs2 = hstates
        
        # Self-retention over actions
        ret, hs1_new = self.retn1(
            key=x, query=x, value=x, hstate=hs1, dones=dones, step_count=step_count
        )
        ret = self.ln1(x + ret)
        
        # Cross-retention over observations and actions
        ret2, hs2_new = self.retn2(
            key=ret, query=obs_rep, value=ret, hstate=hs2, dones=dones, step_count=step_count
        )
        y = self.ln2(obs_rep + ret2)
        output = self.ln3(y + self.ffn(y))
        
        return output, (hs1_new, hs2_new)
        
    def recurrent(
        self,
        x: torch.Tensor,
        obs_rep: torch.Tensor,
        hstates: Tuple[torch.Tensor, torch.Tensor],
        step_count: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Recurrent mode for single-step processing."""
        hs1, hs2 = hstates
        
        # Self-retention over actions
        ret, hs1_new = self.retn1.recurrent(
            key_n=x, query_n=x, value_n=x, hstate=hs1, step_count=step_count
        )
        ret = self.ln1(x + ret)
        
        # Cross-retention over observations and actions
        ret2, hs2_new = self.retn2.recurrent(
            key_n=ret, query_n=obs_rep, value_n=ret, hstate=hs2, step_count=step_count
        )
        y = self.ln2(obs_rep + ret2)
        output = self.ln3(y + self.ffn(y))
        
        return output, (hs1_new, hs2_new)


class Decoder(nn.Module):
    """Multi-block decoder."""
    
    def __init__(self, config: SableConfig, n_agents_per_chunk: int):
        super().__init__()
        self.config = config
        
        self.ln = RMSNorm(config.embed_dim)
        
        # Action encoder - handle both regular actions and actions with start tokens
        if config.action_space_type == "discrete":
            # For discrete: input can be [action_dim] or [action_dim + 1] (with start token)
            max_input_dim = max(config.action_dim, config.action_dim + 1)
            self.action_encoder = nn.Sequential(
                nn.Linear(max_input_dim, config.embed_dim, bias=False),
                nn.GELU()
            )
        else:
            # For continuous: input is [action_dim]
            self.action_encoder = nn.Sequential(
                nn.Linear(config.action_dim, config.embed_dim, bias=True),
                nn.GELU()
            )
        
        # Action head
        self.head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            RMSNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.action_dim)
        )
        
        # Log std for continuous actions
        if config.action_space_type == "continuous":
            self.log_std = nn.Parameter(torch.zeros(config.action_dim))
        else:
            self.log_std = None
            
        # Decoder blocks
        self.blocks = nn.ModuleList([
            DecodeBlock(config, n_agents_per_chunk)
            for _ in range(config.n_block)
        ])
        
    def forward(
        self,
        action: torch.Tensor,
        obs_rep: torch.Tensor,
        hstates: Tuple[torch.Tensor, torch.Tensor],
        dones: Optional[torch.Tensor] = None,
        step_count: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Decoder forward pass."""
        action_embeddings = self.action_encoder(action)
        x = self.ln(action_embeddings)
        
        # Initialize updated hidden states
        updated_hstates = (torch.zeros_like(hstates[0]), torch.zeros_like(hstates[1]))
        
        # Apply decoder blocks
        for i, block in enumerate(self.blocks):
            block_hstates = (hstates[0][:, i], hstates[1][:, i])
            x, block_hstates_new = block(
                x, obs_rep, block_hstates, dones, step_count
            )
            updated_hstates[0][:, i] = block_hstates_new[0]
            updated_hstates[1][:, i] = block_hstates_new[1]
            
        logits = self.head(x)
        return logits, updated_hstates
        
    def recurrent(
        self,
        action: torch.Tensor,
        obs_rep: torch.Tensor,
        hstates: Tuple[torch.Tensor, torch.Tensor],
        step_count: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Recurrent mode for single-step processing."""
        action_embeddings = self.action_encoder(action)
        x = self.ln(action_embeddings)
        
        # Initialize updated hidden states
        updated_hstates = (torch.zeros_like(hstates[0]), torch.zeros_like(hstates[1]))
        
        # Apply decoder blocks in recurrent mode
        for i, block in enumerate(self.blocks):
            block_hstates = (hstates[0][:, i], hstates[1][:, i])
            x, block_hstates_new = block.recurrent(
                x, obs_rep, block_hstates, step_count
            )
            updated_hstates[0][:, i] = block_hstates_new[0]
            updated_hstates[1][:, i] = block_hstates_new[1]
            
        logits = self.head(x)
        return logits, updated_hstates


class SableNetwork(nn.Module):
    """Complete Sable network with proper autoregressive action selection."""
    
    def __init__(self, config: SableConfig):
        super().__init__()
        self.config = config
        config.validate()  # Validate configuration
        
        self.n_agents_per_chunk = config.chunk_size
        
        # Decay kappas for hidden state decay
        decay_kappas = 1 - torch.exp(torch.linspace(
            np.log(1/32), np.log(1/512), config.n_head
        ))
        decay_kappas = decay_kappas * config.decay_scaling_factor
        # Shape: [1, n_head, 1, 1, 1] for broadcasting
        self.register_buffer('decay_kappas', 
                           decay_kappas.view(1, config.n_head, 1, 1, 1))
        
        self.encoder = Encoder(config, self.n_agents_per_chunk)
        self.decoder = Decoder(config, self.n_agents_per_chunk)
        
    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        hstates: Dict[str, torch.Tensor],
        dones: Optional[torch.Tensor] = None,
        legal_actions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward pass with proper shifted action sequences.
        
        Args:
            observation: [B, S, obs_dim] observations (S = T * N)
            action: [B, S] action indices for discrete or [B, S, action_dim] for continuous
            hstates: Dictionary of hidden states
            dones: [B, S] done flags
            legal_actions: [B, S, action_dim] legal action mask
            
        Returns:
            value: [B, S] value estimates
            log_prob: [B, S] action log probabilities
            entropy: [B, S] policy entropy
        """
        B, S, _ = observation.shape
        
        # Create step count for chunked processing
        step_count = torch.arange(S, device=observation.device, dtype=torch.float32)
        step_count = step_count[None, :, None].expand(B, S, 1)  # [B, S, 1]
        
        # Encode observations to get values and representations
        value, obs_rep, _ = self.encoder(
            observation, hstates['encoder'], dones, step_count
        )
        
        # Use proper autoregressive decoder training functions
        if self.config.action_space_type == "discrete":
            action_log_prob, entropy = discrete_train_decoder_fn(
                self.decoder,
                obs_rep,
                action.squeeze(-1) if action.dim() > 2 else action,
                legal_actions,
                (hstates['decoder_self_retn'], hstates['decoder_cross_retn']),
                dones,
                step_count,  # Pass step count for advanced processing
                self.config.n_agents,
                self.config.chunk_size
            )
        else:
            action_log_prob, entropy = continuous_train_decoder_fn(
                self.decoder,
                obs_rep,
                action,
                (hstates['decoder_self_retn'], hstates['decoder_cross_retn']),
                dones,
                step_count,  # Pass step count for advanced processing
                self.config.n_agents,
                self.config.chunk_size,
                self.config.action_dim
            )
        
        return value.squeeze(-1), action_log_prob, entropy
        
    def get_actions(
        self,
        observation: torch.Tensor,
        hstates: Dict[str, torch.Tensor],
        deterministic: bool = False,
        legal_actions: Optional[torch.Tensor] = None,
        key: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Autoregressive action selection - the core of Sable.
        
        Args:
            observation: [B, N, obs_dim] observations
            hstates: Dictionary of hidden states
            deterministic: Whether to sample deterministically (not used with generator)
            legal_actions: [B, N, action_dim] legal action mask
            key: Random generator for reproducible sampling
            
        Returns:
            actions: [B, N] discrete actions or [B, N, action_dim] continuous actions
            log_probs: [B, N] action log probabilities
            values: [B, N] value estimates
            updated_hstates: Updated hidden states
        """
        B, N, _ = observation.shape
        
        # Decay hidden states - key feature of retention mechanism
        decayed_hstates = {}
        for hs_name, hs in hstates.items():
            # Simple decay without broadcasting issues - use a single decay factor for now
            # In the original JAX implementation, this would be more sophisticated
            chunk_decay = self.decay_kappas.mean()  # Average decay across heads
            decayed_hstates[hs_name] = hs * chunk_decay
            
        # Encode observations to get values and representations
        values, obs_rep, updated_enc_hs = self.encoder.recurrent(
            observation.view(B, N, self.config.obs_dim),
            decayed_hstates['encoder']
        )
        
        # Autoregressive action generation - THE CORE SABLE FEATURE
        decoder_hstates = (
            decayed_hstates['decoder_self_retn'], 
            decayed_hstates['decoder_cross_retn']
        )
        
        if self.config.action_space_type == "discrete":
            actions, log_probs, updated_dec_hs = discrete_autoregressive_act(
                self.decoder,
                obs_rep,
                decoder_hstates,
                legal_actions,
                None,  # step_count
                key
            )
        else:
            actions, log_probs, updated_dec_hs = continuous_autoregressive_act(
                self.decoder,
                obs_rep,
                decoder_hstates,
                None,  # step_count
                self.config.action_dim,
                key
            )
        
        # Pack updated hidden states
        updated_hstates = {
            'encoder': updated_enc_hs,
            'decoder_self_retn': updated_dec_hs[0],
            'decoder_cross_retn': updated_dec_hs[1]
        }
        
        return actions, log_probs, values.squeeze(-1), updated_hstates
        
    def init_hidden_states(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Initialize hidden states for the network."""
        head_size = self.config.embed_dim // self.config.n_head
        
        return {
            'encoder': torch.zeros(
                batch_size, self.config.n_block, self.config.n_head, head_size, head_size,
                device=device
            ),
            'decoder_self_retn': torch.zeros(
                batch_size, self.config.n_block, self.config.n_head, head_size, head_size,
                device=device
            ),
            'decoder_cross_retn': torch.zeros(
                batch_size, self.config.n_block, self.config.n_head, head_size, head_size,
                device=device
            )
        }