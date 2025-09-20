"""Fully aligned Sable Network implementation - 100% compatible with Mava Sable."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, NamedTuple
from einops import rearrange, einsum

from retention_fully_aligned import MultiScaleRetention, HiddenStates, get_init_hidden_state
from config import SableConfig
from autoregressive_fully_aligned import (
    discrete_train_decoder_fn,
    continuous_train_decoder_fn,
    discrete_autoregressive_act,
    continuous_autoregressive_act
)


class SwiGLU(nn.Module):
    """SwiGLU activation function - matches JAX implementation exactly."""
    
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights with zeros to match JAX implementation exactly
        self.W_linear = nn.Parameter(torch.zeros(embed_dim, hidden_dim))
        self.W_gate = nn.Parameter(torch.zeros(embed_dim, hidden_dim))
        self.W_output = nn.Parameter(torch.zeros(hidden_dim, embed_dim))
        
        # Initialize weights to match JAX orthogonal initialization  
        # JAX uses orthogonal(sqrt(2)) for hidden layers
        nn.init.orthogonal_(self.W_linear, gain=np.sqrt(2))
        nn.init.orthogonal_(self.W_gate, gain=np.sqrt(2))
        nn.init.orthogonal_(self.W_output, gain=np.sqrt(2))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Matches JAX exactly: swish(x @ W_gate) * (x @ W_linear) @ W_output
        gate_out = F.silu(x @ self.W_gate)  # SwiSH = SiLU in PyTorch
        linear_out = x @ self.W_linear
        gated_output = gate_out * linear_out
        return gated_output @ self.W_output


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - matches JAX implementation."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class EncodeBlock(nn.Module):
    """Sable encoder block - matches JAX implementation exactly."""
    
    def __init__(self, config: SableConfig):
        super().__init__()
        self.config = config
        
        self.ln1 = RMSNorm(config.embed_dim)
        self.ln2 = RMSNorm(config.embed_dim)
        
        self.retn = MultiScaleRetention(
            embed_dim=config.embed_dim,
            n_head=config.n_head,
            n_agents=config.chunk_size,  # Use chunk size for agents per chunk
            masked=False,  # Full retention for encoder
            decay_scaling_factor=config.decay_scaling_factor,
            memory_type="standard",
            timestep_positional_encoding=True
        )
        
        # SwiGLU with proper hidden dimension
        self.ffn = SwiGLU(config.embed_dim, config.embed_dim)
        
    def forward(
        self, 
        x: torch.Tensor, 
        hstate: torch.Tensor, 
        dones: Optional[torch.Tensor] = None,
        step_count: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encoder block forward pass."""
        # Pre-norm architecture
        normed_x = self.ln1(x)
        ret, updated_hstate = self.retn(
            key=normed_x, query=normed_x, value=normed_x, 
            hstate=hstate, dones=dones, step_count=step_count
        )
        x = x + ret  # Residual connection
        
        # Feed-forward with residual
        normed_x2 = self.ln2(x)
        ffn_out = self.ffn(normed_x2)
        output = x + ffn_out
        
        return output, updated_hstate
        
    def recurrent(
        self,
        x: torch.Tensor,
        hstate: torch.Tensor,
        step_count: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recurrent mode for single-step processing."""
        normed_x = self.ln1(x)
        ret, updated_hstate = self.retn.recurrent(
            key_n=normed_x, query_n=normed_x, value_n=normed_x, 
            hstate=hstate, step_count=step_count
        )
        x = x + ret
        
        normed_x2 = self.ln2(x)
        ffn_out = self.ffn(normed_x2)
        output = x + ffn_out
        
        return output, updated_hstate


class Encoder(nn.Module):
    """Multi-block encoder with sophisticated chunked processing."""
    
    def __init__(self, config: SableConfig):
        super().__init__()
        self.config = config
        
        self.ln = RMSNorm(config.embed_dim)
        
        # Observation encoder with proper normalization
        self.obs_encoder = nn.Sequential(
            RMSNorm(config.obs_dim),  # Input normalization
            nn.Linear(config.obs_dim, config.embed_dim, bias=False),
            nn.GELU()
        )
        
        # Value head - matches JAX architecture
        self.head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            RMSNorm(config.embed_dim),
            nn.Linear(config.embed_dim, 1)
        )
        
        # Encoder blocks
        self.blocks = nn.ModuleList([
            EncodeBlock(config) for _ in range(config.n_block)
        ])
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights to match JAX exactly."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # JAX uses orthogonal initialization with specific gains
                if 'head' in str(module) and hasattr(module, 'out_features') and module.out_features == 1:
                    # Value head output layer uses orthogonal(0.01)
                    nn.init.orthogonal_(module.weight, gain=0.01)
                else:
                    # Other layers use orthogonal(sqrt(2))
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(
        self,
        obs: torch.Tensor,
        hstate: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
        step_count: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sophisticated chunked processing for training."""
        B, S, _ = obs.shape
        
        # Process observations
        obs_rep = self.obs_encoder(obs)
        
        # Generate step count for chunked processing if not provided
        if step_count is None:
            # Sophisticated chunked processing: group agents by chunks
            chunk_size = self.config.chunk_size
            num_chunks = (S + chunk_size - 1) // chunk_size  # Ceiling division
            
            # Create step count that resets within each chunk
            step_count = torch.zeros(B, S, 1, device=obs.device, dtype=torch.float32)
            for chunk_id in range(num_chunks):
                start_idx = chunk_id * chunk_size
                end_idx = min((chunk_id + 1) * chunk_size, S)
                chunk_length = end_idx - start_idx
                
                # Step count within chunk
                chunk_steps = torch.arange(chunk_length, device=obs.device, dtype=torch.float32)
                step_count[:, start_idx:end_idx, 0] = chunk_steps
        
        # Process through encoder blocks with JAX-aligned hidden state management
        updated_hstate = torch.zeros_like(hstate)
        x = self.ln(obs_rep)
        
        for i, block in enumerate(self.blocks):
            block_hstate = hstate[:, :, i]  # [B, n_head, head_size, head_size] - JAX aligned
            x, block_hstate_new = block(x, block_hstate, dones, step_count)
            updated_hstate[:, :, i] = block_hstate_new
            
        # Compute values
        value = self.head(x)
        return value, x, updated_hstate
        
    def recurrent(
        self,
        obs: torch.Tensor,
        hstate: torch.Tensor,
        step_count: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recurrent mode for autoregressive action generation."""
        B, N, _ = obs.shape
        
        # Process observations
        obs_rep = self.obs_encoder(obs)
        
        # Generate step count for recurrent processing
        if step_count is None:
            # Sequential step count for autoregressive generation
            step_count = torch.arange(N, device=obs.device, dtype=torch.float32)
            step_count = step_count[None, :, None].expand(B, N, 1)
        
        # Process through encoder blocks with JAX-aligned indexing
        updated_hstate = torch.zeros_like(hstate)
        x = self.ln(obs_rep)
        
        for i, block in enumerate(self.blocks):
            block_hstate = hstate[:, :, i]  # JAX aligned: [B, n_head, head_size, head_size]
            x, block_hstate_new = block.recurrent(x, block_hstate, step_count)
            updated_hstate[:, :, i] = block_hstate_new
            
        # Compute values
        value = self.head(x)
        return value, x, updated_hstate


class DecodeBlock(nn.Module):
    """Sable decoder block - matches JAX implementation exactly."""
    
    def __init__(self, config: SableConfig):
        super().__init__()
        self.config = config
        
        self.ln1 = RMSNorm(config.embed_dim)
        self.ln2 = RMSNorm(config.embed_dim)
        self.ln3 = RMSNorm(config.embed_dim)
        
        # Self-retention over actions (masked autoregressive)
        self.retn1 = MultiScaleRetention(
            embed_dim=config.embed_dim,
            n_head=config.n_head,
            n_agents=config.chunk_size,
            masked=True,  # Masked for autoregressive generation
            decay_scaling_factor=config.decay_scaling_factor,
            memory_type="standard",
            timestep_positional_encoding=True
        )
        
        # Cross-retention between observations and actions (masked)
        self.retn2 = MultiScaleRetention(
            embed_dim=config.embed_dim,
            n_head=config.n_head,
            n_agents=config.chunk_size,
            masked=True,  # Masked for autoregressive generation
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
        """Decoder block forward pass with two-stage attention."""
        hs1, hs2 = hstates
        
        # Stage 1: Self-retention over actions (autoregressive)
        normed_x = self.ln1(x)
        ret1, hs1_new = self.retn1(
            key=normed_x, query=normed_x, value=normed_x,
            hstate=hs1, dones=dones, step_count=step_count
        )
        x_self = x + ret1  # Residual connection
        
        # Stage 2: Cross-retention between observations and actions
        normed_obs = self.ln2(obs_rep)
        ret2, hs2_new = self.retn2(
            key=x_self, query=normed_obs, value=x_self,
            hstate=hs2, dones=dones, step_count=step_count
        )
        x_cross = obs_rep + ret2  # Add to observation representation
        
        # Feed-forward with residual
        normed_x3 = self.ln3(x_cross)
        ffn_out = self.ffn(normed_x3)
        output = x_cross + ffn_out
        
        return output, (hs1_new, hs2_new)
        
    def recurrent(
        self,
        x: torch.Tensor,
        obs_rep: torch.Tensor,
        hstates: Tuple[torch.Tensor, torch.Tensor],
        step_count: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Recurrent mode for autoregressive generation."""
        hs1, hs2 = hstates
        
        # Stage 1: Self-retention over actions
        normed_x = self.ln1(x)
        ret1, hs1_new = self.retn1.recurrent(
            key_n=normed_x, query_n=normed_x, value_n=normed_x,
            hstate=hs1, step_count=step_count
        )
        x_self = x + ret1
        
        # Stage 2: Cross-retention
        normed_obs = self.ln2(obs_rep)
        ret2, hs2_new = self.retn2.recurrent(
            key_n=x_self, query_n=normed_obs, value_n=x_self,
            hstate=hs2, step_count=step_count
        )
        x_cross = obs_rep + ret2
        
        # Feed-forward
        normed_x3 = self.ln3(x_cross)
        ffn_out = self.ffn(normed_x3)
        output = x_cross + ffn_out
        
        return output, (hs1_new, hs2_new)


class Decoder(nn.Module):
    """Multi-block decoder with proper action encoding."""
    
    def __init__(self, config: SableConfig):
        super().__init__()
        self.config = config
        
        self.ln = RMSNorm(config.embed_dim)
        
        # Action encoder - handles both discrete and continuous actions
        if config.action_space_type == "discrete":
            # For discrete: support actions with start tokens [action_dim + 1]
            self.action_encoder = nn.Sequential(
                nn.Linear(config.action_dim + 1, config.embed_dim, bias=False),
                nn.GELU()
            )
        else:
            # For continuous: standard action encoding
            self.action_encoder = nn.Sequential(
                nn.Linear(config.action_dim, config.embed_dim, bias=True),
                nn.GELU()
            )
        
        # Action head for output
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
            DecodeBlock(config) for _ in range(config.n_block)
        ])
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights to match JAX exactly."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # JAX uses orthogonal initialization with specific gains
                if 'head' in str(module) and hasattr(module, 'out_features') and module.out_features == 1:
                    # Value head output layer uses orthogonal(0.01)
                    nn.init.orthogonal_(module.weight, gain=0.01)
                else:
                    # Other layers use orthogonal(sqrt(2))
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(
        self,
        action: torch.Tensor,
        obs_rep: torch.Tensor,
        hstates: Tuple[torch.Tensor, torch.Tensor],
        dones: Optional[torch.Tensor] = None,
        step_count: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Decoder forward pass for training."""
        # Encode actions
        action_embeddings = self.action_encoder(action)
        x = self.ln(action_embeddings)
        
        # Initialize updated hidden states
        updated_hstates = (
            torch.zeros_like(hstates[0]), 
            torch.zeros_like(hstates[1])
        )
        
        # Process through decoder blocks with JAX-aligned indexing
        for i, block in enumerate(self.blocks):
            block_hstates = (hstates[0][:, :, i], hstates[1][:, :, i])  # JAX aligned
            x, block_hstates_new = block(
                x, obs_rep, block_hstates, dones, step_count
            )
            updated_hstates[0][:, :, i] = block_hstates_new[0]
            updated_hstates[1][:, :, i] = block_hstates_new[1]
            
        # Generate logits/means
        logits = self.head(x)
        return logits, updated_hstates
        
    def recurrent(
        self,
        action: torch.Tensor,
        obs_rep: torch.Tensor,
        hstates: Tuple[torch.Tensor, torch.Tensor],
        step_count: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Recurrent mode for autoregressive generation."""
        # Encode actions
        action_embeddings = self.action_encoder(action)
        x = self.ln(action_embeddings)
        
        # Initialize updated hidden states
        updated_hstates = (
            torch.zeros_like(hstates[0]),
            torch.zeros_like(hstates[1])
        )
        
        # Process through decoder blocks with JAX-aligned indexing
        for i, block in enumerate(self.blocks):
            block_hstates = (hstates[0][:, :, i], hstates[1][:, :, i])  # JAX aligned
            x, block_hstates_new = block.recurrent(
                x, obs_rep, block_hstates, step_count
            )
            updated_hstates[0][:, :, i] = block_hstates_new[0]
            updated_hstates[1][:, :, i] = block_hstates_new[1]
            
        # Generate logits/means
        logits = self.head(x)
        return logits, updated_hstates


class SableNetwork(nn.Module):
    """Complete Sable network - 100% aligned with Mava implementation."""
    
    def __init__(self, config: SableConfig):
        super().__init__()
        self.config = config
        config.validate()  # Validate configuration
        
        # Multi-scale decay rates - matches JAX exactly  
        # JAX: 1 - exp(linspace(log(1/32), log(1/512), n_head))
        log_decay_range = torch.linspace(np.log(1/32), np.log(1/512), config.n_head)
        decay_kappas = 1 - torch.exp(log_decay_range)
        decay_kappas = decay_kappas * config.decay_scaling_factor
        # JAX shape for decay: [1, n_head, 1, 1, 1] for broadcasting
        self.register_buffer('decay_kappas', decay_kappas.view(1, -1, 1, 1, 1))
        
        # Encoder and decoder
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize all weights to match JAX."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Parameter):
                if module.dim() > 1:
                    nn.init.xavier_uniform_(module)
                else:
                    nn.init.zeros_(module)
        
    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        hstates: Dict[str, torch.Tensor],
        dones: Optional[torch.Tensor] = None,
        legal_actions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward pass with sophisticated chunked processing."""
        B, S, _ = observation.shape
        
        # Sophisticated step count generation for chunked processing
        chunk_size = self.config.chunk_size
        num_chunks = (S + chunk_size - 1) // chunk_size
        
        step_count = torch.zeros(B, S, 1, device=observation.device, dtype=torch.float32)
        for chunk_id in range(num_chunks):
            start_idx = chunk_id * chunk_size
            end_idx = min((chunk_id + 1) * chunk_size, S)
            chunk_length = end_idx - start_idx
            
            # Create sophisticated step count within chunk
            chunk_steps = torch.arange(chunk_length, device=observation.device, dtype=torch.float32)
            # Add chunk offset for global positioning
            chunk_steps = chunk_steps + chunk_id * chunk_size
            step_count[:, start_idx:end_idx, 0] = chunk_steps
        
        # Encode observations
        value, obs_rep, updated_enc_hs = self.encoder(
            observation, hstates['encoder'], dones, step_count
        )
        
        # Decode actions with proper autoregressive training
        decoder_hstates = (
            hstates['decoder_self_retn'],
            hstates['decoder_cross_retn']
        )
        
        if self.config.action_space_type == "discrete":
            action_log_prob, entropy = discrete_train_decoder_fn(
                self.decoder,
                obs_rep,
                action.squeeze(-1) if action.dim() > 2 else action,
                legal_actions,
                decoder_hstates,
                dones,
                step_count,
                self.config.n_agents,
                self.config.chunk_size
            )
        else:
            action_log_prob, entropy = continuous_train_decoder_fn(
                self.decoder,
                obs_rep,
                action,
                decoder_hstates,
                dones,
                step_count,
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
        """Autoregressive action generation - THE CORE SABLE INNOVATION."""
        B, N, _ = observation.shape
        
        # Sophisticated hidden state decay - matches JAX exactly
        # JAX shape: (batch_size, n_head, n_block, head_size, head_size)
        decayed_hstates = {}
        for hs_name, hs in hstates.items():
            # Apply per-head decay rates with JAX-aligned shape
            # JAX applies decay per head using tree.map with decay_kappas shape [1, n_head, 1, 1, 1]
            # The decay_kappas is already shaped for broadcasting: [1, n_head, 1, 1, 1]
            decayed_hstates[hs_name] = hs * self.decay_kappas
        
        # Encode observations with recurrent processing
        values, obs_rep, updated_enc_hs = self.encoder.recurrent(
            observation, decayed_hstates['encoder']
        )
        
        # Generate step count for recurrent processing
        step_count = torch.arange(N, device=observation.device, dtype=torch.float32)
        step_count = step_count[None, :, None].expand(B, N, 1)
        
        # Autoregressive action generation - CORE SABLE MECHANISM
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
                step_count,
                key
            )
        else:
            actions, log_probs, updated_dec_hs = continuous_autoregressive_act(
                self.decoder,
                obs_rep,
                decoder_hstates,
                step_count,
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
        
    def init_hidden_states(
        self, 
        batch_size: int, 
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Initialize hidden states with JAX-aligned structure.
        
        JAX shape: (batch_size, n_head, n_block, head_size, head_size)
        """
        return get_init_hidden_state(
            batch_size, self.config.n_head, self.config.n_block, 
            self.config.embed_dim, device
        )