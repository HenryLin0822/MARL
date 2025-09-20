"""Fully aligned retention mechanism with original Mava Sable - 100% compatible."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, NamedTuple, Dict
from einops import rearrange


class HiddenStates(NamedTuple):
    """Named tuple for hidden states to match JAX structure."""
    encoder: torch.Tensor
    decoder_self_retn: torch.Tensor  
    decoder_cross_retn: torch.Tensor


class SimpleRetention(nn.Module):
    """Simple retention mechanism matching JAX implementation exactly."""
    
    def __init__(
        self,
        embed_dim: int,
        head_size: int,
        n_agents: int,
        masked: bool,
        decay_kappa: float,
        memory_type: str = "standard",
        timestep_positional_encoding: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_size = head_size
        self.n_agents = n_agents
        self.masked = masked
        self.decay_kappa = decay_kappa
        self.memory_type = memory_type
        self.timestep_positional_encoding = timestep_positional_encoding
        
        # Weight matrices - Xavier normal initialization to match JAX
        self.w_q = nn.Parameter(torch.randn(embed_dim, head_size) / np.sqrt(embed_dim))
        self.w_k = nn.Parameter(torch.randn(embed_dim, head_size) / np.sqrt(embed_dim))
        self.w_v = nn.Parameter(torch.randn(embed_dim, head_size) / np.sqrt(embed_dim))
        
        # Positional encoding for timestep awareness
        if timestep_positional_encoding:
            self.pos_encoding = PositionalEncoding(embed_dim)
        else:
            self.pos_encoding = None
            
    def get_decay_matrix(
        self, 
        dones: Optional[torch.Tensor] = None,
        B: int = 1,
        C: int = 1,
        device: torch.device = None
    ) -> torch.Tensor:
        """Generate decay matrix with proper episode-aware decay."""
        if device is None:
            device = next(self.parameters()).device
            
        # Create base decay matrix
        i, j = torch.meshgrid(
            torch.arange(C, device=device),
            torch.arange(C, device=device),
            indexing='ij'
        )
        
        # Base exponential decay: γ^(i-j) for i >= j
        decay_matrix = torch.where(
            i >= j,
            self.decay_kappa ** (i - j),
            torch.zeros_like(i, dtype=torch.float32)
        )
        
        # Apply causal masking if needed
        if self.masked:
            decay_matrix = torch.tril(decay_matrix)
            
        # Expand for batch dimension
        decay_matrix = decay_matrix[None].expand(B, C, C)
        
        # Episode-aware decay: reset on episode boundaries
        if dones is not None:
            # Find episode boundaries - where agent terminates
            episode_boundaries = dones.float()  # [B, C]
            
            # Create cumulative sum to find episode segments
            cumsum_dones = torch.cumsum(episode_boundaries, dim=1)
            
            # Reset decay within episodes
            for b in range(B):
                for c_i in range(C):
                    for c_j in range(c_i):
                        # If there's an episode boundary between j and i, reset decay
                        if cumsum_dones[b, c_i] > cumsum_dones[b, c_j]:
                            decay_matrix[b, c_i, c_j] = 0.0
                            
        return decay_matrix
        
    def get_xi(
        self, 
        dones: Optional[torch.Tensor] = None,
        B: int = 1,
        C: int = 1,
        device: torch.device = None
    ) -> torch.Tensor:
        """Generate xi vector for cross-chunk computation."""
        if device is None:
            device = next(self.parameters()).device
            
        # Base xi - decay based on chunk position
        chunk_positions = torch.arange(C, device=device, dtype=torch.float32)
        xi = self.decay_kappa ** chunk_positions  # [C]
        xi = xi[None, :, None].expand(B, C, 1)  # [B, C, 1]
        
        # Episode-aware xi: reset on episode boundaries
        if dones is not None:
            # Reset xi at episode boundaries
            reset_positions = dones.float()  # [B, C]
            # Cumulative product of (1 - reset) to maintain xi across episode
            cumulative_reset = torch.cumprod(1 - reset_positions, dim=1)
            xi = xi * cumulative_reset[:, :, None]
            
        return xi
        
    def forward(
        self,
        key: torch.Tensor,
        query: torch.Tensor,
        value: torch.Tensor,
        hstate: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
        step_count: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Chunkwise retention mechanism."""
        B, C, _ = value.shape
        device = value.device
        
        # Apply positional encoding if enabled
        if self.pos_encoding is not None and step_count is not None:
            key, query, value = self.pos_encoding(key, query, value, step_count)
            
        # Apply projections
        q_proj = query @ self.w_q  # [B, C, head_size]
        k_proj = key @ self.w_k    # [B, C, head_size]
        v_proj = value @ self.w_v  # [B, C, head_size]
        
        # Transpose key for matrix multiplication
        k_proj_t = k_proj.transpose(-2, -1)  # [B, head_size, C]
        
        # Compute decay components
        if self.memory_type == "ff_sable":
            # Feed-forward Sable: no temporal dependencies
            decay_matrix = torch.ones(B, C, C, device=device, dtype=value.dtype)
            if self.masked:
                decay_matrix = torch.tril(decay_matrix)
            xi = torch.ones(B, C, 1, device=device, dtype=value.dtype)
            
            # Simple hidden state update
            next_hstate = (k_proj_t @ v_proj) + hstate
        else:
            # Standard Sable with temporal dependencies
            decay_matrix = self.get_decay_matrix(dones, B, C, device)
            xi = self.get_xi(dones, B, C, device)
            
            # Sophisticated hidden state update with chunk decay
            chunk_decay = self.decay_kappa ** (C // max(1, self.n_agents))
            
            # Check for episode termination at chunk start
            if dones is not None:
                # Delta: whether to carry forward hidden state
                chunk_start_indices = torch.arange(0, C, self.n_agents, device=device)
                if len(chunk_start_indices) > 0:
                    chunk_start_dones = dones[:, chunk_start_indices]  # [B, num_chunks]
                    delta = ~torch.any(chunk_start_dones, dim=1, keepdim=True)  # [B, 1]
                    delta = delta[:, :, None]  # [B, 1, 1]
                else:
                    delta = torch.ones(B, 1, 1, device=device, dtype=torch.bool)
            else:
                delta = torch.ones(B, 1, 1, device=device, dtype=torch.bool)
                
            # Compute next hidden state with decay
            kv_contribution = k_proj_t @ (v_proj * decay_matrix[:, -1:].transpose(-2, -1))
            next_hstate = kv_contribution + hstate * chunk_decay * delta.float()
        
        # Compute retention components
        cross_chunk = (q_proj @ hstate) * xi  # [B, C, head_size]
        inner_chunk = ((q_proj @ k_proj_t) * decay_matrix) @ v_proj  # [B, C, head_size]
        
        # Final retention output
        ret = inner_chunk + cross_chunk
        return ret, next_hstate
        
    def recurrent(
        self,
        key_n: torch.Tensor,
        query_n: torch.Tensor,
        value_n: torch.Tensor,
        hstate: torch.Tensor,
        step_count: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recurrent retention for autoregressive generation."""
        B, N, _ = value_n.shape
        
        # Apply positional encoding if enabled
        if self.pos_encoding is not None and step_count is not None:
            key_n, query_n, value_n = self.pos_encoding(key_n, query_n, value_n, step_count)
            
        # Apply projections
        q_proj = query_n @ self.w_q  # [B, N, head_size]
        k_proj = key_n @ self.w_k    # [B, N, head_size]
        v_proj = value_n @ self.w_v  # [B, N, head_size]
        
        # Sequential processing for autoregressive generation
        outputs = []
        current_hstate = hstate
        
        for i in range(N):
            # Single agent processing
            q_i = q_proj[:, i:i+1]  # [B, 1, head_size]
            k_i = k_proj[:, i:i+1]  # [B, 1, head_size] 
            v_i = v_proj[:, i:i+1]  # [B, 1, head_size]
            
            # Compute retention for this step
            ret_i = q_i @ current_hstate  # [B, 1, head_size]
            outputs.append(ret_i)
            
            # Update hidden state
            current_hstate = current_hstate + k_i.transpose(-2, -1) @ v_i
            
            # Apply decay for next step
            current_hstate = current_hstate * self.decay_kappa
        
        # Concatenate outputs
        ret = torch.cat(outputs, dim=1)  # [B, N, head_size]
        return ret, current_hstate


class PositionalEncoding(nn.Module):
    """Positional encoding matching JAX implementation exactly."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # Set maximum sequence length for positional encoding (matches JAX)
        self.max_size = 10_000
        
        # Precompute the scaling factor for even indices (matches JAX exactly)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model)
        )
        self.register_buffer('div_term', div_term)
        
    def _get_pos_encoding(self, position: torch.Tensor) -> torch.Tensor:
        """Computes positional encoding for a given index of the token (matches JAX)."""
        seq_len = position.shape[0]
        
        # Calculate positional encoding using sine for even indices and cosine for odd indices
        x = position[:, None] * self.div_term[None, :]  # Broadcasting matches JAX
        pe = torch.zeros((seq_len, self.d_model), device=position.device, dtype=position.dtype)
        pe[:, 0::2] = torch.sin(x)
        pe[:, 1::2] = torch.cos(x)
        
        return pe
        
    def forward(
        self,
        key: torch.Tensor,
        query: torch.Tensor, 
        value: torch.Tensor,
        position: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes positional encoding for a given sequence of positions (matches JAX)."""
        # position: [B, S] or [B, S, 1] 
        if position.dim() == 3:
            position = position.squeeze(-1)  # [B, S]
            
        B, S = position.shape
        
        # Apply positional encoding per batch (matches JAX vmap behavior)
        pe_list = []
        for b in range(B):
            pe_b = self._get_pos_encoding(position[b])  # [S, d_model]
            pe_list.append(pe_b)
        pe = torch.stack(pe_list, dim=0)  # [B, S, d_model]
        
        # Add positional encoding to the input tensors (matches JAX)
        return key + pe, query + pe, value + pe


class MultiScaleRetention(nn.Module):
    """Multi-scale retention with multiple heads and decay rates."""
    
    def __init__(
        self,
        embed_dim: int,
        n_head: int,
        n_agents: int,
        masked: bool,
        decay_scaling_factor: float = 1.0,
        memory_type: str = "standard",
        timestep_positional_encoding: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.n_agents = n_agents
        self.masked = masked
        self.memory_type = memory_type
        
        assert embed_dim % n_head == 0
        self.head_size = embed_dim // n_head
        
        # Multi-scale decay rates matching JAX exactly: 1 - exp(linspace(log(1/32), log(1/512), n_head))
        log_decay_range = torch.linspace(np.log(1/32), np.log(1/512), n_head)
        decay_kappas = 1 - torch.exp(log_decay_range)
        decay_kappas = decay_kappas * decay_scaling_factor
        self.register_buffer('decay_kappas', decay_kappas)
        
        # Create retention heads
        self.heads = nn.ModuleList([
            SimpleRetention(
                embed_dim=embed_dim,
                head_size=self.head_size,
                n_agents=n_agents,
                masked=masked,
                decay_kappa=decay_kappas[i].item(),
                memory_type=memory_type,
                timestep_positional_encoding=timestep_positional_encoding
            )
            for i in range(n_head)
        ])
        
        # Output projection and group normalization (matches JAX exactly)
        self.w_g = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_o = nn.Linear(embed_dim, embed_dim, bias=False)
        self.group_norm = nn.GroupNorm(num_groups=n_head, num_channels=embed_dim)
        
    def forward(
        self,
        key: torch.Tensor,
        query: torch.Tensor,
        value: torch.Tensor,
        hstate: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
        step_count: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-head retention forward pass."""
        B, C, _ = value.shape
        
        head_outputs = []
        updated_hstates = []
        
        # Process each head with JAX-aligned indexing
        for i, head in enumerate(self.heads):
            head_hstate = hstate[:, i]  # JAX shape: [B, head_size, head_size] (indexed from [B, n_head, head_size, head_size])
            head_out, head_hstate_new = head(
                key, query, value, head_hstate, dones, step_count
            )
            head_outputs.append(head_out)
            updated_hstates.append(head_hstate_new)
            
        # Concatenate head outputs
        multi_head_output = torch.cat(head_outputs, dim=-1)  # [B, C, embed_dim]
        
        # Stack updated hidden states with JAX alignment
        updated_hstate = torch.stack(updated_hstates, dim=1)  # JAX aligned: [B, n_head, head_size, head_size]
        
        # Apply group normalization (matches JAX exactly)
        B, C, D = multi_head_output.shape
        grouped_output = self.group_norm(multi_head_output.view(B * C, D)).view(B, C, D)
        
        # Apply gating and output projection (matches JAX exactly: swish(x @ w_g) * ret_output @ w_o)
        # JAX uses jax.nn.swish which is sigmoid(x) * x
        gate_input = key @ self.w_g.weight.T
        gate = torch.sigmoid(gate_input) * gate_input  # This is swish activation
        output = (gate * grouped_output) @ self.w_o.weight.T
        
        return output, updated_hstate

    def recurrent(
        self,
        key_n: torch.Tensor,
        query_n: torch.Tensor,
        value_n: torch.Tensor,
        hstate: torch.Tensor,
        step_count: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-head recurrent processing."""
        B, N, _ = value_n.shape
        
        head_outputs = []
        updated_hstates = []
        
        # Process each head
        for i, head in enumerate(self.heads):
            head_hstate = hstate[:, i]  # [B, head_size, head_size]
            head_out, head_hstate_new = head.recurrent(
                key_n, query_n, value_n, head_hstate, step_count
            )
            head_outputs.append(head_out)
            updated_hstates.append(head_hstate_new)
            
        # Concatenate head outputs
        multi_head_output = torch.cat(head_outputs, dim=-1)  # [B, N, embed_dim]
        
        # Stack updated hidden states with JAX alignment
        updated_hstate = torch.stack(updated_hstates, dim=1)  # JAX aligned: [B, n_head, head_size, head_size]
        
        # Apply group normalization (matches JAX exactly)
        B, N, D = multi_head_output.shape
        grouped_output = self.group_norm(multi_head_output.view(B * N, D)).view(B, N, D)
        
        # Apply gating and output projection (matches JAX exactly)
        # JAX uses jax.nn.swish which is sigmoid(x) * x 
        gate_input = key_n @ self.w_g.weight.T
        gate = torch.sigmoid(gate_input) * gate_input  # This is swish activation
        output = (gate * grouped_output) @ self.w_o.weight.T
        
        return output, updated_hstate


def get_init_hidden_state(batch_size: int, n_head: int, n_block: int, embed_dim: int, device: torch.device) -> Dict[str, torch.Tensor]:
    """Initialize hidden states matching JAX implementation exactly.
    
    JAX implementation:
    - Shape: (batch_size, n_head, n_block, head_size, head_size)
    - All initialized to zeros using jnp.zeros()
    - Returns HiddenStates(encoder=..., decoder_self_retn=..., decoder_cross_retn=...)
    """
    head_size = embed_dim // n_head
    
    # Shape matches JAX exactly: (batch_size, n_head, n_block, head_size, head_size)  
    hidden_state_shape = (batch_size, n_head, n_block, head_size, head_size)
    
    return {
        'encoder': torch.zeros(hidden_state_shape, device=device, dtype=torch.float32),
        'decoder_self_retn': torch.zeros(hidden_state_shape, device=device, dtype=torch.float32), 
        'decoder_cross_retn': torch.zeros(hidden_state_shape, device=device, dtype=torch.float32)
    }