"""Multi-Scale Retention mechanism for Sable."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np
from einops import rearrange, einsum


class SimpleRetention(nn.Module):
    """Simple retention mechanism for Sable.
    
    Based on the RetNet paper and implementation.
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        head_size: int, 
        n_agents: int, 
        masked: bool = False,
        decay_kappa: float = 0.9
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_size = head_size
        self.n_agents = n_agents
        self.masked = masked
        self.decay_kappa = decay_kappa
        
        # Linear projections
        self.w_q = nn.Linear(embed_dim, head_size, bias=False)
        self.w_k = nn.Linear(embed_dim, head_size, bias=False)
        self.w_v = nn.Linear(embed_dim, head_size, bias=False)
        
        # Initialize weights
        nn.init.normal_(self.w_q.weight, std=1/np.sqrt(embed_dim))
        nn.init.normal_(self.w_k.weight, std=1/np.sqrt(embed_dim))
        nn.init.normal_(self.w_v.weight, std=1/np.sqrt(embed_dim))
        
    def _causal_mask(self, decay_matrix: torch.Tensor) -> torch.Tensor:
        """Apply causal mask to decay matrix."""
        if self.masked:
            B, C, _ = decay_matrix.shape
            mask = torch.triu(torch.ones(C, C, device=decay_matrix.device), diagonal=1)
            decay_matrix = decay_matrix.masked_fill(mask.bool(), 0)
        return decay_matrix
        
    def get_decay_matrix(self, dones: torch.Tensor) -> torch.Tensor:
        """Compute decay matrix based on done flags."""
        B, C = dones.shape
        # Create position matrix
        positions = torch.arange(C, device=dones.device).float()
        pos_diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # (C, C)
        
        # Apply decay
        decay_matrix = self.decay_kappa ** torch.clamp(pos_diff, min=0)
        decay_matrix = decay_matrix.unsqueeze(0).expand(B, -1, -1)
        
        # Apply causal mask
        decay_matrix = self._causal_mask(decay_matrix)
        
        return decay_matrix
        
    def get_xi(self, dones: torch.Tensor) -> torch.Tensor:
        """Compute xi values for cross-chunk retention."""
        B, C = dones.shape
        xi = torch.ones(B, C, 1, device=dones.device)
        return xi
    
    def forward(
        self, 
        key: torch.Tensor, 
        query: torch.Tensor, 
        value: torch.Tensor, 
        hstate: torch.Tensor, 
        dones: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for retention mechanism.
        
        Args:
            key, query, value: [B, C, embed_dim] input tensors
            hstate: [B, head_size, head_size] hidden state
            dones: [B, C] done flags (optional)
            
        Returns:
            ret: [B, C, head_size] retention output
            next_hstate: [B, head_size, head_size] updated hidden state
        """
        B, C, _ = value.shape
        
        # Apply projections
        q_proj = self.w_q(query)  # [B, C, head_size]
        k_proj = self.w_k(key)    # [B, C, head_size]  
        v_proj = self.w_v(value)  # [B, C, head_size]
        
        # Transpose k for matrix multiplication
        k_proj_t = rearrange(k_proj, 'b c h -> b h c')  # [B, head_size, C]
        
        if dones is None:
            # Feed-forward mode (no temporal dependencies)
            decay_matrix = torch.ones(B, C, C, device=value.device)
            decay_matrix = self._causal_mask(decay_matrix)
            xi = torch.ones(B, C, 1, device=value.device)
            next_hstate = einsum(k_proj_t, v_proj, 'b h c, b c d -> b h d') + hstate
        else:
            # Recurrent mode with decay
            decay_matrix = self.get_decay_matrix(dones)
            xi = self.get_xi(dones)
            chunk_decay = self.decay_kappa ** (C // self.n_agents)
            
            # Check if any agents are done in this chunk
            agent_dones = dones[:, ::self.n_agents]  # Sample every n_agents
            delta = (~torch.any(agent_dones, dim=1)).float().unsqueeze(-1).unsqueeze(-1)
            
            # Update hidden state
            last_decay = decay_matrix[:, -1:, :]  # [B, 1, C]
            weighted_v = v_proj * rearrange(last_decay, 'b 1 c -> b c 1')
            next_hstate = einsum(k_proj_t, weighted_v, 'b h c, b c d -> b h d') + hstate * chunk_decay * delta
        
        # Compute retention output
        cross_chunk = einsum(q_proj, hstate, 'b c h, b h d -> b c d') * xi  # [B, C, head_size]
        inner_chunk = einsum(q_proj, k_proj_t, decay_matrix, v_proj, 
                           'b c1 h, b h c2, b c1 c2, b c2 d -> b c1 d')
        
        ret = inner_chunk + cross_chunk
        return ret, next_hstate


class MultiScaleRetention(nn.Module):
    """Multi-scale retention with multiple heads."""
    
    def __init__(
        self,
        embed_dim: int,
        n_head: int,
        n_agents: int,
        masked: bool = False,
        decay_scaling_factor: float = 1.0
    ):
        super().__init__()
        assert embed_dim % n_head == 0, "embed_dim must be divisible by n_head"
        
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.head_size = embed_dim // n_head
        self.n_agents = n_agents
        self.masked = masked
        
        # Decay kappa for each head (different time scales)
        decay_kappas = 1 - torch.exp(torch.linspace(
            np.log(1/32), np.log(1/512), n_head
        ))
        decay_kappas = decay_kappas * decay_scaling_factor
        self.register_buffer('decay_kappas', decay_kappas)
        
        # Create retention heads
        self.heads = nn.ModuleList([
            SimpleRetention(
                embed_dim=embed_dim,
                head_size=self.head_size,
                n_agents=n_agents,
                masked=masked,
                decay_kappa=self.decay_kappas[i].item()
            )
            for i in range(n_head)
        ])
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
    def forward(
        self,
        key: torch.Tensor,
        query: torch.Tensor,
        value: torch.Tensor,
        hstate: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
        step_count: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-head retention forward pass.
        
        Args:
            key, query, value: [B, C, embed_dim] 
            hstate: [B, n_head, head_size, head_size]
            dones: [B, C] done flags
            step_count: [B] step counts (unused in current implementation)
            
        Returns:
            output: [B, C, embed_dim]
            updated_hstate: [B, n_head, head_size, head_size]
        """
        B, C, _ = value.shape
        head_outputs = []
        updated_hstates = []
        
        for i, head in enumerate(self.heads):
            head_hstate = hstate[:, i]  # [B, head_size, head_size]
            head_out, head_hstate_new = head(key, query, value, head_hstate, dones)
            head_outputs.append(head_out)
            updated_hstates.append(head_hstate_new)
        
        # Concatenate head outputs
        output = torch.cat(head_outputs, dim=-1)  # [B, C, embed_dim]
        output = self.out_proj(output)
        
        # Stack updated hidden states
        updated_hstate = torch.stack(updated_hstates, dim=1)  # [B, n_head, head_size, head_size]
        
        return output, updated_hstate
        
    def recurrent(
        self,
        key_n: torch.Tensor,
        query_n: torch.Tensor,
        value_n: torch.Tensor,
        hstate: torch.Tensor,
        step_count: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recurrent mode for single-step processing."""
        # For recurrent mode, we process one step at a time
        return self.forward(key_n, query_n, value_n, hstate, dones=None, step_count=step_count)