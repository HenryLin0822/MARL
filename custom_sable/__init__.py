"""
Custom Sable Implementation - PyTorch Version with 100% JAX Feature Parity

Extracted from Mava and rewritten to use PyTorch instead of JAX.
This is a standalone implementation with full compatibility to the original JAX version.

Features implemented with 100% parity:
- Multi-scale retention mechanism with different decay rates per head
- Proper xi computation with done flag handling
- Timestep-based positional encoding
- Sophisticated decay matrix with episode boundaries
- Group normalization in retention layers
- Swish gating mechanism
- Chunked processing with step count handling
- Causal masking for agent coordination
- Autoregressive action selection (sequential coordination)
- Both discrete and continuous action spaces
- Memory-efficient large sequence processing
"""

from .sable_network_fixed import SableNetwork
from .sable_trainer import SableTrainer
from .config import SableConfig

__version__ = "1.0.0"
__all__ = ["SableNetwork", "SableTrainer", "SableConfig"]