"""
encoder.py - Encoder Layer and Full Encoder Stack.

THE ENCODER'S JOB
-----------------
The encoder takes the source sequence (e.g., an English sentence) and
produces a sequence of contextual representations - one vector per
source position. These representations capture:
  - The meaning of each token
  - Its relationship to every other token in the sequence
  - Its position in the sequence

These representations are then used by the decoder via cross-attention.

ENCODER LAYER STRUCTURE
-----------------------
Each of the N=6 encoder layers applies, in order:

  1. Multi-Head Self-Attention (MHSA):
       Every position can attend to every other position.
       Q = K = V = x (all from the same input)
       With source padding mask applied.

  2. Residual + LayerNorm:
       x = LayerNorm(x + MHSA(x))

  3. Position-wise Feed-Forward Network (FFN):
       Applied independently to each position.

  4. Residual + LayerNorm:
       x = LayerNorm(x + FFN(x))

POST-NORM vs PRE-NORM:
The paper uses post-norm (shown above): normalize AFTER adding residual.
Pre-norm (used in many modern models): normalize BEFORE applying sub-layer:
  x = x + SubLayer(LayerNorm(x))

We implement post-norm to match the paper. Both are available here.

DROPOUT PLACEMENT:
Paper: dropout is applied to the output of each sub-layer, before
it is added to the sub-layer input. So:
  x = LayerNorm(x + Dropout(SubLayer(x)))

FULL ENCODER STACK
------------------
N identical encoder layers (N=6 in the paper), stacked sequentially.
Output of layer i becomes input to layer i+1.
The final output is passed to every decoder layer as memory (keys and values
for cross-attention).
"""

import torch
import torch.nn as nn
from attention import MultiHeadAttention
from feedforward import PositionwiseFeedForward
from config import cfg


class EncoderLayer(nn.Module):
    """
    A single encoder layer with MHSA + FFN, both with residual + LayerNorm.

    PARAMETERS (per layer):
    ----------------------------------------------
    MHSA parameters:
      W_Q, W_K, W_V, W_O: each [d_model, d_model] -> 4 x 512^2 = 1,048,576

    FFN parameters:
      W_1: [d_model, d_ff] = [512, 2048] = 1,048,576
      b_1: [d_ff] = 2048
      W_2: [d_ff, d_model] = [2048, 512] = 1,048,576
      b_2: [d_model] = 512

    LayerNorm parameters (x2, one per sub-layer):
      Y-gamma (gain): [d_model] = 512 (initialized to 1)
      B-beta (bias): [d_model] = 512 (initialized to 0)

    Total per layer: ~3.15M parameters
    Total for 6 layers: ~18.9M parameters (just encoder)
    """

    def __init__(self, d_model: int = cfg.d_model, n_heads: int = cfg.n_heads,
                 d_ff: int = cfg.d_ff, dropout: float = cfg.dropout):
        super().__init__()

        # Sub-layer 1: Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)

        # Sub-layer 2: Position-wise FFN
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # Layer Normalization (post-norm: applied AFTER residual addition)
        # Two separate LayerNorm instances — one per sub-layer.
        # LayerNorm(d_model) normalizes over the last dimension (d_model).
        # It has learned parameters Y-gamma and B-beta of shape [d_model].
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout: Applied to sub-layer output before adding to residual.
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through one encoder layer.

        Parameters
        ----------
        x : shape [B, S, d_model], Input sequence representations.
        mask : torch.Tensor or None, shape [B, 1, 1, S]
            Source padding mask. True = don't attend to that position.

        Returns
        -------
        x : torch.Tensor, shape [B, S, d_model]
            Updated sequence representations (same shape as input).

        Shape trace (B=2, S=5, d_model=512):
        ------------------------------------
        Input x:                [2, 5, 512]

        === Sub-layer 1: Self-Attention ===
        residual = x:           [2, 5, 512]
        attn = MHSA(x, x, x):  [2, 5, 512]  (self-attention: Q=K=V=x)
        dropout(attn):          [2, 5, 512]
        residual + attn:        [2, 5, 512]
        norm1(...):             [2, 5, 512]

        === Sub-layer 2: FFN ===
        residual = x:           [2, 5, 512]
        ffn_out = FFN(x):       [2, 5, 512]
        dropout(ffn_out):       [2, 5, 512]
        residual + ffn_out:     [2, 5, 512]
        norm2(...):             [2, 5, 512]

        Output:                 [2, 5, 512]  <- same as input!
        """

        # === Sub-layer 1: Multi-Head Self-Attention ===
        # Save residual (the "skip connection" input)
        residual = x

        # Self-attention: Q = K = V = x. The encoder can attend to ALL positions (no causal masking needed here), only padding positions are masked.
        attn_output = self.self_attention(x, x, x, mask) # shape: [B, S, d_model]

        # Dropout + residual connection + layer normalization (post-norm)
        x = self.norm1(residual + self.dropout(attn_output)) # shape: [B, S, d_model]

        # === Sub-layer 2: Position-wise FFN ====
        residual = x
        ffn_output = self.feed_forward(x) # shape: [B, S, d_model]
        x = self.norm2(residual + self.dropout(ffn_output)) # shape: [B, S, d_model]

        return x


class Encoder(nn.Module):
    """
    Full encoder: N stacked encoder layers.

    The input flows through N=6 identical (but separately parameterized)
    encoder layers. Each layer refines the contextual representations.

    After N layers, the final output is a rich contextual representation
    of each source token, informed by all other source tokens.

    This output (sometimes called "memory" or "encoder_output") is passed
    to every decoder layer as the source of keys and values for cross-attention.
    """

    def __init__(self, n_layers: int = cfg.n_layers, d_model: int = cfg.d_model,
                 n_heads: int = cfg.n_heads, d_ff: int = cfg.d_ff,
                 dropout: float = cfg.dropout):
        super().__init__()

        # Create N identical-but-independent encoder layers.
        # nn.ModuleList is used (not a Python list) so PyTorch registers the sub-modules and includes their parameters in model.parameters().
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Pass input through all N encoder layers.

        Parameters
        ----------
        x : torch.Tensor, shape [B, S, d_model], Input after embedding + positional encoding.
        mask : torch.Tensor or None, shape [B, 1, 1, S], Source padding mask.

        Returns
        -------
        x : torch.Tensor, shape [B, S, d_model]
            Final encoder output (contextual representations of all source tokens).
            This is passed to each decoder layer as 'memory'.
        """
        # Pass through each encoder layer sequentially
        for layer in self.layers:
            x = layer(x, mask) # shape stays [B, S, d_model] through all layers

        return x
