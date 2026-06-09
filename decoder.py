"""
decoder.py - Decoder Layer and Full Decoder Stack.

THE DECODER'S JOB
-----------------
The decoder generates the target sequence (e.g., a French translation) one
token at a time, conditioned on:
  1. The encoder output (source context - what we're translating FROM)
  2. Previously generated tokens (target context - what we've generated SO FAR)

During training, all target tokens are provided simultaneously (teacher
forcing), with the causal mask preventing future peeking.

During inference, tokens are generated autoregressively: the decoder is
called once per output token.

DECODER LAYER STRUCTURE
-----------------------
Each of the N=6 decoder layers applies, in order:

  1. Masked Multi-Head Self-Attention (causal):
       Each target position attends to all PREVIOUS target positions.
       Q = K = V = target embeddings (with causal + padding mask)

  2. Residual + LayerNorm

  3. Multi-Head Cross-Attention (encoder-decoder attention):
       Each target position attends to ALL source positions.
       Q = output from sub-layer 1 (decoder state)
       K = V = encoder_output (source representations), (with source padding mask)

  4. Residual + LayerNorm

  5. Position-wise FFN

  6. Residual + LayerNorm

KEY DIFFERENCE FROM ENCODER:
  - Has 3 sub-layers (encoder has 2)
  - Sub-layer 1 uses CAUSAL masking (encoder self-attention has none)
  - Sub-layer 2 is CROSS-attention (encoder only has self-attention)
  - The encoder output flows into EVERY decoder layer (not just the first)
"""

import torch
import torch.nn as nn
from attention import MultiHeadAttention
from feedforward import PositionwiseFeedForward
from config import cfg


class DecoderLayer(nn.Module):
    """
    A single decoder layer with 3 sub-layers:
      1. Masked self-attention
      2. Cross-attention (encoder-decoder)
      3. FFN

    All with residual connections and layer normalization.
    """

    def __init__(self, d_model: int = cfg.d_model, n_heads: int = cfg.n_heads,
                 d_ff: int = cfg.d_ff, dropout: float = cfg.dropout):
        super().__init__()

        # === Sub-layer 1: Masked Multi-Head Self-Attention ===
        # Operates on the target sequence. Uses causal masking: position i can only attend to positions 0..i.
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)

        # === Sub-layer 2: Multi-Head Cross-Attention (Encoder-Decoder) ===
        # Q comes from the decoder (previous sub-layer output). K and V come from the encoder output.
        # Allows each target position to look at all source positions.
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)

        # === Sub-layer 3: Position-wise FFN ===
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # === Layer Normalization (one per sub-layer) ===
        self.norm1 = nn.LayerNorm(d_model)  # after masked self-attention
        self.norm2 = nn.LayerNorm(d_model)  # after cross-attention
        self.norm3 = nn.LayerNorm(d_model)  # after FFN

        # === Dropout ===
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through one decoder layer.

        Parameters
        ----------
        tgt : torch.Tensor, shape [B, S_tgt, d_model]
            Target sequence representations (after embedding + PE for first layer, or output of previous decoder layer for subsequent layers).

        memory : torch.Tensor, shape [B, S_src, d_model]
            Encoder output. Same for all decoder layers (passed in every time). This is what allows the decoder to "look at" the source sequence.

        tgt_mask : torch.Tensor or None, shape [B, 1, S_tgt, S_tgt]
            Combined causal + padding mask for the target sequence. True = block. Prevents attending to future or padding positions.

        memory_mask : torch.Tensor or None, shape [B, 1, 1, S_src]
            Padding mask for the source sequence. Prevents attending to padding tokens in the encoder output.

        Returns
        -------
        tgt : torch.Tensor, shape [B, S_tgt, d_model]
            Updated target representations.

        Complete Shape Trace:
        ---------------------
        B=2, S_tgt=4, S_src=5, d_model=512, n_heads=8, d_k=64
        tgt input:   [2, 4, 512]
        memory:      [2, 5, 512]  (from encoder)

        === Sub-layer 1: Masked Self-Attention ===
        residual:           [2, 4, 512]
        Q=K=V=tgt -> MHSA -> [2, 4, 512]
          (Q: [2,8,4,64], K: [2,8,4,64], V: [2,8,4,64])
          (scores: [2,8,4,4], masked by tgt_mask)
          (output: [2,8,4,64] -> combined -> [2,4,512])
        dropout:            [2, 4, 512]
        residual + out:     [2, 4, 512]
        norm1:              [2, 4, 512]

        === Sub-layer 2: Cross-Attention ===
        residual:                [2, 4, 512]
        Q=tgt, K=V=memory -> MHA:
          (Q: [2,8,4,64])        <- from decoder
          (K: [2,8,5,64])        <- from encoder!
          (V: [2,8,5,64])        <- from encoder!
          (scores: [2,8,4,5] masked by memory_mask)   <- 4 target queries attending to 5 source keys!
          (output: [2,8,4,64] -> combined -> [2,4,512])
        dropout:                 [2, 4, 512]
        residual + out:          [2, 4, 512]
        norm2:                   [2, 4, 512]

        === Sub-layer 3: FFN ===
        residual:           [2, 4, 512]
        FFN:                [2, 4, 512]
        dropout:            [2, 4, 512]
        residual + out:     [2, 4, 512]
        norm3:              [2, 4, 512]

        Output:             [2, 4, 512]  <- same shape as tgt input
        """

        # === Sub-layer 1: Masked Multi-Head Self-Attention ===
        residual = tgt
        # Self-attention: Q = K = V = tgt, tgt_mask prevents attending to future target positions (causal) and padding positions
        self_attn_out = self.self_attention(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(residual + self.dropout(self_attn_out)) # shape: [B, S_tgt, d_model]

        # === Sub-layer 2: Multi-Head Cross-Attention ===
        residual = tgt
        # Cross-attention:
        #   Q = tgt (decoder state - "what the decoder is looking for")
        #   K = memory (encoder output - "what each source position represents")
        #   V = memory (encoder output - "what to retrieve from each source position")
        # memory_mask prevents attending to padding in the source
        cross_attn_out = self.cross_attention(tgt, memory, memory, memory_mask)
        tgt = self.norm2(residual + self.dropout(cross_attn_out)) # shape: [B, S_tgt, d_model]

        # === Sub-layer 3: Position-wise FFN ===
        residual = tgt
        ffn_out = self.feed_forward(tgt)
        tgt = self.norm3(residual + self.dropout(ffn_out)) # shape: [B, S_tgt, d_model]

        return tgt


class Decoder(nn.Module):
    """
    Full decoder: N stacked decoder layers.

    The target sequence flows through N=6 decoder layers.
    Each layer refines the target representations using:
      - Causal self-attention on the target
      - Cross-attention to the encoder output (passed in to EVERY layer)

    The SAME encoder output (memory) is passed to all decoder layers.
    """

    def __init__(self, n_layers: int = cfg.n_layers, d_model: int = cfg.d_model,
                 n_heads: int = cfg.n_heads, d_ff: int = cfg.d_ff,
                 dropout: float = cfg.dropout):
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Pass target through all N decoder layers.

        Parameters
        ----------
        tgt : torch.Tensor, shape [B, S_tgt, d_model]
        memory : torch.Tensor, shape [B, S_src, d_model], Encoder output (same for all decoder layers).
        tgt_mask : torch.Tensor or None, Target combined (causal + padding) mask.
        memory_mask : torch.Tensor or None, Source padding mask.

        Returns
        -------
        tgt : torch.Tensor, shape [B, S_tgt, d_model]
        """
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask) # shape stays [B, S_tgt, d_model] through all layers

        return tgt
