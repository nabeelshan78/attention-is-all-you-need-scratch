"""
attention.py - Scaled Dot-Product Attention and Multi-Head Attention.

This file contains the core computational engine of the Transformer.

COMPONENTS:
  1. scaled_dot_product_attention() - The fundamental attention operation. A function (not a module) - it's pure computation with no parameters.
  2. MultiHeadAttention - The full multi-head attention module with learned projection matrices W_Q, W_K, W_V, W_O.

USAGE IN THE TRANSFORMER:
  - Encoder self-attention:       Q=K=V=encoder_input
  - Decoder masked self-attention: Q=K=V=decoder_input (with causal mask)
  - Decoder cross-attention:       Q=decoder_state, K=V=encoder_output

The SAME MultiHeadAttention class is going to handle all three cases -
the difference is just which tensors are passed as Q, K, V.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None,
    dropout: nn.Dropout = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.
    Formula:
        Attention(Q, K, V) = softmax(Q.K^T / sqrt(d_k)).V

    Parameters
    ---------
    Q : shape -> [B, H, S_q, d_k]
        Queries. B=batch, H=heads, S_q=query length, d_k=key/query dim.
    K : shape -> [B, H, S_k, d_k]
        Keys. S_k=key/value length (same as S_q for self-attention).
    V : shape -> [B, H, S_k, d_v]
        Values. d_v=value dim (equal to d_k in this paper).
    mask : torch.Tensor or None, shape broadcastable to [B, H, S_q, S_k]
        Boolean mask. True = "block this position" -> set score to -inf.
        - Padding mask:   shape [B, 1, 1, S_k]   — broadcast over H and S_q
        - Causal mask:    shape [1, 1, S_q, S_k] — broadcast over B and H
        - Combined:       shape [B, 1, S_q, S_k]
    dropout : nn.Dropout or None
        If provided, applied to the attention weights (after softmax). Paper: dropout applied to attention weights (Section 5.4).

    Returns
    -------
    output : torch.Tensor, shape [B, H, S_q, d_v]. Attention-weighted sum of values.
    attention_weights : torch.Tensor, shape [B, H, S_q, S_k]
        Normalized attention weights. Useful for visualization.
    """
    # d_k is the last dimension of Q (and K)
    d_k = Q.size(-1)

    # Step 1: Compute raw attention scores
    # Q shape:  [B, H, S_q, d_k]
    # K.T shape: [B, H, d_k, S_k]  (transpose last two dims)
    # scores shape: [B, H, S_q, S_k]
    # scores[b, h, i, j] = dot product of query i with key j (in head h, batch b)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    # shape: [B, H, S_q, S_k]

    # Step 2: Scale by 1/sqrt(d_k)
    # Without this, dot products have std deviation sqrt(d_k), pushing softmax
    # into saturated regions with near-zero gradients.
    scores = scores / math.sqrt(d_k)
    # shape unchanged: [B, H, S_q, S_k]

    # Step 3: Apply mask
    # Where mask is True, set score to -inf so softmax gives 0 there.
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    # shape unchanged: [B, H, S_q, S_k]

    # Step 4: Softmax over the key dimension
    # For each query position, compute a probability distribution over all key positions.
    # dim=-1 means "normalize over the last dimension" = the S_k dimension.
    attention_weights = F.softmax(scores, dim=-1) # shape: [B, H, S_q, S_k] # Each row (for each query) sums to 1.
    # If a position was masked (-inf), its softmax output is 0.

    # NUMERICAL NOTE: If an entire row is -inf (e.g., a completely masked # query — which can happen for PAD queries), softmax produces NaN.
    # We handle this by replacing NaN with 0.
    attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

    # Step 5: Apply dropout to attention weights # Paper: "We apply dropout to the output of the softmax function."
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # Step 6: Weighted sum of values
    # attention_weights: [B, H, S_q, S_k]
    # V:                 [B, H, S_k, d_v]
    # output:            [B, H, S_q, d_v]
    output = torch.matmul(attention_weights, V) # shape: [B, H, S_q, d_v]

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as described in Section 3.2.2.

    Instead of a single attention function with d_model-dimensional Q, K, V,
    we:
      1. Project Q, K, V to d_k, d_k, d_v dimensions for each of h heads (using learned linear projections - different per head)
      2. Run h scaled dot-product attention operations in parallel
      3. Concatenate the h outputs (shape: h * d_v = d_model)
      4. Apply a final output projection W_O

    By projecting to lower-dimensional subspaces, each head can
    attend to different aspects of the input simultaneously. Head 1 might
    focus on syntactic relationships, Head 2 on semantic similarity, etc.

    PARAMETERS (what gets learned):
    -------------------------------
    W_Q: shape [d_model, d_model] — projects input to all head queries at once
    W_K: shape [d_model, d_model] — projects input to all head keys at once
    W_V: shape [d_model, d_model] — projects input to all head values at once
    W_O: shape [d_model, d_model] — output projection (concatenated heads -> d_model)

    Note: W_Q, W_K, W_V each have shape [d_model, d_model] because they contain all h heads' projections stacked:
      [d_model, d_model] = [d_model, h * d_k]
    We split them into h heads in the forward pass.
    Total parameters: 4 * d_model^2 = 4 * 512^2 = 1,048,576 for base model.
    """

    def __init__(self, d_model: int = cfg.d_model, n_heads: int = cfg.n_heads, dropout: float = cfg.dropout):
        super().__init__()

        # Validate: d_model must be divisible by n_heads
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads   # dimension per head for Q and K
        self.d_v = d_model // n_heads   # dimension per head for V

        # ------------ Linear projection layers ------------
        # W_Q, W_K, W_V: each maps [*, d_model] -> [*, d_model]
        # Internally, this is all h heads' projections combined.
        # We'll split into heads in the forward pass.
        # nn.Linear(in, out) has weight [out, in] and bias [out].
        self.W_Q = nn.Linear(d_model, d_model, bias=True)  # queries projection
        self.W_K = nn.Linear(d_model, d_model, bias=True)  # keys projection
        self.W_V = nn.Linear(d_model, d_model, bias=True)  # values projection
        self.W_O = nn.Linear(d_model, d_model, bias=True)  # output projection

        # Dropout on attention weights
        self.attn_dropout = nn.Dropout(p=dropout)

        # Store last attention weights for visualization/debugging
        self.attention_weights = None

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (n_heads, d_k) and transpose.

        This is how we "separate" the combined projection matrix into
        individual heads without actually having h separate projection matrices.

        Shape transformation:
          [B, S, d_model] -> [B, S, h, d_k] -> [B, h, S, d_k]

        Example:
          d_model=512, n_heads=8, d_k=64
          [2, 5, 512] -> [2, 5, 8, 64] -> [2, 8, 5, 64]
        """
        B, S, _ = x.shape
        # Reshape: treat the d_model dimension as (n_heads, d_k)
        x = x.view(B, S, self.n_heads, self.d_k)
        # Transpose to put head dimension before sequence dimension:
        # [B, S, n_heads, d_k] -> [B, n_heads, S, d_k]
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reverse of split_heads: merge the head and d_k dimensions.

        Shape transformation:
          [B, h, S, d_k] -> [B, S, h, d_k] -> [B, S, h*d_k] = [B, S, d_model]

        Example:
          [2, 8, 5, 64] -> [2, 5, 8, 64] -> [2, 5, 512]
        """
        B, _, S, _ = x.shape
        # Transpose back: [B, h, S, d_k] → [B, S, h, d_k]
        x = x.transpose(1, 2)
        # .contiguous() is needed before .view() if the tensor isn't contiguous
        # in memory after the transpose (which it typically isn't).
        x = x.contiguous()
        # Reshape: merge n_heads and d_k -> d_model
        return x.view(B, S, self.n_heads * self.d_k)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass for Multi-Head Attention.

        Parameters
        ----------
        Q : shape [B, S_q, d_model]
            Query input. In self-attention, same as K and V.
            In cross-attention (decoder), this is the decoder state.
        K : shape [B, S_k, d_model]
            Key input. In self-attention, same as Q.
            In cross-attention, this is the encoder output.
        V : shape [B, S_k, d_model]
            Value input. Same as K in all cases in this paper.
        mask : torch.Tensor or None
            Attention mask. True = block. See masks.py for shapes.

        Returns
        -------
        output : shape [B, S_q, d_model]
            Same shape as Q input. Each position now has a contextual
            representation formed from attending to K/V.

        Complete Shape Trace (B=2, S_q=5, S_k=5, d_model=512, h=8, d_k=64):
        -------------------------------------------------------------------
        Q input:       [2, 5, 512]
        K input:       [2, 5, 512]
        V input:       [2, 5, 512]

        W_Q(Q):        [2, 5, 512]   (linear projection: each position projected)
        W_K(K):        [2, 5, 512]
        W_V(V):        [2, 5, 512]

        split_heads(Q_proj): [2, 8, 5, 64]  (8 heads, each with 64-dim queries)
        split_heads(K_proj): [2, 8, 5, 64]
        split_heads(V_proj): [2, 8, 5, 64]

        attention_output: [2, 8, 5, 64]  (output of scaled dot-product attention)

        combine_heads(): [2, 5, 512]  (concatenate all 8 heads)

        W_O:           [2, 5, 512]  (output projection)
        """

        # Step 1: Linear projections for Q, K, V
        # Project from d_model to d_model (containing all heads combined)
        # W_Q maps [B, S, d_model] -> [B, S, d_model]
        Q_proj = self.W_Q(Q)  # [B, S_q, d_model]
        K_proj = self.W_K(K)  # [B, S_k, d_model]
        V_proj = self.W_V(V)  # [B, S_k, d_model]

        # Step 2: Split into multiple heads
        # [B, S, d_model] -> [B, n_heads, S, d_k]
        Q_heads = self.split_heads(Q_proj)  # [B, n_heads, S_q, d_k]
        K_heads = self.split_heads(K_proj)  # [B, n_heads, S_k, d_k]
        V_heads = self.split_heads(V_proj)  # [B, n_heads, S_k, d_v]

        # Step 3: Scaled dot-product attention for all heads in parallel
        # PyTorch's matmul broadcasts over leading dimensions (B, n_heads),
        # so this runs attention for all batch elements and all heads at once.
        attn_output, self.attention_weights = scaled_dot_product_attention(
            Q_heads, K_heads, V_heads,
            mask=mask,
            dropout=self.attn_dropout
        )
        # attn_output shape: [B, n_heads, S_q, d_v]

        # Step 4: Combine heads
        # [B, n_heads, S_q, d_v] -> [B, S_q, n_heads * d_v] = [B, S_q, d_model]
        combined = self.combine_heads(attn_output)  # [B, S_q, d_model]

        # Step 5: Output projection
        # W_O: [B, S_q, d_model] -> [B, S_q, d_model]
        output = self.W_O(combined)

        return output
