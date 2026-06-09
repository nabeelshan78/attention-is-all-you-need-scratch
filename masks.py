"""
masks.py - Padding mask and look-ahead (causal) mask construction.

WHY MASKS EXIST
--------------
The Transformer's attention mechanism, in its raw form, allows every position
to attend to every other position. We need to selectively "block" certain
attention connections for two reasons:

  1. PADDING MASK: Sequences in a batch have different lengths. Shorter ones
     are right-padded with PAD tokens (ID = 0). We don't want any position to
     attend to a padding token, it carries no meaning and would corrupt the
     representation.

  2. LOOK-AHEAD (CAUSAL) MASK: In the decoder's self-attention, when predicting
     the token at position t, the model must not be able to "see" tokens at
     positions t+1, t+2, ... These are future tokens - not yet generated. Allowing
     the model to see them during training would be "cheating" and would prevent
     it from learning a proper generative model. This mask enforces autoregressive
     behavior.

HOW MASKS WORK
-------------
After computing the raw attention scores (shape [B, H, S_q, S_k]):
  scores = scores.masked_fill(mask == True, float('-inf'))

Where mask has value True at positions to BLOCK. Then:

  attention_weights = softmax(scores, dim=-1)

Since exp(-inf) = 0, masked positions receive exactly 0 attention weight.
They are completely invisible to the attending position.

SHAPE CONVENTIONS
-----------------
Attention scores are shape [B, H, S_q, S_k]:
  B = batch size
  H = number of heads
  S_q = query sequence length
  S_k = key sequence length

Masks must be broadcastable to this shape.

  Padding mask: [B, 1, 1, S_k]
    - We don't differentiate between heads (broadcast over H)
    - We don't differentiate between query positions (broadcast over S_q)
    - We only care about which KEY positions to mask

  Look-ahead mask: [1, 1, S_q, S_k]  (or [S_q, S_k])
    - Same for all batch elements and heads
    - Upper-triangular: masks future key positions for each query position

  Combined decoder mask: max(look_ahead_mask, padding_mask)
    - True wherever either mask says to block
"""

import torch


def make_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Create a padding mask that blocks attention to PAD tokens.

    Parameters
    ----------
    seq : shape [B, S], Token ID sequences. PAD positions have value pad_idx.
    pad_idx : int, The token ID used for padding. Default: 0.

    Returns
    -------
    mask : shape [B, 1, 1, S]
        Boolean tensor. True = "mask this position" (it's a PAD token).
        Shape is expanded for broadcasting over (heads, query positions).

    Example
    -------
    seq = [[45, 732, 1, 0, 0],   # 3 real tokens + 2 PAD
           [12, 74, 395, 2, 100]] # 5 real tokens
    mask = [[[[False, False, False, True, True]]],
            [[[False, False, False, False, False]]]]
    Shape: [2, 1, 1, 5]

    When broadcast against scores [2, 8, 5, 5], this blocks attention TO positions 3 and 4 in the first sequence, for all 8 heads and
    all 5 query positions so that query do not attend to padding tokens.
    """
    # seq shape: [B, S]
    # unsqueeze(1) -> [B, 1, S] -> unsqueeze(2) -> [B, 1, 1, S] <- broadcasts over heads and query positions
    mask = (seq == pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


def make_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a causal (look-ahead) mask for the decoder self-attention.
    Blocks position i from attending to position j when j > i (future).

    Parameters
    ----------
    seq_len : int, Length of the sequence (S).
    device : torch.device, Device to create the mask on.

    Returns
    -------
    mask : shape [1, 1, S, S], Boolean tensor. True = "mask this (i,j) pair."
    Visualization (S=5):
    --------------------

    Q-K  0     1     2     3     4
    0  [F     T     T     T     T] <- pos 0 can only attend to pos 0
    1  [F     F     T     T     T] <- pos 1 can attend to pos 0,1
    2  [F     F     F     T     T] <- pos 2 can attend to pos 0,1,2
    3  [F     F     F     F     T]
    4  [F     F     F     F     F] <- pos 4 can attend to all

    T = True = masked (blocked)
    F = False = allowed

    torch.triu with diagonal=1 gives the strict upper triangle.
    """
    mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1
    )
    # shape: [S, S] -> [1, 1, S, S] for broadcasting over (batch, heads)
    return mask.unsqueeze(0).unsqueeze(0)


def make_src_mask(src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Create source padding mask for the encoder.

    Used in:
    - Encoder self-attention: blocks attention TO padding positions.
    - Decoder cross-attention: same - blocks attention to padding in the source.

    Parameters
    ----------
    src : shape [B, S_src]
    pad_idx : int

    Returns
    -------
    mask : shape [B, 1, 1, S_src]
    """
    return make_padding_mask(src, pad_idx)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Create combined target mask for the decoder self-attention.

    Combines:
    1. Causal mask: prevents attending to future positions.
    2. Padding mask: prevents attending to PAD tokens.

    The combined mask is the logical OR - True if EITHER condition applies.

    Parameters
    ----------
    tgt : shape [B, S_tgt]
    pad_idx : int

    Returns
    -------
    mask : shape [B, 1, S_tgt, S_tgt]
        True = masked (blocked).
    """
    S_tgt = tgt.size(1)
    device = tgt.device

    # Causal mask: shape [1, 1, S_tgt, S_tgt]
    causal = make_causal_mask(S_tgt, device=device)

    # Padding mask: shape [B, 1, 1, S_tgt]
    padding = make_padding_mask(tgt, pad_idx)

    # Combine: torch.logical_or broadcasts [B, 1, 1, S] with [1, 1, S, S]
    # Result: [B, 1, S_tgt, S_tgt]
    combined = torch.logical_or(causal, padding)

    return combined
