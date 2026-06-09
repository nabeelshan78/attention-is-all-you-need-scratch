"""
embeddings.py - Token embeddings and sinusoidal positional encoding.

TWO COMPONENTS:
  1. TokenEmbedding:        Maps integer token IDs -> dense vectors of dim d_model. Learned. Scaled by sqrt(d_model).
  2. PositionalEncoding:    Adds fixed sinusoidal vectors to encode position. Not learned (in the paper's version).
  3. TransformerEmbedding:  Combines both. The full embedding pipeline.

MATHEMATICAL DETAIL: Token Embedding
------------------------------------
Let E: (V, d_model) be the embedding matrix (V = vocab_size). For token ID k, embedding = E[k, :] - the k-th row of E.
For batch input x of shape [B, S] (B sequences, each S tokens):
  emb = E[x] -> shape [B, S, d_model], emb = emb * sqrt(d_model)  -> scaled

WHY SCALE: The embedding matrix is initialized with values ~ N(0, 1/d_model) (PyTorch default for nn.Embedding after we scale them). The positional
encodings have values in [-1, 1]. Without scaling embeddings up, the PE signal would dominate and the learned token meaning would be negligible.

MATHEMATICAL DETAIL: Positional Encoding
----------------------------------------
PE(pos, 2i)   = sin(pos / 10000^{2i / d_model})
PE(pos, 2i+1) = cos(pos / 10000^{2i / d_model})
For position `pos` and dimension index `i`:     frequency ω_i = 1 / (10000 ^ (2i / d_model))
The original paper applies dropout to the sum of embeddings + positional encodings. We will use that.
"""

import math
import torch
import torch.nn as nn
from config import cfg


class TokenEmbedding(nn.Module):
    """
    Learnable token embedding with sqrt(d_model) scaling.
    Wraps nn.Embedding, which is essentially a lookup table: nn.Embedding(vocab_size, d_model) stores a matrix of shape [vocab_size, d_model].

    Parameters (what get learned)
    -----------------------------
    weight: shape [vocab_size, d_model] - one vector per token in vocabulary.
    PyTorch's nn.Embedding initializes weights ~ N(0, 1) by default. We scale the output (not the weights themselves) by sqrt(d_model).
    """

    def __init__(self, vocab_size: int = cfg.vocab_size, d_model: int = cfg.d_model):
        super().__init__()
        self.d_model = d_model
        # nn.Embedding: a simple lookup table that stores embeddings of a fixed dictionary.
        # the embedding for PAD token is kept as zeros and receives no gradient updates
        self.embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=cfg.PAD_IDX
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : shape [B, S], Integer token IDs. Each value must be in [0, vocab_size - 1].

        Returns
        -------
        emb : [B, S, d_model], Scaled embedding vectors.
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding.

    Not learned - the encoding is computed once at initialization using the formulas from the paper and stored as a non-parameter buffer.
    During forward, it's simply added to the input embeddings.

    BUFFER vs PARAMETER:
    - nn.Parameter: included in model.parameters(), gets gradients, updates
    - register_buffer: part of model state (saved/loaded), but no gradients
    We use register_buffer because PE is fixed - it never changes.
    """

    def __init__(self, d_model: int = cfg.d_model, max_seq_len: int = cfg.max_seq_len, dropout: float = cfg.dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Construct the PE matrix of shape [max_seq_len, d_model]
        # pe[pos, i] = sin(pos / 10000^(2*(i//2) / d_model)) if i is even
        #            = cos(pos / 10000^(2*(i//2) / d_model)) if i is odd

        # Initialize with zeros; we'll fill it in
        pe = torch.zeros(max_seq_len, d_model)  # shape: [max_seq_len, d_model]

        # position: column vector of positions [0, 1, 2, ..., max_seq_len-1]
        # shape: [max_seq_len, 1]
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # div_term: the denominators 10000^(2i/d_model) for i = 0, 1, ..., d_model//2-1
        # We compute this in log space for numerical stability:
        #   10000^(2i/d_model) = exp(2i/d_model * log(10000))
        # This gives shape [d_model // 2]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *   # [0, 2, 4, ..., d_model-2]
            (-math.log(10000.0) / d_model)           # scale factor
        )
        # div_term[i] = exp(-2i/d_model * log(10000)) = 1 / 10000^(2i/d_model)

        # Fill even dimensions with sin, odd with cos
        pe[:, 0::2] = torch.sin(position * div_term)  # dims 0,2,4,...
        pe[:, 1::2] = torch.cos(position * div_term)  # dims 1,3,5,...

        # Add batch dimension: [max_seq_len, d_model] -> [1, max_seq_len, d_model] # So it broadcasts with inputs of shape [B, S, d_model]
        pe = pe.unsqueeze(0)

        # Register as a buffer (not a parameter - won't be updated by optimizer)
        self.register_buffer('pe', pe)
        # After this: self.pe has shape [1, max_seq_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input and apply dropout.

        Parameters
        ---------
        x : [B, S, d_model], Token embeddings.

        Returns
        ------
        x : [B, S, d_model], Embeddings with positional information added.
        """
        # self.pe[:, :x.size(1), :] slices PE to length S
        # The addition broadcasts: [B, S, 512] + [1, S, 512] -> [B, S, 512]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    """
    Complete embedding pipeline: token embedding + positional encoding.
    This is the first thing the encoder and decoder apply to their inputs.
    Pipeline:
      token_ids [B, S] -> TokenEmbedding -> [B, S, d_model] -> PositionalEncoding → [B, S, d_model]

    WEIGHT SHARING:
    The paper shares the embedding weight matrix between:
      1. Source token embedding (encoder input)
      2. Target token embedding (decoder input)
      3. Pre-softmax linear projection (decoder output)
    This reduces parameters. In our Transformer class, we pass the same TokenEmbedding to both encoder and decoder embeddings.
    """

    def __init__(self, vocab_size: int = cfg.vocab_size, d_model: int = cfg.d_model,
                 max_seq_len: int = cfg.max_seq_len,
                 dropout: float = cfg.dropout):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ---------
        x : [B, S], Token IDs.

        Returns
        -------
        out : [B, S, d_model], Embedding + positional encoding with dropout.
        """
        # 1. Lookup embeddings and scale
        tok_emb = self.token_embedding(x)     # [B, S, d_model]
        # 2. Add positional encodings + dropout
        out = self.positional_encoding(tok_emb)  # [B, S, d_model]
        return out
