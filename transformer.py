"""
transformer.py - The complete Transformer model.

This ties together all components:
  - Source embedding (token + positional)
  - Target embedding (token + positional)
  - Encoder stack
  - Decoder stack
  - Final linear + softmax projection

WEIGHT SHARING:
The paper shares weights between:
  1. Source token embedding matrix
  2. Target token embedding matrix
  3. The pre-softmax linear projection (output layer)

All three use the SAME weight matrix E of shape [vocab_size, d_model].
For the output linear layer, we use E^T: [d_model, vocab_size].

WHY SHARE WEIGHTS?
  - Reduces parameters significantly.
  - The embedding maps token_id -> vector; the output layer maps vector -> token_id.
    These are inverse operations - sharing weights makes sense intuitively.

PARAMETER INITIALIZATION:
  - Linear layers: Xavier/Glorot uniform (PyTorch default for nn.Linear)
  - Embeddings: N(0, 1) then scaled (we do this implicitly)
  - LayerNorm: gamma=1, beta=0 (PyTorch default)

We use Xavier uniform for all linear layers (this is PyTorch's default).
"""

import torch
import torch.nn as nn
import math
from embeddings import TransformerEmbedding, TokenEmbedding
from encoder import Encoder
from decoder import Decoder
from masks import make_src_mask, make_tgt_mask
from config import cfg


class Transformer(nn.Module):
    """
    The complete Transformer model for sequence-to-sequence tasks.

    FULL FORWARD PASS:
    ------------------
    1. Source tokens -> src_embedding (TokenEmb + PosEnc) -> [B, S_src, d_model]
    2. Encoder(src_emb, src_mask) -> encoder_output [B, S_src, d_model]
    3. Target tokens -> tgt_embedding (TokenEmb + PosEnc) -> [B, S_tgt, d_model]
    4. Decoder(tgt_emb, encoder_output, tgt_mask, src_mask) -> [B, S_tgt, d_model]
    5. output_projection(decoder_out) -> logits [B, S_tgt, vocab_size]

    During training: all steps happen in one forward() call.
    During inference: encode() once, then decode() autoregressively.
    """

    def __init__(
        self,
        vocab_size: int = cfg.vocab_size,
        d_model: int = cfg.d_model,
        n_heads: int = cfg.n_heads,
        n_layers: int = cfg.n_layers,
        d_ff: int = cfg.d_ff,
        dropout: float = cfg.dropout,
        max_seq_len: int = cfg.max_seq_len,
        pad_idx: int = cfg.PAD_IDX
    ):
        super().__init__()

        self.pad_idx = pad_idx
        self.d_model = d_model

        # Embeddings
        # We create ONE shared TokenEmbedding (the weight matrix E). Both src and tgt use it (weight sharing).
        # Each gets its own PositionalEncoding (PE is not shared - different sequences can have different lengths, and PE is fixed anyway).
        self.shared_embedding = TokenEmbedding(vocab_size, d_model)

        self.src_embedding = TransformerEmbedding(vocab_size, d_model, max_seq_len, dropout)
        self.tgt_embedding = TransformerEmbedding(vocab_size, d_model, max_seq_len, dropout)

        # Share weights: src and tgt token embeddings point to the SAME matrix
        self.src_embedding.token_embedding = self.shared_embedding
        self.tgt_embedding.token_embedding = self.shared_embedding

        # Encoder
        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff, dropout)

        # Decoder
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, dropout)

        # Output projection: d_model -> vocab_size. This is a linear layer WITHOUT bias. Weight matrix shape: [vocab_size, d_model]
        # We transpose to apply as [d_model, vocab_size] in the forward pass.
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: output_projection weight = shared embedding weight
        # output_projection.weight has shape [vocab_size, d_model], shared_embedding.embedding.weight has shape [vocab_size, d_model]
        # They are the SAME tensor - not a copy!
        self.output_projection.weight = self.shared_embedding.embedding.weight

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """
        Initialize model parameters.

        For linear layers: Xavier uniform initialization.
        For layer norms: already initialized to gamma=1, beta=0 by PyTorch.
        For embeddings: PyTorch default N(0,1).
        For biases: initialized to zero.
        """
        for name, p in self.named_parameters():
            if p.dim() > 1:
                # 2D+ tensors: weight matrices of linear layers
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
            # LayerNorm params (gamma, beta) and embedding weights keep PyTorch defaults

    def make_masks(self, src: torch.Tensor, tgt: torch.Tensor):
        """
        Create all masks needed for a forward pass.

        Parameters
        ----------
        src : [B, S_src]  — source token IDs
        tgt : [B, S_tgt]  — target token IDs

        Returns
        -------
        src_mask    : [B, 1, 1, S_src]      - padding mask for source
        tgt_mask    : [B, 1, S_tgt, S_tgt]  - causal + padding for target
        """
        src_mask = make_src_mask(src, self.pad_idx)
        tgt_mask = make_tgt_mask(tgt, self.pad_idx)
        return src_mask, tgt_mask

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Run the encoder on the source sequence.
        Called once per inference pass (not per generated token).

        Parameters
        ----------
        src : [B, S_src]            - source token IDs
        src_mask : [B, 1, 1, S_src] - source padding mask

        Returns
        -------
        memory : [B, S_src, d_model] - encoder output
        """
        if src_mask is None:
            src_mask = make_src_mask(src, self.pad_idx)

        src_emb = self.src_embedding(src)       # [B, S_src, d_model]
        memory = self.encoder(src_emb, src_mask) # [B, S_src, d_model]
        return memory

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Run the decoder on target sequence + encoder memory.

        Parameters
        ----------
        tgt : [B, S_tgt]             - target token IDs so far
        memory : [B, S_src, d_model]  - encoder output
        tgt_mask : [B, 1, S_tgt, S_tgt]
        memory_mask : [B, 1, 1, S_src]

        Returns
        -------
        logits : [B, S_tgt, vocab_size] - raw scores for each vocab token
        """
        if tgt_mask is None:
            tgt_mask = make_tgt_mask(tgt, self.pad_idx)

        tgt_emb = self.tgt_embedding(tgt)   # [B, S_tgt, d_model]
        dec_out = self.decoder(tgt_emb, memory, tgt_mask, memory_mask)   # [B, S_tgt, d_model]

        logits = self.output_projection(dec_out)   # [B, S_tgt, vocab_size]

        return logits

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor
    ) -> torch.Tensor:
        """
        Full forward pass: src + tgt -> logits.

        Used during TRAINING (teacher forcing: entire target sequence fed at once).

        Parameters
        ----------
        src : shape [B, S_src], Source token IDs.
        tgt : shape [B, S_tgt] Target input token IDs (decoder input - shifted right, starts with <SOS>).
            In teacher forcing, this is ground-truth target[:-1] with <SOS> prepended.

        Returns
        -------
        logits : shape [B, S_tgt, vocab_size]
            Raw (unnormalized) scores for each vocabulary token at each position.
            Apply softmax to get probabilities; use with cross-entropy loss directly.

        Full Shape Trace (B=2, S_src=5, S_tgt=4, d_model=512, vocab_size=1000):
        src:            [2, 5]
        tgt:            [2, 4]

        src_mask:       [2, 1, 1, 5]
        tgt_mask:       [2, 1, 4, 4]

        src_emb:        [2, 5, 512]
        encoder_out:    [2, 5, 512]

        tgt_emb:        [2, 4, 512]
        decoder_out:    [2, 4, 512]

        logits:         [2, 4, 1000]
        """
        # Build masks
        src_mask, tgt_mask = self.make_masks(src, tgt)

        # Encode source
        memory = self.encode(src, src_mask)       # [B, S_src, d_model]

        # Decode target (with cross-attention to memory)
        logits = self.decode(tgt, memory, tgt_mask, src_mask)   # [B, S_tgt, vocab_size]

        return logits

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_transformer(config=cfg) -> Transformer:
    """
    Factory function: build a Transformer from config.
    Prints a summary of the model.
    """
    model = Transformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
        pad_idx=config.PAD_IDX,
    )
    n_params = model.count_parameters()
    print(f"[Transformer] Built model with {n_params:,} trainable parameters")
    print(f"  d_model={config.d_model}, n_heads={config.n_heads}, "
          f"n_layers={config.n_layers}, d_ff={config.d_ff}")
    return model
