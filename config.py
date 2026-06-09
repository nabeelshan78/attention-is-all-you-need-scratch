"""
config.py - Central configuration for the Transformer.
Every single hyperparameter lives here.
Architecture matches "Attention Is All You Need" (Vaswani et al., 2017) base model.
"""

from dataclasses import dataclass
import torch

@dataclass
class TransformerConfig:
    """
    Holds all hyperparameters for the Transformer model and training.

    ARCHITECTURE HYPERPARAMETERS
    ----------------------------
    d_model:     Dimension of all embeddings and sublayer inputs/outputs throughout the model.
    n_heads:     Number of attention heads in multi-head attention. Must evenly divide d_model (d_k = d_model / n_heads).
    n_layers:    Number of encoder layers AND decoder layers.
    d_ff:        Dimension of the inner (hidden) layer in the position-wise feed-forward networks. Paper: 2048 (= 4 * d_model).
    dropout:     Dropout probability. Applied after each sub-layer output, and to embedding + positional encoding sums. Paper: 0.1 for base model.
    max_seq_len: Maximum sequence length the model can handle. Positional encoding is precomputed up to this length.
    vocab_size:  Size of the shared source+target vocabulary.

    DERIVED DIMENSIONS (computed automatically)
    -------------------------------------------
    d_k:         Dimension of queries and keys per head = d_model / n_heads. Paper: 64.
    d_v:         Dimension of values per head = d_model / n_heads. Paper: 64.

    TRAINING HYPERPARAMETERS
    ------------------------
    batch_size:      Number of training examples per gradient update.
    num_epochs:      Total number of passes over the training set.
    warmup_steps:    Number of steps over which to linearly increase the learning rate before decaying.
    label_smoothing: Epsilon for label smoothing. A small amount of probability mass is spread to all tokens. Paper: 0.1
    PAD_IDX:         Token ID used for padding sequences to equal length. We use 0. This must match the dataset's tokenizer.
    SOS_IDX:         Start-of-sequence token ID. Prepended to decoder input.
    EOS_IDX:         End-of-sequence token ID. Marks end of output.

    DEVICE
    ------
    device:      'cuda' if GPU available, else 'cpu'. Auto-detected.
    """

    # Architecture
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    # dropout: float = 0.1
    dropout: float = 0.2
    # max_seq_len: int = 100
    # vocab_size: int = 1000
    max_seq_len: int = 128      # <--- Locked in after data analysis
    vocab_size: int = 37000     # <--- Locked in to match the trained BPE

    # Derived
    @property
    def d_k(self) -> int:
        """Dimension of queries and keys per attention head."""
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        return self.d_model // self.n_heads

    @property
    def d_v(self) -> int:
        """Dimension of values per attention head (same as d_k in this paper)."""
        return self.d_model // self.n_heads

    # Training
    batch_size: int = 32
    num_epochs: int = 30
    # warmup_steps: int = 4000
    warmup_steps: int = 2000
    label_smoothing: float = 0.1

    # Special tokens
    PAD_IDX: int = 0    # Padding token — sequences shorter than max are padded
    SOS_IDX: int = 1    # Start-of-sequence token — prepended to decoder input
    EOS_IDX: int = 2    # End-of-sequence token — appended to targets

    # Device
    @property
    def device(self) -> torch.device:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validate_config(config: TransformerConfig):
    """Run basic consistency checks on the configuration."""
    assert config.d_model > 0, "d_model must be positive"
    assert config.n_heads > 0, "n_heads must be positive"
    assert config.d_model % config.n_heads == 0, (
        f"d_model must be divisible by n_heads: {config.d_model} % {config.n_heads} != 0"
    )
    assert config.n_layers > 0, "n_layers must be positive"
    assert config.d_ff > 0, "d_ff must be positive"
    assert 0.0 <= config.dropout < 1.0, "dropout must be in [0, 1)"
    assert config.vocab_size > 3, "vocab_size must be > 3 (PAD, SOS, EOS need IDs)"
    assert config.max_seq_len > 0, "max_seq_len must be positive"
    print(f"[Config] Validated. d_model={config.d_model}, n_heads={config.n_heads}, "
          f"d_k={config.d_k}, n_layers={config.n_layers}, d_ff={config.d_ff}")
    print(f"[Config] Device: {config.device}")

# Singleton instance: We will import this throughout the project: `from config import cfg`
cfg = TransformerConfig()
