"""
loss.py - Label Smoothing Cross-Entropy Loss.
WHAT LABEL SMOOTHING IS
-----------------------
Standard cross-entropy loss trains the model against one-hot targets: y = [0, 0, 1, 0, 0, ...]   (1.0 at the correct token, 0 elsewhere)
The model is pushed to put ALL probability mass on the correct token:
  predicted prob of correct token -> 1.0
  predicted prob of all others    -> 0.0

This causes two problems:
  1. OVERCONFIDENCE: The model outputs extreme logit values to satisfy the loss, which uses model capacity inefficiently.
  2. POOR CALIBRATION: The model assigns ~0 probability to alternatives, even sensible ones (many translations are valid!).

Label smoothing distributes a small amount e-epsilon of probability mass uniformly across ALL vocabulary tokens, including the correct one:

  y_smooth = (1 - e) * y_onehot + e / V

With e=0.1 and V=1000:
  - Correct token target: 0.9 + 0.1/1000 = 0.9001
  - Each wrong token target: 0.1/1000 = 0.0001

The model is now trained to be "almost" certain (not fully certain) about the correct token. This is a form of regularization.

EFFECT ON LOSS:
  L = -sum_i y_smooth_i * log(p_i) = -(1-e) * log(p_correct) - e/V * sum_i log(p_i)

The second term penalizes the model for having a very peaked (confident) distribution - it receives a lower loss if it spreads some probability.

THE PAPER'S RESULT:
"Label smoothing of value e_ls = 0.1, hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score."

Perplexity measures confidence -> lower perplexity = more confident. Smoothing makes the model less confident -> higher perplexity.
But it generalizes better -> better BLEU.

PADDING:
We ignore PAD tokens in the loss (they have no meaningful target). The loss is averaged only over non-PAD positions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing and PAD token masking.

    Can be used in two modes:
      1. With PyTorch's built-in label_smoothing (simpler, recommended)
      2. Manual KL-divergence formulation (more explicit, educational)

    We implement both and use PyTorch's built-in for training.
    """

    def __init__(
        self,
        vocab_size: int = cfg.vocab_size,
        pad_idx: int = cfg.PAD_IDX,
        label_smoothing: float = cfg.label_smoothing
    ):
        """
        Parameters
        ----------
        vocab_size : int, Number of tokens in vocabulary (size of logit dimension).
        pad_idx : int, Token ID for padding. These positions are ignored in loss.
        label_smoothing : float, Epsilon e in [0, 1). 0 = no smoothing (standard CE). 0.1 = paper value.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.label_smoothing = label_smoothing

        # PyTorch's CrossEntropyLoss with label_smoothing handles everything:
        # - Softmax is applied internally (expects raw logits, not probabilities)
        # - label_smoothing distributes e mass uniformly
        # - ignore_index masks PAD positions (they contribute 0 to the loss)
        # - reduction='mean' averages over all non-ignored positions
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=pad_idx,
            reduction='mean'
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label-smoothed cross-entropy loss.

        Parameters
        ----------
        logits : shape [B, S, vocab_size], Raw (unnormalized) model outputs. DO NOT apply softmax before this.
            nn.CrossEntropyLoss applies log_softmax internally.
        targets : shape [B, S], Ground-truth token IDs. PAD positions are ignored.
            These should be the SHIFTED target: target[1:] (the tokens to predict).

        Returns
        -------
        loss : scalar, Scalar loss value averaged over all non-PAD positions.

        Shape trace:
          logits:  [B, S, vocab_size]  e.g., [2, 4, 1000]
          targets: [B, S]              e.g., [2, 4]

          nn.CrossEntropyLoss expects:
            input:  [N, C]
            target: [N]
          where C is number of classes.

          We reshape:
            logits:  [B*S, vocab_size]  <- [N, C] format
            targets: [B*S]              <- [N] format

          loss: scalar (mean over non-PAD positions)
        """
        B, S, V = logits.shape

        # Reshape for CrossEntropyLoss:
        logits_flat = logits.reshape(B * S, V)

        # [B, S] -> [B*S]
        targets_flat = targets.reshape(B * S)

        loss = self.criterion(logits_flat, targets_flat)
        return loss


def compute_loss_manual(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int = cfg.PAD_IDX,
    label_smoothing: float = cfg.label_smoothing
) -> torch.Tensor:
    """
    Manual implementation of label-smoothed cross-entropy for understanding.
    The LabelSmoothingLoss class above uses PyTorch's built-in, which is more efficient.

    DERIVATION:
    -----------
    Standard CE:    L = -log(p_{y})
    KL divergence:  L = sum_i y_smooth_i * log(y_smooth_i / p_i)
                      = sum_i y_smooth_i * log(y_smooth_i)  (constant) - sum_i y_smooth_i * log(p_i)  (cross-entropy)

    Ignoring the constant (doesn't affect gradients):
      L = -sum_i y_smooth_i * log(p_i) = -(1-e) * log(p_y) - (e/V) * sum_i log(p_i)
    sum_i log(p_i) = sum of log-probs over all tokens.
    """
    B, S, V = logits.shape

    # Log probabilities: [B, S, vocab_size]
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather log prob of the correct token at each position: [B, S]
    # targets.unsqueeze(-1): [B, S, 1]
    # log_probs.gather(-1, ...): [B, S, 1] -> [B, S]
    log_prob_correct = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # Sum of log probs over all vocab: [B, S]
    log_prob_sum = log_probs.sum(dim=-1)

    # Label-smoothed loss per position:
    # L = -(1-e) * log(p_y) - (e/V) * sum_i log(p_i)
    loss_per_pos = -(1 - label_smoothing) * log_prob_correct \
                   - (label_smoothing / V) * log_prob_sum

    # Mask: 1 for real tokens, 0 for PAD
    non_pad_mask = (targets != pad_idx).float()  # [B, S]

    # Average over non-PAD positions
    loss = (loss_per_pos * non_pad_mask).sum() / non_pad_mask.sum().clamp(min=1)
    return loss
