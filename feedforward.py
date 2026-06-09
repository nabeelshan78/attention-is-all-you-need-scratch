"""
feedforward.py : Position-wise Feed-Forward Network (FFN).

WHAT IT IS
---------
A small two-layer MLP (multi-layer perceptron) applied independently
and identically to each position in the sequence:

    FFN(x) = max(0, x.W_1 + b_1).W_2 + b_2

This is applied AFTER the attention sub-layer in both encoder and decoder.

WHY IT EXISTS
------------
After multi-head attention, each position has gathered information from
all other positions. The attention output is a weighted blend of value
vectors - it aggregates context, but the aggregation is a LINEAR operation
over the values.

The FFN then processes each position's context-enriched representation
nonlinearly. It adds expressive power: the network can "think about" what
it has collected, applying a nonlinear transformation.

The expanded hidden dimension (d_ff = 2048 >> d_model = 512) gives room
    for multiple "features" to be computed simultaneously.

WHY "POSITION-WISE"?
-------------------
The FFN is applied to each position INDEPENDENTLY - no mixing across positions.
The SAME weight matrices (W_1, W_2) are applied at every position.
This is equivalent to two 1x1 convolutions along the sequence.

Position mixing ONLY happens in the attention layers. The FFN is a
per-position transformation.

THE EXPANSION FACTOR 4x:
------------------------
d_ff = 4 * d_model (2048 = 4 * 512 in the paper).

PARAMETERS
----------
W_1: shape [d_model, d_ff]  = [512, 2048]
b_1: shape [d_ff]           = [2048]
W_2: shape [d_ff, d_model]  = [2048, 512]
b_2: shape [d_model]        = [512]
"""

import torch
import torch.nn as nn
from config import cfg


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    The same weight matrices are applied to each position separately.
    PyTorch's nn.Linear handles this naturally when the input has shape
    [B, S, d_model] - it applies the linear transformation to the last
    dimension, independently for each (b, s) position.
    """

    def __init__(self, d_model: int = cfg.d_model, d_ff: int = cfg.d_ff,
                 dropout: float = cfg.dropout):
        """
        Parameters
        ----------
        d_model : int
            Input and output dimension. Both linear layers start and end here.
        d_ff : int
            Hidden (inner) dimension. Larger than d_model (typically 4X).
        dropout : float
            Dropout applied after the ReLU activation.
        """
        super().__init__()

        # First linear layer: d_model -> d_ff
        # Applied to last dimension: [B, S, d_model] -> [B, S, d_ff]
        self.linear1 = nn.Linear(d_model, d_ff)

        # Second linear layer: d_ff -> d_model
        # Projects back to model dimension: [B, S, d_ff] -> [B, S, d_model]
        self.linear2 = nn.Linear(d_ff, d_model)

        # ReLU activation: max(0, x)
        # Applied element-wise after linear1.
        self.relu = nn.ReLU()

        # Dropout for regularization (applied after ReLU, before linear2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply position-wise FFN to input.

        Parameters
        ----------
        x : shape [B, S, d_model]

        Returns
        -------
        output : shape [B, S, d_model]
            Same shape as input.

        Step-by-step shape trace (B=2, S=5, d_model=512, d_ff=2048):
        ------------------------------------------------------------
        x:            [2, 5, 512]
        linear1(x):   [2, 5, 2048]   (expand: each position goes 512->2048)
        relu:         [2, 5, 2048]   (negatives zeroed out)
        dropout:      [2, 5, 2048]   (random zeros, training only)
        linear2:      [2, 5, 512]    (compress: each position goes 2048->512)
        """
        # Step 1: Expand with first linear layer
        # x: [B, S, d_model] -> [B, S, d_ff]
        x = self.linear1(x)

        # Step 2: ReLU nonlinearity
        x = self.relu(x)

        # Step 3: Dropout
        x = self.dropout(x)

        # Step 4: Project back to d_model
        # [B, S, d_ff] -> [B, S, d_model]
        x = self.linear2(x)

        return x
