"""
inference.py - Autoregressive decoding at inference time.

TRAINING vs INFERENCE
---------------------
During TRAINING (teacher forcing):
  - We feed the ENTIRE target sequence to the decoder at once.
  - The causal mask prevents the decoder from seeing future tokens.
  - All output logits are computed in a single forward pass.
  - This is efficient because it's fully parallelized.

During INFERENCE:
  - We don't have the target sequence (that's what we want to generate!).
  - We must generate tokens one at a time, left to right.
  - At step t, we feed tokens [SOS, t1, ..., t_{t-1}] to get logit for t_t.
  - Then append t_t and repeat until <EOS> or max_len.
  - This IS sequential - no parallelism across generation steps.
  - But we can parallelize across multiple sequences in a batch.

GREEDY DECODING vs BEAM SEARCH
------------------------------
GREEDY: At each step, pick the token with the highest probability.
  - Simple and fast
  - Locally optimal at each step, but not globally optimal
  - Often sufficient for simple tasks

BEAM SEARCH: At each step, keep the top-k most likely SEQUENCES (beams).
  - Explores multiple hypotheses simultaneously
  - Much better for real translation (the paper uses beam size 4)
  - More complex to implement

We implement both: greedy for simplicity, beam search for completeness.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional
from config import cfg


@torch.no_grad()
def greedy_decode(
    model: torch.nn.Module,
    src: torch.Tensor,
    max_len: int = cfg.max_seq_len,
    sos_idx: int = cfg.SOS_IDX,
    eos_idx: int = cfg.EOS_IDX,
    pad_idx: int = cfg.PAD_IDX,
    device: torch.device = None
) -> torch.Tensor:
    """
    Greedy autoregressive decoding: always pick the most probable next token.
    Parameters
    ----------
    model : Transformer, The trained model. Must be in eval mode (model.eval()).
    src : torch.Tensor, shape [B, S_src] or [S_src], Source token IDs. If 1D, adds batch dimension automatically.
    max_len : int, Maximum number of output tokens to generate (including EOS).
    sos_idx : int, Start-of-sequence token ID.
    eos_idx : int, End-of-sequence token ID. Generation stops when this is produced.
    pad_idx : int, Padding token ID (for the source mask).
    device : torch.device, Device to run on.

    Returns
    ------
    decoded : shape [B, S_out]
        Generated token IDs for each sequence in the batch. May have different lengths if we pad to the longest.
        Includes EOS (if generated), excludes SOS.

    STEP-BY-STEP:
    -------------
    1. Encode source once -> memory [B, S_src, d_model]
    2. Initialize decoder input = [[SOS], [SOS], ...] shape [B, 1]
    3. For t = 1, 2, ..., max_len:
       a. Run decoder(decoder_input, memory) -> logits [B, t, vocab_size]
       b. Take last position: logits[:, -1, :] -> [B, vocab_size]
       c. Argmax -> next_token [B]
       d. Append to decoder_input
       e. If ALL sequences have produced EOS, stop early
    4. Return decoder_input[:, 1:] (remove SOS prefix)
    """
    if device is None:
        device = next(model.parameters()).device

    # Ensure src has batch dimension
    if src.dim() == 1:
        src = src.unsqueeze(0)  # [1, S_src]
    src = src.to(device)

    B = src.size(0)

    # Step 1: Encode source (done ONCE)
    from masks import make_src_mask
    src_mask = make_src_mask(src, pad_idx).to(device)
    memory = model.encode(src, src_mask)   # [B, S_src, d_model]

    # Step 2: Initialize decoder input with SOS
    # decoder_input starts as [[SOS], [SOS], ...] for each sequence in batch
    decoder_input = torch.full((B, 1), sos_idx, dtype=torch.long, device=device) # shape: [B, 1]

    # Track which sequences have finished (produced EOS)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    # Step 3: Autoregressively generate tokens
    for _ in range(max_len):
        # Run decoder with current decoder_input
        # decoder_input shape: [B, t] where t grows each step
        logits = model.decode(decoder_input, memory, memory_mask=src_mask) # logits shape: [B, t, vocab_size]

        # Take logits for the LAST position only (we just want the next token)
        next_logits = logits[:, -1, :]   # [B, vocab_size]

        # Greedy: pick the token with the highest logit
        next_token = next_logits.argmax(dim=-1)  # [B]

        # For already-finished sequences, replace with PAD (no new meaningful tokens)
        next_token = next_token.masked_fill(finished, pad_idx)

        # Append next token to decoder input, # shape: [B, t+1]
        decoder_input = torch.cat(
            [decoder_input, next_token.unsqueeze(1)], dim=1
        )

        # Mark sequences that just produced EOS as finished
        finished = finished | (next_token == eos_idx)

        # Early stop: if all sequences are finished, we're done
        if finished.all():
            break

    # Remove the SOS token at the beginning (it was only an input cue)
    # Return shape: [B, S_out] where S_out = number of tokens generated
    return decoder_input[:, 1:]


@torch.no_grad()
def beam_search_decode(
    model: torch.nn.Module,
    src: torch.Tensor,
    beam_size: int = 4,
    max_len: int = cfg.max_seq_len,
    sos_idx: int = cfg.SOS_IDX,
    eos_idx: int = cfg.EOS_IDX,
    pad_idx: int = cfg.PAD_IDX,
    length_penalty: float = 0.6,
    device: torch.device = None
) -> List[torch.Tensor]:
    """
    Beam search decoding for a single source sequence.

    Beam search keeps the top `beam_size` most likely sequences at each step, exploring multiple hypotheses before committing to an output.

    LENGTH PENALTY (from the paper, alpha=0.6):
    ---------------------------------------
    Without length penalty, beam search tends to prefer shorter sequences
    because shorter = fewer terms in the log-prob product = less negative and close to 0.

    The paper uses:
      score = log_prob(hypothesis) / length_penalty_fn(length)
    where:
      length_penalty_fn(l) = ((5 + l) / (5 + 1))^alpha

    Higher alpha -> stronger penalty for short sequences -> longer outputs.
    alpha=0 -> no penalty (standard beam search).

    Parameters
    ----------
    model : Transformer (in eval mode)
    src : shape [S_src] or [1, S_src], Single source sequence.
    beam_size : int, Number of beams to maintain (k in top-k).
    max_len : int
    sos_idx, eos_idx, pad_idx : int
    length_penalty : float (aplha in the paper, 0.6)
    device : torch.device

    Returns
    -------
    hypotheses : List[torch.Tensor]
        List of `beam_size` generated sequences (best first, by score). Each tensor has shape [S_out].
    """
    if device is None:
        device = next(model.parameters()).device

    if src.dim() == 1:
        src = src.unsqueeze(0)   # [1, S_src]
    src = src.to(device)

    from masks import make_src_mask

    # Step 1: Encode source
    src_mask = make_src_mask(src, pad_idx).to(device)
    memory = model.encode(src, src_mask)   # [1, S_src, d_model]

    # Initialize beam
    # Each beam is: (cumulative log-prob, sequence_tensor)
    # Start: one beam with just SOS, log-prob = 0
    beams = [(0.0, torch.tensor([sos_idx], dtype=torch.long, device=device))]

    completed = []  # finished beams (produced EOS)

    # Expand each beam step by step
    for step in range(max_len):
        # Collect all candidates from expanding each current beam
        candidates = []

        for score, seq in beams:
            if seq[-1].item() == eos_idx:
                # Already finished - add to completed and don't expand
                completed.append((score, seq))
                continue

            # Run decoder with this beam's sequence # seq shape: [t] -> unsqueeze to [1, t] (batch of 1)
            decoder_input = seq.unsqueeze(0)   # [1, t]
            logits = model.decode(decoder_input, memory, memory_mask=src_mask) # logits: [1, t, vocab_size]

            # Take last position
            last_logits = logits[0, -1, :]   # [vocab_size]
            log_probs = F.log_softmax(last_logits, dim=-1)   # [vocab_size]

            # Get top-k tokens (pruning to beam_size most promising)
            top_log_probs, top_indices = log_probs.topk(beam_size)

            for log_p, idx in zip(top_log_probs, top_indices):
                new_score = score + log_p.item()
                new_seq = torch.cat([seq, idx.unsqueeze(0)])
                candidates.append((new_score, new_seq))

        if not candidates:
            break

        # Score with length penalty
        def length_penalty_fn(length):
            return ((5 + length) / 6.0) ** length_penalty

        def scored(candidate):
            s, seq = candidate
            return s / length_penalty_fn(len(seq))

        # Keep top beam_size candidates
        candidates.sort(key=scored, reverse=True)
        beams = candidates[:beam_size]

        # If all beams are finished, stop
        if all(seq[-1].item() == eos_idx for _, seq in beams):
            completed.extend(beams)
            beams = []
            break

    # Add any remaining non-finished beams to completed
    completed.extend(beams)

    # Sort by score with length penalty, best first
    completed.sort(key=lambda x: x[0] / length_penalty_fn(len(x[1])), reverse=True)

    # Return sequences without SOS token
    return [seq[1:] for _, seq in completed[:beam_size]]


def decode_tokens(token_ids: torch.Tensor, id_to_token: dict = None) -> str:
    """
    Convert a list of token IDs to a human-readable string.
    Parameters
    ----------
    token_ids : torch.Tensor or list of int
    id_to_token : dict {int: str} or None
        If None, just prints the IDs as numbers.

    Returns
    -------
    text : str
    """
    ids = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids

    # Remove EOS and PAD tokens
    ids = [i for i in ids if i not in (cfg.EOS_IDX, cfg.PAD_IDX)]

    if id_to_token is not None:
        return ' '.join(id_to_token.get(i, f'<{i}>') for i in ids)
    else:
        return ' '.join(str(i) for i in ids)
