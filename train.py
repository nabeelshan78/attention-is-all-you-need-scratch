"""
train.py - Full Training Loop for the Transformer.

WHAT THIS FILE DOES
-------------------
Ties together:
  - Dataset / DataLoaders
  - Model (Transformer)
  - Loss function (LabelSmoothingLoss)
  - Optimizer (Adam)
  - LR Scheduler (NoamScheduler)
  - Training loop (forward, loss, backward, step)
  - Validation loop
  - Checkpointing (save/load best model)
  - Logging (loss, LR, accuracy per epoch)

THE TRAINING LOOP IN DETAIL
---------------------------
For each epoch:
  1. Set model to train mode (enables dropout)
  2. For each batch:
     a. Move src, tgt_in, tgt_out to device
     b. Forward pass: logits = model(src, tgt_in)
     c. Compute loss: loss = criterion(logits, tgt_out)
     d. Backward pass: loss.backward()  (computes gradients)
     e. Gradient clipping: clip_grad_norm_(model.parameters(), max_norm=1.0)
     f. Optimizer step: optimizer.step()  (updates parameters)
     g. LR scheduler step: scheduler.step()
     h. Zero gradients: optimizer.zero_grad()
  3. Compute training metrics (loss, token accuracy)
  4. Run validation loop (same as training but no backward/step)
  5. Save checkpoint if validation loss improved
  6. Print epoch summary

TEACHER FORCING:
During training, tgt_in = [SOS, t1, t2, ..., t_{n-1}] (ground truth shifted right).
The model predicts tgt_out = [t1, t2, ..., t_n, EOS].
All positions are computed in parallel (causal mask prevents cheating).

TOKEN ACCURACY:
We measure how many output tokens (excluding PAD) the model correctly predicts.
This is a proxy for sequence accuracy - a sequence is only "correct" if ALL
tokens are correct, but token accuracy is easier to compute and more informative
during early training.

CHECKPOINTING:
We save:
  - model.state_dict(): all parameters
  - optimizer.state_dict(): Adam moment estimates (allows resuming training)
  - scheduler.state_dict(): current step (for LR schedule continuity)
  - epoch, best_val_loss: metadata
"""

import os
import time
import math
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional

from config import cfg
from transformer import build_transformer, Transformer
from loss import LabelSmoothingLoss
from optimizer import build_optimizer, NoamScheduler
from dataset import build_dataloaders


def compute_token_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int = cfg.PAD_IDX
) -> Tuple[int, int]:
    """
    Compute number of correctly predicted tokens (ignoring PAD).

    Parameters
    ----------
    logits : [B, S, vocab_size]
    targets : [B, S]
    pad_idx : int

    Returns
    -------
    correct : int   — number of correct non-PAD predictions
    total : int     — total number of non-PAD tokens
    """
    # predicted: [B, S] - argmax over vocab dimension
    predicted = logits.argmax(dim=-1)

    # Create mask for non-PAD positions
    non_pad = (targets != pad_idx)

    # Count correct predictions at non-PAD positions
    correct = ((predicted == targets) & non_pad).sum().item()
    total = non_pad.sum().item()

    return correct, total


def train_epoch(
    model: Transformer,
    dataloader,
    criterion: LabelSmoothingLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: NoamScheduler,
    device: torch.device,
    clip_grad: float = 1.0
) -> Dict[str, float]:
    """
    Run one training epoch.
    Returns
    -------
    metrics : dict with keys 'loss', 'accuracy', 'lr'
    """
    model.train()   # Enable dropout

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    for batch_idx, (src, tgt_in, tgt_out) in enumerate(dataloader):
        # Move to device
        src = src.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)

        # Forward pass
        # model(src, tgt_in) -> logits [B, S_tgt, vocab_size]
        # src:     [B, S_src]   — source token IDs
        # tgt_in:  [B, S_tgt]   — target input (SOS + target[:-1])
        # tgt_out: [B, S_tgt]   — target output (target[1:] + EOS)
        logits = model(src, tgt_in) # logits: [B, S_tgt, vocab_size]

        # Compute loss, criterion expects logits [B, S, V] and targets [B, S]
        loss = criterion(logits, tgt_out)

        # Backward pass
        loss.backward() # Gradients now accumulated in all parameter .grad attributes

        # Gradient clipping
        # Clip the global L2 norm of all gradients to `clip_grad`.
        # If norm > clip_grad, all gradients are scaled down proportionally.
        # This prevents exploding gradients from destabilizing training.
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        # 1. Calculate and apply the EXACT learning rate for this step
        scheduler.step()

        # 2. Update the weights using that precise learning rate
        optimizer.step()

        # 3. Clear gradients
        optimizer.zero_grad(set_to_none=True)

        # Accumulate metrics
        with torch.no_grad():
            correct, total = compute_token_accuracy(logits, tgt_out, cfg.PAD_IDX)
            total_correct += correct
            total_tokens += total
            total_loss += loss.item()
            num_batches += 1

    return {
        'loss': total_loss / max(num_batches, 1),
        'accuracy': total_correct / max(total_tokens, 1),
        'lr': scheduler.current_lr
    }

@torch.no_grad()
def validate_epoch(
    model: Transformer,
    dataloader,
    criterion: LabelSmoothingLoss,
    device: torch.device
) -> Dict[str, float]:
    """
    Run one validation epoch (no gradient computation).

    Returns
    ───────
    metrics : dict with keys 'loss', 'accuracy'
    """
    model.eval()   # Disable dropout

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    for src, tgt_in, tgt_out in dataloader:
        src = src.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)

        logits = model(src, tgt_in)
        loss = criterion(logits, tgt_out)

        correct, total = compute_token_accuracy(logits, tgt_out, cfg.PAD_IDX)
        total_correct += correct
        total_tokens += total
        total_loss += loss.item()
        num_batches += 1

    return {
        'loss': total_loss / max(num_batches, 1),
        'accuracy': total_correct / max(total_tokens, 1)
    }

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler: NoamScheduler,
    epoch: int,
    val_loss: float,
    history: dict,
    path: str
):
    """Save model checkpoint to disk."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'history': history
    }
    torch.save(checkpoint, path)
    print(f"  [Checkpoint] Saved to {path}")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[NoamScheduler] = None
) -> Tuple[int, dict, float]:
    """
    Load model (and optionally optimizer/scheduler) from checkpoint.

    Returns
    -------
    epoch : int — epoch number when checkpoint was saved
    """
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    val_loss = checkpoint.get('val_loss', float('inf'))

    # Extract history, or return empty template if it doesn't exist
    empty_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
    history = checkpoint.get('history', empty_history)

    print(f"  [Checkpoint] Loaded from {path} (epoch {epoch}, val_loss {val_loss:.4f})")
    return epoch, history, val_loss

def train(
    num_epochs: int = cfg.num_epochs,
    batch_size: int = cfg.batch_size,
    checkpoint_dir: str = './checkpoints',
    resume_from: str = None,
    task: str = 'translation',
    verbose: bool = True
):
    """
    Full training pipeline.

    Parameters
    ----------
    num_epochs : int
    batch_size : int
    checkpoint_dir : str - directory to save checkpoints
    resume_from : str or None - path to checkpoint to resume from
    task : str - Task identifier
    verbose : bool - print per-epoch stats
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = cfg.device
    print(f"\n{'='*60}")
    print(f"TRAINING TRANSFORMER")
    print(f"{'='*60}")
    print(f"  Device:   {device}")
    print(f"  Task:     {task}")
    print(f"  Epochs:   {num_epochs}")
    print(f"  Batch:    {batch_size}")

    # Build components
    model = build_transformer(cfg).to(device)
    criterion = LabelSmoothingLoss(
        vocab_size=cfg.vocab_size,
        pad_idx=cfg.PAD_IDX,
        label_smoothing=cfg.label_smoothing
    )
    optimizer, scheduler = build_optimizer(model, cfg)

    # Initialize State
    start_epoch = 0
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}

    # Optionally resume from checkpoint
    if resume_from and os.path.exists(resume_from):
        start_epoch, history, current_val_loss = load_checkpoint(
            resume_from, model, optimizer, scheduler
        )

        # Find the TRUE historical best from the loaded history list!
        if len(history['val_loss']) > 0:
            best_val_loss = min(history['val_loss'])
        else:
            best_val_loss = current_val_loss

    # Build dataloaders
    train_loader, val_loader = build_dataloaders(
        train_path="data/train.jsonl",
        val_path="data/val.jsonl",
        tokenizer_path="bpe_tokenizer.json",
        batch_size=batch_size
    )

    # Training loop
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}  "
          f"{'Train Acc':>10}  {'Val Acc':>10}  {'LR':>12}  {'Time':>8}")
    print("-" * 78)

    for epoch in range(start_epoch + 1, num_epochs + 1):
        t_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )

        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)

        # Log
        elapsed = time.time() - t_start
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['lr'].append(train_metrics['lr'])

        if verbose:
            print(f"{epoch:>6}  {train_metrics['loss']:>10.4f}  {val_metrics['loss']:>10.4f}  "
                  f"{train_metrics['accuracy']:>10.2%}  {val_metrics['accuracy']:>10.2%}  "
                  f"{train_metrics['lr']:>12.2e}  {elapsed:>6.1f}s")

        # Checkpoint if best
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics['loss'], history,
                os.path.join(checkpoint_dir, 'best_model.pt')
            )

        # Save latest checkpoint every 5 epochs (for resuming)
        if epoch % 5 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics['loss'], history,
                os.path.join(checkpoint_dir, f'epoch_{epoch:04d}.pt')
            )

    print(f"\n{'='*60}")
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"{'='*60}")

    return model, history
