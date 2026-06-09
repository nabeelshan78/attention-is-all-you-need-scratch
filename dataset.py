"""
dataset.py - Sequence-to-Sequence Translation Dataset.

THE TASK: ENGLISH TO GERMAN TRANSLATION
---------------------------------------
This pipeline processes real-world bilingual data. It handles:
  1. Disk I/O (loading from JSONL files)
  2. Subword Tokenization (via trained HuggingFace BPE tokenizer)
  3. Truncation (enforcing cfg.max_seq_len)
  4. Dynamic Padding (in the collate_fn)

SEQUENCE FORMAT
---------------
Source input   (src):     [t1, t2, t3, ...] (Truncated to max_seq_len)
Decoder input  (tgt_in):  [<SOS>, t1, t2, t3, ...]
Decoder target (tgt_out): [t1, t2, t3, ..., <EOS>]

TRUNCATION LOGIC
----------------
Because sequences can be arbitrarily long, we must truncate them to fit our GPU memory budget (max_seq_len = 128).
For the target sequences, we truncate to max_seq_len - 1 to guarantee we always have room to append the <SOS> or <EOS> token without exceeding 128.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple
from tokenizers import Tokenizer
from config import cfg

class TranslationDataset(Dataset):
    """
    PyTorch Dataset for English-to-German Translation.

    Reads a JSONL file where each line is:
    {"translation": {"en": "Hello", "de": "Hallo"}}
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer_path: str = "bpe_tokenizer.json",
        max_len: int = cfg.max_seq_len
    ):
        """
        Parameters
        ----------
        jsonl_path : str
            Path to the train.jsonl, val.jsonl, or test.jsonl file.
        tokenizer_path : str
            Path to the saved BPE tokenizer JSON file.
        max_len : int
            Maximum absolute sequence length (from config).
        """
        self.max_len = max_len

        # 1. Load the Tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        # 2. Map Special Tokens (must match tokenizer training order)
        # Training order: ["<pad>", "<sos>", "<eos>", "<unk>"]
        self.PAD = self.tokenizer.token_to_id("<pad>")
        self.SOS = self.tokenizer.token_to_id("<sos>")
        self.EOS = self.tokenizer.token_to_id("<eos>")
        self.UNK = self.tokenizer.token_to_id("<unk>")

        # 3. Load Data from Disk
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item['translation'])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns one processed sample.
        """
        pair = self.data[idx]
        en_text = pair['en']
        de_text = pair['de']

        # 1. Encode text to integer IDs
        src_ids = self.tokenizer.encode(en_text).ids
        tgt_ids = self.tokenizer.encode(de_text).ids

        # 2. Truncate
        # Source can be truncated exactly to max_len
        src_ids = src_ids[:self.max_len]

        # Target must be truncated to max_len - 1 to leave room for SOS/EOS
        tgt_ids = tgt_ids[:self.max_len - 1]

        # 3. Convert to Tensors and apply offsets
        src = torch.tensor(src_ids, dtype=torch.long)

        # tgt_in: prepend SOS, no EOS
        tgt_in = torch.tensor([self.SOS] + tgt_ids, dtype=torch.long)

        # tgt_out: no SOS, append EOS
        tgt_out = torch.tensor(tgt_ids + [self.EOS], dtype=torch.long)

        return src, tgt_in, tgt_out


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    pad_idx: int = 0  # <pad> is ID 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate a batch of variable-length sequences into padded tensors.
    Uses Dynamic Padding.
    """
    srcs, tgt_ins, tgt_outs = zip(*batch)

    # Pad sequences to the longest item in THIS specific batch
    src_batch = pad_sequence(srcs, batch_first=True, padding_value=pad_idx)
    tgt_in_batch = pad_sequence(tgt_ins, batch_first=True, padding_value=pad_idx)
    tgt_out_batch = pad_sequence(tgt_outs, batch_first=True, padding_value=pad_idx)

    return src_batch, tgt_in_batch, tgt_out_batch


def build_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer_path: str,
    batch_size: int = cfg.batch_size,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    Build training and validation DataLoaders.
    """
    train_dataset = TranslationDataset(train_path, tokenizer_path)
    val_dataset = TranslationDataset(val_path, tokenizer_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,          # Shuffle training data
        collate_fn=lambda b: collate_fn(b, pad_idx=train_dataset.PAD),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,         # No need to shuffle validation data
        collate_fn=lambda b: collate_fn(b, pad_idx=val_dataset.PAD),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    print(f"[Dataset] Train: {len(train_dataset):,} samples | "
          f"Val: {len(val_dataset):,} samples | "
          f"Batch size: {batch_size}")

    return train_loader, val_loader
