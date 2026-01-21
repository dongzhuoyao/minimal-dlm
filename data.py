"""
Data utilities - Simple binary data format following nanoGPT patterns.
"""
import os
import pickle
import requests
import numpy as np
import torch
from typing import Tuple, Dict, Optional


def get_batch(split: str, data_dir: str, batch_size: int, block_size: int,
              device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    """Get random batch from memory-mapped binary data file."""
    data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def load_meta(data_dir: str) -> Optional[Dict]:
    """Load metadata (vocab_size, stoi, itos) from meta.pkl."""
    path = os.path.join(data_dir, "meta.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def prepare_shakespeare_char(data_dir: str = "data/shakespeare_char") -> Dict:
    """Download and prepare character-level Shakespeare dataset."""
    os.makedirs(data_dir, exist_ok=True)
    input_file = os.path.join(data_dir, "input.txt")

    if not os.path.exists(input_file):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Downloading Shakespeare dataset...")
        with open(input_file, "w") as f:
            f.write(requests.get(url).text)

    with open(input_file, "r") as f:
        data = f.read()

    chars = sorted(set(data))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s]
    n = len(data)
    train_ids = np.array(encode(data[:int(n*0.9)]), dtype=np.uint16)
    val_ids = np.array(encode(data[int(n*0.9):]), dtype=np.uint16)

    train_ids.tofile(os.path.join(data_dir, "train.bin"))
    val_ids.tofile(os.path.join(data_dir, "val.bin"))

    meta = {"vocab_size": len(chars), "stoi": stoi, "itos": itos}
    with open(os.path.join(data_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"Prepared {len(chars)} chars, {len(train_ids)} train, {len(val_ids)} val tokens")
    return meta


class TextDataset:
    """Simple wrapper for character-level encoding/decoding."""
    def __init__(self, data_dir: str):
        meta = load_meta(data_dir)
        self.vocab_size = meta["vocab_size"]
        self.stoi, self.itos = meta["stoi"], meta["itos"]

    def encode(self, text: str) -> list:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: list) -> str:
        return "".join([self.itos[i] for i in ids if i in self.itos])
