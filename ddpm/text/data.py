import hashlib
import os
import urllib.request
from collections import Counter

import torch


class Vocab:
    def __init__(self) -> None:
        self.voc: dict[str, int] = {}
        self.inv_voc: dict[int, str] = {}
        self.masked = "_"

    def build(self, in_path: str, from_string: bool = False) -> None:
        """Ids in descending frequency, ties in first-seen order, so rebuilding from the same text
        gives the same ids (checkpoints don't store the vocabulary); mask_id = V."""
        c = Counter[str]()
        if from_string:
            c.update(in_path)
        else:
            with open(in_path, encoding="utf-8") as in_file:
                while chunk := in_file.read(4096):
                    c.update(chunk)
        self.voc = {k[0]: c_i for c_i, k in enumerate(c.most_common())}

        self.mask_id = len(self.voc)

        self.inv_voc = {v: k for k, v in self.voc.items()}

    def stoi(self, s: str) -> int:
        return self.voc[s]

    def itos(self, i: int) -> str:
        return self.inv_voc[i]

    def size(self) -> int:
        return len(self.voc)


def md5_hash_file(file_path: str) -> str:
    md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)

    return md5.hexdigest()


def download_tiny_shakespeare(root: str) -> str:
    """Path to Tiny Shakespeare under root, downloaded if missing or corrupt, MD5-checked."""
    md5_hash = "6fb458f1232090904fb40fe944165e91"
    os.makedirs(root, exist_ok=True)
    out_path = os.path.join(root, "tiny_shakespeare.txt")
    if os.path.isfile(out_path):
        if md5_hash_file(out_path) == md5_hash:
            print("already downloaded ", out_path)
            return out_path
        else:
            print("file corrupted ", out_path)

    print("downloading")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/refs/heads/master/data/tinyshakespeare/input.txt",
        out_path,
    )
    if md5_hash_file(out_path) != md5_hash:
        raise Exception("Failed download")
    print("downloaded, written to ", out_path)

    return out_path


def encode(in_str: str, vocab: Vocab, device: str) -> torch.Tensor:
    out = []
    for c in in_str:
        out.append(vocab.stoi(c))
    out_t = torch.tensor(out, dtype=torch.long, device=device)

    return out_t


def encode_file(in_path: str, vocab: Vocab, device: str) -> torch.Tensor:
    out: list[torch.Tensor] = []
    with open(in_path, encoding="utf-8") as in_file:
        while chunk := in_file.read(4096):
            out.append(encode(chunk, vocab, device))
    out_t = torch.cat(out, dim=0)
    return out_t


def sample_batch(
    encoded: torch.Tensor, batch_size: int, seq_len: int, generator: torch.Generator
) -> torch.Tensor:
    """(batch_size, seq_len) contiguous crops at uniformly random starts."""
    text_len = encoded.shape[0]
    starts = torch.randint(0, text_len - seq_len + 1, (1, batch_size), generator=generator)
    offsets = torch.arange(seq_len, device=encoded.device)
    inds = starts.T.to(encoded.device) + offsets
    return encoded[inds]


def decode(t: torch.Tensor, vocab: Vocab) -> str:
    """(L,) long -> str, mask_id drawn as `vocab.masked`; any other unknown id raises."""
    out = ""

    for c in t.tolist():
        if c in vocab.inv_voc:
            out += vocab.itos(c)
        elif c == vocab.mask_id:
            out += vocab.masked
        else:
            raise Exception(f"token id {c} is neither a character nor mask_id {vocab.mask_id}")

    return out


def split(encoded: torch.Tensor, train_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    split_idx = int(encoded.shape[0] * train_ratio)
    encoded_train = encoded[:split_idx]
    encoded_val = encoded[split_idx:]

    return encoded_train, encoded_val


def val_chunks(encoded: torch.Tensor, seq_len: int) -> torch.Tensor:
    """(n, seq_len) back-to-back chunks; a tail shorter than seq_len is dropped."""
    n = encoded.shape[0] // seq_len
    return encoded[: n * seq_len].view(n, seq_len)
