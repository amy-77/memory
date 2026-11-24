"""Offline, unquantized Top-K teacher targets aligned to response prediction positions."""
import hashlib
import json
from pathlib import Path

import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_key(row):
    return hashlib.sha256(str(row["id"]).encode()).hexdigest()


def token_digest(batch):
    ids = batch["input_ids"].tolist(), batch["labels"].tolist()
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


@torch.no_grad()
def make_topk_targets(hidden, head, labels, top_k, temperature=1., chunk=128):
    if not 1 <= top_k <= head.weight.shape[0] or temperature <= 0 or chunk < 1:
        raise ValueError("invalid teacher cache parameters")
    selected = hidden[:, :-1][labels[:, 1:] != -100]
    indices, log_probs = [], []
    for offset in range(0, len(selected), chunk):
        logits = head(selected[offset:offset+chunk]).float() / temperature
        values, index = logits.topk(top_k, dim=-1)
        indices.append(index.cpu())
        log_probs.append(values.log_softmax(-1).cpu())
    if not indices:
        raise ValueError("teacher cache requires supervised response tokens")
    return torch.cat(indices), torch.cat(log_probs)


class TeacherCache:
    def __init__(self, directory, data_path, model, top_k, temperature):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "manifest.json").read_text())
        expected = {"data_sha256": file_sha256(data_path), "model": model,
                    "top_k": top_k, "temperature": temperature, "format_version": 1}
        for key, value in expected.items():
            if self.metadata.get(key) != value:
                raise ValueError(f"teacher cache {key} does not match this training run")

    def get(self, row, batch):
        record = torch.load(self.directory / (row_key(row) + ".pt"), map_location="cpu", weights_only=True)
        if record["token_digest"] != token_digest(batch):
            raise ValueError(f"teacher cache token alignment mismatch for {row['id']}")
        indices, log_probs = record["indices"], record["log_probs"]
        shape = (int((batch["labels"][:, 1:] != -100).sum()), self.metadata["top_k"])
        if tuple(indices.shape) != shape or tuple(log_probs.shape) != shape:
            raise ValueError("teacher cache target shape mismatch")
        if indices.dtype != torch.long or (indices < 0).any() or not torch.isfinite(log_probs).all():
            raise ValueError("invalid teacher cache values")
        if not torch.allclose(log_probs.logsumexp(-1), torch.zeros(shape[0]), atol=1e-5):
            raise ValueError("teacher Top-K probabilities must be normalized")
        return indices, log_probs
