"""Publish complete checkpoints atomically; select steps by numeric basename."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import re
import shutil
import tempfile

import torch

from .modeling import CONFIG_NAME, WEIGHTS_NAME, save_memory

OPTIMIZER_NAME = "optimizer.pt"
RUNTIME_NAME = "training_state.pt"


def save_training_checkpoint(model, optimizer, destination, metadata, *, runtime_state=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        version = 3 if runtime_state is not None else 2
        save_memory(model, str(staging), {**metadata, "training_version": version})
        torch.save(optimizer.state_dict(), staging / OPTIMIZER_NAME)
        if runtime_state is not None:
            torch.save(runtime_state, staging / RUNTIME_NAME)
        (staging / "COMPLETE").write_text(f"{version}\n")
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def latest_checkpoint(root):
    """Allow old checkpoints and MOVED.txt pointers; never choose weight-only snapshots."""
    candidates = []
    for entry in Path(root).glob("step_*"):
        match = re.fullmatch(r"step_(\d+)", entry.name)
        if match:
            candidates.append((int(match[1]), entry))
    for step, entry in sorted(candidates, reverse=True):
        moved = entry / "MOVED.txt"
        if moved.is_file():
            target = Path(moved.read_text().strip())
            entry = target if target.is_absolute() else entry / target
        required = [entry / name for name in (CONFIG_NAME, WEIGHTS_NAME, OPTIMIZER_NAME)]
        if not all(p.is_file() and p.stat().st_size > 0 for p in required):
            continue
        try:
            metadata = json.loads((entry / CONFIG_NAME).read_text())
        except (OSError, ValueError):
            continue
        if metadata.get("step") != step:
            continue
        if metadata.get("training_version", 1) >= 2 and not (entry / "COMPLETE").is_file():
            continue
        if metadata.get("training_version", 1) >= 3:
            runtime = entry / RUNTIME_NAME
            if not runtime.is_file() or runtime.stat().st_size == 0:
                continue
        return str(entry.resolve())
    return None


def archive_checkpoint(source, destination, *, move=False):
    """Copy a complete checkpoint to a mounted store, repairing partial previous copies.

    Keep optimizer states. Local large files are removed only with explicit move=True,
    after all files verify and an atomic MOVED.txt pointer has been written.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("archive source and destination must differ")
    if not (source / "COMPLETE").is_file():
        raise ValueError("only completed checkpoints may be archived")
    if (source / "MOVED.txt").exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "COMPLETE").unlink(missing_ok=True)

    def digest(path):
        result = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                result.update(block)
        return result.digest()

    for file in source.iterdir():
        if not file.is_file() or file.name in ("COMPLETE", "MOVED.txt") or file.name.startswith("."):
            continue
        target = destination / file.name
        expected = digest(file)
        if target.is_file() and digest(target) == expected:
            continue
        temporary = destination / ("." + file.name + ".copying")
        shutil.copyfile(file, temporary)
        if digest(temporary) != expected:
            raise OSError(f"archive checksum mismatch: {file.name}")
        os.replace(temporary, target)
    marker = destination / ".COMPLETE.tmp"
    marker.write_text((source / "COMPLETE").read_text())
    os.replace(marker, destination / "COMPLETE")
    if move:
        pointer = source / ".MOVED.tmp"
        pointer.write_text(str(destination.resolve()) + "\n")
        os.replace(pointer, source / "MOVED.txt")
        for name in (WEIGHTS_NAME, OPTIMIZER_NAME):
            (source / name).unlink()
