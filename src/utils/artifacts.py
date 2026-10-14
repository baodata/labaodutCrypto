"""Immutable run directories, atomic artifacts, and reproducibility metadata."""

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch


def new_run_directory(root: Path, name: str) -> Path:
    path = Path(root) / f"{name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def file_hashes(paths, root: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted({Path(p).resolve() for p in paths}):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        key = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        hashes[key] = digest.hexdigest()
    return hashes


def atomic_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def atomic_csv(frame, path: Path, **kwargs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{uuid4().hex}.tmp")
    frame.to_csv(tmp, **kwargs)
    tmp.replace(path)


def rng_state() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    s = state["numpy"]
    np.random.set_state((s[0], np.asarray(s[1], dtype=np.uint32), s[2], s[3], s[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
