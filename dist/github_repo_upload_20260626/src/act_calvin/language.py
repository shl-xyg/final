from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize_language(text: str) -> list[str]:
    """Tokenize CALVIN language goals for deterministic feature hashing."""

    return TOKEN_RE.findall(text.lower())


def encode_language(text: str, dim: int) -> np.ndarray:
    """Encode a language goal as a fixed-size signed hashing vector.

    This avoids adding a separate language model dependency while preserving a
    language-conditioned ACT interface for CALVIN's short task instructions.
    """

    if dim < 1:
        raise ValueError("language embedding dim must be >= 1")
    vector = np.zeros((dim,), dtype=np.float32)
    for token in tokenize_language(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="little", signed=False)
        index = value % dim
        sign = 1.0 if ((value >> 63) & 1) == 0 else -1.0
        vector[index] += sign
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def fit_language_dim(vector: np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    if dim < 1:
        raise ValueError("language embedding dim must be >= 1")
    if arr.shape[0] == dim:
        return arr
    fitted = np.zeros((dim,), dtype=np.float32)
    length = min(dim, arr.shape[0])
    fitted[:length] = arr[:length]
    return fitted


def default_auto_lang_path(episode_env_map: str | Path, split: str) -> Path:
    """Infer the official CALVIN auto language annotation path from the map path."""

    return (
        Path(episode_env_map).expanduser().resolve().parent
        / "extracted"
        / "task_ABC_D"
        / split.lower()
        / "lang_annotations"
        / "auto_lang_ann.npy"
    )


def default_task_embeddings_path(episode_env_map: str | Path) -> Path:
    return (
        Path(episode_env_map).expanduser().resolve().parent
        / "extracted"
        / "task_ABC_D"
        / "validation"
        / "lang_annotations"
        / "embeddings.npy"
    )


def load_auto_lang_embeddings(path: str | Path) -> np.ndarray:
    auto_path = Path(path).expanduser().resolve()
    if not auto_path.exists():
        raise FileNotFoundError(auto_path)
    payload = np.load(auto_path, allow_pickle=True).item()
    embeddings = np.asarray(payload["language"]["emb"], dtype=np.float32)
    return embeddings.reshape(embeddings.shape[0], -1)


def load_task_embeddings(path: str | Path, dim: int | None = None) -> dict[str, np.ndarray]:
    embeddings_path = Path(path).expanduser().resolve()
    if not embeddings_path.exists():
        raise FileNotFoundError(embeddings_path)
    payload: dict[str, Any] = np.load(embeddings_path, allow_pickle=True).item()
    result: dict[str, np.ndarray] = {}
    for task, value in payload.items():
        vector = np.asarray(value["emb"], dtype=np.float32).reshape(-1)
        result[str(task)] = fit_language_dim(vector, dim) if dim is not None else vector
    return result


def language_text(*, task: str, annotation: str, source: str) -> str:
    if source == "annotation":
        return annotation
    if source == "task":
        return task.replace("_", " ")
    if source == "task_annotation":
        return f"{task.replace('_', ' ')}: {annotation}"
    raise ValueError(f"Unsupported language source: {source!r}")
