from __future__ import annotations

import json
import math
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .language import default_auto_lang_path, encode_language, fit_language_dim, language_text, load_auto_lang_embeddings


EPISODE_RE = re.compile(r"episode_(\d{7})\.npz$")


@dataclass(frozen=True)
class SceneRange:
    scene: str
    start: int
    end: int

    @property
    def env_label(self) -> str:
        suffix = self.scene.rsplit("_", 1)[-1].upper()
        return suffix if suffix in {"A", "B", "C", "D"} else self.scene

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass
class DatasetSummary:
    root: str
    split: str
    environments: list[str]
    scenes: list[dict]
    chunk_size: int
    image_size: int
    action_key: str
    total_files: int
    total_samples: int
    max_samples: int | None
    state_dim: int
    language_conditioning: bool
    language_embedding_dim: int
    language_embedding_backend: str
    language_source: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


@dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    split: str
    env: str
    length: int
    task: str
    annotation: str
    language_embedding_index: int


def _episode_id(path: Path) -> int:
    match = EPISODE_RE.match(path.name)
    if match is None:
        raise ValueError(f"Not a CALVIN episode file: {path}")
    return int(match.group(1))


def available_episode_ids(split_dir: Path) -> set[int]:
    return {_episode_id(path) for path in split_dir.glob("episode_*.npz")}


def load_scene_ranges(split_dir: Path) -> list[SceneRange]:
    scene_info = split_dir / "scene_info.npy"
    if scene_info.exists():
        raw = np.load(scene_info, allow_pickle=True).item()
        ranges = []
        for scene, bounds in raw.items():
            if len(bounds) != 2:
                raise ValueError(f"Unexpected scene range for {scene}: {bounds}")
            ranges.append(SceneRange(scene=str(scene), start=int(bounds[0]), end=int(bounds[1])))
        return sorted(ranges, key=lambda r: (r.env_label, r.start))

    ids = sorted(available_episode_ids(split_dir))
    if not ids:
        raise FileNotFoundError(f"No episode_*.npz files found in {split_dir}")
    return [SceneRange(scene="unknown", start=ids[0], end=ids[-1])]


def normalize_environment_filter(environments: str | Iterable[str] | None) -> set[str] | None:
    if environments is None:
        return None
    if isinstance(environments, str):
        parts = [p.strip().upper() for p in environments.split(",") if p.strip()]
    else:
        parts = [str(p).strip().upper() for p in environments if str(p).strip()]
    if not parts or "ALL" in parts:
        return None
    return set(parts)


def filter_scene_ranges(ranges: list[SceneRange], environments: str | Iterable[str] | None) -> list[SceneRange]:
    requested = normalize_environment_filter(environments)
    if requested is None:
        return ranges
    selected = [scene for scene in ranges if scene.env_label in requested or scene.scene.upper() in requested]
    if not selected:
        available = ", ".join(f"{r.scene}({r.env_label})" for r in ranges)
        raise ValueError(f"No scenes match environments {sorted(requested)}. Available scenes: {available}")
    return selected


def _load_image_array(arr: np.ndarray, image_size: int) -> torch.Tensor:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr)
    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).contiguous().float() / 255.0
    return tensor


def _load_image_cell(cell: object, image_size: int) -> torch.Tensor:
    if isinstance(cell, dict):
        payload = cell.get("bytes")
        if payload is None and cell.get("path"):
            payload = Path(str(cell["path"])).read_bytes()
        if payload is None:
            raise ValueError("Image cell has neither bytes nor path")
        with Image.open(BytesIO(payload)) as image:
            arr = np.array(image.convert("RGB"), copy=True)
        return _load_image_array(arr, image_size)
    if isinstance(cell, (bytes, bytearray)):
        with Image.open(BytesIO(cell)) as image:
            arr = np.array(image.convert("RGB"), copy=True)
        return _load_image_array(arr, image_size)
    if isinstance(cell, np.ndarray):
        return _load_image_array(cell, image_size)
    raise TypeError(f"Unsupported image cell type: {type(cell)!r}")


def _normalize_action_array(action: np.ndarray) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32).copy()
    arr[..., -1] = np.where(arr[..., -1] == 0.0, -1.0, arr[..., -1])
    return arr


def _limit_samples(indices: list, max_samples: int | None) -> list:
    if max_samples is None or len(indices) <= max_samples:
        return indices
    if max_samples < 1:
        raise ValueError("max_samples must be >= 1 when provided")
    selected = np.linspace(0, len(indices) - 1, num=max_samples, dtype=np.int64)
    return [indices[int(i)] for i in selected]


class CalvinActionChunkDataset(Dataset):
    """CALVIN raw `.npz` dataset that returns LeRobot ACT-ready action chunks.

    CALVIN stores one transition per `.npz` file. ACT expects the current
    observation and a future chunk of actions. This dataset constructs that
    chunk lazily and masks padded entries at the end of a scene range.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        environments: str | Iterable[str] | None = None,
        chunk_size: int = 100,
        image_size: int = 96,
        action_key: str = "rel_actions",
        max_samples: int | None = None,
        sample_stride: int = 1,
        language_conditioning: bool = False,
        language_embedding_dim: int = 0,
        language_embedding_backend: str = "hash",
        language_source: str = "annotation",
    ) -> None:
        if language_conditioning:
            raise ValueError("language_conditioning is only supported for hf_parquet datasets")
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.split_dir = self.root / split
        self.chunk_size = int(chunk_size)
        self.image_size = int(image_size)
        self.action_key = action_key
        self.max_samples = max_samples
        self.sample_stride = max(1, int(sample_stride))
        self.language_conditioning = False
        self.language_embedding_dim = 0
        self.language_embedding_backend = language_embedding_backend
        self.language_source = language_source
        self.state_dim = 15

        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if self.image_size < 16:
            raise ValueError("image_size must be >= 16")
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"CALVIN split directory not found: {self.split_dir}")

        self.available_ids = available_episode_ids(self.split_dir)
        if not self.available_ids:
            raise FileNotFoundError(f"No CALVIN episode files found in {self.split_dir}")

        ranges = load_scene_ranges(self.split_dir)
        self.scene_ranges = filter_scene_ranges(ranges, environments)
        self.indices: list[tuple[int, SceneRange]] = []
        for scene in self.scene_ranges:
            for episode_id in range(scene.start, scene.end + 1, self.sample_stride):
                if episode_id in self.available_ids:
                    self.indices.append((episode_id, scene))

        self.indices = _limit_samples(self.indices, max_samples)

        if not self.indices:
            raise ValueError(f"No usable samples found for {self.split_dir}")

    def episode_path(self, episode_id: int) -> Path:
        return self.split_dir / f"episode_{episode_id:07d}.npz"

    def _load_npz(self, episode_id: int) -> np.lib.npyio.NpzFile:
        path = self.episode_path(episode_id)
        if not path.exists():
            raise FileNotFoundError(path)
        return np.load(path, allow_pickle=True)

    def _load_image(self, arr: np.ndarray) -> torch.Tensor:
        return _load_image_array(arr, self.image_size)

    def _load_action_chunk(self, start_id: int, scene: SceneRange) -> tuple[torch.Tensor, torch.Tensor]:
        actions = np.zeros((self.chunk_size, 7), dtype=np.float32)
        is_pad = np.ones((self.chunk_size,), dtype=bool)
        for offset in range(self.chunk_size):
            episode_id = start_id + offset
            if episode_id > scene.end or episode_id not in self.available_ids:
                continue
            with self._load_npz(episode_id) as data:
                if self.action_key not in data:
                    raise KeyError(f"Action key {self.action_key!r} missing in {self.episode_path(episode_id)}")
                actions[offset] = _normalize_action_array(data[self.action_key])
            is_pad[offset] = False
        return torch.from_numpy(actions), torch.from_numpy(is_pad)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        episode_id, scene = self.indices[idx]
        with self._load_npz(episode_id) as data:
            state = torch.from_numpy(data["robot_obs"].astype(np.float32))
            static = self._load_image(data["rgb_static"])
            gripper = self._load_image(data["rgb_gripper"])
        actions, is_pad = self._load_action_chunk(episode_id, scene)
        return {
            "observation.state": state,
            "observation.images.static": static,
            "observation.images.gripper": gripper,
            "action": actions,
            "action_is_pad": is_pad,
        }

    def summary(self) -> DatasetSummary:
        return DatasetSummary(
            root=str(self.root),
            split=self.split,
            environments=sorted({scene.env_label for scene in self.scene_ranges}),
            scenes=[asdict(scene) for scene in self.scene_ranges],
            chunk_size=self.chunk_size,
            image_size=self.image_size,
            action_key=self.action_key,
            total_files=len(self.available_ids),
            total_samples=len(self.indices),
            max_samples=self.max_samples,
            state_dim=self.state_dim,
            language_conditioning=self.language_conditioning,
            language_embedding_dim=self.language_embedding_dim,
            language_embedding_backend=self.language_embedding_backend,
            language_source=self.language_source,
        )

    def environment_coverage(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, scene in self.indices:
            counts[scene.env_label] = counts.get(scene.env_label, 0) + 1
        return counts


def split_train_val_samples(total: int, val_fraction: float) -> tuple[int, int]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    val = max(1, int(math.floor(total * val_fraction)))
    train = max(1, total - val)
    return train, val


class CalvinParquetActionChunkDataset(Dataset):
    """CALVIN LeRobot-style per-episode Parquet dataset.

    This loader is for converted CALVIN ABC-D datasets whose episode order
    matches the official CALVIN language annotations. The required
    ``episode_env_map`` is generated from the official ``scene_info.npy`` and
    ``auto_lang_ann.npy`` files, so A/B/C/D filtering remains tied to the
    benchmark metadata rather than inferred from converted filenames.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        environments: str | Iterable[str] | None = None,
        episode_env_map: str | Path | None = None,
        chunk_size: int = 100,
        image_size: int = 96,
        action_key: str = "action",
        max_samples: int | None = None,
        sample_stride: int = 1,
        top_image_key: str = "observation.images.top",
        wrist_image_key: str = "observation.images.wrist",
        state_key: str = "observation.state",
        cache_size: int = 16,
        language_conditioning: bool = False,
        language_embedding_dim: int = 64,
        language_embedding_backend: str = "hash",
        language_source: str = "annotation",
        language_embeddings_path: str | Path | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.chunk_size = int(chunk_size)
        self.image_size = int(image_size)
        self.action_key = action_key
        self.max_samples = max_samples
        self.sample_stride = max(1, int(sample_stride))
        self.top_image_key = top_image_key
        self.wrist_image_key = wrist_image_key
        self.state_key = state_key
        self.cache_size = max(1, int(cache_size))
        self.language_conditioning = bool(language_conditioning)
        self.language_embedding_backend = language_embedding_backend
        self.language_embedding_dim = int(language_embedding_dim) if self.language_conditioning else 0
        self.language_source = language_source
        self.language_embeddings_path = Path(language_embeddings_path).expanduser().resolve() if language_embeddings_path else None
        self._language_embeddings: np.ndarray | None = None
        self._episode_cache: OrderedDict[int, dict[str, object]] = OrderedDict()

        if self.language_conditioning and self.language_embedding_dim < 1:
            raise ValueError("language_embedding_dim must be >= 1 when language_conditioning is enabled")
        if self.language_embedding_backend not in {"hash", "calvin_sbert"}:
            raise ValueError(f"Unsupported language embedding backend: {self.language_embedding_backend}")

        if episode_env_map is None:
            raise ValueError("episode_env_map is required for hf_parquet datasets")
        self.episode_env_map = Path(episode_env_map).expanduser().resolve()
        if not self.episode_env_map.exists():
            raise FileNotFoundError(self.episode_env_map)
        if not (self.root / "data").is_dir():
            raise FileNotFoundError(f"LeRobot Parquet data directory not found: {self.root / 'data'}")
        episodes_path = self.root / "meta" / "episodes.jsonl"
        if not episodes_path.exists():
            raise FileNotFoundError(episodes_path)

        lengths: dict[int, int] = {}
        with episodes_path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                lengths[int(row["episode_index"])] = int(row["length"])

        requested_envs = normalize_environment_filter(environments)
        requested_split = split.lower()
        split_language_indices: dict[str, int] = {}
        self.episodes: list[EpisodeRecord] = []
        with self.episode_env_map.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                env = str(row["env"]).upper()
                row_split = str(row["split"]).lower()
                language_embedding_index = split_language_indices.get(row_split, 0)
                split_language_indices[row_split] = language_embedding_index + 1
                if requested_split not in {"all", row_split}:
                    continue
                if requested_envs is not None and env not in requested_envs:
                    continue
                episode_index = int(row["episode_index"])
                if episode_index not in lengths:
                    raise KeyError(f"Episode {episode_index} missing from {episodes_path}")
                self.episodes.append(
                    EpisodeRecord(
                        episode_index=episode_index,
                        split=row_split,
                        env=env,
                        length=lengths[episode_index],
                        task=str(row.get("task", "")),
                        annotation=str(row.get("annotation", "")),
                        language_embedding_index=language_embedding_index,
                    )
                )

        if not self.episodes:
            raise ValueError(f"No episodes selected for split={split!r}, environments={environments!r}")

        self.indices: list[tuple[int, int]] = []
        for local_episode_idx, record in enumerate(self.episodes):
            for frame_idx in range(0, record.length, self.sample_stride):
                self.indices.append((local_episode_idx, frame_idx))

        self.indices = _limit_samples(self.indices, max_samples)
        if not self.indices:
            raise ValueError("No usable Parquet frame samples selected")
        first_episode = self._load_episode(self.episodes[0].episode_index)
        base_state = np.asarray(first_episode["state"], dtype=np.float32)
        self.robot_state_dim = int(base_state.shape[1])
        if self.language_conditioning and self.language_embedding_backend == "calvin_sbert":
            path = self.language_embeddings_path or default_auto_lang_path(self.episode_env_map, split)
            self._language_embeddings = load_auto_lang_embeddings(path)
            self.language_embeddings_path = path
            if self._language_embeddings.shape[1] != self.language_embedding_dim:
                self.language_embedding_dim = int(self._language_embeddings.shape[1])
        self.state_dim = self.robot_state_dim + self.language_embedding_dim

    def episode_path(self, episode_index: int) -> Path:
        chunk = episode_index // 1000
        return self.root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"

    def _load_episode(self, episode_index: int) -> dict[str, object]:
        cached = self._episode_cache.get(episode_index)
        if cached is not None:
            self._episode_cache.move_to_end(episode_index)
            return cached

        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("pyarrow is required for hf_parquet CALVIN datasets") from exc

        path = self.episode_path(episode_index)
        if not path.exists():
            raise FileNotFoundError(path)
        table = pq.read_table(
            path,
            columns=[self.top_image_key, self.wrist_image_key, self.state_key, self.action_key],
        )
        rows = table.to_pylist()
        episode = {
            "top": [row[self.top_image_key] for row in rows],
            "wrist": [row[self.wrist_image_key] for row in rows],
            "state": np.asarray([row[self.state_key] for row in rows], dtype=np.float32),
            "action": _normalize_action_array(np.asarray([row[self.action_key] for row in rows], dtype=np.float32)),
        }
        self._episode_cache[episode_index] = episode
        self._episode_cache.move_to_end(episode_index)
        while len(self._episode_cache) > self.cache_size:
            self._episode_cache.popitem(last=False)
        return episode

    def _load_action_chunk(self, actions_arr: np.ndarray, start_frame: int) -> tuple[torch.Tensor, torch.Tensor]:
        actions = np.zeros((self.chunk_size, 7), dtype=np.float32)
        is_pad = np.ones((self.chunk_size,), dtype=bool)
        for offset in range(self.chunk_size):
            frame_idx = start_frame + offset
            if frame_idx >= len(actions_arr):
                continue
            actions[offset] = actions_arr[frame_idx]
            is_pad[offset] = False
        return torch.from_numpy(actions), torch.from_numpy(is_pad)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        local_episode_idx, frame_idx = self.indices[idx]
        record = self.episodes[local_episode_idx]
        episode = self._load_episode(record.episode_index)
        state_arr = episode["state"]
        action_arr = episode["action"]
        top_images = episode["top"]
        wrist_images = episode["wrist"]
        state = torch.from_numpy(state_arr[frame_idx].astype(np.float32, copy=False))
        if self.language_conditioning:
            if self.language_embedding_backend == "calvin_sbert":
                if self._language_embeddings is None:
                    raise RuntimeError("CALVIN language embeddings were not loaded")
                lang_arr = fit_language_dim(self._language_embeddings[record.language_embedding_index], self.language_embedding_dim)
            else:
                text = language_text(task=record.task, annotation=record.annotation, source=self.language_source)
                lang_arr = encode_language(text, self.language_embedding_dim)
            lang = torch.from_numpy(lang_arr)
            state = torch.cat([state, lang], dim=0)
        static = _load_image_cell(top_images[frame_idx], self.image_size)
        gripper = _load_image_cell(wrist_images[frame_idx], self.image_size)
        actions, is_pad = self._load_action_chunk(action_arr, frame_idx)
        return {
            "observation.state": state,
            "observation.images.static": static,
            "observation.images.gripper": gripper,
            "action": actions,
            "action_is_pad": is_pad,
        }

    def summary(self) -> DatasetSummary:
        env_counts = self.environment_coverage()
        return DatasetSummary(
            root=str(self.root),
            split=self.split,
            environments=sorted(env_counts),
            scenes=[
                {"scene": f"calvin_scene_{env}", "env_label": env, "episodes": count}
                for env, count in sorted(env_counts.items())
            ],
            chunk_size=self.chunk_size,
            image_size=self.image_size,
            action_key=self.action_key,
            total_files=len(self.episodes),
            total_samples=len(self.indices),
            max_samples=self.max_samples,
            state_dim=self.state_dim,
            language_conditioning=self.language_conditioning,
            language_embedding_dim=self.language_embedding_dim,
            language_embedding_backend=self.language_embedding_backend,
            language_source=self.language_source,
        )

    def environment_coverage(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.episodes:
            counts[record.env] = counts.get(record.env, 0) + 1
        return counts
