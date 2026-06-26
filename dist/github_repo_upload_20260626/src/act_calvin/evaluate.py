from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .calvin_dataset import CalvinActionChunkDataset, CalvinParquetActionChunkDataset
from .train import evaluate
from .utils import configure_torch_runtime, json_dump, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a LeRobot ACT checkpoint on CALVIN raw npz data.")
    parser.add_argument("--checkpoint", required=True, help="Directory containing model.safetensors and config.")
    parser.add_argument("--dataset-format", default="raw_npz", choices=["raw_npz", "hf_parquet"])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--episode-env-map", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--envs", default="D")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--action-key", default="auto", choices=["auto", "rel_actions", "actions", "action", "action.relative"])
    parser.add_argument("--top-image-key", default="observation.images.top")
    parser.add_argument("--wrist-image-key", default="observation.images.wrist")
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--episode-cache-size", type=int, default=64)
    parser.add_argument("--language-conditioning", action="store_true")
    parser.add_argument("--language-embedding-dim", type=int, default=384)
    parser.add_argument("--language-embedding-backend", default="calvin_sbert", choices=["hash", "calvin_sbert"])
    parser.add_argument("--language-embeddings-path", default=None)
    parser.add_argument(
        "--language-source",
        default="annotation",
        choices=["annotation", "task", "task_annotation"],
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_dataset(args: argparse.Namespace):
    action_key = args.action_key
    if action_key == "auto":
        action_key = "rel_actions" if args.dataset_format == "raw_npz" else "action"
    if args.dataset_format == "raw_npz":
        return CalvinActionChunkDataset(
            root=args.data_root,
            split=args.split,
            environments=args.envs,
            chunk_size=args.chunk_size,
            image_size=args.image_size,
            action_key=action_key,
            max_samples=args.max_samples,
            sample_stride=args.sample_stride,
            language_conditioning=args.language_conditioning,
            language_embedding_dim=args.language_embedding_dim,
            language_embedding_backend=args.language_embedding_backend,
            language_source=args.language_source,
        )
    return CalvinParquetActionChunkDataset(
        root=args.data_root,
        split=args.split,
        environments=args.envs,
        episode_env_map=args.episode_env_map,
        chunk_size=args.chunk_size,
        image_size=args.image_size,
        action_key=action_key,
        max_samples=args.max_samples,
        sample_stride=args.sample_stride,
        top_image_key=args.top_image_key,
        wrist_image_key=args.wrist_image_key,
        state_key=args.state_key,
        cache_size=args.episode_cache_size,
        language_conditioning=args.language_conditioning,
        language_embedding_dim=args.language_embedding_dim,
        language_embedding_backend=args.language_embedding_backend,
        language_source=args.language_source,
        language_embeddings_path=args.language_embeddings_path,
    )


def main() -> None:
    args = parse_args()
    configure_torch_runtime()
    set_seed(args.seed)
    device = resolve_device(args.device)
    from lerobot.policies.act import ACTPolicy

    policy = ACTPolicy.from_pretrained(Path(args.checkpoint), local_files_only=True, strict=False)
    policy.config.device = str(device)
    policy.to(device)
    ds = make_dataset(args)
    expected_state_dim = int(policy.config.robot_state_feature.shape[0])
    actual_state_dim = ds.summary().state_dim
    if expected_state_dim != actual_state_dim:
        raise ValueError(
            f"Checkpoint expects state dim {expected_state_dim}, but dataset provides {actual_state_dim}. "
            "Use --language-conditioning with the same --language-embedding-dim/--language-source as training."
        )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    metrics = evaluate(policy, loader, device)
    json_dump(
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "data_root": str(Path(args.data_root).resolve()),
            "split": args.split,
            "envs": args.envs,
            "dataset": ds.summary().__dict__,
            "environment_coverage": ds.environment_coverage(),
            "metrics": metrics,
            "metric_type": "offline_action_chunk_l1_proxy",
            "success_rate": None,
            "success_rate_note": "Closed-loop CALVIN rollout is required for Success Rate; this script reports offline chunk L1.",
        },
        args.output_json,
    )


if __name__ == "__main__":
    main()
