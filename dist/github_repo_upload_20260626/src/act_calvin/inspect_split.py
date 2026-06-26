from __future__ import annotations

import argparse
from pathlib import Path

from .calvin_dataset import CalvinActionChunkDataset, load_scene_ranges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect CALVIN raw split coverage.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="training")
    parser.add_argument("--envs", default="all")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=96)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dir = Path(args.data_root).expanduser().resolve() / args.split
    print(f"Split: {split_dir}")
    print("Scene ranges:")
    for scene in load_scene_ranges(split_dir):
        print(f"  {scene.scene:20s} env={scene.env_label:>4s} start={scene.start} end={scene.end} len={scene.length}")
    ds = CalvinActionChunkDataset(
        root=args.data_root,
        split=args.split,
        environments=args.envs,
        chunk_size=args.chunk_size,
        image_size=args.image_size,
    )
    print(ds.summary().to_json())
    sample = ds[0]
    print("Sample tensors:")
    for key, value in sample.items():
        print(f"  {key:28s} shape={tuple(value.shape)} dtype={value.dtype}")


if __name__ == "__main__":
    main()

