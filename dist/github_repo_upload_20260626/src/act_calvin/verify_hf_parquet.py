from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calvin_dataset import CalvinParquetActionChunkDataset
from .utils import json_dump


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify selected CALVIN HF/LeRobot parquet episode files exist.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--episode-env-map", required=True)
    parser.add_argument("--split", default="training")
    parser.add_argument("--envs", default="A,B,C,D")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = CalvinParquetActionChunkDataset(
        root=args.data_root,
        split=args.split,
        environments=args.envs,
        episode_env_map=args.episode_env_map,
        chunk_size=1,
        image_size=96,
        max_samples=1,
    )
    missing = [str(ds.episode_path(record.episode_index)) for record in ds.episodes if not ds.episode_path(record.episode_index).exists()]
    payload = {
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "episode_env_map": str(Path(args.episode_env_map).expanduser().resolve()),
        "split": args.split,
        "envs": args.envs,
        "selected_episodes": len(ds.episodes),
        "missing_episodes": len(missing),
        "missing_paths": missing[:100],
        "missing_paths_truncated": len(missing) > 100,
        "environment_coverage": ds.environment_coverage(),
    }
    if args.output_json:
        json_dump(payload, args.output_json)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
