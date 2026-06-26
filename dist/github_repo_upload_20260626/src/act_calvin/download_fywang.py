from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm


REPO_ID = "fywang/calvin-task-ABC-D-lerobot"


def episode_filename(episode_index: int) -> str:
    return f"data/chunk-{episode_index // 1000:03d}/episode_{episode_index:06d}.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download selected fywang CALVIN ABC-D parquet episodes by env map.")
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--episode-env-map", required=True)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--training-envs", default="A,B,C")
    parser.add_argument("--validation-envs", default="D")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def parse_envs(value: str) -> set[str]:
    return {part.strip().upper() for part in value.split(",") if part.strip()}


def load_targets(map_path: str | Path, training_envs: set[str], validation_envs: set[str]) -> list[tuple[int, str]]:
    priorities = {"A": 0, "B": 1, "C": 2, "D": 3}
    targets: list[tuple[int, str]] = []
    with Path(map_path).expanduser().open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            split = str(row["split"]).lower()
            env = str(row["env"]).upper()
            if split == "training" and env not in training_envs:
                continue
            if split == "validation" and env not in validation_envs:
                continue
            episode_index = int(row["episode_index"])
            priority = priorities.get(env, 99)
            targets.append((priority, episode_filename(episode_index)))
    return sorted(targets)


def download_one(repo_id: str, local_dir: Path, filename: str) -> str:
    final_path = local_dir / filename
    if final_path.exists() and final_path.stat().st_size > 0:
        return "cached"
    hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename, local_dir=local_dir)
    return "downloaded"


def main() -> None:
    args = parse_args()
    local_dir = Path(args.local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    for meta_file in ("meta/episodes.jsonl", "meta/info.json"):
        hf_hub_download(repo_id=args.repo_id, repo_type="dataset", filename=meta_file, local_dir=local_dir)

    targets = load_targets(
        args.episode_env_map,
        training_envs=parse_envs(args.training_envs),
        validation_envs=parse_envs(args.validation_envs),
    )
    filenames = [filename for _, filename in targets]
    if args.limit is not None:
        filenames = filenames[: args.limit]

    counts = {"cached": 0, "downloaded": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_file = {
            executor.submit(download_one, args.repo_id, local_dir, filename): filename for filename in filenames
        }
        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="fywang-calvin-download"):
            filename = future_to_file[future]
            try:
                status = future.result()
            except Exception as exc:
                counts["failed"] += 1
                tqdm.write(f"FAILED {filename}: {type(exc).__name__}: {exc}")
            else:
                counts[status] += 1
    print(json.dumps({"local_dir": str(local_dir), "targets": len(filenames), **counts}, indent=2, sort_keys=True))
    if counts["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
