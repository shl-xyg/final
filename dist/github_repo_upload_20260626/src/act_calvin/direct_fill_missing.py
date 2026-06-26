from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct-curl missing fywang CALVIN parquet files.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--episode-env-map", required=True)
    parser.add_argument("--training-envs", default="A,B,C")
    parser.add_argument("--validation-envs", default="D")
    parser.add_argument("--max-workers", type=int, default=48)
    parser.add_argument("--repo-url", default="https://huggingface.co/datasets/fywang/calvin-task-ABC-D-lerobot/resolve/main")
    return parser.parse_args()


def parse_envs(value: str) -> set[str]:
    return {part.strip().upper() for part in value.split(",") if part.strip()}


def episode_path(root: Path, episode_index: int) -> Path:
    return root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"


def episode_filename(episode_index: int) -> str:
    return f"data/chunk-{episode_index // 1000:03d}/episode_{episode_index:06d}.parquet"


def collect_missing(root: Path, map_path: Path, training_envs: set[str], validation_envs: set[str]) -> list[int]:
    missing: list[int] = []
    with map_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            split = str(row["split"]).lower()
            env = str(row["env"]).upper()
            if split == "training" and env not in training_envs:
                continue
            if split == "validation" and env not in validation_envs:
                continue
            idx = int(row["episode_index"])
            path = episode_path(root, idx)
            if not path.exists():
                missing.append(idx)
    return missing


def fetch(root: Path, repo_url: str, episode_index: int) -> str:
    path = episode_path(root, episode_index)
    if path.exists() and path.stat().st_size > 0:
        return "cached"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmpcurl")
    tmp.unlink(missing_ok=True)
    url = f"{repo_url}/{episode_filename(episode_index)}?download=true"
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--http1.1",
        "--retry",
        "10",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "--max-time",
        "600",
        "--speed-time",
        "120",
        "--speed-limit",
        "1024",
        "-o",
        str(tmp),
        url,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        return f"failed:{episode_index}:{proc.stderr[-300:].replace(chr(10), ' ')}"
    if path.exists() and path.stat().st_size > 0:
        tmp.unlink(missing_ok=True)
        return "race_cached"
    os.replace(tmp, path)
    return "downloaded"


def main() -> None:
    args = parse_args()
    root = Path(args.data_root).expanduser().resolve()
    map_path = Path(args.episode_env_map).expanduser().resolve()
    missing = collect_missing(root, map_path, parse_envs(args.training_envs), parse_envs(args.validation_envs))
    print(json.dumps({"targets": len(missing), "first": missing[:20]}, indent=2), flush=True)

    counts: dict[str, int] = {}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(fetch, root, args.repo_url, idx): idx for idx in missing}
        for future in tqdm(as_completed(futures), total=len(futures), desc="direct-fill"):
            result = future.result()
            key = result.split(":", 1)[0]
            counts[key] = counts.get(key, 0) + 1
            if key == "failed":
                failures.append(result)
                tqdm.write(result)

    print(json.dumps({"counts": counts, "failure_count": len(failures), "failures": failures[:50]}, indent=2), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
