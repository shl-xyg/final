from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from act_calvin.language import default_task_embeddings_path


RUN_A = "act_a_only_formal_88k_v1"
RUN_ABC = "act_abc_joint_formal_88k_v1"
RUN_LANG_A = "act_lang_a_only_formal_88k_v1"
RUN_LANG_ABC = "act_lang_abc_joint_formal_88k_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for data and run the ACT CALVIN formal protocol locally.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--episode-env-map", required=True)
    parser.add_argument("--calvin-root", default="/home/zzh/Titan/v12/repos/reference/calvin")
    parser.add_argument("--output-dir", default="outputs/train")
    parser.add_argument("--steps", type=int, default=88000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--val-samples", type=int, default=4096)
    parser.add_argument("--eval-d-samples", type=int, default=None)
    parser.add_argument("--rollout-sequences", type=int, default=50)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--device-a", default="cuda:0")
    parser.add_argument("--device-abc", default="cuda:1")
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--language-conditioning", dest="language_conditioning", action="store_true", default=True)
    parser.add_argument("--no-language-conditioning", dest="language_conditioning", action="store_false")
    parser.add_argument("--language-embedding-dim", type=int, default=384)
    parser.add_argument("--language-embedding-backend", default="calvin_sbert", choices=["hash", "calvin_sbert"])
    parser.add_argument("--language-embeddings-path", default=None)
    parser.add_argument("--task-embeddings-path", default=None)
    parser.add_argument("--temporal-ensemble-coeff", type=float, default=0.01)
    parser.add_argument(
        "--language-source",
        default="annotation",
        choices=["annotation", "task", "task_annotation"],
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run_names(args: argparse.Namespace) -> tuple[str, str]:
    if args.language_conditioning:
        return RUN_LANG_A, RUN_LANG_ABC
    return RUN_A, RUN_ABC


def maybe_language_args(args: argparse.Namespace) -> list[str]:
    if not args.language_conditioning:
        return []
    result = [
        "--language-conditioning",
        "--language-embedding-dim",
        str(args.language_embedding_dim),
        "--language-embedding-backend",
        args.language_embedding_backend,
        "--language-source",
        args.language_source,
    ]
    if args.language_embeddings_path is not None:
        result.extend(["--language-embeddings-path", args.language_embeddings_path])
    return result


def episode_path(root: Path, episode_index: int) -> Path:
    return root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"


def load_selected_episodes(map_path: Path, split: str, envs: set[str]) -> list[int]:
    selected: list[int] = []
    with map_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if str(row["split"]).lower() == split and str(row["env"]).upper() in envs:
                selected.append(int(row["episode_index"]))
    return selected


def missing_count(root: Path, map_path: Path, split: str, envs: set[str]) -> tuple[int, int]:
    episodes = load_selected_episodes(map_path, split, envs)
    missing = sum(1 for idx in episodes if not episode_path(root, idx).exists())
    return len(episodes) - missing, missing


def wait_until_ready(root: Path, map_path: Path, split: str, envs: set[str], poll_seconds: int) -> None:
    label = f"{split}:{','.join(sorted(envs))}"
    while True:
        present, missing = missing_count(root, map_path, split, envs)
        print(f"[data] {label} present={present} missing={missing}", flush=True)
        if missing == 0:
            return
        time.sleep(poll_seconds)


def run_command(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[run]", " ".join(cmd), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)


def train_cmd(
    *,
    run_name: str,
    train_envs: str,
    val_envs: str,
    device: str,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/train_act_calvin.py",
        "--dataset-format",
        "hf_parquet",
        "--data-root",
        args.data_root,
        "--episode-env-map",
        args.episode_env_map,
        "--train-split",
        "training",
        "--val-split",
        "training",
        "--train-envs",
        train_envs,
        "--val-envs",
        val_envs,
        "--output-dir",
        args.output_dir,
        "--run-name",
        run_name,
        "--device",
        device,
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--log-every",
        "200",
        "--eval-every",
        "2000",
        "--save-every",
        "10000",
        "--max-val-samples",
        str(args.val_samples),
        "--chunk-size",
        "10",
        "--image-size",
        "224",
        "--dim-model",
        "512",
        "--n-heads",
        "8",
        "--dim-feedforward",
        "3200",
        "--n-encoder-layers",
        "4",
        "--n-decoder-layers",
        "1",
        "--lr",
        "1e-5",
        "--weight-decay",
        "1e-4",
        "--kl-weight",
        "10",
        "--episode-cache-size",
        "128",
        "--wandb",
        "--wandb-mode",
        args.wandb_mode,
    ]
    cmd.extend(maybe_language_args(args))
    return cmd


def eval_cmd(run_name: str, device: str, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/evaluate_act_calvin.py",
        "--dataset-format",
        "hf_parquet",
        "--checkpoint",
        str(Path(args.output_dir) / run_name / "best"),
        "--data-root",
        args.data_root,
        "--episode-env-map",
        args.episode_env_map,
        "--split",
        "validation",
        "--envs",
        "D",
        "--output-json",
        str(Path(args.output_dir) / run_name / "eval_D_offline_l1.json"),
        "--device",
        device,
        "--batch-size",
        "64",
        "--num-workers",
        str(args.num_workers),
        "--chunk-size",
        "10",
        "--image-size",
        "224",
        "--episode-cache-size",
        "128",
    ]
    if args.eval_d_samples is not None:
        cmd.extend(["--max-samples", str(args.eval_d_samples)])
    cmd.extend(maybe_language_args(args))
    return cmd


def rollout_cmd(run_name: str, device: str, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/rollout_act_calvin.py",
        "--checkpoint",
        str(Path(args.output_dir) / run_name / "best"),
        "--output-json",
        str(Path(args.output_dir) / run_name / "rollout_D_success.json"),
        "--calvin-root",
        args.calvin_root,
        "--device",
        device,
        "--num-sequences",
        str(args.rollout_sequences),
        "--ep-len",
        "360",
        "--image-size",
        "224",
    ]
    if args.language_conditioning:
        cmd.extend(["--language-source", args.language_source])
        cmd.extend(["--language-embedding-backend", args.language_embedding_backend])
        if args.language_embedding_backend == "calvin_sbert":
            task_embeddings_path = args.task_embeddings_path or str(default_task_embeddings_path(args.episode_env_map))
            cmd.extend(["--task-embeddings-path", task_embeddings_path])
    if args.temporal_ensemble_coeff is not None:
        cmd.extend(["--temporal-ensemble-coeff", str(args.temporal_ensemble_coeff)])
    return cmd


def main() -> None:
    args = parse_args()
    root = Path(args.data_root).expanduser().resolve()
    map_path = Path(args.episode_env_map).expanduser().resolve()
    output_dir = Path(args.output_dir)
    logs_dir = output_dir / "formal_logs"
    run_a, run_abc = run_names(args)
    env = os.environ.copy()
    repo_root = Path.cwd()
    calvin_root = Path(args.calvin_root).expanduser().resolve()
    extra_pythonpath = [
        str(repo_root / "src"),
        str(calvin_root / "calvin_env"),
        str(calvin_root / "calvin_models"),
    ]
    env["PYTHONPATH"] = ":".join(extra_pythonpath + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))

    wait_until_ready(root, map_path, "training", {"A"}, args.poll_seconds)
    if not args.skip_existing or not (output_dir / run_a / "best" / "model.safetensors").exists():
        run_command(train_cmd(run_name=run_a, train_envs="A", val_envs="A", device=args.device_a, args=args), logs_dir / f"{run_a}.log", env)

    wait_until_ready(root, map_path, "training", {"A", "B", "C"}, args.poll_seconds)
    wait_until_ready(root, map_path, "validation", {"D"}, args.poll_seconds)
    if not args.skip_existing or not (output_dir / run_abc / "best" / "model.safetensors").exists():
        run_command(train_cmd(run_name=run_abc, train_envs="A,B,C", val_envs="A,B,C", device=args.device_abc, args=args), logs_dir / f"{run_abc}.log", env)

    for run_name, device in ((run_a, args.device_a), (run_abc, args.device_abc)):
        run_command(eval_cmd(run_name, device, args), logs_dir / f"{run_name}_eval_D.log", env)
        run_command(rollout_cmd(run_name, device, args), logs_dir / f"{run_name}_rollout_D.log", env)

    plot_cmd = [
        sys.executable,
        "scripts/plot_metrics.py",
        "--metrics",
        str(output_dir / run_a / "metrics.csv"),
        str(output_dir / run_abc / "metrics.csv"),
        "--labels",
        "A_only",
        "ABC_joint",
        "--output-dir",
        "reports/figures/formal_protocol",
    ]
    run_command(plot_cmd, logs_dir / "plot_metrics.log", env)


if __name__ == "__main__":
    main()
