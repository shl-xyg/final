from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .calvin_dataset import CalvinActionChunkDataset, CalvinParquetActionChunkDataset
from .policy import make_act_config
from .utils import configure_torch_runtime, ensure_dir, json_dump, move_batch_to_device, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LeRobot ACT on CALVIN raw npz data.")
    parser.add_argument("--dataset-format", default="raw_npz", choices=["raw_npz", "hf_parquet"])
    parser.add_argument("--data-root", required=True, help="CALVIN dataset root containing training/validation.")
    parser.add_argument("--episode-env-map", default=None, help="JSONL map from official CALVIN metadata for hf_parquet.")
    parser.add_argument("--train-split", default="training")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--train-envs", default="A", help="Comma-separated CALVIN env labels, e.g. A or A,B,C.")
    parser.add_argument("--val-envs", default="D", help="Comma-separated validation env labels.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume-from", default=None, help="Optional ACTPolicy checkpoint directory to continue from.")
    parser.add_argument("--start-step", type=int, default=0, help="Global step already completed by --resume-from.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--n-action-steps", type=int, default=None)
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
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dim-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--dim-feedforward", type=int, default=3200)
    parser.add_argument("--n-encoder-layers", type=int, default=4)
    parser.add_argument("--n-decoder-layers", type=int, default=1)
    parser.add_argument("--kl-weight", type=float, default=10.0)
    parser.add_argument("--no-vae", action="store_true")
    parser.add_argument("--no-pretrained-backbone", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="act-calvin-generalization")
    parser.add_argument("--wandb-mode", default="offline", choices=["offline", "online", "disabled"])
    return parser.parse_args()


def make_dataset(
    *,
    args: argparse.Namespace,
    split: str,
    environments: str,
    max_samples: int | None,
):
    action_key = args.action_key
    if action_key == "auto":
        action_key = "rel_actions" if args.dataset_format == "raw_npz" else "action"
    if args.dataset_format == "raw_npz":
        return CalvinActionChunkDataset(
            root=args.data_root,
            split=split,
            environments=environments,
            chunk_size=args.chunk_size,
            image_size=args.image_size,
            action_key=action_key,
            max_samples=max_samples,
            sample_stride=args.sample_stride,
            language_conditioning=args.language_conditioning,
            language_embedding_dim=args.language_embedding_dim,
            language_embedding_backend=args.language_embedding_backend,
            language_source=args.language_source,
        )
    return CalvinParquetActionChunkDataset(
        root=args.data_root,
        split=split,
        environments=environments,
        episode_env_map=args.episode_env_map,
        chunk_size=args.chunk_size,
        image_size=args.image_size,
        action_key=action_key,
        max_samples=max_samples,
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


@torch.no_grad()
def evaluate(policy, loader: DataLoader, device: torch.device, max_batches: int | None = None) -> dict[str, float]:
    was_training = policy.training
    # LeRobot 0.5.2 ACT skips the VAE encoder in eval mode, while policy.forward
    # still reports the KL term. Keep train mode for teacher-forced validation
    # loss, but disable gradients with no_grad().
    policy.train()
    totals = {"loss": 0.0, "l1_loss": 0.0, "kld_loss": 0.0}
    count = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        loss, out = policy(batch)
        totals["loss"] += float(loss.item())
        totals["l1_loss"] += float(out.get("l1_loss", 0.0))
        totals["kld_loss"] += float(out.get("kld_loss", 0.0))
        count += 1
    if not was_training:
        policy.eval()
    if count == 0:
        return {key: float("nan") for key in totals}
    return {key: value / count for key, value in totals.items()}


def append_metrics(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    configure_torch_runtime()
    set_seed(args.seed)
    device = resolve_device(args.device)
    run_dir = ensure_dir(Path(args.output_dir) / args.run_name)
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    metrics_path = run_dir / "metrics.csv"

    train_ds = make_dataset(
        args=args,
        split=args.train_split,
        environments=args.train_envs,
        max_samples=args.max_train_samples,
    )
    val_ds = make_dataset(
        args=args,
        split=args.val_split,
        environments=args.val_envs,
        max_samples=args.max_val_samples,
    )
    train_summary = train_ds.summary()
    val_summary = val_ds.summary()
    if train_summary.state_dim != val_summary.state_dim:
        raise ValueError(f"Train/validation state dims differ: {train_summary.state_dim} != {val_summary.state_dim}")

    cfg = make_act_config(
        image_size=args.image_size,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        device=str(device),
        dim_model=args.dim_model,
        n_heads=args.n_heads,
        dim_feedforward=args.dim_feedforward,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        use_vae=not args.no_vae,
        kl_weight=args.kl_weight,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pretrained_backbone_weights=None if args.no_pretrained_backbone else "ResNet18_Weights.IMAGENET1K_V1",
        state_dim=train_summary.state_dim,
    )
    from lerobot.policies.act import ACTPolicy

    if args.resume_from:
        policy = ACTPolicy.from_pretrained(args.resume_from, config=cfg, local_files_only=True).to(device)
    else:
        policy = ACTPolicy(cfg).to(device)
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    json_dump(
        {
            "args": vars(args),
            "device": str(device),
            "train_dataset": train_summary.__dict__,
            "val_dataset": val_summary.__dict__,
            "train_environment_coverage": train_ds.environment_coverage(),
            "val_environment_coverage": val_ds.environment_coverage(),
        },
        run_dir / "run_config.json",
    )

    wandb_run = None
    if args.wandb and args.wandb_mode != "disabled":
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            dir=str(run_dir),
            mode=args.wandb_mode,
            config=vars(args),
        )

    best_val = float("inf")
    if args.resume_from:
        resume_metrics_path = Path(args.resume_from) / "best_metrics.json"
        if resume_metrics_path.exists():
            with resume_metrics_path.open(encoding="utf-8") as f:
                resume_metrics = json.load(f)
            best_val = float(resume_metrics.get("best_val_l1_loss", best_val))
        best_dir = run_dir / "best"
        policy.save_pretrained(best_dir)
        json_dump({"step": args.start_step, "best_val_l1_loss": best_val}, best_dir / "best_metrics.json")

    if args.steps <= args.start_step:
        raise ValueError(f"--steps ({args.steps}) must be greater than --start-step ({args.start_step})")

    start = time.time()
    loader_iter = iter(train_loader)
    progress = tqdm(range(args.start_step + 1, args.steps + 1), desc=args.run_name)
    for step in progress:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        policy.train()
        batch = move_batch_to_device(batch, device)
        loss, out = policy(batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        row = {
            "step": step,
            "split": "train",
            "loss": float(loss.item()),
            "l1_loss": float(out.get("l1_loss", 0.0)),
            "kld_loss": float(out.get("kld_loss", 0.0)),
            "grad_norm": float(grad_norm.item()),
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": time.time() - start,
        }
        if step % args.log_every == 0 or step == 1:
            append_metrics(metrics_path, row)
            progress.set_postfix({"loss": f"{row['loss']:.4f}", "l1": f"{row['l1_loss']:.4f}"})
            if wandb_run is not None:
                wandb_run.log({f"train/{k}": v for k, v in row.items() if k not in {"split"}}, step=step)

        if step % args.eval_every == 0 or step == args.steps:
            val = evaluate(policy, val_loader, device)
            val_row = {
                "step": step,
                "split": "validation",
                "loss": val["loss"],
                "l1_loss": val["l1_loss"],
                "kld_loss": val["kld_loss"],
                "grad_norm": 0.0,
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_s": time.time() - start,
            }
            append_metrics(metrics_path, val_row)
            if wandb_run is not None:
                wandb_run.log({f"validation/{k}": v for k, v in val.items()}, step=step)
            if val["l1_loss"] < best_val:
                best_val = val["l1_loss"]
                best_dir = run_dir / "best"
                policy.save_pretrained(best_dir)
                json_dump({"step": step, "best_val_l1_loss": best_val}, best_dir / "best_metrics.json")

        if step % args.save_every == 0:
            ckpt_dir = checkpoints_dir / f"step_{step:06d}"
            policy.save_pretrained(ckpt_dir)

    final_dir = run_dir / "final"
    policy.save_pretrained(final_dir)
    json_dump({"best_val_l1_loss": best_val, "total_elapsed_s": time.time() - start}, run_dir / "summary.json")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
