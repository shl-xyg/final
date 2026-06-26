from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils import json_dump


RUN_LANG_A_88K = "act_lang_a_only_100k_b128_v1"
RUN_LANG_ABC_88K = "act_lang_abc_joint_100k_b128_v1"

RUNS = (
    {
        "label": "ACT_Lang_A_only_88k_b128_TE",
        "run_name": RUN_LANG_A_88K,
        "train_envs": "A",
        "val_envs": "A",
        "language": "yes",
        "rollout_file": "rollout_D_success_temporal_ens.json",
    },
    {
        "label": "ACT_Lang_ABC_joint_88k_b128_TE",
        "run_name": RUN_LANG_ABC_88K,
        "train_envs": "A,B,C",
        "val_envs": "A,B,C",
        "language": "yes",
        "rollout_file": "rollout_D_success_temporal_ens.json",
    },
)

HPARAM_KEYS = (
    "dataset_format",
    "chunk_size",
    "image_size",
    "dim_model",
    "n_heads",
    "dim_feedforward",
    "n_encoder_layers",
    "n_decoder_layers",
    "n_action_steps",
    "batch_size",
    "lr",
    "weight_decay",
    "kl_weight",
    "language_conditioning",
    "language_embedding_backend",
    "language_embedding_dim",
    "steps",
    "optimizer",
    "loss_function",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect formal ACT-CALVIN metrics and package best weights.")
    parser.add_argument("--output-dir", default="outputs/train")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument("--skip-packaging", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_metrics(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def last_row(rows: list[dict[str, str]], split: str) -> dict[str, str] | None:
    selected = [row for row in rows if row.get("split") == split]
    if not selected:
        return None
    return max(selected, key=lambda row: int(float(row.get("step", "0") or 0)))


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_cell(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if value != value:
            return "NA"
        return f"{value:.6g}"
    return str(value)


def latex_escape(value: Any) -> str:
    text = format_cell(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_results_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        ("label", "Model"),
        ("train_envs", "Train envs"),
        ("language", "Lang"),
        ("best_step", "Best step"),
        ("best_train_val_l1", "Train-val L1"),
        ("d_offline_l1", "D offline L1"),
        ("rollout_1_task_sr", "D SR@1"),
        ("rollout_avg_seq_len", "Avg len"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{llllllll}\n\\toprule\n")
        f.write(" & ".join(header for _, header in columns) + " \\\\\n\\midrule\n")
        for row in rows:
            f.write(" & ".join(latex_escape(row.get(key)) for key, _ in columns) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def write_hparams_tex(path: Path, hparams: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{ll}\n\\toprule\nItem & Value \\\\\n\\midrule\n")
        for key in HPARAM_KEYS:
            if key in hparams:
                f.write(f"{latex_escape(key)} & {latex_escape(hparams[key])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_best(run_dir: Path, weights_dir: Path) -> tuple[Path | None, str | None]:
    best_dir = run_dir / "best"
    if not best_dir.exists():
        return None, None
    weights_dir.mkdir(parents=True, exist_ok=True)
    package = weights_dir / f"{run_dir.name}_best.tar.gz"
    with tarfile.open(package, "w:gz") as tar:
        tar.add(best_dir, arcname=f"{run_dir.name}_best")
    return package, sha256_file(package)


def collect_run(output_dir: Path, run: dict[str, str]) -> dict[str, Any]:
    run_dir = output_dir / run["run_name"]
    summary = read_json(run_dir / "summary.json") or {}
    best = read_json(run_dir / "best" / "best_metrics.json") or {}
    eval_d = read_json(run_dir / "eval_D_offline_l1.json") or {}
    rollout = read_json(run_dir / run.get("rollout_file", "rollout_D_success.json")) or {}
    metrics_rows = read_metrics(run_dir / "metrics.csv")
    train_row = last_row(metrics_rows, "train")
    val_row = last_row(metrics_rows, "validation")
    eval_metrics = eval_d.get("metrics", {}) if isinstance(eval_d.get("metrics"), dict) else {}
    chain_success = rollout.get("chain_success", {}) if isinstance(rollout.get("chain_success"), dict) else {}
    return {
        "label": run["label"],
        "run_name": run["run_name"],
        "train_envs": run["train_envs"],
        "train_validation_envs": run["val_envs"],
        "language": run.get("language", "no"),
        "rollout_file": run.get("rollout_file", "rollout_D_success.json"),
        "temporal_ensemble_coeff": rollout.get("temporal_ensemble_coeff"),
        "best_step": best.get("step"),
        "best_train_val_l1": as_float(best.get("best_val_l1_loss", summary.get("best_val_l1_loss"))),
        "final_train_l1": as_float(train_row.get("l1_loss") if train_row else None),
        "final_train_loss": as_float(train_row.get("loss") if train_row else None),
        "final_validation_l1": as_float(val_row.get("l1_loss") if val_row else None),
        "final_validation_loss": as_float(val_row.get("loss") if val_row else None),
        "d_offline_l1": as_float(eval_metrics.get("l1_loss")),
        "d_offline_loss": as_float(eval_metrics.get("loss")),
        "d_offline_kld": as_float(eval_metrics.get("kld_loss")),
        "rollout_num_sequences": rollout.get("num_sequences"),
        "rollout_avg_seq_len": as_float(rollout.get("avg_seq_len")),
        "rollout_1_task_sr": as_float(chain_success.get("1")),
        "complete": bool(best and eval_d and rollout),
    }


def load_hparams(output_dir: Path) -> dict[str, Any]:
    run_config = read_json(output_dir / RUN_LANG_A_88K / "run_config.json")
    args = dict(run_config.get("args", {})) if run_config else {}
    args["optimizer"] = "AdamW"
    args["loss_function"] = "Action L1 reconstruction + KL regularization"
    if run_config and args.get("run_name") == RUN_LANG_A_88K:
        args["steps"] = "88000 effective stop; command cap 100000"
    return {key: args.get(key) for key in HPARAM_KEYS if key in args}


def percent(value: Any) -> float | None:
    number = as_float(value)
    if number is None:
        return None
    return number * 100.0


def write_success_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        ("model", "Model"),
        ("split", "Split"),
        ("sr1_percent", "SR@1"),
        ("avg_len", "Avg len"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{llll}\n\\toprule\n")
        f.write(" & ".join(header for _, header in columns) + " \\\\\n\\midrule\n")
        for row in rows:
            f.write(" & ".join(latex_escape(row.get(key)) for key, _ in columns) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def build_success_rows(formal_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        ("ACT+Lang A 88k b128 TE", "ACT_Lang_A_only_88k_b128_TE", "A->D"),
        ("ACT+Lang ABC 88k b128 TE", "ACT_Lang_ABC_joint_88k_b128_TE", "ABC->D"),
    ]
    by_label = {row["label"]: row for row in formal_rows}
    rows: list[dict[str, Any]] = []
    for model, label, split in selected:
        row = by_label.get(label)
        if not row:
            continue
        rows.append(
            {
                "model": model,
                "run_name": row["run_name"],
                "split": split,
                "sr1_percent": percent(row.get("rollout_1_task_sr")),
                "avg_len": row.get("rollout_avg_seq_len"),
            }
        )
    return rows


def write_success_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["model"]) for row in rows]
    values = [as_float(row.get("sr1_percent")) or 0.0 for row in rows]
    colors = ["#5B8DEF" for _ in labels]
    plt.figure(figsize=(9.6, 4.6))
    bars = plt.bar(range(len(labels)), values, color=colors)
    plt.ylabel("D rollout SR@1 (%)")
    plt.ylim(0, max(50, max(values, default=0) + 12))
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, values, strict=True):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.0f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(path.with_suffix(".png"), dpi=220)
    plt.savefig(path.with_suffix(".pdf"))
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    reports_dir = Path(args.reports_dir)
    weights_dir = Path(args.weights_dir)
    tables_dir = reports_dir / "tables"

    rows = [collect_run(output_dir, run) for run in RUNS]
    if args.require_complete and not all(row["complete"] for row in rows):
        incomplete = [row["run_name"] for row in rows if not row["complete"]]
        raise SystemExit(f"Incomplete formal runs: {', '.join(incomplete)}")

    write_csv(tables_dir / "formal_results.csv", rows)
    write_results_tex(tables_dir / "formal_results.tex", rows)
    hparams = load_hparams(output_dir)
    write_csv(tables_dir / "formal_hyperparameters.csv", [{"item": key, "value": value} for key, value in hparams.items()])
    write_hparams_tex(tables_dir / "formal_hyperparameters.tex", hparams)
    success_rows = build_success_rows(rows)
    write_csv(tables_dir / "strong_policy_results.csv", success_rows)
    write_success_tex(tables_dir / "strong_policy_results.tex", success_rows)
    write_success_plot(reports_dir / "figures" / "formal_protocol" / "success_rate_comparison", success_rows)

    packages = []
    if not args.skip_packaging:
        packaged_runs: set[str] = set()
        for run in RUNS:
            if run["run_name"] in packaged_runs:
                continue
            packaged_runs.add(run["run_name"])
            package, digest = package_best(output_dir / run["run_name"], weights_dir)
            if package and digest:
                packages.append({"file": str(package), "sha256": digest})
        if packages:
            with (weights_dir / "SHA256SUMS.txt").open("w", encoding="utf-8") as f:
                for item in packages:
                    f.write(f"{item['sha256']}  {Path(item['file']).name}\n")

    summary = {
        "runs": rows,
        "hyperparameters": hparams,
        "success_comparison": success_rows,
        "packages": packages,
        "complete": all(row["complete"] for row in rows),
    }
    json_dump(summary, reports_dir / "formal_results_summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
