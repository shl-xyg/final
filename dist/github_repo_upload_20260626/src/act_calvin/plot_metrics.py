from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ACT training metrics.")
    parser.add_argument("--metrics", nargs="+", required=True, help="One or more metrics.csv files.")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = args.labels or [Path(path).parent.name for path in args.metrics]
    if len(labels) != len(args.metrics):
        raise ValueError("--labels length must match --metrics length")

    frames = []
    for path, label in zip(args.metrics, labels, strict=True):
        df = pd.read_csv(path)
        df["run"] = label
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    for metric in ["loss", "l1_loss", "kld_loss"]:
        plt.figure(figsize=(7.2, 4.4))
        for (run, split), group in data.groupby(["run", "split"]):
            group = group.sort_values("step")
            linestyle = "-" if split == "train" else "--"
            plt.plot(group["step"], group[metric], linestyle=linestyle, marker="o", markersize=2.5, label=f"{run} {split}")
        plt.xlabel("Training step")
        plt.ylabel(metric)
        plt.title(f"ACT {metric}")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}.png", dpi=220)
        plt.savefig(output_dir / f"{metric}.pdf")
        plt.close()


if __name__ == "__main__":
    main()

