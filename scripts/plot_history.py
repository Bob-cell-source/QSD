import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_history(path: str | Path):
    with Path(path).open("rt", encoding="utf-8") as f:
        history = json.load(f)
    if not history:
        raise ValueError(f"No rows found in {path}")
    return history


def plot_history(history_path: str | Path, output_path: str | Path) -> None:
    history = load_history(history_path)
    epochs = [row["epoch"] for row in history]
    losses = [row["loss"] for row in history]
    ndcg10 = [row.get("NDCG@10") for row in history]
    hr10 = [row.get("HR@10") for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, losses, marker="o", linewidth=2, markersize=4)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    if any(value is not None for value in ndcg10):
        axes[1].plot(epochs, ndcg10, marker="o", linewidth=2, markersize=4, label="NDCG@10")
    if any(value is not None for value in hr10):
        axes[1].plot(epochs, hr10, marker="s", linewidth=2, markersize=4, label="HR@10")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print({"history": str(history_path), "output": str(output_path)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot_history(args.history, args.output)


if __name__ == "__main__":
    main()
