import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from qsdrec.io_utils import read_json


def load_embeddings(path: Path) -> np.ndarray:
    emb = np.load(path).astype("float32")
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {emb.shape}")
    return emb


def load_item_ids(path: Path):
    return [str(x) for x in read_json(path)]


def prefix_labels(semantic_path: Path, item_ids, prefix_level: int):
    obj = read_json(semantic_path)
    sem_ids = {str(k): list(map(int, v)) for k, v in obj["semantic_ids"].items()}
    labels = []
    prefixes = []
    for item_id in item_ids:
        sid = sem_ids[item_id]
        prefix = tuple(sid[:prefix_level])
        prefixes.append(prefix)
        labels.append("|".join(map(str, prefix)))
    return labels, prefixes


def reduce_2d(emb: np.ndarray, method: str, seed: int) -> np.ndarray:
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(emb)
    if method == "tsne":
        init = PCA(n_components=2, random_state=seed).fit_transform(emb)
        return TSNE(
            n_components=2,
            perplexity=30,
            init=init,
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(emb)
    raise ValueError(f"Unknown reduction method: {method}")


def select_points(emb, item_ids, labels, prefixes, max_points: int, seed: int):
    if max_points <= 0 or len(item_ids) <= max_points:
        idx = np.arange(len(item_ids))
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(item_ids), size=max_points, replace=False))
    return emb[idx], [item_ids[i] for i in idx], [labels[i] for i in idx], [prefixes[i] for i in idx]


def encode_top_labels(labels, top_groups: int):
    counts = Counter(labels)
    top = {label for label, _ in counts.most_common(top_groups)}
    encoded = [label if label in top else "other" for label in labels]
    unique = sorted(set(encoded), key=lambda x: (x == "other", x))
    label_to_id = {label: idx for idx, label in enumerate(unique)}
    colors = np.array([label_to_id[label] for label in encoded])
    return encoded, colors, unique


def plot_clusters(coords, colors, unique_labels, title: str, output: Path) -> None:
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=colors,
        s=8,
        alpha=0.75,
        cmap="tab20",
        linewidths=0,
    )
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True, alpha=0.2)

    if len(unique_labels) <= 21:
        handles, _ = scatter.legend_elements(num=len(unique_labels))
        display_labels = [label if len(label) <= 18 else label[:15] + "..." for label in unique_labels]
        plt.legend(
            handles,
            display_labels,
            title="Prefix group",
            loc="best",
            fontsize=8,
            title_fontsize=9,
            framealpha=0.85,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()


def write_group_summary(labels, output: Path) -> None:
    counts = Counter(labels)
    rows = [{"label": label, "count": count} for label, count in counts.most_common()]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--item-ids", required=True)
    parser.add_argument("--semantic-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefix-level", type=int, default=2)
    parser.add_argument("--method", choices=["pca", "tsne"], default="pca")
    parser.add_argument("--max-points", type=int, default=5000)
    parser.add_argument("--top-groups", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()

    embeddings = load_embeddings(Path(args.embeddings))
    item_ids = load_item_ids(Path(args.item_ids))
    if len(item_ids) != embeddings.shape[0]:
        raise ValueError("item-ids length must match embedding rows.")

    labels, prefixes = prefix_labels(Path(args.semantic_ids), item_ids, args.prefix_level)
    embeddings, item_ids, labels, prefixes = select_points(
        embeddings,
        item_ids,
        labels,
        prefixes,
        max_points=args.max_points,
        seed=args.seed,
    )
    encoded_labels, colors, unique_labels = encode_top_labels(labels, args.top_groups)
    coords = reduce_2d(embeddings, args.method, args.seed)

    title = (
        f"Semantic Clusters ({args.method.upper()}), "
        f"prefix level={args.prefix_level}, points={len(item_ids)}"
    )
    plot_clusters(coords, colors, unique_labels, title, Path(args.output))

    summary_path = Path(args.summary) if args.summary else Path(args.output).with_suffix(".groups.json")
    write_group_summary(encoded_labels, summary_path)
    print({"output": args.output, "summary": str(summary_path), "points": len(item_ids)})


if __name__ == "__main__":
    main()
