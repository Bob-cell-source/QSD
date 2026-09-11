"""Read-only diagnostics on an already trained pure Hard SID checkpoint.

This deliberately does not train or modify a model.  It uses the saved
positive-target signatures and the full-catalog Hard SID test observations.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pair_values(g, pairs):
    if len(pairs) == 0:
        return np.empty(0, dtype=np.float32)
    return np.einsum("ij,ij->i", g[pairs[:, 0]], g[pairs[:, 1]])


def unique_pairs(values):
    if not values:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(sorted(values), dtype=np.int64).reshape(-1, 2)


def make_pairs(hard, neighbors, counts, min_count=1, seed=2026):
    n, levels = hard.shape
    eligible = counts >= min_count
    # Pair groups use only items with a nonzero profiling signature and the
    # requested minimum number of positive examples.
    # The caller filters signatures separately; this function only groups SID.
    same, semantic = [], []
    for level in range(levels):
        for token in np.unique(hard[:, level]):
            ids = np.where((hard[:, level] == token) & eligible)[0]
            a, b = np.triu_indices(len(ids), 1)
            same.extend((int(i), int(j)) for i, j in zip(ids[a], ids[b]))
    # Semantic-neighbor control is level-specific: an item can share a token
    # at one level yet use a different token at another level.  A previous
    # all-level filter would remove every posting-list neighbor by definition.
    for level in range(levels):
        for i, row in enumerate(neighbors):
            for item in row[row > 0]:
                j = int(item) - 1
                if j >= i or not eligible[i] or not eligible[j] or hard[i, level] == hard[j, level]:
                    continue
                semantic.append((j, i))
    same = unique_pairs(same)
    semantic = unique_pairs(semantic)
    rng = np.random.default_rng(seed + min_count)
    pool = np.where(eligible)[0]
    a = rng.integers(len(pool), size=len(same))
    b = rng.integers(len(pool) - 1, size=len(same))
    b += b >= a
    random_pairs = np.column_stack((pool[a], pool[b]))
    return same, semantic, random_pairs


def stats(values):
    if len(values) == 0:
        return {"pairs": 0, "negative_fraction": None, "quantiles": []}
    return {"pairs": int(len(values)),
            "negative_fraction": float(np.mean(values < 0)),
            "below_minus_001_fraction": float(np.mean(values < -.01)),
            "mean": float(np.mean(values)),
            "quantiles": np.quantile(values, [0,.01,.05,.25,.5,.75,.95,.99,1]).tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="runs/office/gcss_20260910")
    p.add_argument("--output", default="runs/office/gcss_gradient_diagnostic_20260910")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--signature", default=None)
    p.add_argument("--observations", default=None)
    p.add_argument("--artifact", default=None)
    args = p.parse_args()
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    seed_root = source / f"seed{args.seed}"
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else seed_root / "stage1/hard_sid_checkpoint.pt"
    signature_path = Path(args.signature) if args.signature else seed_root / "grad_signature.pt"
    observation_path = Path(args.observations) if args.observations else source.parent / "gcss_diagnostic_abc_20260910/hard_test_observations.npz"
    artifact_path = Path(args.artifact) if args.artifact else source / "semantic_candidates.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    hard = checkpoint["model"]["item_encoder.hard_sid_table"][1:].cpu().numpy()
    signatures = torch.load(signature_path, map_location="cpu", weights_only=True)
    g = signatures["grad_signature"].numpy()
    observations = np.load(observation_path)
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    neighbors = artifact["neighbor_ids"].numpy()
    n, levels = hard.shape
    training_config = checkpoint.get("training_config", {})
    semantic_path = Path(training_config.get("semantic_ids", ""))
    codebook_sizes = json.loads(semantic_path.read_text())["codebook_sizes"] if semantic_path.exists() else [None] * levels

    level_stats = []
    for level in range(levels):
        tokens, counts = np.unique(hard[:, level], return_counts=True)
        level_stats.append({"level": level + 1, "codebook_size": codebook_sizes[level], "unique_tokens": int(len(tokens)),
                            "utilization_fraction": None if codebook_sizes[level] is None else float(len(tokens) / codebook_sizes[level]),
                            "mean_items_per_used_token": float(counts.mean()),
                            "max_items_per_token": int(counts.max()),
                            "p50_items_per_token": float(np.quantile(counts, .5)),
                            "p95_items_per_token": float(np.quantile(counts, .95))})
    full_codes, full_counts = np.unique(hard, axis=0, return_counts=True)
    collision = {"items": n, "unique_full_sids": int(len(full_codes)),
                 "collision_items": int(np.sum(full_counts[full_counts > 1])),
                 "collision_rate": float(np.mean(full_counts[np.searchsorted(full_codes, hard, axis=0)] > 1)) if False else float(np.sum(full_counts[full_counts > 1]) / n),
                 "collision_groups": int(np.sum(full_counts > 1)),
                 "max_items_per_full_sid": int(full_counts.max())}

    pair_report = {}
    grad_counts = signatures["grad_count"].numpy()
    same, semantic, random = make_pairs(hard, neighbors, grad_counts, min_count=1, seed=args.seed)
    for label, pairs in (("same_token", same), ("semantic_neighbor_different_token", semantic), ("random_pair", random)):
        pair_report[label] = stats(pair_values(g, pairs))
    pair_report_by_min_count = {}
    for minimum in (1, 5):
        s, m, r = make_pairs(hard, neighbors, grad_counts, min_count=minimum, seed=args.seed)
        pair_report_by_min_count[str(minimum)] = {label: stats(pair_values(g, pairs)) for label, pairs in
                                                  (("same_token", s), ("semantic_neighbor_different_token", m), ("random_pair", r))}

    # Item-level C_i: deduplicated union of all items sharing any aligned token.
    posting = {}
    for level in range(levels):
        for token in np.unique(hard[:, level]):
            posting[level, int(token)] = np.where(hard[:, level] == token)[0]
    item_c = np.zeros(n, dtype=np.float32)
    item_neighbors = np.zeros(n, dtype=np.int64)
    for i in range(n):
        ids = set()
        for level, token in enumerate(hard[i]):
            ids.update(int(j) for j in posting[level, int(token)] if j != i)
        ids = np.asarray(sorted(ids), dtype=np.int64)
        item_neighbors[i] = len(ids)
        if len(ids):
            item_c[i] = np.mean(g[i] @ g[ids].T)
    targets = observations["target"].astype(np.int64) - 1
    test_ndcg = observations["ndcg"]
    item_metric = np.full(n, np.nan)
    for item in np.unique(targets):
        item_metric[item] = float(test_ndcg[targets == item].mean())
    eligible = np.isfinite(item_metric) & (item_neighbors > 0)
    order = np.argsort(item_c[eligible])
    eligible_ids = np.where(eligible)[0][order]
    quantiles = []
    for q, ids in enumerate(np.array_split(eligible_ids, 5), 1):
        if len(ids) == 0: continue
        quantiles.append({"quantile_low_to_high": q, "items": int(len(ids)),
                          "mean_C": float(item_c[ids].mean()),
                          "mean_shared_neighbors": float(item_neighbors[ids].mean()),
                          "test_targets": int(np.sum(np.isin(targets, ids))),
                          "hard_ndcg_at_10": float(np.nanmean(item_metric[ids]))})
    np.savez(output / "item_compatibility.npz", compatibility=item_c,
             shared_neighbors=item_neighbors, test_ndcg=item_metric)
    payload = {"source_checkpoint": str(checkpoint_path),
               "gradient_signature_shape": list(g.shape),
               "profiled_items": int((signatures["grad_count"] > 0).sum()),
               "collision": collision, "per_level": level_stats,
               "pair_compatibility": pair_report,
               "pair_compatibility_by_min_profile_count": pair_report_by_min_count,
               "item_C_quantiles": quantiles,
               "definition": "C_i is the mean dot product with the deduplicated union of all same-token items across levels; g is the saved positive-target directional mean and is not outer-normalized.",
               "atomic_id": None}
    (output / "diagnostic.json").write_text(json.dumps(payload, indent=2))
    np.savez(output / "pair_compatibility.npz", same_token=pair_values(g, same), semantic_neighbor_different_token=pair_values(g, semantic), random_pair=pair_values(g, random))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    arrays = [pair_values(g, same), pair_values(g, semantic), pair_values(g, random)]
    labels = ["same SID token", "semantic neighbor / different token", "random pair"]
    ax.hist(arrays, bins=50, density=True, histtype="step", linewidth=1.8, label=labels)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"gradient compatibility $g_i^\top g_j$")
    ax.set_ylabel("density")
    ax.set_title("Hard SID positive-target gradient compatibility")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / "pair_compatibility.png", dpi=180); fig.savefig(output / "pair_compatibility.pdf"); plt.close(fig)
    print(json.dumps({"output": str(output), "collision": collision, "pair_report": pair_report, "quantiles": quantiles}, indent=2))


if __name__ == "__main__":
    main()
