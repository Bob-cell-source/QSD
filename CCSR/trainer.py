import argparse
import math
import random
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from LoCoRec.locorec.data import NextItemDataset, collate_eval, collate_train
from LoCoRec.locorec.io import read_json, write_json
from LoCoRec.locorec.soft_sid import build_semantic_table
from .model import CCSR, load_hard_checkpoint


class NegativeSampler:
    def __init__(self, num_items, num_negatives):
        self.num_items, self.num_negatives = num_items, num_negatives

    def sample(self, target, known_train_positives):
        used = {target, *known_train_positives}
        available = self.num_items - len(used)
        if available < self.num_negatives:
            raise ValueError(f"Only {available} eligible negatives; requested {self.num_negatives}.")
        if available < 2 * self.num_negatives:
            pool = [i for i in range(1, self.num_items + 1) if i not in used]
            return [target] + random.sample(pool, self.num_negatives)
        result = [target]
        while len(result) <= self.num_negatives:
            item = random.randint(1, self.num_items)
            if item not in used:
                result.append(item)
                used.add(item)
        return result


class RouterStats:
    def __init__(self, cost):
        self.cost = cost.detach().double().cpu()
        self.mass = torch.zeros_like(self.cost)
        self.hard = torch.zeros_like(self.cost)
        self.entropy = 0.0
        self.count = 0

    def update(self, pi):
        pi = pi.detach().double().cpu()
        self.mass += pi.sum(0)
        self.hard += torch.bincount(pi.argmax(-1), minlength=pi.size(-1))
        self.entropy += float(-(pi * pi.clamp_min(1e-30).log()).sum())
        self.count += pi.size(0)

    def result(self):
        if not self.count:
            return {"count": 0}
        mass = self.mass / self.count
        return {"count": self.count, "probability_distribution": mass.tolist(),
                "selected_distribution": (self.hard / self.count).tolist(),
                "mean_expected_resolution": float(mass @ torch.arange(1, len(mass) + 1, dtype=mass.dtype)),
                "mean_cost": float(mass @ self.cost), "router_entropy": self.entropy / self.count}


def training_stage(epoch, warmup_epochs, lambda_res, ramp_epochs=5, fixed_resolution=None):
    if fixed_resolution is not None:
        return False, 0.0
    uniform = epoch <= warmup_epochs
    router_epoch = epoch - warmup_epochs
    scale = min(1.0, max(0.0, (router_epoch - 1) / max(ramp_epochs - 1, 1)))
    if ramp_epochs == 1 and not uniform:
        scale = 1.0
    return uniform, 0.0 if uniform else lambda_res * scale


@torch.no_grad()
def evaluate(model, loader, device, num_items, chunk_size, group_labels=None,
             fixed_resolution=None, cutoffs=(5, 10, 20)):
    model.eval()
    modes = ["hard", "soft"] + [str(r) for r in range(1, model.depth + 2)]
    totals = {mode: {f"{metric}@{k}": 0.0 for metric in ("HR", "NDCG") for k in cutoffs} for mode in modes}
    router = RouterStats(model.item_encoder.resolution_cost)
    groups = {}
    # Cache every catalog representation once per evaluation. CPU storage bounds GPU use.
    cache = []
    for start in range(1, num_items + 1, chunk_size):
        items = torch.arange(start, min(start + chunk_size, num_items + 1), device=device)
        cache.append((start, model.item_encoder(items).cpu()))
    total = 0
    top_k = min(max(cutoffs), num_items)
    for sequences, targets, histories in loader:
        sequences, targets = sequences.to(device), targets.to(device)
        h = model.encode_sequence(sequences)
        pi = model.stopping_distribution(h, fixed_resolution=fixed_resolution)
        router.update(pi)
        if group_labels is not None:
            labels = group_labels[total:total + len(sequences)]
            for label in set(labels):
                group = groups.setdefault(label, RouterStats(model.item_encoder.resolution_cost))
                mask = torch.tensor([v == label for v in labels], device=device)
                group.update(pi[mask])
        best_scores = {mode: h.new_empty(len(h), 0) for mode in modes}
        best_ids = {mode: torch.empty(len(h), 0, dtype=torch.long, device=device) for mode in modes}
        seen_rows, seen_ids = [], []
        for row, (history, target) in enumerate(zip(histories, targets.tolist())):
            seen = set(history) - {target, 0}
            seen_rows.extend([row] * len(seen))
            seen_ids.extend(seen)
        seen_rows = torch.tensor(seen_rows, device=device, dtype=torch.long)
        seen_ids = torch.tensor(seen_ids, device=device, dtype=torch.long)
        for start, cpu_vectors in cache:
            vectors = cpu_vectors.to(device)
            in_chunk = (seen_ids >= start) & (seen_ids < start + len(vectors))
            rows, cols = seen_rows[in_chunk], seen_ids[in_chunk] - start
            item_ids = torch.arange(start, start + len(vectors), device=device).expand(len(h), -1)
            for mode in modes:
                scores = model.score_catalog(h, vectors, pi, mode)
                scores[rows, cols] = -torch.inf
                combined_scores = torch.cat([best_scores[mode], scores], 1)
                combined_ids = torch.cat([best_ids[mode], item_ids], 1)
                values, indices = combined_scores.topk(min(top_k, combined_scores.size(1)), dim=1)
                best_scores[mode], best_ids[mode] = values, combined_ids.gather(1, indices)
        for mode in modes:
            for k in cutoffs:
                matches = best_ids[mode][:, :k].eq(targets[:, None])
                hit = matches.any(-1)
                rank = matches.float().argmax(-1) + 1
                totals[mode][f"HR@{k}"] += float(hit.float().sum())
                totals[mode][f"NDCG@{k}"] += float((hit / torch.log2(rank.float() + 1)).sum())
        total += len(h)
    if total == 0:
        raise ValueError("Evaluation split is empty.")
    metrics = {mode: {key: value / total for key, value in values.items()} for mode, values in totals.items()}
    for values in metrics.values():
        values.update({f"Recall@{k}": values[f"HR@{k}"] for k in cutoffs})
    return {"ranking": metrics, "router": router.result(),
            "groups": {label: stats.result() for label, stats in groups.items()}}


def train(args):
    if args.epochs <= args.warmup_epochs and args.fixed_resolution is None:
        raise ValueError("epochs must include at least one epoch after resolution warm-up.")
    for name in ("epochs", "batch_size", "eval_candidate_chunk_size", "max_len", "dim",
                 "num_heads", "num_layers", "num_random_negatives", "lambda_warmup_epochs", "early_stop_patience"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive.")
    if args.warmup_epochs < 0 or args.lambda_res < 0 or args.num_workers < 0:
        raise ValueError("Warm-up epochs, lambda_res and num_workers must be nonnegative.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequences = read_json(Path(args.dataset_dir) / "sequences.json")
    num_items = int(read_json(Path(args.dataset_dir) / "stats.json")["num_items"])
    semantic = read_json(args.semantic_ids)
    sizes = semantic["codebook_sizes"]
    for codes in semantic["semantic_ids"].values():
        if len(codes) != len(sizes) or any(not 0 <= int(c) < int(s) for c, s in zip(codes, sizes)):
            raise ValueError("SID codes must lie within their level's codebook range.")
    if any(not 1 <= int(item) <= num_items for row in sequences for item in row["items"]):
        raise ValueError("Sequence item IDs must be in 1..num_items.")
    hard_table, _, num_tokens = build_semantic_table(semantic, num_items)
    model = CCSR(num_items, num_tokens, hard_table, args.dim, args.max_len,
                 args.num_heads, args.num_layers, args.dropout)
    if args.fixed_resolution is not None and not 1 <= args.fixed_resolution <= model.depth + 1:
        raise ValueError("fixed_resolution must be in 1..L+1.")
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        write_json(output_dir / "initialization.json", load_hard_checkpoint(model, checkpoint))
    else:
        write_json(output_dir / "initialization.json", {"from_scratch": True})
    encoder = model.item_encoder
    write_json(output_dir / "resolution_preprocess.json", {
        "num_items": num_items, "depth": model.depth,
        "prefix_entropy": encoder.prefix_entropy.tolist(), "resolution_cost": encoder.resolution_cost.tolist(),
        "prefix_group_counts": [int(encoder.prefix_group_id[:, r].max()) for r in range(model.depth)],
        "last_sid_equals_id": bool(encoder.resolution_cost[-2] == 1),
        "cost_definition": "catalog prefix entropy / log(N); item identity cost = 1",
    })
    model.to(device)
    datasets = {split: NextItemDataset(sequences, args.max_len, split) for split in ("train", "valid", "test")}
    if any(len(ds) == 0 for ds in datasets.values()):
        raise ValueError("Train, validation and test splits must all be nonempty.")
    sampler = NegativeSampler(num_items, args.num_random_negatives)
    # Fail before starting workers rather than hanging when unique negatives are unavailable.
    for _, target, known in datasets["train"].samples:
        if num_items - len({target, *known}) < args.num_random_negatives:
            raise ValueError("Too few unseen catalog items for the requested unique negatives.")
    loaders = {split: DataLoader(ds, batch_size=args.batch_size, shuffle=split == "train",
               num_workers=args.num_workers, collate_fn=partial(collate_train, sampler=sampler)
               if split == "train" else collate_eval) for split, ds in datasets.items()}
    labels = read_json(args.group_labels) if args.group_labels else {}
    for split, values in labels.items():
        if split not in ("valid", "test") or len(values) != len(datasets[split]) or any(not isinstance(v, str) for v in values):
            raise ValueError("Group labels must be string arrays aligned with valid/test dataset samples.")
    # No weight decay or extra regularizers: exactly expected CE + lambda * expected cost.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_valid, best_epoch, stale = -math.inf, None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        uniform, coefficient = training_stage(epoch, args.warmup_epochs, args.lambda_res,
                                               args.lambda_warmup_epochs, args.fixed_resolution)
        router_trainable = not uniform and args.fixed_resolution is None
        for parameter in model.stop_head.parameters():
            parameter.requires_grad_(router_trainable)
        router = RouterStats(model.item_encoder.resolution_cost)
        totals = {key: 0.0 for key in ("loss", "loss_pred", "loss_res")}
        per_resolution = torch.zeros(model.depth + 1)
        count = 0
        for sequence, candidates in loaders["train"]:
            sequence, candidates = sequence.to(device), candidates.to(device)
            output = model(sequence, candidates, uniform, args.fixed_resolution)
            losses = model.objective(output, coefficient)
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError("Non-finite training loss.")
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            batch = len(sequence)
            for key in totals:
                totals[key] += float(losses[key].detach()) * batch
            per_resolution += losses["loss_per_resolution"].detach().cpu().sum(0)
            router.update(output["pi"])
            count += batch
        valid = evaluate(model, loaders["valid"], device, num_items, args.eval_candidate_chunk_size,
                         labels.get("valid"), args.fixed_resolution)
        record = {"epoch": epoch, "stage": "warmup" if uniform else "fixed" if args.fixed_resolution else "router",
                  "lambda_res": coefficient, **{k: v / count for k, v in totals.items()},
                  "loss_per_resolution": (per_resolution / count).tolist(),
                  "train_router": router.result(), "valid": valid}
        history.append(record)
        write_json(output_dir / "history.json", history)
        print(record, flush=True)
        # Warm-up checkpoints cannot become the final adaptive result.
        metric = valid["ranking"]["hard"]["NDCG@10"]
        if not uniform and metric > best_valid:
            best_valid, best_epoch, stale = metric, epoch, 0
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch,
                        "best_valid_NDCG@10": metric}, output_dir / "best.pt")
        elif not uniform:
            # Do not stop early before the cost ramp finishes.
            if args.fixed_resolution is not None or epoch >= args.warmup_epochs + args.lambda_warmup_epochs:
                stale += 1
                if stale >= args.early_stop_patience:
                    break
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    test = evaluate(model, loaders["test"], device, num_items, args.eval_candidate_chunk_size,
                    labels.get("test"), args.fixed_resolution)
    result = {"test": test, "best_epoch": best_epoch, "best_valid_NDCG@10": best_valid,
              "selection_mode": "hard", "args": vars(args)}
    write_json(output_dir / "test_metrics.json", result)
    print(result, flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Hard SID Context-Conditioned Sharing Resolution")
    for name in ("dataset-dir", "semantic-ids", "output-dir"):
        parser.add_argument(f"--{name}", required=True)
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument("--init-checkpoint", help="LoCoRec Hard SID ablation checkpoint with shared residual")
    initialization.add_argument("--from-scratch", action="store_true", help="Explicit scratch control; bypass Stage 0")
    parser.add_argument("--group-labels", help='JSON {"valid": ["Mem", "Gen", ...], "test": [...]}')
    parser.add_argument("--fixed-resolution", type=int, help="Train a separate fixed baseline, numbered 1..L+1")
    parser.add_argument("--device", default="cuda")
    for name, default in (("epochs", 100), ("warmup-epochs", 3), ("lambda-warmup-epochs", 5),
                          ("early-stop-patience", 10), ("batch-size", 256), ("eval-candidate-chunk-size", 2048),
                          ("num-workers", 0), ("max-len", 50), ("dim", 128), ("num-heads", 2),
                          ("num-layers", 2), ("num-random-negatives", 100), ("seed", 2026)):
        parser.add_argument(f"--{name}", type=int, default=default)
    for name, default in (("dropout", 0.2), ("lr", 1e-3), ("grad-clip", 5.0), ("lambda-res", 0.03)):
        parser.add_argument(f"--{name}", type=float, default=default)
    return parser


def main():
    train(build_parser().parse_args())
