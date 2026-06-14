import argparse
import random
import time
from functools import partial
from pathlib import Path
from typing import Dict, Sequence

import torch
from torch.utils.data import DataLoader

from .data import NextItemDataset, RandomNegativeSampler, collate_eval, collate_train
from .io import read_json, write_json
from .model import LoCoRec
from .soft_sid import (
    SoftSIDConfig,
    build_semantic_table,
    build_soft_sid_table,
    build_train_item_frequency,
)


def sampled_cross_entropy(scores: torch.Tensor) -> torch.Tensor:
    labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
    return torch.nn.functional.cross_entropy(scores, labels)


@torch.no_grad()
def evaluate_full_ranking(
    model: LoCoRec,
    loader: DataLoader,
    device: torch.device,
    num_items: int,
    candidate_chunk_size: int,
    cutoffs: Sequence[int] = (5, 10, 20),
) -> Dict[str, float]:
    model.eval()
    hits = {cutoff: 0.0 for cutoff in cutoffs}
    ndcgs = {cutoff: 0.0 for cutoff in cutoffs}
    total = 0
    all_items = torch.arange(1, num_items + 1, device=device)
    max_cutoff = max(cutoffs)

    for sequences, targets, full_histories in loader:
        sequences = sequences.to(device)
        targets = targets.to(device)
        user_vectors, _ = model.encode_sequence(sequences)
        score_chunks = []
        for start in range(0, num_items, candidate_chunk_size):
            candidates = all_items[start : start + candidate_chunk_size]
            candidate_vectors = model.item_encoder(candidates)["vectors"]
            score_chunks.append(user_vectors @ candidate_vectors.transpose(0, 1))
        scores = torch.cat(score_chunks, dim=1)

        for row, (history, target) in enumerate(zip(full_histories, targets.tolist())):
            seen_items = {int(item) for item in history if int(item) != target}
            if seen_items:
                columns = torch.tensor(
                    [item - 1 for item in seen_items],
                    dtype=torch.long,
                    device=device,
                )
                scores[row, columns] = float("-inf")

        top_items = scores.topk(k=max_cutoff, dim=1).indices + 1
        target_column = targets.unsqueeze(1)
        for cutoff in cutoffs:
            matches = top_items[:, :cutoff].eq(target_column)
            hit = matches.any(dim=1)
            rank = matches.float().argmax(dim=1) + 1
            hits[cutoff] += hit.float().sum().item()
            ndcgs[cutoff] += (
                hit.float() / torch.log2(rank.float() + 1.0)
            ).sum().item()
        total += sequences.size(0)

    metrics: Dict[str, float] = {}
    for cutoff in cutoffs:
        metrics[f"HR@{cutoff}"] = hits[cutoff] / max(total, 1)
        metrics[f"Recall@{cutoff}"] = metrics[f"HR@{cutoff}"]
        metrics[f"NDCG@{cutoff}"] = ndcgs[cutoff] / max(total, 1)
    return metrics


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequences = read_json(Path(args.dataset_dir) / "sequences.json")
    stats = read_json(Path(args.dataset_dir) / "stats.json")
    semantic_obj = read_json(args.semantic_ids)
    num_items = int(stats["num_items"])

    hard_table, item_codes, num_semantic_tokens = build_semantic_table(
        semantic_obj, num_items
    )
    preprocess_start = time.perf_counter()
    soft_ids, candidate_prior, local_consistency, hard_consistency = build_soft_sid_table(
        hard_table,
        item_codes,
        SoftSIDConfig(
            top_m=args.soft_top_m,
            loo_min_overlap_slots=args.loo_min_overlap_slots,
            min_support=args.soft_min_support,
            max_neighbors=args.soft_max_neighbors,
            tie_break_seed=args.seed,
        ),
    )
    write_json(
        output_dir / "soft_sid_preprocess.json",
        {
            "elapsed_seconds": time.perf_counter() - preprocess_start,
            "local_consistency_mean": float(local_consistency[1:].mean()),
            "local_consistency_min": float(local_consistency[1:].min()),
            "local_consistency_max": float(local_consistency[1:].max()),
            "hard_consistency_mean": float(hard_consistency[1:].mean()),
            "hard_consistency_min": float(hard_consistency[1:].min()),
            "hard_consistency_max": float(hard_consistency[1:].max()),
            "mismatch_score_mean": float(
                (local_consistency[1:] - hard_consistency[1:]).mean()
            ),
        },
    )
    item_frequency = build_train_item_frequency(sequences, num_items)

    train_dataset = NextItemDataset(sequences, args.max_len, "train")
    valid_dataset = NextItemDataset(sequences, args.max_len, "valid")
    test_dataset = NextItemDataset(sequences, args.max_len, "test")
    sampler = RandomNegativeSampler(num_items, args.num_random_negatives)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=partial(collate_train, sampler=sampler),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_eval,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_eval,
    )

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    model = LoCoRec(
        num_items=num_items,
        num_semantic_tokens=num_semantic_tokens,
        soft_sid_table=soft_ids,
        candidate_prior=candidate_prior,
        local_consistency=local_consistency,
        item_frequency=item_frequency,
        dim=args.dim,
        max_len=args.max_len,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        tail_tau=args.tail_tau,
        residual_scale=args.residual_scale,
        gate_correction_scale=args.gate_correction_scale,
        gate_private_margin=args.gate_private_margin,
    ).to(device)

    gate_parameters = list(model.item_encoder.residual_gate.parameters())
    gate_ids = {id(parameter) for parameter in gate_parameters}
    base_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in gate_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": base_parameters, "lr": args.lr},
            {"params": gate_parameters, "lr": args.lr * args.gate_lr_scale},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_valid = -1.0
    best_path = output_dir / "best.pt"
    history = []
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        gate_trainable = epoch > args.gate_warmup_epochs
        for parameter in gate_parameters:
            parameter.requires_grad_(gate_trainable)

        totals = {
            "loss": 0.0,
            "rec_loss": 0.0,
            "attention_entropy": 0.0,
            "gate_kl": 0.0,
            "private_penalty": 0.0,
        }
        gate_mean = torch.zeros(3)
        steps = 0
        for sequences_batch, candidates in train_loader:
            sequences_batch = sequences_batch.to(device)
            candidates = candidates.to(device)
            output = model(sequences_batch, candidates)
            rec_loss = sampled_cross_entropy(output["score"])
            loss = (
                rec_loss
                + args.gate_kl_weight * output["gate_kl"]
                + args.gate_private_weight * output["private_penalty"]
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            totals["loss"] += loss.item()
            totals["rec_loss"] += rec_loss.item()
            totals["attention_entropy"] += output["attention_entropy"].item()
            totals["gate_kl"] += output["gate_kl"].item()
            totals["private_penalty"] += output["private_penalty"].item()
            gate_mean += output["gate_mean"].detach().cpu()
            steps += 1

        valid_metrics = evaluate_full_ranking(
            model,
            valid_loader,
            device,
            num_items,
            args.eval_candidate_chunk_size,
        )
        record = {
            "epoch": epoch,
            **{key: value / max(steps, 1) for key, value in totals.items()},
            "gate_trainable": gate_trainable,
            "gate_basis": float(gate_mean[0] / max(steps, 1)),
            "gate_shared": float(gate_mean[1] / max(steps, 1)),
            "gate_private": float(gate_mean[2] / max(steps, 1)),
            **valid_metrics,
        }
        history.append(record)
        print(record)

        if valid_metrics["NDCG@10"] > best_valid:
            best_valid = valid_metrics["NDCG@10"]
            stale_epochs = 0
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
        else:
            stale_epochs = 0 if not gate_trainable else stale_epochs + 1
            if stale_epochs >= args.early_stop_patience:
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate_full_ranking(
        model,
        test_loader,
        device,
        num_items,
        args.eval_candidate_chunk_size,
    )
    result = {
        "test": test_metrics,
        "best_valid_NDCG@10": best_valid,
        "learned_prior_beta": float(
            torch.nn.functional.softplus(model.item_encoder.prior_beta_raw).detach().cpu()
        ),
        "learned_gate_statistics": model.item_encoder.statistics(),
        "args": vars(args),
    }
    write_json(output_dir / "history.json", history)
    write_json(output_dir / "test_metrics.json", result)
    print(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--semantic-ids", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-candidate-chunk-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-random-negatives", type=int, default=100)
    parser.add_argument("--soft-top-m", type=int, default=4)
    parser.add_argument("--loo-min-overlap-slots", type=int, default=2)
    parser.add_argument("--soft-min-support", type=float, default=0.05)
    parser.add_argument("--soft-max-neighbors", type=int, default=50)
    parser.add_argument("--tail-tau", type=float, default=20.0)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--gate-correction-scale", type=float, default=0.3)
    parser.add_argument("--gate-kl-weight", type=float, default=0.05)
    parser.add_argument("--gate-private-weight", type=float, default=0.1)
    parser.add_argument("--gate-private-margin", type=float, default=0.05)
    parser.add_argument("--gate-warmup-epochs", type=int, default=10)
    parser.add_argument("--gate-lr-scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main() -> None:
    train(build_parser().parse_args())
