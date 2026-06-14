import argparse
import random
from pathlib import Path
import time
from typing import Dict, Sequence

import torch
from torch.utils.data import DataLoader

from .data import NextItemDataset, RandomNegativeSampler, collate_eval, collate_train
from .io import read_json, write_json
from .model import LCSoftCRSID
from .soft_sid import (
    SoftSIDConfig,
    build_semantic_table,
    build_soft_sid_table,
    build_text_knn_neighbors,
    build_train_item_frequency,
)


def sampled_cross_entropy(scores: torch.Tensor) -> torch.Tensor:
    labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
    return torch.nn.functional.cross_entropy(scores, labels)


@torch.no_grad()
def evaluate_full_ranking(
    model: LCSoftCRSID,
    loader: DataLoader,
    device: torch.device,
    num_items: int,
    candidate_chunk_size: int,
    cutoffs: Sequence[int] = (5, 10, 20),
    mask_seen_items: bool = True,
) -> Dict[str, float]:
    model.eval()
    hits = {cutoff: 0.0 for cutoff in cutoffs}
    ndcgs = {cutoff: 0.0 for cutoff in cutoffs}
    mrrs = {cutoff: 0.0 for cutoff in cutoffs}
    total = 0
    all_items = torch.arange(1, num_items + 1, device=device)
    max_cutoff = max(cutoffs)

    for sequences, targets in loader:
        sequences = sequences.to(device)
        targets = targets.to(device)
        batch_size = sequences.size(0)
        score_chunks = []
        for start in range(0, num_items, candidate_chunk_size):
            candidates = all_items[start : start + candidate_chunk_size]
            candidates = candidates.unsqueeze(0).expand(batch_size, -1)
            score_chunks.append(model(sequences, candidates)["score"])
        scores = torch.cat(score_chunks, dim=1)

        seen_mask = sequences.gt(0)
        if mask_seen_items and seen_mask.any():
            rows, columns = seen_mask.nonzero(as_tuple=True)
            scores[rows, sequences[rows, columns] - 1] = float("-inf")

        top_items = scores.topk(k=max_cutoff, dim=1).indices + 1
        target_column = targets.unsqueeze(1)
        for cutoff in cutoffs:
            matches = top_items[:, :cutoff].eq(target_column)
            hit = matches.any(dim=1)
            hits[cutoff] += hit.float().sum().item()
            rank = matches.float().argmax(dim=1) + 1
            ndcgs[cutoff] += (hit.float() / torch.log2(rank.float() + 1.0)).sum().item()
            mrrs[cutoff] += (hit.float() / rank.float()).sum().item()
        total += batch_size

    metrics: Dict[str, float] = {}
    for cutoff in cutoffs:
        metrics[f"HR@{cutoff}"] = hits[cutoff] / max(total, 1)
        metrics[f"Recall@{cutoff}"] = metrics[f"HR@{cutoff}"]
        metrics[f"NDCG@{cutoff}"] = ndcgs[cutoff] / max(total, 1)
        metrics[f"MRR@{cutoff}"] = mrrs[cutoff] / max(total, 1)
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

    hard_sid_table, item_codes, num_semantic_tokens = build_semantic_table(semantic_obj, num_items)
    soft_config = SoftSIDConfig(
        top_m=args.soft_top_m,
        min_overlap_slots=args.soft_min_overlap_slots,
        min_support=args.soft_min_support,
        reliability_floor=args.soft_reliability_floor,
        max_neighbors=args.soft_max_neighbors,
        candidate_construction=(
            "uniform_topk"
            if args.candidate_weight_mode == "neighborhood_learned"
            else "local_prior"
        ),
    )
    soft_preprocess_start = time.perf_counter()
    base_neighbors = None
    neighbor_report: Dict[str, float | int | str] = {
        "neighbor_source": args.soft_neighbor_source,
        "candidate_construction": soft_config.candidate_construction,
    }
    if args.soft_neighbor_source == "text_knn":
        if not args.soft_text_embeddings or not args.soft_text_item_ids:
            raise ValueError(
                "text_knn requires --soft-text-embeddings and --soft-text-item-ids."
            )
        base_neighbors, text_report = build_text_knn_neighbors(
            embeddings_path=args.soft_text_embeddings,
            item_ids_path=args.soft_text_item_ids,
            num_items=num_items,
            max_neighbors=args.soft_max_neighbors,
            chunk_size=args.soft_text_knn_chunk_size,
        )
        neighbor_report.update(text_report)
    soft_sid_table, soft_sid_weights, reliability = build_soft_sid_table(
        semantic_table=hard_sid_table,
        item_codes=item_codes,
        config=soft_config,
        base_neighbors=base_neighbors,
    )
    tensor_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (soft_sid_table, soft_sid_weights, reliability)
    )
    write_json(
        output_dir / "soft_sid_preprocess.json",
        {
            **neighbor_report,
            "total_elapsed_seconds": time.perf_counter() - soft_preprocess_start,
            "soft_table_bytes": tensor_bytes,
            "reliability_mean": float(reliability[1:].mean().item()),
            "reliability_min": float(reliability[1:].min().item()),
            "reliability_max": float(reliability[1:].max().item()),
        },
    )
    item_frequency = build_train_item_frequency(sequences, num_items)

    train_dataset = NextItemDataset(sequences, args.max_len, "train")
    valid_dataset = NextItemDataset(sequences, args.max_len, "valid")
    test_dataset = NextItemDataset(sequences, args.max_len, "test")
    sampler = RandomNegativeSampler(num_items, args.num_random_negatives)
    train_collate = (
        collate_eval
        if args.train_objective == "full_softmax"
        else lambda batch: collate_train(batch, sampler)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_collate,
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

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = LCSoftCRSID(
        num_items=num_items,
        num_semantic_tokens=num_semantic_tokens,
        soft_sid_table=soft_sid_table,
        soft_sid_weights=soft_sid_weights,
        semantic_reliability=reliability,
        item_frequency=item_frequency,
        dim=args.dim,
        max_len=args.max_len,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        tail_tau=args.tail_tau,
        alpha_mode=args.alpha_mode,
        fusion_mode=args.fusion_mode,
        residual_scale=args.residual_scale,
        gate_correction_scale=args.gate_correction_scale,
        gate_private_margin=args.gate_private_margin,
        candidate_weight_mode=args.candidate_weight_mode,
        disable_semantic_basis=args.disable_semantic_basis,
        disable_shared_residual=args.disable_shared_residual,
        disable_private_residual=args.disable_private_residual,
    ).to(device)
    gate_parameters = (
        list(model.item_encoder.fusion_gate.parameters())
        if model.item_encoder.fusion_gate is not None
        else []
    )
    gate_parameter_ids = {id(parameter) for parameter in gate_parameters}
    base_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in gate_parameter_ids
    ]
    parameter_groups = [{"params": base_parameters, "lr": args.lr}]
    if gate_parameters:
        parameter_groups.append(
            {"params": gate_parameters, "lr": args.lr * args.gate_lr_scale}
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    catalog_items = torch.arange(1, num_items + 1, device=device)

    best_valid = -1.0
    best_path = output_dir / "best.pt"
    history = []
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        gate_trainable = epoch > args.gate_warmup_epochs
        for parameter in gate_parameters:
            parameter.requires_grad_(gate_trainable)
        total_loss = 0.0
        total_rec_loss = 0.0
        total_attention_kl = 0.0
        total_attention_entropy = 0.0
        total_gate_kl = 0.0
        total_gate_private_penalty = 0.0
        total_gate_mean = torch.zeros(3)
        steps = 0
        for sequence, candidates in train_loader:
            sequence = sequence.to(device)
            candidates = candidates.to(device)
            if args.train_objective == "full_softmax":
                output = model.full_catalog_forward(sequence, catalog_items)
                if not args.keep_seen_items:
                    seen_mask = sequence.gt(0)
                    if seen_mask.any():
                        rows, columns = seen_mask.nonzero(as_tuple=True)
                        output["score"][rows, sequence[rows, columns] - 1] = float("-inf")
                rec_loss = torch.nn.functional.cross_entropy(
                    output["score"], candidates - 1
                )
            else:
                output = model(sequence, candidates)
                rec_loss = sampled_cross_entropy(output["score"])
            loss = (
                rec_loss
                + args.gate_kl_weight * output["gate_kl"]
                + args.gate_private_weight * output["gate_private_penalty"]
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item()
            total_rec_loss += rec_loss.item()
            total_attention_kl += output["attention_kl"].item()
            total_attention_entropy += output["attention_entropy"].item()
            total_gate_kl += output["gate_kl"].item()
            total_gate_private_penalty += output["gate_private_penalty"].item()
            total_gate_mean += output["gate_mean"].detach().cpu()
            steps += 1

        valid_metrics = evaluate_full_ranking(
            model=model,
            loader=valid_loader,
            device=device,
            num_items=num_items,
            candidate_chunk_size=args.eval_candidate_chunk_size,
            mask_seen_items=not args.keep_seen_items,
        )
        record = {
            "epoch": epoch,
            "loss": total_loss / max(steps, 1),
            "rec_loss": total_rec_loss / max(steps, 1),
            "attention_kl": total_attention_kl / max(steps, 1),
            "attention_entropy": total_attention_entropy / max(steps, 1),
            "gate_kl": total_gate_kl / max(steps, 1),
            "gate_private_penalty": total_gate_private_penalty / max(steps, 1),
            "gate_trainable": gate_trainable,
            "gate_basis": float(total_gate_mean[0] / max(steps, 1)),
            "gate_shared": float(total_gate_mean[1] / max(steps, 1)),
            "gate_private": float(total_gate_mean[2] / max(steps, 1)),
            **valid_metrics,
        }
        history.append(record)
        print(record)

        valid_metric = valid_metrics[args.early_stop_metric]
        if valid_metric > best_valid:
            best_valid = valid_metric
            stale_epochs = 0
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
        else:
            # Do not consume early-stopping patience before the gate is opened.
            stale_epochs = 0 if not gate_trainable else stale_epochs + 1
            if stale_epochs >= args.early_stop_patience:
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate_full_ranking(
        model=model,
        loader=test_loader,
        device=device,
        num_items=num_items,
        candidate_chunk_size=args.eval_candidate_chunk_size,
        mask_seen_items=not args.keep_seen_items,
    )
    best_valid_ndcg = max((row.get("NDCG@10", 0.0) for row in history), default=0.0)
    result = {
        "test": test_metrics,
        "best_valid_NDCG@10": best_valid_ndcg,
        "early_stop_metric": args.early_stop_metric,
        "best_valid_metric": best_valid,
        "learned_prior_beta": model.item_encoder.prior_beta(),
        "learned_alpha_parameters": model.item_encoder.alpha_parameters(),
        "learned_gate_statistics": model.item_encoder.gate_statistics(),
        "args": vars(args),
    }
    write_json(output_dir / "history.json", history)
    write_json(output_dir / "test_metrics.json", result)
    print(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the clean LC-SoftCRSID implementation.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--semantic-ids", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument(
        "--early-stop-metric",
        choices=["NDCG@10", "MRR@10"],
        default="NDCG@10",
    )
    parser.add_argument(
        "--keep-seen-items",
        action="store_true",
        help="Keep previously interacted items in full-softmax training and ranking.",
    )
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
    parser.add_argument(
        "--train-objective",
        choices=["sampled", "full_softmax"],
        default="sampled",
    )
    parser.add_argument("--tail-tau", type=float, default=20.0)
    parser.add_argument(
        "--alpha-mode",
        choices=["fixed", "learnable_monotonic"],
        default="fixed",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=["fixed", "prior_guided_gate", "hierarchical_residual_gate"],
        default="fixed",
    )
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--gate-kl-weight", type=float, default=0.0)
    parser.add_argument("--gate-private-weight", type=float, default=0.0)
    parser.add_argument("--gate-private-margin", type=float, default=0.0)
    parser.add_argument("--gate-correction-scale", type=float, default=1.0)
    parser.add_argument("--gate-warmup-epochs", type=int, default=0)
    parser.add_argument("--gate-lr-scale", type=float, default=1.0)
    parser.add_argument(
        "--candidate-weight-mode",
        choices=["fixed", "learned", "prior_guided", "neighborhood_learned"],
        default="prior_guided",
    )
    parser.add_argument("--soft-top-m", type=int, default=4)
    parser.add_argument("--soft-min-overlap-slots", type=int, default=3)
    parser.add_argument("--soft-min-support", type=float, default=0.05)
    parser.add_argument("--soft-reliability-floor", type=float, default=0.10)
    parser.add_argument("--soft-max-neighbors", type=int, default=50)
    parser.add_argument(
        "--soft-neighbor-source",
        choices=["sid_overlap", "text_knn"],
        default="sid_overlap",
    )
    parser.add_argument("--soft-text-embeddings")
    parser.add_argument("--soft-text-item-ids")
    parser.add_argument("--soft-text-knn-chunk-size", type=int, default=256)
    parser.add_argument("--disable-semantic-basis", action="store_true")
    parser.add_argument("--disable-shared-residual", action="store_true")
    parser.add_argument("--disable-private-residual", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main() -> None:
    train(build_parser().parse_args())
