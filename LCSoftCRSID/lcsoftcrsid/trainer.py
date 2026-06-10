import argparse
import random
from pathlib import Path
from typing import Dict, Sequence

import torch
from torch.utils.data import DataLoader

from .data import NextItemDataset, RandomNegativeSampler, collate_eval, collate_train
from .io import read_json, write_json
from .model import LCSoftCRSID
from .soft_sid import SoftSIDConfig, build_semantic_table, build_soft_sid_table, build_train_item_frequency


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
) -> Dict[str, float]:
    model.eval()
    hits = {cutoff: 0.0 for cutoff in cutoffs}
    ndcgs = {cutoff: 0.0 for cutoff in cutoffs}
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
        if seen_mask.any():
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
        total += batch_size

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

    hard_sid_table, item_codes, num_semantic_tokens = build_semantic_table(semantic_obj, num_items)
    soft_config = SoftSIDConfig(
        top_m=args.soft_top_m,
        min_overlap_slots=args.soft_min_overlap_slots,
        min_support=args.soft_min_support,
        support_eta=args.soft_support_eta,
        hard_token_prior=args.soft_hard_token_prior,
        reliability_floor=args.soft_reliability_floor,
        max_neighbors=args.soft_max_neighbors,
    )
    soft_sid_table, soft_sid_weights, reliability = build_soft_sid_table(
        semantic_table=hard_sid_table,
        item_codes=item_codes,
        config=soft_config,
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
        collate_fn=lambda batch: collate_train(batch, sampler),
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
        residual_scale=args.residual_scale,
        frequency_transform=args.frequency_transform,
        alpha_mode=args.alpha_mode,
        candidate_weight_mode=args.candidate_weight_mode,
        prior_beta_init=args.prior_beta_init,
        disable_semantic_basis=args.disable_semantic_basis,
        disable_shared_residual=args.disable_shared_residual,
        disable_private_residual=args.disable_private_residual,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_valid = -1.0
    best_path = output_dir / "best.pt"
    history = []
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rec_loss = 0.0
        total_attention_kl = 0.0
        total_attention_entropy = 0.0
        steps = 0
        for sequence, candidates in train_loader:
            sequence = sequence.to(device)
            candidates = candidates.to(device)
            output = model(sequence, candidates)
            rec_loss = sampled_cross_entropy(output["score"])
            loss = rec_loss + args.attention_kl_weight * output["attention_kl"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item()
            total_rec_loss += rec_loss.item()
            total_attention_kl += output["attention_kl"].item()
            total_attention_entropy += output["attention_entropy"].item()
            steps += 1

        valid_metrics = evaluate_full_ranking(
            model=model,
            loader=valid_loader,
            device=device,
            num_items=num_items,
            candidate_chunk_size=args.eval_candidate_chunk_size,
        )
        record = {
            "epoch": epoch,
            "loss": total_loss / max(steps, 1),
            "rec_loss": total_rec_loss / max(steps, 1),
            "attention_kl": total_attention_kl / max(steps, 1),
            "attention_entropy": total_attention_entropy / max(steps, 1),
            **valid_metrics,
        }
        history.append(record)
        print(record)

        valid_ndcg = valid_metrics["NDCG@10"]
        if valid_ndcg > best_valid:
            best_valid = valid_ndcg
            stale_epochs = 0
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
        else:
            stale_epochs += 1
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
    )
    result = {
        "test": test_metrics,
        "best_valid_NDCG@10": best_valid,
        "learned_prior_beta": model.item_encoder.prior_beta(),
        "learned_alpha_parameters": model.item_encoder.alpha_parameters(),
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
    parser.add_argument("--tail-tau", type=float, default=20.0)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--frequency-transform", choices=["raw", "log"], default="raw")
    parser.add_argument(
        "--alpha-mode",
        choices=["fixed", "learnable_monotonic"],
        default="fixed",
    )
    parser.add_argument(
        "--candidate-weight-mode",
        choices=["fixed", "learned", "prior_guided"],
        default="fixed",
    )
    parser.add_argument("--prior-beta-init", type=float, default=1.0)
    parser.add_argument("--attention-kl-weight", type=float, default=0.0)
    parser.add_argument("--soft-top-m", type=int, default=4)
    parser.add_argument("--soft-min-overlap-slots", type=int, default=2)
    parser.add_argument("--soft-min-support", type=float, default=0.05)
    parser.add_argument("--soft-support-eta", type=float, default=2.0)
    parser.add_argument("--soft-hard-token-prior", type=float, default=1.0)
    parser.add_argument("--soft-reliability-floor", type=float, default=0.10)
    parser.add_argument("--soft-max-neighbors", type=int, default=50)
    parser.add_argument("--disable-semantic-basis", action="store_true")
    parser.add_argument("--disable-shared-residual", action="store_true")
    parser.add_argument("--disable-private-residual", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main() -> None:
    train(build_parser().parse_args())
