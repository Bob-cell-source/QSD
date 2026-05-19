import argparse
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .io_utils import read_json, write_json
from .model import QSDRec


class NextItemDataset(Dataset):
    def __init__(self, sequences: List[Dict], max_len: int, split: str) -> None:
        self.max_len = max_len
        self.samples: List[Tuple[List[int], int]] = []
        for row in sequences:
            items = row["items"]
            if len(items) < 3:
                continue
            if split == "train":
                for idx in range(1, len(items) - 2):
                    self.samples.append((items[:idx], items[idx]))
            elif split == "valid":
                self.samples.append((items[:-2], items[-2]))
            elif split == "test":
                self.samples.append((items[:-1], items[-1]))
            else:
                raise ValueError(split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        seq, target = self.samples[idx]
        seq = seq[-self.max_len :]
        padded = [0] * (self.max_len - len(seq)) + seq
        return torch.tensor(padded, dtype=torch.long), int(target)


class CandidateSampler:
    def __init__(
        self,
        num_items: int,
        item_semantic_ids: Dict[int, List[int]],
        prefix_level: int,
        hard_neg_mode: str,
        min_overlap_slots: int,
        num_random_neg: int,
        num_hard_neg: int,
    ) -> None:
        self.num_items = num_items
        self.num_random_neg = num_random_neg
        self.num_hard_neg = num_hard_neg
        self.prefix_level = prefix_level
        self.hard_neg_mode = hard_neg_mode
        self.min_overlap_slots = min_overlap_slots
        self.prefix_groups: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
        self.slot_groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for item, sid in item_semantic_ids.items():
            key = tuple(sid[:prefix_level])
            self.prefix_groups[key].append(item)
            for slot, code in enumerate(sid):
                self.slot_groups[(slot, int(code))].append(item)
        self.item_semantic_ids = item_semantic_ids
        self.all_items = list(range(1, num_items + 1))

    def hard_pool(self, target: int, seen: set[int], used: set[int]) -> List[int]:
        sid = self.item_semantic_ids.get(target)
        if sid is None or self.num_hard_neg <= 0:
            return []
        if self.hard_neg_mode == "prefix":
            group = self.prefix_groups.get(tuple(sid[: self.prefix_level]), [])
            return [x for x in group if x not in used and x not in seen]
        if self.hard_neg_mode == "overlap":
            counts: Counter[int] = Counter()
            for slot, code in enumerate(sid):
                counts.update(self.slot_groups.get((slot, int(code)), []))
            return [
                item
                for item, count in counts.items()
                if count >= self.min_overlap_slots and item not in used and item not in seen
            ]
        raise ValueError(f"Unsupported hard_neg_mode: {self.hard_neg_mode}")

    def sample_one(self, target: int, seen: set[int] | None = None) -> List[int]:
        seen = seen or set()
        candidates = [target]
        used = {target}
        hard_pool = self.hard_pool(target, seen, used)
        if hard_pool:
            hard_negs = random.sample(hard_pool, min(self.num_hard_neg, len(hard_pool)))
            candidates.extend(hard_negs)
            used.update(hard_negs)
        while len(candidates) < 1 + self.num_hard_neg + self.num_random_neg:
            neg = random.randint(1, self.num_items)
            if neg not in used and neg not in seen:
                candidates.append(neg)
                used.add(neg)
        return candidates


def collate_train(batch, sampler: CandidateSampler):
    seqs, targets = zip(*batch)
    seq_tensor = torch.stack(seqs)
    cand = []
    for seq, target in zip(seqs, targets):
        seen = {int(x) for x in seq.tolist() if int(x) > 0}
        cand.append(sampler.sample_one(int(target), seen))
    return seq_tensor, torch.tensor(cand, dtype=torch.long)


def collate_full_eval(batch):
    seqs, targets = zip(*batch)
    return torch.stack(seqs), torch.tensor(targets, dtype=torch.long)


def build_semantic_table(
    semantic_obj: Dict,
    num_items: int,
) -> Tuple[torch.Tensor, Dict[int, List[int]], int]:
    raw = {int(k): list(map(int, v)) for k, v in semantic_obj["semantic_ids"].items()}
    sizes = list(map(int, semantic_obj["codebook_sizes"]))
    offsets = [1]
    for size in sizes[:-1]:
        offsets.append(offsets[-1] + size)
    table = torch.zeros(num_items + 1, len(sizes), dtype=torch.long)
    offset_sem_ids: Dict[int, List[int]] = {}
    for item in range(1, num_items + 1):
        codes = raw.get(item, [0] * len(sizes))
        token_ids = [offsets[idx] + int(code) for idx, code in enumerate(codes)]
        table[item] = torch.tensor(token_ids, dtype=torch.long)
        offset_sem_ids[item] = codes
    return table, offset_sem_ids, sum(sizes)


def cross_entropy_first_positive(scores: torch.Tensor) -> torch.Tensor:
    labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
    return torch.nn.functional.cross_entropy(scores, labels)


def full_softmax_loss(
    model: QSDRec,
    seq: torch.Tensor,
    targets: torch.Tensor,
    num_items: int,
    sem_weight: float,
    candidate_chunk_size: int,
) -> torch.Tensor:
    batch_size = seq.size(0)
    all_items = torch.arange(1, num_items + 1, dtype=torch.long, device=seq.device)
    score_chunks = []
    for start in range(0, num_items, candidate_chunk_size):
        cand = all_items[start : start + candidate_chunk_size]
        cand = cand.unsqueeze(0).expand(batch_size, -1)
        score_chunks.append(model(seq, cand, sem_weight=sem_weight)["score"])
    scores = torch.cat(score_chunks, dim=1)

    seen_mask = seq.gt(0)
    if seen_mask.any():
        history_rows, history_cols = seen_mask.nonzero(as_tuple=True)
        history_item_idx = seq[history_rows, history_cols] - 1
        scores[history_rows, history_item_idx] = float("-inf")

    labels = targets - 1
    return torch.nn.functional.cross_entropy(scores, labels)


@torch.no_grad()
def evaluate_full_ranking(
    model: QSDRec,
    loader: DataLoader,
    device: torch.device,
    num_items: int,
    sem_weight: float,
    ks: Sequence[int] = (5, 10, 20),
    batch_eval_size: int = 1024,
) -> Dict[str, float]:
    model.eval()
    hits = {k: 0.0 for k in ks}
    ndcgs = {k: 0.0 for k in ks}
    total = 0

    all_items = torch.arange(1, num_items + 1, dtype=torch.long, device=device)
    max_k = max(ks)

    for seq, targets in loader:
        seq = seq.to(device)
        targets = targets.to(device)
        batch_size = seq.size(0)

        score_chunks = []
        for start in range(0, num_items, batch_eval_size):
            cand = all_items[start : start + batch_eval_size]
            cand = cand.unsqueeze(0).expand(batch_size, -1)
            out = model(seq, cand, sem_weight=sem_weight)
            score_chunks.append(out["score"])
        scores = torch.cat(score_chunks, dim=1)

        # Exclude historical interactions so evaluation follows leave-one-out
        # full ranking instead of rewarding already-consumed items.
        seen_mask = seq.gt(0)
        if seen_mask.any():
            history_rows, history_cols = seen_mask.nonzero(as_tuple=True)
            history_item_idx = seq[history_rows, history_cols] - 1
            scores[history_rows, history_item_idx] = float("-inf")

        topk_idx = scores.topk(k=max_k, dim=1).indices + 1
        target_col = targets.unsqueeze(1)

        for k in ks:
            topk = topk_idx[:, :k]
            match = topk.eq(target_col)
            hit = match.any(dim=1)
            hits[k] += hit.float().sum().item()

            pos = match.float().argmax(dim=1) + 1
            ndcg = hit.float() / torch.log2(pos.float() + 1.0)
            ndcgs[k] += ndcg.sum().item()

        total += batch_size

    result = {}
    for k in ks:
        hr = hits[k] / max(total, 1)
        ndcg = ndcgs[k] / max(total, 1)
        result[f"HR@{k}"] = hr
        result[f"Recall@{k}"] = hr
        result[f"NDCG@{k}"] = ndcg
    return result


def train(args) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = read_json(Path(args.dataset_dir) / "sequences.json")
    stats = read_json(Path(args.dataset_dir) / "stats.json")
    semantic_obj = read_json(args.semantic_ids)
    num_items = int(stats["num_items"])
    semantic_table, item_semantic_ids, num_semantic_tokens = build_semantic_table(semantic_obj, num_items)

    train_data = NextItemDataset(sequences, args.max_len, "train")
    valid_data = NextItemDataset(sequences, args.max_len, "valid")
    test_data = NextItemDataset(sequences, args.max_len, "test")
    sampler = CandidateSampler(
        num_items,
        item_semantic_ids,
        args.prefix_level,
        args.hard_neg_mode,
        args.min_overlap_slots,
        args.num_random_neg,
        args.num_hard_neg,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_train(b, sampler),
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_full_eval,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_full_eval,
    )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = QSDRec(
        num_items=num_items,
        num_semantic_tokens=num_semantic_tokens,
        semantic_id_table=semantic_table,
        dim=args.dim,
        max_len=args.max_len,
        num_interests=args.num_interests,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        interest_router=args.interest_router,
        prefix_level=args.prefix_level,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_metric = -1.0
    best_path = output_dir / "best.pt"
    history = []
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for seq, candidates in train_loader:
            seq = seq.to(device)
            if args.train_objective == "full_softmax":
                targets = candidates[:, 0].to(device)
                loss = full_softmax_loss(
                    model=model,
                    seq=seq,
                    targets=targets,
                    num_items=num_items,
                    sem_weight=args.sem_weight,
                    candidate_chunk_size=args.train_candidate_chunk_size,
                )
                if args.div_weight > 0:
                    h_id, _ = model.encoder(seq)
                    queries = model.user_queries(seq, h_id)
                    loss = loss + args.div_weight * model.diversity_loss(queries)
            else:
                candidates = candidates.to(device)
                out = model(seq, candidates, sem_weight=args.sem_weight)
                rec_loss = cross_entropy_first_positive(out["score"])
                dis_loss = cross_entropy_first_positive(out["sem_score"])
                div_loss = model.diversity_loss(out["queries"])
                loss = rec_loss + args.dis_weight * dis_loss + args.div_weight * div_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            total_loss += loss.item()
            steps += 1
        valid = evaluate_full_ranking(
            model=model,
            loader=valid_loader,
            device=device,
            num_items=num_items,
            sem_weight=args.sem_weight,
            ks=(5, 10, 20),
            batch_eval_size=args.eval_batch_eval_size,
        )
        row = {"epoch": epoch, "loss": total_loss / max(steps, 1), **valid}
        history.append(row)
        print(row)
        metric = valid.get("NDCG@10", 0.0)
        if metric > best_metric:
            best_metric = metric
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.early_stop_patience:
                print(
                    {
                        "early_stop": True,
                        "epoch": epoch,
                        "best_valid_NDCG@10": best_metric,
                        "patience": args.early_stop_patience,
                    }
                )
                break

    if best_path.exists():
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model"])
    test = evaluate_full_ranking(
        model=model,
        loader=test_loader,
        device=device,
        num_items=num_items,
        sem_weight=args.sem_weight,
        ks=(5, 10, 20),
        batch_eval_size=args.eval_batch_eval_size,
    )
    result = {
        "test": test,
        "best_valid_NDCG@10": best_metric,
        "args": vars(args),
    }
    write_json(output_dir / "history.json", history)
    write_json(output_dir / "test_metrics.json", result)
    print(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--semantic-ids", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--num-interests", type=int, default=4)
    parser.add_argument("--interest-router", choices=["semantic", "prefix"], default="semantic")
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-random-neg", type=int, default=100)
    parser.add_argument("--num-hard-neg", type=int, default=20)
    parser.add_argument("--hard-neg-mode", choices=["prefix", "overlap"], default="prefix")
    parser.add_argument("--min-overlap-slots", type=int, default=2)
    parser.add_argument("--train-objective", choices=["sampled", "full_softmax"], default="sampled")
    parser.add_argument("--train-candidate-chunk-size", type=int, default=4096)
    parser.add_argument("--eval-batch-eval-size", type=int, default=1024)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--prefix-level", type=int, default=2)
    parser.add_argument("--sem-weight", type=float, default=1.0)
    parser.add_argument("--dis-weight", type=float, default=0.2)
    parser.add_argument("--div-weight", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
