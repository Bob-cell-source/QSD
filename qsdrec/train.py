import argparse
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .io_utils import read_json, write_json
from .model import CRSIDRec, GRU4Rec, QSDRec


CRSID_VARIANTS = {"crsid", "crsid_semhub", "crsid_soft", "gru4rec_lcsoft"}
SOFT_SID_VARIANTS = {"crsid_soft", "gru4rec_lcsoft"}


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


def build_semantic_hubness(semantic_table: torch.Tensor, num_semantic_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
    token_counts = torch.zeros(num_semantic_tokens + 1, dtype=torch.float)
    item_sids = semantic_table[1:]
    valid = item_sids.gt(0)
    token_counts.scatter_add_(0, item_sids[valid], torch.ones_like(item_sids[valid], dtype=torch.float))
    token_hub = torch.log1p(token_counts)
    if token_hub.max() > 0:
        token_hub = token_hub / token_hub.max()
    item_hub = token_hub[semantic_table].sum(dim=1)
    if item_hub.max() > 0:
        item_hub = item_hub / item_hub.max()
    item_hub[0] = 0.0
    token_hub[0] = 0.0
    return token_hub, item_hub


def build_soft_semantic_table(
    semantic_table: torch.Tensor,
    item_semantic_ids: Dict[int, List[int]],
    num_items: int,
    top_m: int,
    min_overlap_slots: int,
    min_support: float,
    support_eta: float,
    hard_token_prior: float,
    reliability_floor: float,
    max_neighbors: int,
    lift_kappa: float = 0.0,
    lift_clip: float = 5.0,
    lift_eps: float = 1e-6,
    decouple_reliability: bool = False,
    behavior_neighbors: Dict[int, List[int]] | None = None,
    behavior_neighbor_weight: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    depth = semantic_table.size(1)
    top_m = max(int(top_m), 1)
    inverted: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    slot_token_counts: List[Counter[int]] = [Counter() for _ in range(depth)]
    for item, sid in item_semantic_ids.items():
        for slot, code in enumerate(sid):
            token = int(code)
            inverted[(slot, token)].append(item)
            slot_token_counts[slot][token] += 1
    global_denom = max(len(item_semantic_ids), 1)
    lift_eps = max(float(lift_eps), 1e-12)
    lift_clip = max(float(lift_clip), 0.0)

    def local_lift(slot: int, token: int, support: float) -> float:
        if lift_kappa == 0.0:
            return 0.0
        global_prob = slot_token_counts[slot].get(int(token), 0) / global_denom
        lift = math.log((support + lift_eps) / (global_prob + lift_eps))
        return min(max(lift, 0.0), lift_clip)

    soft_ids = torch.zeros(num_items + 1, depth, top_m, dtype=torch.long)
    soft_weights = torch.zeros(num_items + 1, depth, top_m, dtype=torch.float)
    reliability = torch.zeros(num_items + 1, dtype=torch.float)

    for item in range(1, num_items + 1):
        sid = item_semantic_ids[item]
        overlap_counts: Counter[int] = Counter()
        for slot, code in enumerate(sid):
            overlap_counts.update(inverted[(slot, int(code))])
        neighbors = [
            (neighbor, count)
            for neighbor, count in overlap_counts.items()
            if neighbor != item and count >= min_overlap_slots
        ]
        neighbors.sort(key=lambda row: row[1], reverse=True)
        if max_neighbors > 0:
            neighbors = neighbors[:max_neighbors]
        neighbor_weights: Dict[int, float] = {neighbor: 1.0 for neighbor, _ in neighbors}
        if behavior_neighbors is not None and behavior_neighbor_weight > 0.0:
            for neighbor in behavior_neighbors.get(item, []):
                if neighbor == item:
                    continue
                neighbor_weights[neighbor] = neighbor_weights.get(neighbor, 0.0) + float(behavior_neighbor_weight)
        denom = max(sum(neighbor_weights.values()), 1.0)

        item_reliability = 0.0
        for slot in range(depth):
            hard_token = int(semantic_table[item, slot].item())
            counts: Counter[int] = Counter()
            for neighbor, weight in neighbor_weights.items():
                counts[int(semantic_table[neighbor, slot].item())] += float(weight)
            candidates: Dict[int, float] = {
                token: count
                for token, count in counts.items()
                if token > 0 and (count / denom) >= min_support
            }
            candidates[hard_token] = candidates.get(hard_token, 0) + max(1.0, hard_token_prior * denom)

            scored = []
            for token, weight_count in candidates.items():
                support = counts.get(int(token), 0) / denom
                lift = local_lift(slot, int(token), support)
                score = max(float(weight_count), 1e-6) ** support_eta
                if lift_kappa != 0.0:
                    score *= math.exp(float(lift_kappa) * lift)
                scored.append((int(token), float(weight_count), float(score)))
            ranked = sorted(scored, key=lambda row: (row[2], row[0] == hard_token), reverse=True)[:top_m]
            scores = torch.tensor([score for _, _, score in ranked], dtype=torch.float)
            weights = scores / scores.sum().clamp_min(1e-6)

            rel_weights = weights.tolist()
            if decouple_reliability:
                rel_scores = []
                for token, _, _ in ranked:
                    support = counts.get(int(token), 0) / denom
                    if support <= 0.0:
                        rel_scores.append(0.0)
                        continue
                    lift = local_lift(slot, int(token), support)
                    rel_score = max(float(counts.get(int(token), 0)), 1e-6) ** support_eta
                    if lift_kappa != 0.0:
                        rel_score *= math.exp(float(lift_kappa) * lift)
                    rel_scores.append(float(rel_score))
                rel_total = sum(rel_scores)
                if rel_total > 0.0:
                    rel_weights = [score / rel_total for score in rel_scores]
                else:
                    rel_weights = [0.0 for _ in rel_scores]

            slot_support = 0.0
            for idx, ((token, _, _), weight) in enumerate(zip(ranked, weights.tolist())):
                soft_ids[item, slot, idx] = int(token)
                soft_weights[item, slot, idx] = float(weight)
            for (token, _, _), rel_weight in zip(ranked, rel_weights):
                slot_support += float(rel_weight) * (counts.get(int(token), 0) / denom)
            item_reliability += slot_support

        reliability[item] = max(float(reliability_floor), item_reliability / max(depth, 1))

    reliability[0] = 0.0
    return soft_ids, soft_weights, reliability.clamp(0.0, 1.0)


def build_log_prior(table: torch.Tensor, num_tokens: int, alpha: float = 1.0) -> torch.Tensor:
    token_counts = torch.zeros(num_tokens + 1, dtype=torch.float)
    item_sids = table[1:]
    valid = item_sids.gt(0)
    token_counts.scatter_add_(0, item_sids[valid], torch.ones_like(item_sids[valid], dtype=torch.float))
    denom = token_counts[1:].sum() + alpha * max(num_tokens, 1)
    log_prior = torch.log((token_counts + alpha) / denom.clamp_min(1.0))
    log_prior[0] = 0.0
    return log_prior


def load_mini_cluster_table(
    mini_cluster_path: str | None,
    num_items: int,
    fallback_table: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not mini_cluster_path:
        num_tokens = int(fallback_table.max().item())
        return fallback_table.clone(), build_log_prior(fallback_table, num_tokens)

    obj = read_json(mini_cluster_path)
    raw = {int(k): list(map(int, v)) for k, v in obj["mini_cluster_ids"].items()}
    depth = fallback_table.size(1)
    num_tokens = int(obj.get("num_mini_clusters", 0))
    table = torch.zeros(num_items + 1, depth, dtype=torch.long)
    for item in range(1, num_items + 1):
        codes = raw.get(item)
        if codes is None:
            table[item] = fallback_table[item]
        else:
            table[item] = torch.tensor(codes, dtype=torch.long)
            num_tokens = max(num_tokens, max(codes) if codes else 0)
    return table, build_log_prior(table, num_tokens)


def build_behavior_neighbors(
    sequences: List[Dict],
    num_items: int,
    window_size: int,
    min_count: int,
    max_neighbors: int,
) -> Dict[int, List[int]]:
    if window_size <= 0 or max_neighbors <= 0:
        return {}

    edge_counts: Dict[int, Counter[int]] = defaultdict(Counter)
    for row in sequences:
        items = [int(x) for x in row["items"][:-2] if 0 < int(x) <= num_items]
        for idx, item in enumerate(items):
            start = max(0, idx - window_size)
            end = min(len(items), idx + window_size + 1)
            for pos in range(start, end):
                if pos == idx:
                    continue
                neighbor = items[pos]
                if neighbor != item:
                    edge_counts[item][neighbor] += 1

    neighbors: Dict[int, List[int]] = {}
    min_count = max(int(min_count), 1)
    for item, counts in edge_counts.items():
        ranked = [
            neighbor
            for neighbor, count in sorted(counts.items(), key=lambda row: (row[1], row[0]), reverse=True)
            if count >= min_count
        ]
        if ranked:
            neighbors[item] = ranked[:max_neighbors]
    return neighbors


def build_train_item_frequency(sequences: List[Dict], num_items: int) -> torch.Tensor:
    freq = torch.zeros(num_items + 1, dtype=torch.float)
    for row in sequences:
        for item in row["items"][:-2]:
            if 0 < int(item) <= num_items:
                freq[int(item)] += 1.0
    return freq


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
    item_frequency = build_train_item_frequency(sequences, num_items)
    semantic_token_hubness, semantic_item_hubness = build_semantic_hubness(semantic_table, num_semantic_tokens)
    soft_semantic_table = None
    soft_semantic_weight = None
    semantic_reliability = None
    behavior_neighbors = None
    if args.model_variant in SOFT_SID_VARIANTS:
        if args.cr_soft_behavior_weight > 0.0:
            behavior_neighbors = build_behavior_neighbors(
                sequences=sequences,
                num_items=num_items,
                window_size=args.cr_soft_behavior_window,
                min_count=args.cr_soft_behavior_min_count,
                max_neighbors=args.cr_soft_max_behavior_neighbors,
            )
        soft_semantic_table, soft_semantic_weight, semantic_reliability = build_soft_semantic_table(
            semantic_table=semantic_table,
            item_semantic_ids=item_semantic_ids,
            num_items=num_items,
            top_m=args.cr_soft_top_m,
            min_overlap_slots=args.cr_soft_min_overlap_slots,
            min_support=args.cr_soft_min_support,
            support_eta=args.cr_soft_support_eta,
            hard_token_prior=args.cr_soft_hard_token_prior,
            reliability_floor=args.cr_soft_reliability_floor,
            max_neighbors=args.cr_soft_max_neighbors,
            lift_kappa=args.cr_soft_lift_kappa,
            lift_clip=args.cr_soft_lift_clip,
            lift_eps=args.cr_soft_lift_eps,
            decouple_reliability=args.cr_soft_decouple_reliability,
            behavior_neighbors=behavior_neighbors,
            behavior_neighbor_weight=args.cr_soft_behavior_weight,
        )
    semantic_token_log_prior = build_log_prior(semantic_table, num_semantic_tokens)
    mini_cluster_table, mini_cluster_log_prior = load_mini_cluster_table(
        args.mini_clusters,
        num_items,
        semantic_table,
    )

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
    if args.model_variant in CRSID_VARIANTS:
        model = CRSIDRec(
            num_items=num_items,
            num_semantic_tokens=num_semantic_tokens,
            semantic_id_table=semantic_table,
            item_frequency=item_frequency,
            soft_semantic_id_table=soft_semantic_table,
            soft_semantic_id_weight=soft_semantic_weight,
            semantic_reliability=semantic_reliability,
            dim=args.dim,
            max_len=args.max_len,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            tail_tau=args.cr_tail_tau,
            residual_scale=args.cr_residual_scale,
            alpha_mode="semantic_hubness" if args.model_variant == "crsid_semhub" else "item_frequency",
            semantic_token_hubness=semantic_token_hubness,
            hub_alpha_floor=args.cr_hub_alpha_floor,
            hub_alpha_gamma=args.cr_hub_alpha_gamma,
            disable_semantic_basis=args.cr_disable_semantic_basis,
            disable_shared_residual=args.cr_disable_shared_residual,
            disable_private_residual=args.cr_disable_private_residual,
            alpha_override=args.cr_alpha_override,
            alpha_frequency_transform=args.cr_alpha_frequency_transform,
            encoder_type="gru4rec" if args.model_variant == "gru4rec_lcsoft" else "sasrec",
        ).to(device)
    elif args.model_variant == "gru4rec":
        model = GRU4Rec(
            num_items=num_items,
            dim=args.dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
    else:
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
            semantic_token_hubness=semantic_token_hubness,
            semantic_item_hubness=semantic_item_hubness,
            semantic_token_log_prior=semantic_token_log_prior,
            mini_cluster_table=mini_cluster_table,
            mini_cluster_log_prior=mini_cluster_log_prior,
            hub_score_weight=args.hub_score_weight,
            hub_attn_weight=args.hub_attn_weight,
            evidence_gate=args.evidence_gate,
            evidence_floor=args.evidence_floor,
            evidence_recency_weight=args.evidence_recency_weight,
            evidence_hub_weight=args.evidence_hub_weight,
            evidence_cross_weight=args.evidence_cross_weight,
            prior_lift_alpha=args.prior_lift_alpha,
            prior_lift_tau=args.prior_lift_tau,
            prior_lift_eta=args.prior_lift_eta,
            hub_penalty_weight=args.hub_penalty_weight,
            semantic_fusion=args.semantic_fusion,
            fusion_floor=args.fusion_floor,
            contrastive_alpha=args.contrastive_alpha,
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
                if args.model_variant == "qsdrec" and args.div_weight > 0:
                    h_id, _ = model.encoder(seq)
                    queries = model.user_queries(seq, h_id)
                    loss = loss + args.div_weight * model.diversity_loss(queries)
            else:
                candidates = candidates.to(device)
                out = model(seq, candidates, sem_weight=args.sem_weight)
                rec_loss = cross_entropy_first_positive(out["score"])
                if args.model_variant in CRSID_VARIANTS:
                    loss = rec_loss + args.cr_residual_reg * out["residual_l2"]
                elif args.model_variant == "gru4rec":
                    loss = rec_loss
                else:
                    dis_loss = cross_entropy_first_positive(out["sem_score"])
                    div_loss = model.diversity_loss(out["queries"])
                    loss = (
                        rec_loss
                        + args.dis_weight * dis_loss
                        + args.div_weight * div_loss
                        + args.hub_loss_weight * out["hub_loss"]
                    )
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
    if args.skip_test_evaluation:
        test = {}
    else:
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
        "test_evaluated": not args.skip_test_evaluation,
        "args": vars(args),
    }
    write_json(output_dir / "history.json", history)
    write_json(output_dir / "test_metrics.json", result)
    print(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-variant",
        choices=["qsdrec", "crsid", "crsid_semhub", "crsid_soft", "gru4rec", "gru4rec_lcsoft"],
        default="qsdrec",
    )
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
    parser.add_argument("--hub-score-weight", type=float, default=0.0)
    parser.add_argument("--hub-attn-weight", type=float, default=0.0)
    parser.add_argument("--hub-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--evidence-gate",
        choices=[
            "none",
            "history_overlap",
            "reliability",
            "hub_reliability",
            "strength",
            "strength_idf",
            "cross_strength_idf",
            "learnable",
            "prior_lift",
            "mini_lift",
        ],
        default="none",
    )
    parser.add_argument("--mini-clusters", default=None)
    parser.add_argument("--evidence-floor", type=float, default=0.1)
    parser.add_argument("--evidence-recency-weight", type=float, default=0.0)
    parser.add_argument("--evidence-hub-weight", type=float, default=0.0)
    parser.add_argument("--evidence-cross-weight", type=float, default=0.2)
    parser.add_argument("--prior-lift-alpha", type=float, default=0.1)
    parser.add_argument("--prior-lift-tau", type=float, default=1.0)
    parser.add_argument("--prior-lift-eta", type=float, default=1.0)
    parser.add_argument("--hub-penalty-weight", type=float, default=0.0)
    parser.add_argument("--semantic-fusion", choices=["fixed", "evidence_coverage"], default="fixed")
    parser.add_argument("--fusion-floor", type=float, default=0.0)
    parser.add_argument("--contrastive-alpha", type=float, default=0.0)
    parser.add_argument("--cr-tail-tau", type=float, default=20.0)
    parser.add_argument("--cr-residual-scale", type=float, default=1.0)
    parser.add_argument("--cr-residual-reg", type=float, default=0.0)
    parser.add_argument("--cr-hub-alpha-floor", type=float, default=0.05)
    parser.add_argument("--cr-hub-alpha-gamma", type=float, default=1.0)
    parser.add_argument("--cr-disable-semantic-basis", action="store_true")
    parser.add_argument("--cr-disable-shared-residual", action="store_true")
    parser.add_argument("--cr-disable-private-residual", action="store_true")
    parser.add_argument("--cr-alpha-override", type=float, default=None)
    parser.add_argument("--cr-alpha-frequency-transform", choices=["raw", "log"], default="raw")
    parser.add_argument("--cr-soft-top-m", type=int, default=4)
    parser.add_argument("--cr-soft-min-overlap-slots", type=int, default=2)
    parser.add_argument("--cr-soft-min-support", type=float, default=0.05)
    parser.add_argument("--cr-soft-support-eta", type=float, default=1.0)
    parser.add_argument("--cr-soft-hard-token-prior", type=float, default=1.0)
    parser.add_argument("--cr-soft-reliability-floor", type=float, default=0.10)
    parser.add_argument("--cr-soft-max-neighbors", type=int, default=50)
    parser.add_argument("--cr-soft-lift-kappa", type=float, default=0.0)
    parser.add_argument("--cr-soft-lift-clip", type=float, default=5.0)
    parser.add_argument("--cr-soft-lift-eps", type=float, default=1e-6)
    parser.add_argument("--cr-soft-decouple-reliability", action="store_true")
    parser.add_argument("--cr-soft-behavior-weight", type=float, default=0.0)
    parser.add_argument("--cr-soft-behavior-window", type=int, default=5)
    parser.add_argument("--cr-soft-behavior-min-count", type=int, default=2)
    parser.add_argument("--cr-soft-max-behavior-neighbors", type=int, default=50)
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
    parser.add_argument(
        "--skip-test-evaluation",
        action="store_true",
        help="Tune on validation only; leave test metrics empty until the final selected configuration.",
    )
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
