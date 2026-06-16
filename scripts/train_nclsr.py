#!/usr/bin/env python3
import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsdrec.model import SASRecDynamicEncoder
from qsdrec.train import NextItemDataset, collate_full_eval, write_json


def read_json(path: str | Path) -> Any:
    with Path(path).open("rt", encoding="utf-8") as f:
        return json.load(f)


def load_ordered_embeddings(
    embeddings_path: str | Path,
    item_ids_path: str | Path,
    num_items: int,
) -> torch.Tensor:
    embeddings = np.load(embeddings_path).astype("float32")
    item_ids = [int(x) for x in read_json(item_ids_path)]
    if embeddings.shape[0] != len(item_ids):
        raise ValueError("Embedding rows and item id count differ.")
    if set(item_ids) != set(range(1, num_items + 1)):
        raise ValueError("Embedding item ids must cover each internal item id exactly once.")
    row_for_item = {item: row for row, item in enumerate(item_ids)}
    ordered = embeddings[[row_for_item[item] for item in range(1, num_items + 1)]]
    ordered = ordered / np.maximum(np.linalg.norm(ordered, axis=1, keepdims=True), 1e-12)
    padded = np.zeros((num_items + 1, ordered.shape[1]), dtype="float32")
    padded[1:] = ordered
    return torch.from_numpy(padded)


def build_synonym_table(
    item_embeddings: torch.Tensor,
    top_k: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    emb = torch.nn.functional.normalize(item_embeddings[1:].float(), dim=-1)
    num_items = emb.size(0)
    k = min(max(int(top_k), 1) + 1, num_items)
    ids = torch.zeros(num_items + 1, k - 1, dtype=torch.long)
    sims = torch.zeros(num_items + 1, k - 1, dtype=torch.float)
    start = time.perf_counter()
    chunk_size = max(int(chunk_size), 1)
    for begin in range(0, num_items, chunk_size):
        end = min(begin + chunk_size, num_items)
        score = emb[begin:end] @ emb.T
        local_k = min(k, score.size(1))
        vals, idx = score.topk(local_k, dim=1)
        for row in range(end - begin):
            item = begin + row + 1
            keep = idx[row] + 1
            keep_vals = vals[row]
            mask = keep.ne(item)
            keep = keep[mask][: k - 1]
            keep_vals = keep_vals[mask][: k - 1]
            ids[item, : keep.numel()] = keep.cpu()
            sims[item, : keep_vals.numel()] = keep_vals.cpu()
    return ids, sims, {
        "num_items": num_items,
        "embedding_dim": int(emb.size(1)),
        "top_k": int(k - 1),
        "elapsed_seconds": time.perf_counter() - start,
    }


class NCLSR(nn.Module):
    def __init__(
        self,
        item_embeddings: torch.Tensor,
        synonym_ids: torch.Tensor,
        synonym_sims: torch.Tensor,
        dim: int,
        max_len: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        dp_epsilon: float,
        score_temperature: float,
        matrix_log_order: int,
        mce_mu: float,
        mce_lambda: float,
        dp_output: str,
        exclude_latest_replace: bool,
    ) -> None:
        super().__init__()
        self.register_buffer("item_embeddings", item_embeddings.float())
        self.register_buffer("synonym_ids", synonym_ids.long())
        self.register_buffer("synonym_sims", synonym_sims.float())
        emb_dim = item_embeddings.size(1)
        if dim == emb_dim:
            self.input_proj = nn.Identity()
            self.target_proj = nn.Identity()
        else:
            self.input_proj = nn.Linear(emb_dim, dim)
            self.target_proj = nn.Linear(emb_dim, dim)
            self.target_proj.load_state_dict(self.input_proj.state_dict())
            for parameter in self.target_proj.parameters():
                parameter.requires_grad = False
        self.encoder = SASRecDynamicEncoder(dim, max_len, num_heads, num_layers, dropout)
        self.predictor = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
        )
        self.out_norm = nn.LayerNorm(dim, eps=1e-8)
        self.dropout = nn.Dropout(dropout)
        self.dp_epsilon = float(dp_epsilon)
        self.score_temperature = float(score_temperature)
        self.matrix_log_order = int(matrix_log_order)
        self.mce_mu = float(mce_mu)
        self.mce_lambda = float(mce_lambda)
        self.dp_output = dp_output
        self.exclude_latest_replace = exclude_latest_replace

    def item_raw(self, items: torch.Tensor) -> torch.Tensor:
        return self.item_embeddings[items]

    def project_raw(self, raw: torch.Tensor, item_mask: torch.Tensor) -> torch.Tensor:
        projected = self.out_norm(self.input_proj(raw))
        return self.dropout(projected) * item_mask.unsqueeze(-1)

    def encode_online_from_raw(self, seq: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        item_repr = self.project_raw(raw, seq.ne(0))
        user_repr, _ = self.encoder(seq, item_repr)
        user_repr = self.predictor(user_repr)
        return torch.nn.functional.normalize(user_repr, dim=-1)

    def encode_target_from_raw(self, seq: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        mask = seq.ne(0).float().unsqueeze(-1)
        profile = (raw * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        with torch.no_grad():
            profile = self.target_proj(profile)
            profile = self.out_norm(profile)
            profile = torch.nn.functional.normalize(profile, dim=-1)
        return profile

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        return self.encode_online_from_raw(seq, self.item_raw(seq))

    def candidate_repr(self, candidates: torch.Tensor) -> torch.Tensor:
        raw = self.item_raw(candidates)
        projected = self.out_norm(self.input_proj(raw))
        return torch.nn.functional.normalize(projected, dim=-1)

    def forward(self, seq: torch.Tensor, candidates: torch.Tensor) -> dict[str, torch.Tensor]:
        user_repr = self.encode(seq)
        cand_repr = self.candidate_repr(candidates)
        score = torch.einsum("bd,bcd->bc", user_repr, cand_repr) / max(self.score_temperature, 1e-6)
        return {"score": score, "user_repr": user_repr}

    def augmented_raw(self, seq: torch.Tensor, replace_count: int) -> torch.Tensor:
        raw = self.item_raw(seq).clone()
        valid = seq.gt(0)
        if replace_count <= 0 or not valid.any():
            return raw
        batch, length = seq.shape
        for row in range(batch):
            positions = valid[row].nonzero(as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            if self.exclude_latest_replace and positions.numel() > 1:
                positions = positions[:-1]
            if positions.numel() == 0:
                continue
            count = min(int(replace_count), int(positions.numel()))
            selected = positions[torch.randperm(positions.numel(), device=seq.device)[:count]]
            items = seq[row, selected]
            syn = self.synonym_ids[items]
            sims = self.synonym_sims[items]
            syn_mask = syn.gt(0)
            utility = torch.exp(sims)
            sensitivity = math.e - (1.0 / math.e)
            logits = self.dp_epsilon * utility / (2.0 * sensitivity)
            logits = logits.masked_fill(~syn_mask, float("-inf"))
            no_syn = ~syn_mask.any(dim=1)
            if no_syn.any():
                logits[no_syn] = 0.0
            weights = torch.softmax(logits, dim=1) * syn_mask.float()
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
            expected = (self.item_embeddings[syn] * weights.unsqueeze(-1)).sum(dim=1)
            if self.dp_output == "expected":
                raw[row, selected] = expected
            elif self.dp_output == "nearest":
                expected = torch.nn.functional.normalize(expected, dim=-1)
                all_emb = torch.nn.functional.normalize(self.item_embeddings[1:], dim=-1)
                scores = expected @ all_emb.T
                original = items.unsqueeze(1)
                nearest = scores.topk(k=min(2, all_emb.size(0)), dim=1).indices + 1
                replacement = nearest[:, 0]
                if nearest.size(1) > 1:
                    replacement = torch.where(replacement.eq(items), nearest[:, 1], replacement)
                raw[row, selected] = self.item_embeddings[replacement]
            else:
                raise ValueError(f"Unsupported dp_output: {self.dp_output}")
        return raw

    def centering_matrix(self, m: int, device: torch.device) -> torch.Tensor:
        eye = torch.eye(m, device=device)
        ones = torch.ones((m, 1), device=device)
        return eye - (ones @ ones.T) / max(m, 1)

    def matrix_log(self, matrix: torch.Tensor) -> torch.Tensor:
        n = matrix.size(0)
        q = matrix - torch.eye(n, device=matrix.device)
        cur = q
        out = torch.zeros_like(q)
        for order in range(1, self.matrix_log_order + 1):
            if order % 2 == 1:
                out = out + cur / float(order)
            else:
                out = out - cur / float(order)
            cur = cur @ q
        return out

    def alignment_mce(self, p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        m, n = z.shape
        center = self.centering_matrix(m, z.device).detach()
        eye = torch.eye(n, device=z.device)
        cov_p = (p.T @ center @ p) / max(m, 1) + self.mce_mu * eye
        cov_z = (z.T @ center @ z) / max(m, 1) + self.mce_mu * eye
        return torch.trace(-cov_p @ self.matrix_log(cov_z))

    def uniformity_mce(self, p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        m, n = z.shape
        center = self.centering_matrix(m, z.device).detach()
        target = self.mce_lambda * torch.eye(n, device=z.device)
        cross_cov = (p.T @ center @ z) / max(m, 1) + self.mce_mu * torch.eye(n, device=z.device)
        return torch.trace(-target @ self.matrix_log(cross_cov))

    def ncl_losses(
        self,
        seq: torch.Tensor,
        replace_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_raw = self.item_raw(seq)
        aug_raw = self.augmented_raw(seq, replace_count)
        p_base = self.encode_target_from_raw(seq, base_raw)
        p_aug = self.encode_target_from_raw(seq, aug_raw)
        z_base = self.encode_online_from_raw(seq, base_raw)
        z_aug = self.encode_online_from_raw(seq, aug_raw)

        align = self.alignment_mce(p_base, z_aug) + self.alignment_mce(p_aug, z_base)
        uniform = self.uniformity_mce(p_base, z_aug) + self.uniformity_mce(p_aug, z_base)
        return uniform, align


def full_softmax_loss(
    model: NCLSR,
    seq: torch.Tensor,
    targets: torch.Tensor,
    num_items: int,
    chunk_size: int,
    mask_seen_items: bool,
) -> torch.Tensor:
    batch_size = seq.size(0)
    all_items = torch.arange(1, num_items + 1, dtype=torch.long, device=seq.device)
    chunks = []
    for start in range(0, num_items, chunk_size):
        candidates = all_items[start : start + chunk_size].unsqueeze(0).expand(batch_size, -1)
        chunks.append(model(seq, candidates)["score"])
    scores = torch.cat(chunks, dim=1)
    if mask_seen_items:
        rows, cols = seq.gt(0).nonzero(as_tuple=True)
        scores[rows, seq[rows, cols] - 1] = float("-inf")
    return torch.nn.functional.cross_entropy(scores, targets - 1)


def sampled_loss(model: NCLSR, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    score = model(seq, candidates)["score"]
    labels = torch.zeros(score.size(0), dtype=torch.long, device=score.device)
    return torch.nn.functional.cross_entropy(score, labels)


@torch.no_grad()
def evaluate_full_ranking(
    model: NCLSR,
    loader: DataLoader,
    device: torch.device,
    num_items: int,
    batch_eval_size: int,
    cutoffs: Sequence[int] = (5, 10, 20),
    mask_seen_items: bool = True,
) -> dict[str, float]:
    model.eval()
    hits = {k: 0.0 for k in cutoffs}
    ndcgs = {k: 0.0 for k in cutoffs}
    mrrs = {k: 0.0 for k in cutoffs}
    total = 0
    all_items = torch.arange(1, num_items + 1, dtype=torch.long, device=device)
    max_k = max(cutoffs)
    for seq, targets in loader:
        seq = seq.to(device)
        targets = targets.to(device)
        chunks = []
        for start in range(0, num_items, batch_eval_size):
            cand = all_items[start : start + batch_eval_size].unsqueeze(0).expand(seq.size(0), -1)
            chunks.append(model(seq, cand)["score"])
        scores = torch.cat(chunks, dim=1)
        if mask_seen_items:
            rows, cols = seq.gt(0).nonzero(as_tuple=True)
            scores[rows, seq[rows, cols] - 1] = float("-inf")
        top = scores.topk(max_k, dim=1).indices + 1
        match_target = targets.unsqueeze(1)
        for k in cutoffs:
            match = top[:, :k].eq(match_target)
            hit = match.any(dim=1)
            rank = match.float().argmax(dim=1) + 1
            hits[k] += hit.float().sum().item()
            ndcgs[k] += (hit.float() / torch.log2(rank.float() + 1.0)).sum().item()
            mrrs[k] += (hit.float() / rank.float()).sum().item()
        total += seq.size(0)
    out = {}
    for k in cutoffs:
        out[f"HR@{k}"] = hits[k] / max(total, 1)
        out[f"Recall@{k}"] = out[f"HR@{k}"]
        out[f"NDCG@{k}"] = ndcgs[k] / max(total, 1)
        out[f"MRR@{k}"] = mrrs[k] / max(total, 1)
    return out


class RandomSampler:
    def __init__(self, num_items: int, num_negatives: int) -> None:
        self.num_items = int(num_items)
        self.num_negatives = int(num_negatives)

    def sample(self, target: int, seen: set[int]) -> list[int]:
        out = [int(target)]
        used = {int(target), *seen}
        while len(out) < self.num_negatives + 1:
            item = random.randint(1, self.num_items)
            if item not in used:
                out.append(item)
                used.add(item)
        return out


def collate_sampled(batch, sampler: RandomSampler):
    seqs, targets = zip(*batch)
    seq_tensor = torch.stack(seqs)
    cands = []
    for seq, target in zip(seqs, targets):
        seen = {int(x) for x in seq.tolist() if int(x) > 0}
        cands.append(sampler.sample(int(target), seen))
    return seq_tensor, torch.tensor(cands, dtype=torch.long)


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequences = read_json(Path(args.dataset_dir) / "sequences.json")
    stats = read_json(Path(args.dataset_dir) / "stats.json")
    num_items = int(stats["num_items"])
    item_embeddings = load_ordered_embeddings(args.item_embeddings, args.embedding_item_ids, num_items)
    model_dim = int(args.dim) if int(args.dim) > 0 else int(item_embeddings.shape[1])
    synonym_ids, synonym_sims, synonym_report = build_synonym_table(
        item_embeddings=item_embeddings,
        top_k=args.synonym_top_k,
        chunk_size=args.synonym_chunk_size,
    )
    write_json(output_dir / "synonym_preprocess.json", synonym_report)

    train_data = NextItemDataset(sequences, args.max_len, "train")
    valid_data = NextItemDataset(sequences, args.max_len, "valid")
    test_data = NextItemDataset(sequences, args.max_len, "test")
    sampler = RandomSampler(num_items, args.num_random_negatives)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_full_eval if args.train_objective == "full_softmax" else lambda b: collate_sampled(b, sampler),
    )
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_full_eval)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_full_eval)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = NCLSR(
        item_embeddings=item_embeddings,
        synonym_ids=synonym_ids,
        synonym_sims=synonym_sims,
        dim=model_dim,
        max_len=args.max_len,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        dp_epsilon=args.dp_epsilon,
        score_temperature=args.score_temperature,
        matrix_log_order=args.mce_order,
        mce_mu=args.mce_mu,
        mce_lambda=args.mce_lambda,
        dp_output=args.dp_output,
        exclude_latest_replace=not args.replace_latest,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_metric = -1.0
    bad_epochs = 0
    best_path = output_dir / "best.pt"
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        rec_total = 0.0
        uniform_total = 0.0
        align_total = 0.0
        steps = 0
        for seq, payload in train_loader:
            seq = seq.to(device)
            if args.train_objective == "full_softmax":
                targets = payload.to(device)
                rec = full_softmax_loss(
                    model=model,
                    seq=seq,
                    targets=targets,
                    num_items=num_items,
                    chunk_size=args.train_candidate_chunk_size,
                    mask_seen_items=not args.keep_seen_items,
                )
            else:
                candidates = payload.to(device)
                rec = sampled_loss(model, seq, candidates)
            uniform, align = model.ncl_losses(
                seq=seq,
                replace_count=args.replace_count,
            )
            loss = rec + args.uniform_weight * uniform + args.align_weight * align
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total += loss.item()
            rec_total += rec.item()
            uniform_total += uniform.item()
            align_total += align.item()
            steps += 1

        valid = evaluate_full_ranking(
            model=model,
            loader=valid_loader,
            device=device,
            num_items=num_items,
            batch_eval_size=args.eval_batch_eval_size,
            mask_seen_items=not args.keep_seen_items,
        )
        row = {
            "epoch": epoch,
            "loss": total / max(steps, 1),
            "rec_loss": rec_total / max(steps, 1),
            "uniform_loss": uniform_total / max(steps, 1),
            "align_loss": align_total / max(steps, 1),
            **valid,
        }
        history.append(row)
        print(row)
        metric = float(valid.get(args.early_stop_metric, 0.0))
        if metric > best_metric:
            best_metric = metric
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.early_stop_patience:
                break

    if best_path.exists():
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model"])
    test = evaluate_full_ranking(
        model=model,
        loader=test_loader,
        device=device,
        num_items=num_items,
        batch_eval_size=args.eval_batch_eval_size,
        mask_seen_items=not args.keep_seen_items,
    )
    result = {
        "test": test,
        "best_valid_metric": best_metric,
        "early_stop_metric": args.early_stop_metric,
        "best_valid_NDCG@10": max((row.get("NDCG@10", 0.0) for row in history), default=0.0),
        "args": vars(args),
    }
    write_json(output_dir / "history.json", history)
    write_json(output_dir / "test_metrics.json", result)
    print(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NCL-SR under the local leave-one-out protocol.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--item-embeddings", required=True)
    parser.add_argument("--embedding-item-ids", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-metric", choices=["NDCG@10", "MRR@10"], default="NDCG@10")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--dim", type=int, default=0, help="Model dimension. 0 uses the BGE embedding dimension.")
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--train-objective", choices=["sampled", "full_softmax"], default="sampled")
    parser.add_argument("--num-random-negatives", type=int, default=100)
    parser.add_argument("--train-candidate-chunk-size", type=int, default=4096)
    parser.add_argument("--eval-batch-eval-size", type=int, default=1024)
    parser.add_argument("--keep-seen-items", action="store_true")
    parser.add_argument("--synonym-top-k", type=int, default=20)
    parser.add_argument("--synonym-chunk-size", type=int, default=512)
    parser.add_argument("--replace-count", type=int, default=3)
    parser.add_argument("--dp-epsilon", type=float, default=1.0)
    parser.add_argument("--uniform-weight", type=float, default=0.05)
    parser.add_argument("--align-weight", type=float, default=0.1)
    parser.add_argument("--mce-order", type=int, default=4)
    parser.add_argument("--mce-mu", type=float, default=1.0)
    parser.add_argument("--mce-lambda", type=float, default=1.0)
    parser.add_argument("--dp-output", choices=["nearest", "expected"], default="nearest")
    parser.add_argument("--replace-latest", action="store_true")
    parser.add_argument("--score-temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=2026)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
