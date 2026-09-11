"""Re-evaluate a saved CCSR checkpoint without retraining."""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from LoCoRec.locorec.data import NextItemDataset, collate_eval
from LoCoRec.locorec.io import read_json, write_json
from CCSR.model import CCSR
from CCSR.trainer import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-dir", help="Defaults to the training dataset")
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--candidate-chunk-size", type=int, default=2048)
    parser.add_argument("--group-labels")
    args = parser.parse_args()
    if args.batch_size < 1 or args.candidate_chunk_size < 1:
        raise ValueError("Batch and chunk sizes must be positive.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg, state = checkpoint["args"], checkpoint["model"]
    table = state["item_encoder.hard_sid_table"]
    n = table.size(0) - 1
    tokens = state["item_encoder.semantic_basis_embedding.weight"].size(0) - 1
    model = CCSR(n, tokens, table, **{key: cfg[key] for key in
                 ("dim", "max_len", "num_heads", "num_layers", "dropout")})
    model.load_state_dict(state)
    device = torch.device(args.device)
    model.to(device)
    dataset_dir = Path(args.dataset_dir or cfg["dataset_dir"])
    if int(read_json(dataset_dir / "stats.json")["num_items"]) != n:
        raise ValueError("Evaluation catalog size differs from the checkpoint.")
    dataset = NextItemDataset(read_json(dataset_dir / "sequences.json"), cfg["max_len"], args.split)
    labels = read_json(args.group_labels)[args.split] if args.group_labels else None
    if labels is not None and (len(labels) != len(dataset) or any(not isinstance(v, str) for v in labels)):
        raise ValueError("Group labels must be strings aligned with evaluation samples.")
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_eval)
    result = evaluate(model, loader, device, n, args.candidate_chunk_size, labels, cfg.get("fixed_resolution"))
    write_json(args.output, {"checkpoint": args.checkpoint, "split": args.split, **result})
    print(result)


if __name__ == "__main__":
    main()
