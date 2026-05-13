"""
统计语义ID完全相同的物品，并生成对应的原始meta数据json文件。

用法:
    python scripts/analyze_semantic_id_collisions.py --dataset office
    python scripts/analyze_semantic_id_collisions.py --dataset beauty
    python scripts/analyze_semantic_id_collisions.py --dataset toys
    python scripts/analyze_semantic_id_collisions.py --dataset games
    python scripts/analyze_semantic_id_collisions.py --dataset all   # 遍历所有数据集
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsdrec.io_utils import read_json, write_json


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def analyze_dataset(dataset_name: str) -> None:
    dataset_dir = get_project_root() / "runs" / dataset_name

    semantic_id_path = dataset_dir / "semantic_ids_rq.json"
    item_meta_path = dataset_dir / "item_meta.json"
    output_dir = dataset_dir / "collision_meta"

    if not semantic_id_path.exists():
        print(f"[{dataset_name}] 语义ID文件不存在: {semantic_id_path}")
        return
    if not item_meta_path.exists():
        print(f"[{dataset_name}] 物品meta文件不存在: {item_meta_path}")
        return

    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*60}")

    sem_ids = read_json(semantic_id_path)["semantic_ids"]
    item_meta = read_json(item_meta_path)

    groups: dict[tuple, list] = defaultdict(list)
    for item_id, sid in sem_ids.items():
        groups[tuple(sid)].append(item_id)

    collision_groups = {k: v for k, v in groups.items() if len(v) > 1}
    sorted_groups = sorted(collision_groups.items(), key=lambda x: len(x[1]), reverse=True)

    print(f"总物品数       : {len(sem_ids)}")
    print(f"唯一语义ID组数 : {len(groups)}")
    print(f"存在碰撞的组数 : {len(collision_groups)}")
    print(f"碰撞物品总数   : {sum(len(v) for v in collision_groups.values())}")
    print(f"碰撞率         : {sum(len(v) for v in collision_groups.values()) / max(len(sem_ids), 1):.4%}")
    print()

    if not sorted_groups:
        print("  无碰撞")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'组序号':<6} {'语义ID':<40} {'物品数':<8} {'物品ID列表'}")
    print("-" * 120)

    summary_report = []

    for idx, (sid_tuple, item_ids) in enumerate(sorted_groups, 1):
        group_meta = {item_id: item_meta.get(item_id, {}) for item_id in item_ids}
        group_file = output_dir / f"collision_group_{idx:04d}.json"
        write_json(group_file, group_meta)

        display_ids = item_ids[:10]
        extra = f" ... (+{len(item_ids)-10})" if len(item_ids) > 10 else ""
        titles = [item_meta.get(i, {}).get("title", "N/A") for i in item_ids]

        print(f"{idx:<6} {str(list(sid_tuple)):<40} {len(item_ids):<8} {display_ids}{extra}")

        summary_report.append(
            {
                "group_index": idx,
                "semantic_id": list(sid_tuple),
                "size": len(item_ids),
                "item_ids": item_ids,
                "all_same_title": len(set(titles)) == 1,
                "meta_file": str(group_file.name),
            }
        )

    write_json(output_dir / "collision_summary.json", summary_report)
    print(f"\n输出目录: {output_dir}")
    print(f"汇总报告: {output_dir / 'collision_summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="统计语义ID碰撞并生成meta文件")
    parser.add_argument(
        "--dataset",
        "-d",
        default="office",
        help="数据集名称，如 office/beauty/toys/games（默认: office）",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="遍历 runs/ 目录下所有子目录",
    )
    args = parser.parse_args()

    project_root = get_project_root()
    runs_dir = project_root / "runs"

    if args.all:
        datasets = sorted(d.name for d in runs_dir.iterdir() if d.is_dir())
        print(f"发现数据集: {datasets}")
        for ds in datasets:
            analyze_dataset(ds)
    else:
        analyze_dataset(args.dataset)


if __name__ == "__main__":
    main()
