import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def short(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def item_line(item: Dict[str, Any]) -> str:
    return (
        f"{short(item.get('title', ''), 90)} | "
        f"sid={item.get('semantic_id')} | "
        f"train={item.get('train_count')} | "
        f"prefix={item.get('prefix_group_size')} | "
        f"overlap={item.get('overlap_group_size')}"
    )


def top_titles(items: List[Dict[str, Any]], limit: int) -> List[str]:
    return [short(item.get("title", ""), 70) for item in items[:limit]]


def render_case(case: Dict[str, Any], top_n: int) -> List[str]:
    target = case["target"]
    lines = [
        f"- sample_index={case.get('sample_index')} base_rank={case.get('base_rank_at_k')} compare_rank={case.get('compare_rank_at_k')}",
        f"  target: {item_line(target)}",
        "  history_tail:",
    ]
    for item in case.get("history_tail", [])[-5:]:
        lines.append(f"    - {item_line(item)}")
    lines.append("  base_top:")
    for title in top_titles(case.get("base_top_items", []), top_n):
        lines.append(f"    - {title}")
    lines.append("  compare_top:")
    for title in top_titles(case.get("compare_top_items", []), top_n):
        lines.append(f"    - {title}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases-per-type", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    data = json.load(open(args.input, "r", encoding="utf-8"))
    lines = [
        "# Badcase 摘要",
        "",
        f"Input: `{args.input}`",
        f"Top-K: `{data.get('top_k')}`",
        f"Filters: `{json.dumps(data.get('filters', {}), ensure_ascii=False)}`",
        "",
    ]
    for case_type, cases in data["cases"].items():
        lines.extend([f"## {case_type}", "", f"count={len(cases)}", ""])
        for case in cases[: args.cases_per_type]:
            lines.extend(render_case(case, args.top_n))
            lines.append("")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
